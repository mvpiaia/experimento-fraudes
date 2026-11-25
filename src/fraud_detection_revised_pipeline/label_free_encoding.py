"""Codificacao numerica sem rotulo para os detectores de anomalia.

As bases nativas guardam as categorias como texto. O CatBoost e o XGBoost as
tratam internamente; o Isolation Forest e o DevNet precisam de numeros. As
codificacoes do bloco classico (estatistica do alvo, WoE) estao fora de
cogitacao: elas constroem o atributo a partir dos rotulos de todo o treino, o
que tornaria o Isolation Forest supervisionado pela porta dos fundos e daria ao
DevNet, que deveria ver so k fraudes rotuladas, a informacao de todas.

Cada categoria vira sua frequencia relativa no treino da dobra. A frequencia e
monotonica com a raridade da categoria, que e exatamente o sinal que um
detector de anomalias deve ver; uma codificacao ordinal imporia uma ordem
arbitraria que o Isolation Forest usaria para cortes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from monetary_transform import log1p_columns


MISSING_CATEGORY_VALUE = "__missing__"
MISSING_INDICATOR_SUFFIX = "__is_missing"
# Categoria que nao ocorreu no treino: frequencia observada zero.
UNSEEN_CATEGORY_FREQUENCY = 0.0


def _category_values(series: pd.Series) -> pd.Series:
    return series.astype("string").fillna(MISSING_CATEGORY_VALUE).astype(str)


@dataclass(frozen=True)
class LabelFreeEncoder:
    """Transformacao ajustada so com os atributos do treino, nunca com o alvo.

    ``impute_missing`` e ``standardize`` existem para o DevNet, cuja rede nao
    aceita ausentes e depende da escala. O Isolation Forest dispensa as duas:
    aceita ausentes.

    Com ``missing_indicators``, cada atributo com ausentes no treino ganha um
    indicador ``<atributo>__is_missing`` (0/1), calculado antes da imputacao,
    como no FT-Transformer. Sem ele, a imputacao pela mediana tornava o braco
    ``native`` (ausentes deixados ao estimador) identico ao ``median`` da
    preparacao. So entram os atributos com ausentes no treino: nos demais o
    indicador seria constante, e os bracos sem ausentes recebem a mesma
    entrada de antes.

    O Isolation Forest NAO e invariante a transformacoes monotonicas por
    atributo: ele sorteia cada ponto de corte uniformemente entre o minimo e o
    maximo observados, de modo que a escala muda como os pontos sao isolados.
    Com o valor bruto, uma compra de 25 mil se isola em poucos cortes e recebe
    escore alto de anomalia, embora as compras acima do p99,9 tenham 0% de
    fraude na ULB e na Sparkov. Por isso as colunas monetarias recebem
    ``log1p`` aqui, no Isolation Forest e no DevNet (item G14).
    """

    feature_columns: tuple[str, ...]
    category_frequencies: dict[str, dict[str, float]]
    fill_values: dict[str, float] | None
    means: dict[str, float] | None
    scales: dict[str, float] | None
    monetary_columns: tuple[str, ...] = ()
    missing_indicator_columns: tuple[str, ...] = ()

    @classmethod
    def fit(
        cls,
        train_features: pd.DataFrame,
        categorical_columns: list[str],
        impute_missing: bool = False,
        standardize: bool = False,
        monetary_columns: tuple[str, ...] = (),
        missing_indicators: bool = False,
    ) -> "LabelFreeEncoder":
        category_frequencies = {
            column: _category_values(train_features[column])
            .value_counts(normalize=True)
            .to_dict()
            for column in categorical_columns
        }
        encoder = cls(
            feature_columns=tuple(train_features.columns),
            category_frequencies=category_frequencies,
            fill_values=None,
            means=None,
            scales=None,
            monetary_columns=tuple(monetary_columns),
        )
        encoded = encoder._encode_categories(train_features)
        missing_indicator_columns = (
            tuple(column for column in encoded.columns if encoded[column].isna().any())
            if missing_indicators else ()
        )

        fill_values = None
        if impute_missing:
            # Mediana do treino; uma coluna toda ausente no treino vira zero.
            fill_values = encoded.median().fillna(0.0).to_dict()
            encoded = encoded.fillna(fill_values)

        means = scales = None
        if standardize:
            means = encoded.mean().fillna(0.0).to_dict()
            deviations = encoded.std(ddof=0).fillna(0.0)
            # Atributo constante no treino: sem escala a corrigir.
            scales = deviations.where(deviations > 0, 1.0).to_dict()

        return cls(
            feature_columns=encoder.feature_columns,
            category_frequencies=category_frequencies,
            fill_values=fill_values,
            means=means,
            scales=scales,
            monetary_columns=encoder.monetary_columns,
            missing_indicator_columns=missing_indicator_columns,
        )

    def _encode_categories(self, features: pd.DataFrame) -> pd.DataFrame:
        encoded = features.reindex(columns=list(self.feature_columns)).copy()
        for column, frequencies in self.category_frequencies.items():
            encoded[column] = (
                _category_values(encoded[column])
                .map(frequencies)
                .fillna(UNSEEN_CATEGORY_FREQUENCY)
            )
        # Infinito vira ausente, como no bloco classico.
        numeric = (
            encoded.apply(pd.to_numeric, errors="coerce")
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        # Antes da imputacao e da padronizacao, que dependem da escala.
        return log1p_columns(numeric, self.monetary_columns)

    def transform(self, features: pd.DataFrame | None) -> pd.DataFrame | None:
        if features is None:
            return None
        encoded = self._encode_categories(features)
        indicators = pd.DataFrame(
            {
                f"{column}{MISSING_INDICATOR_SUFFIX}": encoded[column].isna().astype(float)
                for column in self.missing_indicator_columns
            },
            index=encoded.index,
        )
        if self.fill_values is not None:
            encoded = encoded.fillna(self.fill_values)
        if self.means is not None and self.scales is not None:
            encoded = (encoded - pd.Series(self.means)) / pd.Series(self.scales)
        if self.missing_indicator_columns:
            encoded = pd.concat([encoded, indicators], axis=1)
        return encoded
