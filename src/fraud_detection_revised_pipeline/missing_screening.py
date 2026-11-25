"""Triagem de colunas antes da remocao de registros incompletos (item G15c).

Remover toda linha com algum faltante, sem olhar as colunas, elimina quase
todo o treino nas bases com colunas muito vazias: na IEEE-CIS ha atributos
ausentes em mais de 90% das transacoes. Por isso o braco
``drop-incomplete-rows`` primeiro remove as colunas cuja taxa de faltantes no
treino passa de um limiar e so entao remove as linhas incompletas.

Nenhum limiar fixo serve a todas as bases (a 40%, a IEEE-CIS-transacional
preserva 56% das linhas mas so 31% das fraudes). O limiar e escolhido por
dobra, sobre o treino: o MAIOR entre 0, 5, ..., 100 que preserva ao menos 75%
das fraudes do treino. "Maior" mantem o maximo de colunas entre as opcoes que
respeitam o piso; o limiar 0 remove todas as colunas com faltantes e sempre
preserva todas as linhas, entao sempre ha solucao.

O modulo nao importa o executor nem o armazenamento: a regra entra na
identidade da configuracao (``prepared_dataset_storage``) e e aplicada na
imputacao (``lr_nb_knn_specific``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


DROP_INCOMPLETE_ROWS_STRATEGY = "drop-incomplete-rows"
DROP_SCREENING_MIN_FRAUD_RETENTION = 0.75
DROP_SCREENING_THRESHOLD_STEP = 5
# Texto gravado no manifesto e na identidade da configuracao.
DROP_SCREENING_RULE = (
    f"min-fraud-retention={DROP_SCREENING_MIN_FRAUD_RETENTION};"
    f"step={DROP_SCREENING_THRESHOLD_STEP}"
)
# Tolerancia para comparar taxas com o limiar sem erro de ponto flutuante.
_RATE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class MissingScreening:
    """Limiar escolhido numa dobra e o efeito da triagem sobre o treino."""

    threshold_pct: int
    screened_out_columns: tuple[str, ...]
    screened_out_missing_rates: dict[str, float]
    rows_before: int
    positives_before: int
    rows_after: int
    positives_after: int
    # Fraudes preservadas em cada limiar candidato: permite verificar depois
    # que o escolhido e o maior admissivel (verificacao P13).
    positives_by_threshold: dict[int, int]
    # Base cujo treino decidiu limiar e linhas. Numa variante de ablacao e a
    # base de origem, com todas as colunas (item G16).
    source_dataset_id: str | None = None

    def to_report(self) -> dict:
        return {
            "rule": DROP_SCREENING_RULE,
            "source_dataset_id": self.source_dataset_id,
            "threshold_pct": self.threshold_pct,
            "screened_out_columns": list(self.screened_out_columns),
            "screened_out_missing_rates": self.screened_out_missing_rates,
            "rows_before": self.rows_before,
            "positives_before": self.positives_before,
            "rows_after": self.rows_after,
            "positives_after": self.positives_after,
            "positives_by_threshold": {
                str(threshold): kept for threshold, kept in self.positives_by_threshold.items()
            },
        }


def select_missing_screening(
    train_data: pd.DataFrame,
    target_column: str,
) -> MissingScreening:
    """Escolhe o limiar da dobra usando apenas o treino.

    Uma linha sobrevive ao limiar t se TODAS as suas colunas ausentes tiverem
    taxa maior que t (isto e, foram removidas). Basta, entao, a menor taxa
    entre as colunas ausentes de cada linha, calculada uma unica vez.
    """
    features = train_data.drop(columns=[target_column]).replace([np.inf, -np.inf], np.nan)
    rates = features.isna().mean()
    # Coluna a coluna, para nao materializar uma matriz linhas x colunas
    # (mais de 1 GB no treino da IEEE-CIS-transacional).
    row_minimum_rate = np.full(len(features), np.inf)
    for column in rates.index[rates > 0]:
        column_missing = features[column].isna().to_numpy()
        row_minimum_rate[column_missing] = np.minimum(
            row_minimum_rate[column_missing], rates[column]
        )
    positives = (train_data[target_column] == 1).to_numpy()
    total_positives = int(positives.sum())

    positives_by_threshold = {
        threshold_pct: int(positives[row_minimum_rate > threshold_pct / 100 + _RATE_TOLERANCE].sum())
        for threshold_pct in range(0, 101, DROP_SCREENING_THRESHOLD_STEP)
    }
    chosen_threshold = max(
        threshold_pct
        for threshold_pct, kept in positives_by_threshold.items()
        if kept >= DROP_SCREENING_MIN_FRAUD_RETENTION * total_positives
    )

    threshold = chosen_threshold / 100 + _RATE_TOLERANCE
    screened = sorted(rates.index[rates > threshold])
    surviving = row_minimum_rate > threshold
    return MissingScreening(
        threshold_pct=chosen_threshold,
        screened_out_columns=tuple(screened),
        screened_out_missing_rates={column: float(rates[column]) for column in screened},
        rows_before=len(train_data),
        positives_before=total_positives,
        rows_after=int(surviving.sum()),
        positives_after=int(positives[surviving].sum()),
        positives_by_threshold=positives_by_threshold,
    )


def surviving_row_mask(
    train_data: pd.DataFrame, target_column: str, screening: MissingScreening
) -> np.ndarray:
    """Linhas do treino sem faltante nas colunas que sobrevivem a triagem."""
    features = train_data.drop(
        columns=[target_column, *screening.screened_out_columns], errors="ignore"
    ).replace([np.inf, -np.inf], np.nan)
    return ~features.isna().any(axis=1).to_numpy()


def decide_fold_removal(
    fold: dict,
    target_column: str,
    ablated_columns: tuple[str, ...] = (),
    source_dataset_id: str | None = None,
) -> dict:
    """Decide limiar e linhas de uma dobra e tira as colunas ablacionadas.

    A decisao usa o treino com TODAS as colunas que ele traz. Numa variante de
    ablacao, a preparacao mantem as colunas ablacionadas ate aqui, de modo que
    variante e base de origem escolhem o mesmo limiar e as mesmas linhas;
    so depois elas saem de todos os splits. Devolve a dobra com
    ``missing_screening`` e ``retained_train_positions`` (posicoes no treino
    da particao), que a imputacao aplica.
    """
    train_data = fold["train_data"]
    screening = select_missing_screening(train_data, target_column)
    positions = np.flatnonzero(surviving_row_mask(train_data, target_column, screening))
    ablated = set(ablated_columns)
    screening = replace(
        screening,
        screened_out_columns=tuple(
            column for column in screening.screened_out_columns if column not in ablated
        ),
        screened_out_missing_rates={
            column: rate
            for column, rate in screening.screened_out_missing_rates.items()
            if column not in ablated
        },
        source_dataset_id=source_dataset_id,
    )
    decided = {**fold, "missing_screening": screening, "retained_train_positions": positions}
    for split_name in ("train_data", "validation_data", "test_data"):
        split_data = fold.get(split_name)
        if split_data is not None and ablated:
            decided[split_name] = split_data.drop(
                columns=[column for column in split_data.columns if column in ablated]
            )
    return decided


def decide_incomplete_row_removal(
    split_information: dict,
    target_column: str,
    ablated_columns: tuple[str, ...] = (),
    source_dataset_id: str | None = None,
) -> dict:
    """Aplica ``decide_fold_removal`` a todas as dobras."""
    return {
        **split_information,
        "test_info": [
            decide_fold_removal(fold, target_column, ablated_columns, source_dataset_id)
            for fold in split_information["test_info"]
        ],
    }


def adopt_paired_removal(fold: dict, paired_fold: dict) -> dict:
    """Aplica a uma dobra a remocao decidida na dobra da base pareada.

    O par da Home Credit tem os mesmos clientes, mas nao as mesmas linhas.
    A decisao (limiar, colunas triadas e linhas) e tomada na completa, e a
    somente-aplicacao mantem no treino so os clientes com ao menos uma linha
    sobrevivente la, sem as colunas triadas que tiver (item G16). Falha se as
    duas dobras nao tiverem os mesmos clientes em cada split.
    """
    units = fold.get("unit_partition_md5") or {}
    paired_units = paired_fold.get("unit_partition_md5") or {}
    if not units or units != paired_units:
        raise ValueError(
            "The paired dataset does not split the same evaluation units; the "
            "removal decision cannot be shared"
        )
    screening = paired_fold["missing_screening"]
    surviving_units = set(
        paired_fold["evaluation_units"]["train_data"]
        .iloc[paired_fold["retained_train_positions"]]
    )
    positions = np.flatnonzero(
        fold["evaluation_units"]["train_data"].isin(surviving_units).to_numpy()
    )
    own_columns = set(fold["train_data"].columns)
    screening = replace(
        screening,
        screened_out_columns=tuple(
            column for column in screening.screened_out_columns if column in own_columns
        ),
        screened_out_missing_rates={
            column: rate
            for column, rate in screening.screened_out_missing_rates.items()
            if column in own_columns
        },
    )
    return {**fold, "missing_screening": screening, "retained_train_positions": positions}


def adopt_paired_incomplete_row_removal(
    split_information: dict, paired_split_information: dict
) -> dict:
    """Aplica ``adopt_paired_removal`` dobra a dobra."""
    folds = split_information["test_info"]
    paired_folds = paired_split_information["test_info"]
    if len(folds) != len(paired_folds):
        raise ValueError(
            f"The paired dataset has {len(paired_folds)} folds; expected {len(folds)}"
        )
    return {
        **split_information,
        "test_info": [
            adopt_paired_removal(fold, paired_fold)
            for fold, paired_fold in zip(folds, paired_folds)
        ],
    }
