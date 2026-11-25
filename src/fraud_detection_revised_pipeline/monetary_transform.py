"""log1p dos valores monetarios (item G14).

``Amount`` (ULB), ``amt`` (Sparkov) e ``TransactionAmt`` (IEEE-CIS) tem
assimetria de 13 a 37. Mesmo padronizada, uma transacao chega a 80-173
desvios-padrao da media e decide sozinha a distancia de KNN, SMOTE e Tomek
Links; no Isolation Forest, que sorteia cada corte entre o minimo e o maximo do
atributo, ela recebe escore alto de anomalia. Acima do p99,9 a ULB e a Sparkov
nao tem nenhuma fraude, entao esse destaque e um erro.

``log1p`` comprime a cauda sem mudar a ordem dos valores e nao tem parametro a
ajustar: aplica-lo antes ou depois da particao da o mesmo resultado. Truncar no
p99,9 foi descartado porque, na Sparkov, juntaria a faixa sem fraude a faixa
com 27,8% de fraude.

Arvores (CatBoost, XGBoost) nao mudam com uma transformacao monotonica, e os
Transformers tem a normalizacao numerica como eixo proprio; por isso so o bloco
classico, o Isolation Forest e o DevNet usam este modulo.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


class NegativeMonetaryValueError(ValueError):
    """Valor monetario negativo: log1p nao esta definido abaixo de -1."""


def log1p_columns(
    dataframe: pd.DataFrame | None,
    monetary_columns: Iterable[str],
    dataset_id: str = "",
) -> pd.DataFrame | None:
    """Aplica ``log1p`` as colunas monetarias presentes, sem alterar o original."""
    if dataframe is None:
        return None
    present = [column for column in monetary_columns if column in dataframe.columns]
    if not present:
        return dataframe
    transformed = dataframe.copy()
    for column in present:
        values = pd.to_numeric(transformed[column], errors="coerce")
        if (values < 0).any():
            origin = f" da base {dataset_id}" if dataset_id else ""
            raise NegativeMonetaryValueError(
                f"A coluna monetaria {column}{origin} tem valores negativos; "
                "log1p exige valores maiores ou iguais a zero"
            )
        transformed[column] = np.log1p(values)
    return transformed


def log1p_data_splits(
    split_information: dict,
    monetary_columns: Iterable[str],
    dataset_id: str = "",
) -> dict:
    """Aplica ``log1p`` a treino, validacao e teste de cada dobra."""
    monetary_columns = tuple(monetary_columns)
    if not monetary_columns:
        return split_information
    transformed_information = {**split_information, "test_info": []}
    for fold in split_information["test_info"]:
        transformed_fold = dict(fold)
        for split_name in ("train_data", "validation_data", "test_data"):
            transformed_fold[split_name] = log1p_columns(
                fold.get(split_name), monetary_columns, dataset_id
            )
        transformed_information["test_info"].append(transformed_fold)
    return transformed_information
