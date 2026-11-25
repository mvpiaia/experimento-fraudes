"""Procedencia das colunas de um arquivo preparado (item G14).

Cada configuracao preparada grava, ao lado de cada CSV, um JSON com as colunas
do arquivo e a historia das demais: o que veio da fonte, o que foi derivado, o
que foi removido e por que (identificador, constante, ablacao), o que recebeu
``log1p`` e o que e categorico. O manifesto guarda so o hash do esquema.

Este modulo monta a parte comum a todos os arquivos da configuracao; as colunas
e o hash de cada arquivo sao acrescentados na gravacao
(``prepared_dataset_storage.write_column_registry``).
"""

from __future__ import annotations

import pandas as pd

from experiment_datasets import DatasetDefinition, identifier_columns_to_drop

CATEGORICAL_DTYPES = ("object", "string", "category")


def categorical_columns_of(frame: pd.DataFrame, target_column: str) -> list[str]:
    """Colunas categoricas pelo tipo, antes de qualquer codificacao."""
    return [
        column
        for column in frame.select_dtypes(include=list(CATEGORICAL_DTYPES)).columns
        if column != target_column
    ]


def build_column_registry(
    definition: DatasetDefinition,
    provenance: dict,
    categorical_columns: list[str],
    log_transformed_columns: list[str],
    pca_applied: bool,
    retained_columns: tuple[str, ...] = (),
) -> dict:
    """Parte do registro comum a todos os arquivos de uma configuracao.

    ``retained_columns`` sao as chaves de sequencia que o TabBERT mantem nos
    arquivos: nao foram removidas, entao nao entram como identificador.
    """
    constant_columns = list(provenance.get("constant_columns", []))
    derived_columns = list(provenance.get("derived_columns", []))
    source_columns = list(provenance.get("source_columns", []))
    loaded = set(source_columns) | set(derived_columns) | set(constant_columns)
    return {
        "dataset_id": definition.dataset_id,
        "source_columns": source_columns,
        "derived_columns": derived_columns,
        "dropped_columns": {
            "identifier": [
                column
                for column in identifier_columns_to_drop(definition)
                if column in loaded and column not in retained_columns
            ],
            "constant": constant_columns,
            "ablation": [
                column for column in definition.ablated_columns if column in loaded
            ],
        },
        # Colunas monetarias transformadas ANTES da imputacao, da padronizacao
        # e da PCA; com PCA elas nao aparecem mais nas colunas do arquivo.
        "log_transformed_columns": list(log_transformed_columns),
        "pca_applied": bool(pca_applied),
        "categorical_columns": list(categorical_columns),
    }
