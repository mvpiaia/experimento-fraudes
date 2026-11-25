"""Relatorio de preparacao por dobra (item G15 da revisao).

O revisor pediu verificacoes de sanidade relatadas em apendice: faltantes
antes e depois da imputacao, linhas e fraudes antes e depois da remocao,
componentes retidos pela PCA. Nada disso era gravado, e foi essa ausencia que
deixou passar um eixo de imputacao sem efeito.

Cada dobra carrega, na chave ``preparation_report`` do seu dicionario, um
relatorio que a preparacao preenche etapa a etapa. Ele e gravado como
``<configuration_md5>.fold<k>.preparation.json`` ao lado dos CSVs, e o
manifesto guarda o MD5 do arquivo em ``preparation_report_md5``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_partitioning import partition_md5, unit_partition_md5
from prepared_dataset_storage import FIRST_FOLD_INDEX, compute_file_content_md5


PREPARATION_REPORT_KEY = "preparation_report"
PREPARATION_REPORT_FILE_SUFFIX = ".preparation.json"
REPORT_SPLIT_NAMES = {
    "train_data": "train",
    "validation_data": "validation",
    "test_data": "test",
}

# Etapas registradas por split. "after_imputation" mostra, no treino, o efeito
# da remocao de registros; "output" e o que foi gravado (no treino, depois da
# reamostragem).
STAGE_INPUT = "input"
STAGE_AFTER_IMPUTATION = "after_imputation"
STAGE_OUTPUT = "output"


def preparation_report_path(directory: str | Path, configuration_md5: str, fold_index: int) -> Path:
    return Path(directory) / f"{configuration_md5}.fold{fold_index}{PREPARATION_REPORT_FILE_SUFFIX}"


def split_counts(dataframe: pd.DataFrame | None, target_column: str) -> dict | None:
    """Linhas, positivos e faltantes de um split; ``inf`` conta como faltante."""
    if dataframe is None:
        return None
    features = dataframe.drop(columns=[target_column], errors="ignore")
    missing = features.isna()
    numeric = features.select_dtypes(include=[np.number])
    if not numeric.empty:
        missing[numeric.columns] |= np.isinf(numeric.to_numpy(dtype=float))
    rows = int(len(dataframe))
    positives = int((dataframe[target_column] == 1).sum()) if target_column in dataframe else 0
    return {
        "rows": rows,
        "positives": positives,
        "positive_rate": positives / rows if rows else None,
        "missing_cells": int(missing.to_numpy().sum()),
        "rows_with_missing": int(missing.any(axis=1).sum()),
    }


def new_fold_report(categorical_handling_mode: str) -> dict:
    return {
        "categorical_handling_mode": categorical_handling_mode,
        "splits": {label: {} for label in REPORT_SPLIT_NAMES.values()},
        "input_feature_columns": [],
        "output_feature_columns": [],
        "missing_screening": None,
        # Conjunto de linhas brutas do treino efetivo, depois da remocao de
        # incompletos e antes da reamostragem (item G16).
        "train_population_md5": None,
        # Clientes do treino efetivo, nas bases com unidade de avaliacao: e o
        # que pareia o treino do par da Home Credit (item G16).
        "train_unit_population_md5": None,
        "train_unit_population_size": None,
        "encoding_non_finite_replaced": {},
        "encoded_categorical_md5": None,
        "pca": None,
    }


def start_reports(
    split_information: dict,
    target_column: str,
    categorical_handling_mode: str,
) -> dict:
    """Cria o relatorio de cada dobra e registra a etapa de entrada."""
    folds = []
    for fold in split_information["test_info"]:
        report = new_fold_report(categorical_handling_mode)
        report["input_feature_columns"] = [
            column for column in fold["train_data"].columns if column != target_column
        ]
        folds.append({**fold, PREPARATION_REPORT_KEY: report})
    return record_stage({**split_information, "test_info": folds}, STAGE_INPUT, target_column)


def record_stage(
    split_information: dict,
    stage: str,
    target_column: str,
    split_names: tuple[str, ...] = tuple(REPORT_SPLIT_NAMES),
) -> dict:
    """Acrescenta os contadores de uma etapa ao relatorio de cada dobra."""
    for fold in split_information["test_info"]:
        report = fold[PREPARATION_REPORT_KEY]
        for split_name in split_names:
            counts = split_counts(fold.get(split_name), target_column)
            if counts is not None:
                report["splits"][REPORT_SPLIT_NAMES[split_name]][stage] = counts
    return split_information


def require_common_schema(fold: dict, fold_index: int) -> None:
    """Treino, validacao e teste da dobra precisam ter as mesmas colunas."""
    train_columns = list(fold["train_data"].columns)
    for split_name in ("validation_data", "test_data"):
        held_out = fold.get(split_name)
        if held_out is None or held_out.empty:
            continue
        if list(held_out.columns) != train_columns:
            only_train = sorted(set(train_columns) - set(held_out.columns))
            only_held_out = sorted(set(held_out.columns) - set(train_columns))
            raise ValueError(
                f"Fold {fold_index}: {split_name} schema differs from training "
                f"(only in train: {only_train[:10]}, only in {split_name}: "
                f"{only_held_out[:10]}, or a different column order)"
            )


def write_preparation_report(
    directory: str | Path,
    configuration_md5: str,
    fold_index: int,
    report: dict,
) -> tuple[str, Path]:
    """Grava o relatorio da dobra e devolve o MD5 do conteudo e o caminho."""
    path = preparation_report_path(directory, configuration_md5, fold_index)
    payload = {"configuration_md5": configuration_md5, "fold_index": fold_index, **report}
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    return compute_file_content_md5(path), path


def train_population_md5(fingerprints, retained_positions=None) -> str:
    """``partition_md5`` das linhas do treino que sobrevivem a remocao.

    Sem remocao (``retained_positions`` ausente), e o proprio ``partition_md5``
    do treino.
    """
    fingerprints = np.asarray(fingerprints)
    if retained_positions is not None:
        fingerprints = fingerprints[np.asarray(retained_positions, dtype=int)]
    return partition_md5(fingerprints)


def train_unit_population_md5(units, retained_positions=None) -> str:
    """``unit_partition_md5`` dos clientes do treino que sobrevivem a remocao."""
    units = pd.Series(units).reset_index(drop=True)
    if retained_positions is not None:
        units = units.iloc[np.asarray(retained_positions, dtype=int)]
    return unit_partition_md5(units)


def absorb_step_details(split_information: dict) -> dict:
    """Move para o relatorio os detalhes que as etapas deixam na dobra.

    A imputacao deixa ``missing_screening`` (``MissingScreening``) e as
    ``retained_train_positions``; a codificacao deixa
    ``encoding_non_finite_replaced``; a PCA deixa ``pca``. A particao deixa
    ``row_fingerprints``, de onde sai o ``train_population_md5``, e, nas
    bases com unidade de avaliacao, as ``evaluation_units``, de onde sai o
    ``train_unit_population_md5``.
    """
    folds = []
    for fold in split_information["test_info"]:
        fold = dict(fold)
        report = fold[PREPARATION_REPORT_KEY]
        screening = fold.pop("missing_screening", None)
        if screening is not None:
            report["missing_screening"] = screening.to_report()
        retained_positions = fold.pop("retained_train_positions", None)
        fingerprints = fold.pop("row_fingerprints", {}).get("train_data")
        if fingerprints is not None:
            report["train_population_md5"] = train_population_md5(
                fingerprints, retained_positions
            )
            fold["train_population_md5"] = report["train_population_md5"]
        train_units = fold.get("evaluation_units", {}).get("train_data")
        if train_units is not None:
            report["train_unit_population_md5"] = train_unit_population_md5(
                train_units, retained_positions
            )
            fold["train_unit_population_md5"] = report["train_unit_population_md5"]
            retained_units = (
                train_units if retained_positions is None
                else train_units.iloc[np.asarray(retained_positions, dtype=int)]
            )
            report["train_unit_population_size"] = int(retained_units.nunique())
        encoded_md5 = fold.pop("encoded_categorical_md5", None)
        if encoded_md5 is not None:
            report["encoded_categorical_md5"] = encoded_md5
        replaced = fold.pop("encoding_non_finite_replaced", None)
        if replaced is not None:
            report["encoding_non_finite_replaced"] = replaced
        pca_summary = fold.pop("pca", None)
        if pca_summary is not None:
            report["pca"] = pca_summary
        folds.append(fold)
    return {**split_information, "test_info": folds}


def finish_reports(split_information: dict, target_column: str) -> dict:
    """Registra a saida, confere o esquema comum e fixa as colunas de saida."""
    split_information = record_stage(split_information, STAGE_OUTPUT, target_column)
    for fold_position, fold in enumerate(split_information["test_info"]):
        require_common_schema(fold, fold_position + FIRST_FOLD_INDEX)
        fold[PREPARATION_REPORT_KEY]["output_feature_columns"] = [
            column for column in fold["train_data"].columns if column != target_column
        ]
    return split_information
