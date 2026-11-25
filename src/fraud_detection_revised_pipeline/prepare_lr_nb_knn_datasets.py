"""Prepare and persist dataset splits consumed by lr_nb_knn_specific.py.

A ordem das operacoes e o ponto central desta reimplementacao. Toda
transformacao que depende dos dados e ajustada **somente** no treinamento de
cada particao e so depois aplicada a validacao e ao teste:

1. carga bruta e derivacao de atributos relativos (hora, dia da semana,
   idade, distancia) -- nada dependente do rotulo;
2. particionamento, respeitando agrupamento e ordem temporal da base;
3. descarte dos identificadores, do agrupamento e da marca de tempo;
4. imputacao de faltantes, com estatisticas do treino;
5. codificacao categorica supervisionada, ajustada no treino;
6. padronizacao, com media e desvio do treino;
7. PCA de 95% da variancia, ajustada no treino, nas bases em que ela faz
   sentido;
8. reamostragem, aplicada **apenas** ao treino.

A versao anterior aplicava PCA e, em algumas bases, imputacao e codificacao
sobre a tabela inteira **antes** de particionar, o que vazava informacao do
teste para o treino (itens G14 e G15 da revisao). O passo 7 so roda nas bases
marcadas em ``experiment_datasets``: a ULB fica de fora porque ``V1``-``V28``
ja sao componentes principais publicados.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import data_partitioning
import execution_provenance
import experiment_datasets
import lr_nb_knn_specific as pipeline
import missing_screening
import preparation_report
import preparation_timing
from preparation_timing import PreparationTimer
from experiment_datasets import (
    STRATIFIED_HOLDOUT,
    TEMPORAL_HOLDOUT,
    get_dataset_definition,
    records_history_strata,
    resolve_base_definition,
    sequence_compatible_protocols,
)
from column_registry import build_column_registry, categorical_columns_of
from monetary_transform import log1p_data_splits
from prepared_dataset_storage import (
    CATEGORICAL_HANDLING_MODE_NATIVE,
    CATEGORICAL_HANDLING_MODE_SEQUENCE,
    CATEGORICAL_HANDLING_MODE_TRANSFORMER,
    DEFAULT_NUMERIC_BINS,
    DEFAULT_PREPARED_DATA_DIRECTORY,
    DEFAULT_RANDOM_SEED,
    FIRST_FOLD_INDEX,
    PreparedDatasetFile,
    build_prepared_dataset_md5,
    normalize_categorical_handling_mode,
    save_prepared_data_splits,
    save_fold_vocabulary,
    save_window_index,
    transformer_identity_parameters,
)
from transformer_feature_schema import resolve_feature_types
from transformer_preprocessing import fit_transformer_preprocessor
from transformer_sequence_windows import (
    assign_row_identity,
    build_sequence_window_index,
    split_chronologically_by_identity,
)


NATIVE_MISSING_CATEGORY = "__missing__"
# Nos modos de Transformer a ausencia e resolvida pelo proprio
# pre-processamento: media do treinamento mais indicador no FT-Transformer,
# token proprio no TabBERT.
TRANSFORMER_MANAGED_FILL_NA_STRATEGY = "model-managed"
# No bloco nativo, deixar a ausencia para o CatBoost/XGBoost. Tem nome proprio
# para nao se confundir com a remocao de incompletos do bloco classico nem com
# o "none" das bases sem faltantes (item G16).
NATIVE_MISSING_HANDLING_STRATEGY = "native"
SEQUENCE_PARTITION_STRATEGY = "sequence-chronological"


def _release_fold_memory() -> None:
    """Coleta a dobra concluida e devolve arenas livres ao cgroup no Linux."""
    gc.collect()
    malloc_trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if malloc_trim is not None:
        malloc_trim(0)


def _normalize_categorical_columns(dataframe):
    prepared_dataframe = dataframe.copy()
    categorical_columns = pipeline._categorical_columns(prepared_dataframe)
    for column in categorical_columns:
        prepared_dataframe[column] = (
            prepared_dataframe[column].astype("string").fillna(NATIVE_MISSING_CATEGORY)
        )
    return prepared_dataframe


def _fit_native_imputation_values(train_data, fillna_strategy, target_column):
    if fillna_strategy in (None, NATIVE_MISSING_HANDLING_STRATEGY):
        return None

    feature_data = train_data.drop(columns=[target_column], errors="ignore")
    fill_values = {}
    for column in feature_data.columns:
        series = feature_data[column]
        if fillna_strategy in {"mean", "median"} and not pd.api.types.is_numeric_dtype(series):
            continue
        fill_values[column] = pipeline._resolve_fill_value(series, fillna_strategy)
    return fill_values


def _apply_native_imputation_values(dataframe, fill_values, target_column):
    if dataframe is None:
        return None

    imputed_data = dataframe.copy().replace([np.inf, -np.inf], np.nan)
    if target_column in imputed_data.columns:
        imputed_data = imputed_data.dropna(subset=[target_column])

    if fill_values:
        applicable_values = {
            column: value
            for column, value in fill_values.items()
            if column in imputed_data.columns
        }
        imputed_data = imputed_data.fillna(applicable_values)

    return _normalize_categorical_columns(imputed_data)


def _impute_native_data_splits(split_information, fillna_strategy, target_column):
    imputed_information = {**split_information, "test_info": []}
    for split in split_information["test_info"]:
        raw_train_data = split["train_data"].replace([np.inf, -np.inf], np.nan)
        if target_column in raw_train_data.columns:
            raw_train_data = raw_train_data.dropna(subset=[target_column])
        fill_values = _fit_native_imputation_values(
            raw_train_data,
            fillna_strategy,
            target_column,
        )
        imputed_split = {
            **split,
            "train_data": _apply_native_imputation_values(
                raw_train_data,
                fill_values,
                target_column,
            ),
        }
        for split_name in ("validation_data", "test_data"):
            imputed_split[split_name] = _apply_native_imputation_values(
                split.get(split_name),
                fill_values,
                target_column,
            )
        imputed_information["test_info"].append(imputed_split)
    return imputed_information


def _apply_pca_to_data_splits(split_information: dict, prediction_column: str) -> dict:
    """Fit PCA on each training split and transform its held-out splits."""
    transformed_information = {**split_information, "test_info": []}
    for split in split_information["test_info"]:
        train_data, validation_data, test_data, pca_summary = pipeline._apply_pca(
            split["train_data"],
            split.get("validation_data"),
            split.get("test_data"),
            prediction_column,
        )
        transformed_information["test_info"].append(
            {
                **split,
                "train_data": train_data,
                "validation_data": validation_data,
                "test_data": test_data,
                "pca": pca_summary,
            }
        )
    return transformed_information


def _attach_preparation_reports(
    created_files: list[PreparedDatasetFile],
    split_information: dict,
    output_directory: str,
    configuration_md5: str,
    first_fold_index: int = FIRST_FOLD_INDEX,
) -> list[PreparedDatasetFile]:
    """Grava o relatorio de cada dobra e leva seu resumo as linhas do manifesto."""
    report_fields_by_fold = {}
    for fold_index, fold in enumerate(
        split_information["test_info"], start=first_fold_index
    ):
        report = fold[preparation_report.PREPARATION_REPORT_KEY]
        report_md5, _path = preparation_report.write_preparation_report(
            output_directory, configuration_md5, fold_index, report
        )
        screening = report["missing_screening"] or {}
        report_fields_by_fold[fold_index] = {
            "preparation_report_md5": report_md5,
            "missing_screening_threshold": str(screening.get("threshold_pct", "")),
            "missing_screening_rule": screening.get("rule", ""),
            "screened_out_columns": ",".join(screening.get("screened_out_columns", [])),
        }
    return [
        replace(created_file, **report_fields_by_fold[created_file.fold_index])
        for created_file in created_files
    ]


def _build_transformer_partition(
    database_train,
    database_auxiliary,
    args: argparse.Namespace,
    categorical_handling_mode: str,
) -> dict:
    """Particiona para os modos de Transformer.

    FT-Transformer e TabBERT usam os mesmos protocolos e a mesma funcao de
    particao do restante do pipeline, de modo que suas metricas sejam
    pareaveis com as dos demais algoritmos sobre as mesmas linhas. O TabBERT so
    aceita os protocolos que preservam as janelas por titular
    (sequence_compatible_protocols) e mantem nos arquivos o titular e a ordem,
    que o indice de janelas usa e o esquema de atributos exclui do modelo.

    ``sequence-chronological``, o corte proprio dentro do historico de cada
    titular, continua disponivel para reproduzir as configuracoes antigas, mas
    nao e pareavel com os outros blocos.
    """
    definition = get_dataset_definition(args.dataset)

    if (
        categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE
        and args.test_strategy == SEQUENCE_PARTITION_STRATEGY
    ):
        complete_data = database_train
        if database_auxiliary is not None and not database_auxiliary.empty:
            complete_data = pd.concat(
                [database_train, database_auxiliary], ignore_index=True
            )
        return split_chronologically_by_identity(
            complete_data,
            definition.dataset_id,
            classification_column_name=args.prediction_column,
            random_seed=args.random_seed,
        )

    retained_columns = (
        definition.sequence_keys
        if categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE
        else ()
    )
    return data_partitioning.build_partitions(
        database_train,
        database_auxiliary,
        definition,
        args.test_strategy,
        args.random_seed,
        retained_columns=retained_columns,
        history_window=_history_window(definition, args, categorical_handling_mode),
    )


def _history_window(definition, args: argparse.Namespace, categorical_handling_mode: str):
    """(historico, passo) para gravar o estrato de historico, ou None (item C3.19).

    So nos modos nativo e do FT-Transformer, nas bases e no protocolo do
    TabBERT: sao os modelos que nao usam historico e que o C3.19 pede para
    cortar pelo mesmo estrato. O TabBERT ja grava o estrato no indice de
    janelas, e o bloco classico nao entra na comparacao.
    """
    if categorical_handling_mode not in {
        CATEGORICAL_HANDLING_MODE_NATIVE,
        CATEGORICAL_HANDLING_MODE_TRANSFORMER,
    }:
        return None
    if not records_history_strata(definition, args.test_strategy):
        return None
    return (args.history_length, args.window_stride)


def _plan_paired_removal(definition, args: argparse.Namespace):
    """Planeja uma vez a particao da base que decide a remocao pareada."""
    paired_definition = get_dataset_definition(definition.paired_dataset_id)
    paired_train, paired_auxiliary, _ = pipeline.load_data(paired_definition.dataset_id)
    if not args.auxiliary_data_is_test_data:
        paired_auxiliary = None
    paired_plan = data_partitioning.plan_partitions(
        paired_train,
        paired_auxiliary,
        paired_definition,
        args.test_strategy,
        args.random_seed,
    )
    return paired_plan


def _adopt_paired_removal(
    fold: dict,
    paired_plan: data_partitioning.PartitionPlan,
    fold_position: int,
    args: argparse.Namespace,
) -> dict:
    """Materializa e adota a decisao pareada somente para uma dobra."""
    paired_fold = data_partitioning.materialize_fold(paired_plan, fold_position)
    paired_fold = missing_screening.decide_fold_removal(
        paired_fold,
        args.prediction_column,
        source_dataset_id=paired_plan.definition.dataset_id,
    )
    return missing_screening.adopt_paired_removal(fold, paired_fold)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load, transform, split, undersample, and save experiment datasets."
    )
    parser.add_argument(
        "--dataset", "--dataset-id", "--database-type", "--database_type", required=True
    )
    parser.add_argument(
        "--categorical-encoding",
        "--categorical-encoding-strategy",
        "--categorical_encoding_strategy",
        default=None,
        type=pipeline._parse_nullable_value,
    )
    parser.add_argument(
        "--fill-na-strategy",
        "--fill_na_strategy",
        default=None,
        type=pipeline._parse_nullable_value,
    )
    parser.add_argument(
        "--categorical-handling-mode",
        "--categorical_handling_mode",
        default="encoded",
        type=normalize_categorical_handling_mode,
    )
    parser.add_argument(
        "--test-strategy",
        "--test_strategy",
        required=True,
        help=(
            "Evaluation protocol: stratified-holdout, stratified-k-fold or "
            "temporal-holdout. The old test_data_size axis no longer exists."
        ),
    )
    parser.add_argument(
        "--random-seed",
        "--random_seed",
        default=DEFAULT_RANDOM_SEED,
        type=int,
        help="Seed for every pseudo-random partition. Part of the identity.",
    )
    parser.add_argument(
        "--prediction-column",
        "--prediction-column-name",
        "--prediction_column_name",
        required=True,
    )
    parser.add_argument(
        "--undersampling-technique",
        "--undersampling_technique",
        default=None,
        type=pipeline._parse_nullable_value,
        help=(
            "Sampling technique applied only to training data. RUS+SMOTE "
            "applies Random Under-Sampling first and SMOTE second."
        ),
    )
    parser.add_argument(
        "--undersampling-parameters",
        "--undersampling-parameter-values",
        "--undersampling_parameter_values",
        default=None,
        type=pipeline._parse_nullable_value,
    )
    parser.add_argument(
        "--auxiliary-data-is-test-data",
        "--auxiliary_data_is_test_data",
        default=True,
        type=pipeline._parse_bool,
    )
    parser.add_argument(
        "--output-directory",
        default=str(DEFAULT_PREPARED_DATA_DIRECTORY),
    )
    parser.add_argument("--numeric-normalization", default="quantile")
    parser.add_argument("--high-cardinality-strategy", default="minimum-frequency")
    parser.add_argument("--high-cardinality-parameter", type=int, default=10)
    parser.add_argument("--numeric-bins", type=int, default=DEFAULT_NUMERIC_BINS)
    parser.add_argument("--window-length", type=int, default=15)
    parser.add_argument("--history-length", type=int, default=10)
    parser.add_argument("--window-stride", type=int, default=5)
    parser.add_argument(
        "--print-configuration-md5-only",
        action="store_true",
        help=(
            "Print configuration_md5 for these arguments and exit, without "
            "loading data or writing any file."
        ),
    )
    return parser.parse_args()


def resolve_configuration_identity(
    args: argparse.Namespace,
) -> tuple[str, str, bool, bool]:
    """Validate CLI arguments and compute the configuration MD5, without loading data."""
    if args.undersampling_parameters and not args.undersampling_technique:
        raise ValueError(
            "undersampling_technique is required when undersampling_parameters is set"
        )

    definition = get_dataset_definition(args.dataset)
    if definition.prediction_column != args.prediction_column:
        raise ValueError(
            f"{args.dataset} uses prediction column "
            f"'{definition.prediction_column}', not '{args.prediction_column}'"
        )

    normalized_encoding = pipeline._normalize_categorical_encoding_strategy(
        args.categorical_encoding
    )
    categorical_handling_mode = normalize_categorical_handling_mode(
        args.categorical_handling_mode
    )
    uses_native_categorical_handling = (
        categorical_handling_mode == CATEGORICAL_HANDLING_MODE_NATIVE
    )
    uses_transformer_handling = categorical_handling_mode in {
        CATEGORICAL_HANDLING_MODE_TRANSFORMER,
        CATEGORICAL_HANDLING_MODE_SEQUENCE,
    }
    if (uses_native_categorical_handling or uses_transformer_handling) and normalized_encoding is not None:
        raise ValueError(
            "categorical_encoding must be none for native and Transformer modes"
        )
    if (
        args.fill_na_strategy == NATIVE_MISSING_HANDLING_STRATEGY
        and not uses_native_categorical_handling
    ):
        raise ValueError(
            f"fill_na_strategy={NATIVE_MISSING_HANDLING_STRATEGY} exists only in the "
            "native boosting block; the classic block removes rows with "
            f"{pipeline.DROP_INCOMPLETE_ROWS_STRATEGY}"
        )
    if uses_transformer_handling and args.fill_na_strategy not in {
        None,
        TRANSFORMER_MANAGED_FILL_NA_STRATEGY,
    }:
        raise ValueError(
            "fill_na_strategy must be "
            f"{TRANSFORMER_MANAGED_FILL_NA_STRATEGY} for Transformer modes: the "
            "model preprocessing handles missing values"
        )
    if categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE:
        supported = (
            *sequence_compatible_protocols(definition),
            SEQUENCE_PARTITION_STRATEGY,
        )
        if args.test_strategy not in supported:
            raise ValueError(
                f"{args.dataset} does not support protocol '{args.test_strategy}' "
                f"for sequence-tokenized preparation. Supported: {', '.join(supported)}"
            )
        if args.undersampling_technique:
            raise ValueError(
                "sequence-tokenized preparation cannot resample chronological rows"
            )
        if args.window_length != args.history_length + args.window_stride:
            raise ValueError("window_length must equal history_length + window_stride")
    else:
        if args.test_strategy not in definition.protocols:
            supported = ", ".join(definition.protocols)
            raise ValueError(
                f"{args.dataset} does not support protocol '{args.test_strategy}'. "
                f"Supported: {supported}"
            )

    configuration_md5 = build_prepared_dataset_md5(
        definition.dataset_id,
        args.categorical_encoding,
        args.fill_na_strategy,
        args.test_strategy,
        args.prediction_column,
        args.undersampling_technique,
        args.undersampling_parameters,
        args.auxiliary_data_is_test_data,
        categorical_handling_mode,
        args.random_seed,
        *transformer_identity_parameters(
            categorical_handling_mode,
            args.numeric_normalization,
            args.high_cardinality_strategy,
            args.high_cardinality_parameter,
            args.numeric_bins,
            args.window_length,
            args.history_length,
            args.window_stride,
        ),
    )
    return (
        configuration_md5,
        categorical_handling_mode,
        uses_native_categorical_handling,
        uses_transformer_handling,
    )


def _prepare_tabular_folds(
    plan: data_partitioning.PartitionPlan,
    definition,
    column_provenance: dict,
    args: argparse.Namespace,
    configuration_md5: str,
    categorical_handling_mode: str,
    uses_native_categorical_handling: bool,
    decides_removal: bool,
    timer: PreparationTimer,
) -> list[PreparedDatasetFile]:
    """Transforma e persiste uma dobra por vez nos blocos classico e nativo."""
    paired_plan = (
        _plan_paired_removal(definition, args)
        if decides_removal and definition.paired_dataset_id
        else None
    )
    column_registry = None
    created_files: list[PreparedDatasetFile] = []

    for fold_position in range(len(plan.fold_positions)):
        fold_index = fold_position + FIRST_FOLD_INDEX
        fold = data_partitioning.materialize_fold(plan, fold_position)

        if decides_removal:
            with timer.step(preparation_timing.STEP_SCREENING):
                if paired_plan is not None:
                    fold = _adopt_paired_removal(
                        fold, paired_plan, fold_position, args
                    )
                else:
                    fold = missing_screening.decide_fold_removal(
                        fold,
                        args.prediction_column,
                        ablated_columns=definition.ablated_columns,
                        source_dataset_id=resolve_base_definition(definition).dataset_id,
                    )

        split_information = {
            "train_test_strategy": plan.test_strategy,
            "test_info": [fold],
        }
        if column_registry is None:
            first_train = fold["train_data"]
            categorical_columns = categorical_columns_of(
                first_train, args.prediction_column
            )
            column_registry = build_column_registry(
                definition,
                column_provenance,
                categorical_columns,
                (
                    []
                    if uses_native_categorical_handling
                    else list(definition.monetary_columns)
                ),
                pca_applied=(
                    not uses_native_categorical_handling
                    and definition.applies_pca_in_classic_block
                ),
            )
            del first_train

        split_information = preparation_report.start_reports(
            split_information,
            args.prediction_column,
            categorical_handling_mode,
        )
        # A partir daqui o dicionario da preparacao e o unico dono da dobra;
        # nao manter a versao bruta viva ao lado das copias de cada etapa.
        del fold
        if uses_native_categorical_handling:
            with timer.step(preparation_timing.STEP_IMPUTATION):
                split_information = _impute_native_data_splits(
                    split_information,
                    args.fill_na_strategy,
                    args.prediction_column,
                )
            split_information = preparation_report.record_stage(
                split_information,
                preparation_report.STAGE_AFTER_IMPUTATION,
                args.prediction_column,
            )
        else:
            with timer.step(preparation_timing.STEP_MONETARY_TRANSFORM):
                split_information = log1p_data_splits(
                    split_information,
                    definition.monetary_columns,
                    definition.dataset_id,
                )
            with timer.step(preparation_timing.STEP_IMPUTATION):
                split_information = pipeline._impute_data_splits(
                    split_information,
                    args.fill_na_strategy,
                    args.prediction_column,
                )
            split_information = preparation_report.record_stage(
                split_information,
                preparation_report.STAGE_AFTER_IMPUTATION,
                args.prediction_column,
            )
            with timer.step(preparation_timing.STEP_ENCODING):
                split_information = pipeline._encode_supervised_data_splits(
                    split_information,
                    args.categorical_encoding,
                    args.prediction_column,
                )
            with timer.step(preparation_timing.STEP_STANDARDIZATION):
                split_information = pipeline.standardize_data_splits(
                    split_information,
                    args.prediction_column,
                )
            if definition.applies_pca_in_classic_block:
                with timer.step(preparation_timing.STEP_PCA):
                    split_information = _apply_pca_to_data_splits(
                        split_information,
                        args.prediction_column,
                    )

        if args.undersampling_technique:
            with timer.step(preparation_timing.STEP_RESAMPLING):
                split_information["test_info"][0]["train_data"] = (
                    pipeline.resample_dataframe(
                        split_information["test_info"][0]["train_data"],
                        args.prediction_column,
                        args.undersampling_technique,
                        args.undersampling_parameters,
                    )
                )

        split_information = preparation_report.absorb_step_details(
            split_information
        )
        split_information = preparation_report.finish_reports(
            split_information, args.prediction_column
        )
        split_information = {
            **split_information,
            "column_registry": column_registry,
        }
        timer.start(preparation_timing.STEP_WRITING)
        fold_files = save_prepared_data_splits(
            split_information,
            args.output_directory,
            configuration_md5,
            first_fold_index=fold_index,
        )
        fold_files = _attach_preparation_reports(
            fold_files,
            split_information,
            args.output_directory,
            configuration_md5,
            first_fold_index=fold_index,
        )
        timer.stop(preparation_timing.STEP_WRITING)
        created_files.extend(fold_files)

        del split_information, fold_files
        _release_fold_memory()

    timer.write(args.output_directory, configuration_md5, len(plan.fold_positions))
    return created_files


def prepare_datasets(
    args: argparse.Namespace,
) -> tuple[str, list[PreparedDatasetFile]]:
    (
        configuration_md5,
        categorical_handling_mode,
        uses_native_categorical_handling,
        uses_transformer_handling,
    ) = resolve_configuration_identity(args)

    definition = get_dataset_definition(args.dataset)
    dataset_id = definition.dataset_id
    # Tempo de cada etapa (item G18), gravado ao lado dos CSVs.
    timer = PreparationTimer()
    with timer.step(preparation_timing.STEP_STRUCTURAL_LOAD):
        database_train, database_auxiliary, _ = pipeline.load_data(dataset_id)
    column_provenance = dict(database_train.attrs.get("column_provenance", {}))
    if not args.auxiliary_data_is_test_data:
        database_auxiliary = None

    if uses_transformer_handling:
        with timer.step(preparation_timing.STEP_PARTITION):
            split_information = _build_transformer_partition(
                database_train,
                database_auxiliary,
                args,
                categorical_handling_mode,
            )
    else:
        # Na remocao de incompletos, uma variante de ablacao guarda as colunas
        # ablacionadas ate a decisao, para remover as mesmas linhas que a base
        # de origem (item G16).
        decides_removal = (
            not uses_native_categorical_handling
            and args.fill_na_strategy == pipeline.DROP_INCOMPLETE_ROWS_STRATEGY
        )
        with timer.step(preparation_timing.STEP_PARTITION):
            partition_plan = data_partitioning.plan_partitions(
                database_train,
                database_auxiliary,
                definition,
                args.test_strategy,
                args.random_seed,
                retained_columns=definition.ablated_columns if decides_removal else (),
                history_window=_history_window(definition, args, categorical_handling_mode),
            )
        # O plano possui a unica base concatenada necessaria daqui em diante.
        del database_train, database_auxiliary
        created_files = _prepare_tabular_folds(
            partition_plan,
            definition,
            column_provenance,
            args,
            configuration_md5,
            categorical_handling_mode,
            uses_native_categorical_handling,
            decides_removal,
            timer,
        )
        return configuration_md5, created_files

    # Antes de qualquer codificacao: as colunas categoricas como a base as traz.
    first_train = split_information["test_info"][0]["train_data"]
    if uses_transformer_handling:
        categorical_columns = resolve_feature_types(
            dataset_id, first_train, args.prediction_column
        )[0]
    else:
        categorical_columns = categorical_columns_of(first_train, args.prediction_column)
    applies_log1p = not (uses_native_categorical_handling or uses_transformer_handling)
    column_registry = build_column_registry(
        definition,
        column_provenance,
        categorical_columns,
        list(definition.monetary_columns) if applies_log1p else [],
        pca_applied=applies_log1p and definition.applies_pca_in_classic_block,
        retained_columns=(
            tuple(definition.sequence_keys or ())
            if categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE
            else ()
        ),
    )

    # Relatorio de preparacao por dobra nos blocos classico e nativo (item G15).
    writes_report = not uses_transformer_handling
    if writes_report:
        split_information = preparation_report.start_reports(
            split_information, args.prediction_column, categorical_handling_mode
        )

    fold_preprocessors = []
    fold_window_indexes = []
    if uses_transformer_handling:
        if categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE:
            split_information = assign_row_identity(split_information)
        transformed_information = {**split_information, "test_info": []}
        timer.start(preparation_timing.STEP_TRANSFORMER_PREPROCESSING)
        for fold in split_information["test_info"]:
            preprocessor = fit_transformer_preprocessor(
                fold["train_data"],
                dataset_id,
                args.prediction_column,
                categorical_handling_mode,
                args.numeric_normalization,
                args.high_cardinality_strategy,
                args.high_cardinality_parameter,
                args.numeric_bins,
            )
            fold_preprocessors.append(preprocessor)
            if categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE:
                fold_window_indexes.append(
                    build_sequence_window_index(
                        fold,
                        dataset_id,
                        args.history_length,
                        args.window_stride,
                        args.window_stride,
                    )
                )
            transformed_information["test_info"].append(
                {
                    **fold,
                    "train_data": preprocessor.transform(fold["train_data"]),
                    "validation_data": preprocessor.transform(fold.get("validation_data")),
                    "test_data": preprocessor.transform(fold.get("test_data")),
                }
            )
        timer.stop(preparation_timing.STEP_TRANSFORMER_PREPROCESSING)
        split_information = transformed_information
    elif uses_native_categorical_handling:
        with timer.step(preparation_timing.STEP_IMPUTATION):
            split_information = _impute_native_data_splits(
                split_information,
                args.fill_na_strategy,
                args.prediction_column,
            )
        split_information = preparation_report.record_stage(
            split_information, preparation_report.STAGE_AFTER_IMPUTATION, args.prediction_column
        )
    else:
        # log1p dos valores monetarios antes de imputar, padronizar e aplicar
        # a PCA (item G14): a cauda longa de Amount/amt/TransactionAmt
        # dominava a distancia euclidiana mesmo depois de padronizada.
        with timer.step(preparation_timing.STEP_MONETARY_TRANSFORM):
            split_information = log1p_data_splits(
                split_information, definition.monetary_columns, dataset_id
            )
        with timer.step(preparation_timing.STEP_IMPUTATION):
            split_information = pipeline._impute_data_splits(
                split_information,
                args.fill_na_strategy,
                args.prediction_column,
            )
        split_information = preparation_report.record_stage(
            split_information, preparation_report.STAGE_AFTER_IMPUTATION, args.prediction_column
        )
        with timer.step(preparation_timing.STEP_ENCODING):
            split_information = pipeline._encode_supervised_data_splits(
                split_information,
                args.categorical_encoding,
                args.prediction_column,
            )
        # Padronizacao para TODAS as configuracoes classicas, e nao apenas
        # para as que recebem PCA (item G14a da revisao).
        with timer.step(preparation_timing.STEP_STANDARDIZATION):
            split_information = pipeline.standardize_data_splits(
                split_information,
                args.prediction_column,
            )
        if definition.applies_pca_in_classic_block:
            with timer.step(preparation_timing.STEP_PCA):
                split_information = _apply_pca_to_data_splits(
                    split_information,
                    args.prediction_column,
                )

    if args.undersampling_technique:
        # Reamostragem toca apenas o treino; validacao e teste preservam a
        # prevalencia original da base.
        with timer.step(preparation_timing.STEP_RESAMPLING):
            for split in split_information["test_info"]:
                split["train_data"] = pipeline.resample_dataframe(
                    split["train_data"],
                    args.prediction_column,
                    args.undersampling_technique,
                    args.undersampling_parameters,
                )

    if writes_report:
        split_information = preparation_report.absorb_step_details(split_information)
        split_information = preparation_report.finish_reports(
            split_information, args.prediction_column
        )

    split_information = {**split_information, "column_registry": column_registry}
    timer.start(preparation_timing.STEP_WRITING)
    created_files = save_prepared_data_splits(
        split_information,
        args.output_directory,
        configuration_md5,
    )
    if writes_report:
        created_files = _attach_preparation_reports(
            created_files, split_information, args.output_directory, configuration_md5
        )
    if uses_transformer_handling:
        artifacts = {}
        for fold_position, preprocessor in enumerate(fold_preprocessors):
            fold_index = fold_position + FIRST_FOLD_INDEX
            vocabulary_md5, vocabulary_path = save_fold_vocabulary(
                args.output_directory,
                configuration_md5,
                fold_index,
                preprocessor.metadata(),
            )
            window_path = None
            if categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE:
                window_path = save_window_index(
                    args.output_directory,
                    configuration_md5,
                    fold_index,
                    fold_window_indexes[fold_position],
                )
            artifacts[fold_index] = (vocabulary_md5, vocabulary_path, window_path)
        created_files = [
            replace(
                created_file,
                vocabulary_md5=artifacts[created_file.fold_index][0],
                vocabulary_path=artifacts[created_file.fold_index][1],
                window_index_path=artifacts[created_file.fold_index][2],
            )
            for created_file in created_files
        ]
    timer.stop(preparation_timing.STEP_WRITING)
    timer.write(
        args.output_directory, configuration_md5, len(split_information["test_info"])
    )
    return configuration_md5, created_files


def main() -> None:
    execution_provenance.guard_environment_or_exit()
    pipeline.RELATIVE_DATA_PATH = "../datasets"
    pipeline.CURRENT_PATH = str(Path(__file__).resolve().parent)
    experiment_datasets.RELATIVE_DATA_PATH = pipeline.RELATIVE_DATA_PATH
    experiment_datasets.CURRENT_PATH = pipeline.CURRENT_PATH
    args = parse_arguments()
    if args.print_configuration_md5_only:
        configuration_md5, *_ = resolve_configuration_identity(args)
        print(f"configuration_md5={configuration_md5}")
        return
    configuration_md5, created_files = prepare_datasets(args)
    print(f"configuration_md5={configuration_md5}")
    for created_file in created_files:
        print(
            "prepared_dataset="
            f"{created_file.dataset_md5}|{created_file.fold_index}|"
            f"{created_file.split_label}|{created_file.content_md5}|"
            f"{created_file.path}|{created_file.partition_md5}|"
            f"{created_file.feature_schema_md5}|{created_file.train_population_md5}|"
            f"{created_file.unit_partition_md5}|{created_file.train_unit_population_md5}"
        )
        if created_file.preparation_report_md5:
            print(
                "preparation_report="
                f"{created_file.fold_index}|{created_file.dataset_md5}|"
                f"{created_file.preparation_report_md5}|"
                f"{created_file.missing_screening_threshold}|"
                f"{created_file.missing_screening_rule}|"
                f"{created_file.screened_out_columns}"
            )
        if created_file.vocabulary_path is not None:
            print(
                "prepared_artifacts="
                f"{created_file.fold_index}|{created_file.vocabulary_md5}|"
                f"{created_file.vocabulary_path}|{created_file.window_index_path or ''}"
            )


if __name__ == "__main__":
    main()
