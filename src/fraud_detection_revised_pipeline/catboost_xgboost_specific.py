"""Run native categorical CatBoost/XGBoost experiments from prepared splits."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time

import numpy as np
import pandas as pd

import execution_provenance
from lr_nb_knn_specific import (
    PREPARED_STORAGE_FAILURE_EXIT_CODE,
    PreparedDatasetStorageError,
    _parse_bool,
    _parse_nullable_value,
    CLASS_WEIGHT_INTENSITIES,
    CLASS_WEIGHT_BALANCED,
    CLASS_WEIGHT_NONE,
    CLASS_WEIGHT_SQRT_BALANCED,
    DEFAULT_CLASS_WEIGHT_INTENSITY,
    _parse_strategy,
    _score_split,
    apply_decision_thresholds_by_unit,
    compute_class_imbalance_ratio,
    is_experiment_already_recorded,
    record_algorithm_failure_to_csv,
    record_algorithm_result_to_csv,
    stage_prepared_dataset_files,
    validate_class_weight_arm,
)
from decision_threshold import (
    DEFAULT_THRESHOLD_STRATEGY,
    REQUESTABLE_THRESHOLD_STRATEGIES,
)
from model_metadata import describe_model_hyperparameters, summarize_training
from execution_failures import check_loaded_folds
from experiment_datasets import get_dataset_definition
from history_strata import (
    HISTORY_STRATUM_NAME,
    append_history_strata_record,
    history_strata_file,
    stratum_metrics,
)
from prepared_dataset_storage import (
    CATEGORICAL_HANDLING_MODE_NATIVE,
    DEFAULT_PREPARED_DATA_DIRECTORY,
    DEFAULT_RANDOM_SEED,
    FIRST_FOLD_INDEX,
    build_prepared_dataset_md5,
    load_prepared_data_splits,
    normalize_categorical_handling_mode,
)


ALGORITHM_ALL = "all"
ALGORITHM_CATBOOST = "catboost"
ALGORITHM_XGBOOST = "xgboost"
SUPPORTED_ALGORITHMS = (ALGORITHM_CATBOOST, ALGORITHM_XGBOOST)
RANDOM_SEED = 42
UNKNOWN_CATEGORY_VALUE = "__unknown__"
MISSING_CATEGORY_VALUE = "__missing__"


@dataclass(frozen=True)
class DatasetConfiguration:
    database_type: str
    dataset_filename: str
    prediction_column_name: str
    auxiliary_data_is_test_data: bool
    categorical_encoding_strategy: str | None
    fill_na_strategy: str | None
    undersampling_technique: str | None
    undersampling_parameter_values: str | None
    test_strategy: str
    # Intensidade da ponderacao de classe. Parametro de EXECUCAO: nao entra no
    # configuration_md5, porque nao muda os arquivos preparados, mas entra na
    # linha de resultado e na chave de ja-executado.
    class_weight_intensity: str
    # validation-f2 (padrao) ou fixed; ver decision_threshold.py (item G11-i).
    threshold_strategy: str
    # test_data_size saiu da identidade: a reserva nao alterava os dados em
    # duas das quatro politicas antigas (item G9c da revisao). random_seed
    # entrou, para que repeticoes com sementes distintas sejam configuracoes
    # distintas e comparaveis (item G13).
    random_seed: int
    configuration_md5: str


@dataclass(frozen=True)
class FoldData:
    train_features: pd.DataFrame
    train_target: pd.Series
    validation_features: pd.DataFrame | None
    validation_target: pd.Series | None
    test_features: pd.DataFrame | None
    test_target: pd.Series | None
    categorical_columns: list[str]
    class_values: list
    split_sizes: dict[str, int]
    # Indice 1-based da dobra e MD5 do conteudo de cada arquivo lido, ambos
    # propagados do manifesto ate a linha de resultado.
    fold_index: int
    split_md5: dict[str, str]
    partition_md5: dict[str, str]
    # Chaves de pareamento do item G16, lidas do manifesto.
    train_population_md5: str = ""
    unit_partition_md5: dict[str, str] = field(default_factory=dict)
    train_unit_population_md5: str = ""
    # Cliente de cada linha de validacao/teste, nas bases com unidade de
    # avaliacao (item G7); vazio nas demais.
    evaluation_units: dict[str, pd.Series] = field(default_factory=dict)
    # Estrato de historico de cada linha de validacao/teste, nas bases e no
    # protocolo do TabBERT (item C3.19); vazio nas demais.
    history_strata: dict[str, pd.Series] = field(default_factory=dict)


class MissingBoostingDependencyError(ImportError):
    """Raised when a requested boosting dependency is not installed."""


def load_catboost_classifier():
    try:
        from catboost import CatBoostClassifier
    except ImportError as error:
        raise MissingBoostingDependencyError(
            "CatBoost is required to run algorithm 'catboost'. "
            "Install the experiment requirements or run: pip install catboost"
        ) from error
    return CatBoostClassifier


def load_xgboost_classifier():
    try:
        from xgboost import XGBClassifier
    except ImportError as error:
        raise MissingBoostingDependencyError(
            "XGBoost is required to run algorithm 'xgboost'. "
            "Install the experiment requirements or run: pip install 'xgboost>=3.1.0'"
        ) from error
    return XGBClassifier


def normalize_algorithm_selection(algorithm: str) -> list[str]:
    normalized_algorithm = str(algorithm).strip().lower()
    if normalized_algorithm == ALGORITHM_ALL:
        return list(SUPPORTED_ALGORITHMS)
    if normalized_algorithm in SUPPORTED_ALGORITHMS:
        return [normalized_algorithm]
    supported_values = ", ".join((ALGORITHM_ALL, *SUPPORTED_ALGORITHMS))
    raise ValueError(f"algorithm must be one of: {supported_values}")


def result_csv_path(results_directory: str | Path, algorithm: str) -> Path:
    return Path(results_directory) / f"resultados_{algorithm}.csv"


def validate_native_manifest_configuration(
    manifest_path: str | Path,
    configuration_md5: str,
) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
    required_columns = {
        "configuration_md5",
        "categorical_handling_mode",
        "categorical_encoding_strategy",
    }
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise ValueError(
            "Prepared dataset manifest is missing columns: "
            f"{', '.join(missing_columns)}"
        )

    normalized_configuration_md5 = str(configuration_md5).strip().lower()
    matching_rows = manifest[
        manifest["configuration_md5"].str.strip().str.lower()
        == normalized_configuration_md5
    ]
    if matching_rows.empty:
        raise FileNotFoundError(
            f"No manifest rows matched configuration MD5 '{normalized_configuration_md5}'"
        )

    handling_modes = {
        normalize_categorical_handling_mode(value)
        for value in matching_rows["categorical_handling_mode"].unique()
    }
    if handling_modes != {CATEGORICAL_HANDLING_MODE_NATIVE}:
        raise ValueError(
            "Boosting native execution accepts only configurations with "
            "categorical_handling_mode=native"
        )

    encoding_values = {
        str(value).strip().lower()
        for value in matching_rows["categorical_encoding_strategy"].unique()
    }
    allowed_encoding_values = {"", "none", "null"}
    if not encoding_values or not encoding_values.issubset(allowed_encoding_values):
        raise ValueError(
            "Boosting native execution accepts only categorical_encoding_strategy=none"
        )

    return matching_rows


def split_features_and_target(
    dataframe: pd.DataFrame | None,
    prediction_column_name: str,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame | None, pd.Series | None]:
    if dataframe is None or dataframe.empty:
        return None, None
    if prediction_column_name not in dataframe.columns:
        raise ValueError(
            f"{prediction_column_name} does not exist in prepared dataframe"
        )

    target = dataframe[prediction_column_name].copy()
    features = dataframe.drop(columns=[prediction_column_name]).copy()
    if feature_columns is not None:
        missing_columns = [
            column for column in feature_columns if column not in features.columns
        ]
        if missing_columns:
            raise ValueError(
                "Prepared split is missing training feature columns: "
                f"{', '.join(missing_columns)}"
            )
        features = features.reindex(columns=feature_columns)
    return features, target


def discover_categorical_columns(features: pd.DataFrame) -> list[str]:
    categorical_columns = features.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    return list(categorical_columns)


def prepare_fold_data(
    split_information: dict,
    prediction_column_name: str,
) -> FoldData:
    train_features, train_target = split_features_and_target(
        split_information.get("train_data"),
        prediction_column_name,
    )
    if train_features is None or train_target is None:
        raise ValueError("Prepared fold does not contain non-empty train data")

    feature_columns = list(train_features.columns)
    validation_features, validation_target = split_features_and_target(
        split_information.get("validation_data"),
        prediction_column_name,
        feature_columns,
    )
    test_features, test_target = split_features_and_target(
        split_information.get("test_data"),
        prediction_column_name,
        feature_columns,
    )
    categorical_columns = discover_categorical_columns(train_features)
    class_values = sorted(pd.Series(train_target).dropna().unique().tolist())
    if len(class_values) < 2:
        raise ValueError("Training fold must contain at least two target classes")

    return FoldData(
        train_features=train_features,
        train_target=train_target,
        validation_features=validation_features,
        validation_target=validation_target,
        test_features=test_features,
        test_target=test_target,
        categorical_columns=categorical_columns,
        class_values=class_values,
        split_sizes={
            split_name: len(split_information.get(split_name))
            if split_information.get(split_name) is not None
            else 0
            for split_name in ["train_data", "validation_data", "test_data"]
        },
        fold_index=split_information.get("fold_index", FIRST_FOLD_INDEX),
        split_md5=split_information.get("split_md5", {}),
        partition_md5=split_information.get("partition_md5", {}),
        train_population_md5=split_information.get("train_population_md5", ""),
        unit_partition_md5=split_information.get("unit_partition_md5", {}),
        train_unit_population_md5=split_information.get("train_unit_population_md5", ""),
        evaluation_units=split_information.get("evaluation_units", {}),
        history_strata=split_information.get("history_strata", {}),
    )


def prepare_catboost_dataframe(
    features: pd.DataFrame | None,
    categorical_columns: list[str],
) -> pd.DataFrame | None:
    if features is None:
        return None

    prepared_features = features.copy()
    for column in categorical_columns:
        prepared_features[column] = (
            prepared_features[column]
            .astype("string")
            .fillna(MISSING_CATEGORY_VALUE)
            .astype(str)
        )
    return prepared_features


def build_xgboost_category_domains(
    train_features: pd.DataFrame,
    categorical_columns: list[str],
) -> dict[str, list[str]]:
    domains: dict[str, list[str]] = {}
    for column in categorical_columns:
        values = (
            train_features[column]
            .astype("string")
            .fillna(MISSING_CATEGORY_VALUE)
            .astype(str)
        )
        categories = sorted(values.unique().tolist())
        if UNKNOWN_CATEGORY_VALUE not in categories:
            categories.append(UNKNOWN_CATEGORY_VALUE)
        domains[column] = categories
    return domains


def prepare_xgboost_dataframe(
    features: pd.DataFrame | None,
    categorical_columns: list[str],
    category_domains: dict[str, list[str]],
) -> pd.DataFrame | None:
    if features is None:
        return None

    prepared_features = features.copy()
    for column in categorical_columns:
        categories = category_domains[column]
        values = (
            prepared_features[column]
            .astype("string")
            .fillna(MISSING_CATEGORY_VALUE)
            .astype(str)
        )
        values = values.where(values.isin(categories), UNKNOWN_CATEGORY_VALUE)
        prepared_features[column] = pd.Categorical(values, categories=categories)
    return prepared_features


# Traducao da intensidade de ponderacao para o parametro de cada biblioteca.
# Os tres niveis sao os MESMOS da regressao logistica, de modo que a
# comparacao entre familias use a mesma correcao relativa do desbalanceamento.
CATBOOST_AUTO_CLASS_WEIGHTS = {
    CLASS_WEIGHT_NONE: None,
    CLASS_WEIGHT_SQRT_BALANCED: "SqrtBalanced",
    CLASS_WEIGHT_BALANCED: "Balanced",
}


def resolve_scale_pos_weight(intensity: str, imbalance_ratio: float | None) -> float:
    """scale_pos_weight do XGBoost: 1, sqrt(R) ou R.

    A documentacao do XGBoost recomenda negativos/positivos como valor tipico
    para o balanceamento completo. R vem do treino desta dobra, que nos bracos
    ponderados nao e reamostrado e tem a prevalencia natural.
    """
    normalized = str(intensity or CLASS_WEIGHT_NONE).strip().lower()
    if normalized not in CLASS_WEIGHT_INTENSITIES:
        raise ValueError(
            f"class_weight_intensity must be one of {CLASS_WEIGHT_INTENSITIES}, "
            f"got {intensity!r}"
        )
    if normalized == CLASS_WEIGHT_NONE or imbalance_ratio is None:
        return 1.0
    if normalized == CLASS_WEIGHT_SQRT_BALANCED:
        return float(imbalance_ratio) ** 0.5
    return float(imbalance_ratio)


def resolve_auto_class_weights(intensity: str) -> str | None:
    """auto_class_weights do CatBoost, que implementa SqrtBalanced nativamente."""
    normalized = str(intensity or CLASS_WEIGHT_NONE).strip().lower()
    if normalized not in CLASS_WEIGHT_INTENSITIES:
        raise ValueError(
            f"class_weight_intensity must be one of {CLASS_WEIGHT_INTENSITIES}, "
            f"got {intensity!r}"
        )
    return CATBOOST_AUTO_CLASS_WEIGHTS[normalized]


def create_catboost_classifier(
    categorical_columns: list[str],
    class_weight_intensity: str = DEFAULT_CLASS_WEIGHT_INTENSITY,
):
    CatBoostClassifier = load_catboost_classifier()
    parameters = dict(
        random_seed=RANDOM_SEED,
        allow_writing_files=False,
        verbose=False,
        thread_count=1,
        cat_features=categorical_columns,
    )
    auto_class_weights = resolve_auto_class_weights(class_weight_intensity)
    if auto_class_weights is not None:
        parameters["auto_class_weights"] = auto_class_weights
    return CatBoostClassifier(**parameters)


def create_xgboost_classifier(
    class_weight_intensity: str = DEFAULT_CLASS_WEIGHT_INTENSITY,
    imbalance_ratio: float | None = None,
):
    XGBClassifier = load_xgboost_classifier()
    return XGBClassifier(
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=0,
        scale_pos_weight=resolve_scale_pos_weight(
            class_weight_intensity, imbalance_ratio
        ),
    )


def probability_scores(model, features: pd.DataFrame) -> np.ndarray | None:
    if not hasattr(model, "predict_proba"):
        return None
    return model.predict_proba(features)


def predict_split(
    model,
    features: pd.DataFrame | None,
    target: pd.Series | None,
    split_name: str,
    class_values: list,
) -> dict | None:
    """Scores de um split; o limiar e aplicado depois, em apply_decision_thresholds."""
    if features is None or target is None:
        return None
    return _score_split(
        target,
        model.predict_proba(features),
        model.classes_,
        class_values,
        split_name,
        "boosting_probability",
    )


def execute_catboost_fold(
    fold_data: FoldData,
    class_weight_intensity: str = DEFAULT_CLASS_WEIGHT_INTENSITY,
) -> dict:
    train_features = prepare_catboost_dataframe(
        fold_data.train_features,
        fold_data.categorical_columns,
    )
    validation_features = prepare_catboost_dataframe(
        fold_data.validation_features,
        fold_data.categorical_columns,
    )
    test_features = prepare_catboost_dataframe(
        fold_data.test_features,
        fold_data.categorical_columns,
    )
    model = create_catboost_classifier(
        fold_data.categorical_columns, class_weight_intensity
    )
    fit_started_at = time.perf_counter()
    model.fit(train_features, fold_data.train_target)
    return build_fold_execution_result(
        model,
        fold_data,
        train_features,
        validation_features,
        test_features,
        fit_time=time.perf_counter() - fit_started_at,
    )


def execute_xgboost_fold(
    fold_data: FoldData,
    class_weight_intensity: str = DEFAULT_CLASS_WEIGHT_INTENSITY,
) -> dict:
    category_domains = build_xgboost_category_domains(
        fold_data.train_features,
        fold_data.categorical_columns,
    )
    train_features = prepare_xgboost_dataframe(
        fold_data.train_features,
        fold_data.categorical_columns,
        category_domains,
    )
    validation_features = prepare_xgboost_dataframe(
        fold_data.validation_features,
        fold_data.categorical_columns,
        category_domains,
    )
    test_features = prepare_xgboost_dataframe(
        fold_data.test_features,
        fold_data.categorical_columns,
        category_domains,
    )
    model = create_xgboost_classifier(
        class_weight_intensity,
        compute_class_imbalance_ratio(fold_data.train_target, fold_data.class_values),
    )
    fit_started_at = time.perf_counter()
    model.fit(train_features, fold_data.train_target)
    return build_fold_execution_result(
        model,
        fold_data,
        train_features,
        validation_features,
        test_features,
        fit_time=time.perf_counter() - fit_started_at,
    )


def build_fold_execution_result(
    model,
    fold_data: FoldData,
    train_features: pd.DataFrame,
    validation_features: pd.DataFrame | None,
    test_features: pd.DataFrame | None,
    fit_time: float | None = None,
) -> dict:
    scoring_started_at = time.perf_counter()
    split_scores = {}
    for split_name, features, target in [
        ("train_data", train_features, fold_data.train_target),
        ("validation_data", validation_features, fold_data.validation_target),
        ("test_data", test_features, fold_data.test_target),
    ]:
        split_result = predict_split(
            model,
            features,
            target,
            split_name,
            fold_data.class_values,
        )
        if split_result is not None:
            split_scores[split_name] = split_result
    scoring_time = time.perf_counter() - scoring_started_at

    return {
        # Tempos da dobra (item G18): so o ajuste do modelo e a pontuacao dos
        # tres conjuntos; a preparacao dos dataframes fica de fora.
        "fit_time": "" if fit_time is None else round(fit_time, 6),
        "scoring_time": round(scoring_time, 6),
        "split_sizes": fold_data.split_sizes,
        "fold_index": fold_data.fold_index,
        "split_md5": fold_data.split_md5,
        "partition_md5": fold_data.partition_md5,
        "train_population_md5": fold_data.train_population_md5,
        "unit_partition_md5": fold_data.unit_partition_md5,
        "train_unit_population_md5": fold_data.train_unit_population_md5,
        "evaluation_units": fold_data.evaluation_units,
        "history_strata": fold_data.history_strata,
        "class_values": fold_data.class_values,
        "split_scores": split_scores,
        # O modelo nao fica na dobra; o que importa dele sai daqui ja extraido.
        "model_hyperparameters": describe_model_hyperparameters(model),
        "training_summary": summarize_training(model),
    }


def execute_boosting_algorithm(
    algorithm: str,
    test_data_information: dict,
    prediction_column_name: str,
    undersampling_technique: str | None = None,
    class_weight_intensity: str = DEFAULT_CLASS_WEIGHT_INTENSITY,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
) -> dict:
    normalized_algorithm = str(algorithm).strip().lower()
    fold_executor = {
        ALGORITHM_CATBOOST: execute_catboost_fold,
        ALGORITHM_XGBOOST: execute_xgboost_fold,
    }[normalized_algorithm]

    execution_started_at = time.perf_counter()
    results = []
    for split_information in test_data_information.get("test_info", []):
        fold_data = prepare_fold_data(split_information, prediction_column_name)
        fold_result = fold_executor(fold_data, class_weight_intensity)
        # R efetivo desta dobra, para que a linha de resultado permita
        # reconstruir o peso aplicado, e nao so o rotulo da intensidade.
        fold_result["class_weight_intensity"] = class_weight_intensity
        fold_result["class_weight_ratio"] = compute_class_imbalance_ratio(
            fold_data.train_target, fold_data.class_values
        )
        results.append(fold_result)
    # O limiar so e escolhido depois de pontuar todas as dobras: no k-fold o
    # de cada dobra vem da validacao das outras.
    results = apply_decision_thresholds_by_unit(results, threshold_strategy)

    return {
        "algorithm": normalized_algorithm,
        "train_test_strategy": test_data_information.get("train_test_strategy"),
        "undersampling_technique": undersampling_technique,
        "results": results,
        "algorithm_time": round(time.perf_counter() - execution_started_at, 6),
    }


def build_dataset_configuration(args: argparse.Namespace) -> DatasetConfiguration:
    categorical_handling_mode = normalize_categorical_handling_mode(
        args.categorical_handling_mode
    )
    expected_configuration_md5 = build_prepared_dataset_md5(
        args.database_type,
        args.categorical_encoding_strategy,
        args.fill_na_strategy,
        args.test_strategy,
        args.prediction_column_name,
        args.undersampling_technique,
        args.undersampling_parameter_values,
        args.auxiliary_data_is_test_data,
        categorical_handling_mode,
        args.random_seed,
    )
    requested_configuration_md5 = (
        args.prepared_dataset_md5 or args.configuration_md5 or expected_configuration_md5
    ).lower()
    validate_class_weight_arm(args.class_weight_intensity, args.undersampling_technique)
    if requested_configuration_md5 != expected_configuration_md5:
        raise ValueError(
            "configuration_md5 does not match the informed dataset preparation "
            "parameters"
        )

    return DatasetConfiguration(
        database_type=args.database_type,
        dataset_filename=args.dataset_filename,
        prediction_column_name=args.prediction_column_name,
        auxiliary_data_is_test_data=args.auxiliary_data_is_test_data,
        categorical_encoding_strategy=args.categorical_encoding_strategy,
        fill_na_strategy=args.fill_na_strategy,
        undersampling_technique=args.undersampling_technique,
        undersampling_parameter_values=args.undersampling_parameter_values,
        random_seed=args.random_seed,
        class_weight_intensity=args.class_weight_intensity,
        threshold_strategy=args.threshold_strategy,
        test_strategy=args.test_strategy,
        configuration_md5=requested_configuration_md5,
    )


def record_failure(
    results_directory: str | Path,
    dataset_configuration: DatasetConfiguration,
    algorithm: str,
    elapsed_time: float,
    error: BaseException | None = None,
) -> None:
    record_algorithm_failure_to_csv(
        result_csv_path(results_directory, algorithm),
        dataset_configuration.dataset_filename,
        algorithm,
        dataset_configuration.categorical_encoding_strategy,
        dataset_configuration.fill_na_strategy,
        dataset_configuration.test_strategy,
        undersampling_technique=dataset_configuration.undersampling_technique,
        random_seed=dataset_configuration.random_seed,
        class_weight_intensity=dataset_configuration.class_weight_intensity,
        threshold_strategy=dataset_configuration.threshold_strategy,
        elapsed_time=elapsed_time,
        configuration_md5=dataset_configuration.configuration_md5,
        database_type=dataset_configuration.database_type,
        error=error,
    )


def history_strata_records(
    dataset_configuration: DatasetConfiguration,
    algorithm_execution_result: dict,
) -> list[dict]:
    """Um registro por dobra e split avaliado com estrato de historico (item C3.19).

    So as dobras por linha: o estrato e da transacao, e nao do cliente. O
    limiar e o da propria linha de resultado.
    """
    records = []
    dataset_id = get_dataset_definition(dataset_configuration.database_type).dataset_id
    for fold_result in algorithm_execution_result.get("results", []):
        if fold_result.get("evaluation_unit", "row") not in ("", "row"):
            continue
        strata_by_split = fold_result.get("history_strata") or {}
        if not strata_by_split:
            continue
        positive_class = sorted(fold_result["class_values"])[-1]
        for split_name, strata in strata_by_split.items():
            split_result = (fold_result.get("split_results") or {}).get(split_name)
            if split_result is None:
                continue
            predictions = split_result["predictions"]
            targets = (predictions["real_value"].to_numpy() == positive_class).astype(int)
            records.append(
                {
                    "configuration_md5": dataset_configuration.configuration_md5,
                    "dataset_id": dataset_id,
                    "test_strategy": dataset_configuration.test_strategy,
                    "random_seed": dataset_configuration.random_seed,
                    "fold_index": int(fold_result["fold_index"]),
                    "algorithm": algorithm_execution_result["algorithm"],
                    "regime": "",
                    "class_weight_intensity": dataset_configuration.class_weight_intensity,
                    "threshold_strategy": fold_result.get("threshold_strategy"),
                    "split": split_name.removesuffix("_data"),
                    "history_strata": {
                        HISTORY_STRATUM_NAME: stratum_metrics(
                            targets,
                            predictions["boosting_probability"].to_numpy(),
                            np.asarray(strata),
                            fold_result["classification_threshold"],
                        )
                    },
                }
            )
    return records


def record_success(
    results_directory: str | Path,
    dataset_configuration: DatasetConfiguration,
    algorithm_execution_result: dict,
    elapsed_time: float,
) -> None:
    # Os estratos sao calculados antes de gravar a linha de resultado: um erro
    # aqui vira falha da execucao, e nao uma linha de sucesso sem estratos.
    strata_records = history_strata_records(dataset_configuration, algorithm_execution_result)
    record_algorithm_result_to_csv(
        result_csv_path(
            results_directory,
            algorithm_execution_result["algorithm"],
        ),
        dataset_configuration.dataset_filename,
        algorithm_execution_result,
        dataset_configuration.categorical_encoding_strategy,
        dataset_configuration.fill_na_strategy,
        elapsed_time,
        algorithm_time=algorithm_execution_result.get("algorithm_time"),
        random_seed=dataset_configuration.random_seed,
        class_weight_intensity=dataset_configuration.class_weight_intensity,
        threshold_strategy=dataset_configuration.threshold_strategy,
        configuration_md5=dataset_configuration.configuration_md5,
        database_type=dataset_configuration.database_type,
    )
    strata_path = history_strata_file(results_directory, algorithm_execution_result["algorithm"])
    for record in strata_records:
        append_history_strata_record(strata_path, record)


def pending_algorithms(
    results_directory: str | Path,
    configuration_md5: str,
    requested_algorithms: list[str],
    class_weight_intensity: str = DEFAULT_CLASS_WEIGHT_INTENSITY,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
) -> list[str]:
    return [
        algorithm
        for algorithm in requested_algorithms
        if not is_experiment_already_recorded(
            results_directory,
            configuration_md5,
            algorithm,
            class_weight_intensity=class_weight_intensity,
            threshold_strategy=threshold_strategy,
        )
    ]


def run_configuration(args: argparse.Namespace) -> int:
    execution_started_at = time.perf_counter()
    results_directory = Path(args.results_directory)
    dataset_configuration = build_dataset_configuration(args)
    requested_algorithms = normalize_algorithm_selection(args.algorithm)
    pending = (
        requested_algorithms
        if args.force_reprocess
        else pending_algorithms(
            results_directory,
            dataset_configuration.configuration_md5,
            requested_algorithms,
            dataset_configuration.class_weight_intensity,
            dataset_configuration.threshold_strategy,
        )
    )
    if not pending:
        print(
            "Skipping experiment already recorded: "
            f"configuration_md5={dataset_configuration.configuration_md5}, "
            f"algorithms={','.join(requested_algorithms)}"
        )
        return 0

    prepared_dataset_manifest = args.prepared_dataset_manifest or (
        Path(args.prepared_data_directory) / "prepared_dataset_configurations.tsv"
    )
    failed_algorithms = []
    local_prepared_data_root = Path(args.local_staging_directory)
    try:
        validate_native_manifest_configuration(
            prepared_dataset_manifest,
            dataset_configuration.configuration_md5,
        )
        with stage_prepared_dataset_files(
            args.prepared_data_directory,
            prepared_dataset_manifest,
            dataset_configuration.configuration_md5,
            local_prepared_data_root,
        ) as staged_data_directory:
            test_data_information = load_prepared_data_splits(
                staged_data_directory,
                prepared_dataset_manifest,
                dataset_configuration.configuration_md5,
                dataset_configuration.test_strategy,
            )
            check_loaded_folds(
                test_data_information, dataset_configuration.prediction_column_name
            )
            if (
                normalize_categorical_handling_mode(
                    test_data_information.get("categorical_handling_mode")
                )
                != CATEGORICAL_HANDLING_MODE_NATIVE
            ):
                raise ValueError("Loaded prepared data is not native categorical data")

            for algorithm in pending:
                algorithm_started_at = time.perf_counter()
                try:
                    execution_result = execute_boosting_algorithm(
                        algorithm,
                        test_data_information,
                        dataset_configuration.prediction_column_name,
                        dataset_configuration.undersampling_technique,
                        dataset_configuration.class_weight_intensity,
                        dataset_configuration.threshold_strategy,
                    )
                    record_success(
                        results_directory,
                        dataset_configuration,
                        execution_result,
                        round(time.perf_counter() - execution_started_at, 6),
                    )
                except Exception as error:
                    failed_algorithms.append(algorithm)
                    print(
                        "Error running boosting experiment: "
                        f"configuration_md5={dataset_configuration.configuration_md5}, "
                        f"algorithm={algorithm}: {error}"
                    )
                    record_failure(
                        results_directory,
                        dataset_configuration,
                        algorithm,
                        round(time.perf_counter() - algorithm_started_at, 6),
                        error,
                    )
    except PreparedDatasetStorageError:
        raise
    except Exception as error:
        print(
            "Error loading native boosting configuration: "
            f"configuration_md5={dataset_configuration.configuration_md5}: {error}"
        )
        for algorithm in pending:
            failed_algorithms.append(algorithm)
            record_failure(
                results_directory,
                dataset_configuration,
                algorithm,
                round(time.perf_counter() - execution_started_at, 6),
                error,
            )

    return 1 if failed_algorithms else 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CatBoost/XGBoost native categorical prepared experiments."
    )
    parser.add_argument(
        "--algorithm",
        default=ALGORITHM_ALL,
        choices=[ALGORITHM_ALL, *SUPPORTED_ALGORITHMS],
    )
    parser.add_argument(
        "--dataset-id", "--database-type", "--database_type",
        dest="database_type", required=True,
    )
    parser.add_argument("--dataset-filename", "--dataset_filename", required=True)
    parser.add_argument("--prediction-column-name", "--prediction_column_name", required=True)
    parser.add_argument(
        "--auxiliary-data-is-test-data",
        "--auxiliary_data_is_test_data",
        default=True,
        type=_parse_bool,
    )
    parser.add_argument(
        "--categorical-encoding-strategy",
        "--categorical_encoding_strategy",
        default=None,
        type=_parse_nullable_value,
    )
    parser.add_argument(
        "--categorical-handling-mode",
        "--categorical_handling_mode",
        default=CATEGORICAL_HANDLING_MODE_NATIVE,
        type=_parse_nullable_value,
    )
    parser.add_argument(
        "--fill-na-strategy",
        "--fill_na_strategy",
        "--fillna-strategy",
        "--fillna_strategy",
        default=None,
        type=_parse_nullable_value,
    )
    parser.add_argument("--test-strategy", "--test_strategy", required=True)
    parser.add_argument(
        "--class-weight-intensity",
        "--class_weight_intensity",
        default=DEFAULT_CLASS_WEIGHT_INTENSITY,
        choices=list(CLASS_WEIGHT_INTENSITIES),
        help=(
            "none, sqrt-balanced or balanced. Maps to CatBoost auto_class_weights "
            "and XGBoost scale_pos_weight, with R measured on this fold's training "
            "split. sqrt-balanced and balanced are balancing arms of their own and "
            "only run on configurations without resampling."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        "--threshold_strategy",
        default=DEFAULT_THRESHOLD_STRATEGY,
        choices=list(REQUESTABLE_THRESHOLD_STRATEGIES),
        help=(
            "validation-f2 (default) picks the F2-maximising threshold on the "
            "validation predictions and applies it to the test split; fixed uses 0.5."
        ),
    )
    parser.add_argument(
        "--random-seed",
        "--random_seed",
        default=DEFAULT_RANDOM_SEED,
        type=int,
        help="Seed used when preparing this configuration; part of its identity.",
    )
    parser.add_argument(
        "--undersampling-technique",
        "--undersampling_technique",
        default=None,
        type=_parse_nullable_value,
    )
    parser.add_argument(
        "--undersampling-parameter-values",
        "--undersampling_parameter_values",
        default=None,
        type=_parse_nullable_value,
    )
    parser.add_argument(
        "--prepared-data-directory",
        "--prepared_data_directory",
        default=str(DEFAULT_PREPARED_DATA_DIRECTORY),
    )
    parser.add_argument(
        "--prepared-dataset-md5",
        "--prepared_dataset_md5",
        default=None,
    )
    parser.add_argument(
        "--configuration-md5",
        "--configuration_md5",
        default=None,
    )
    parser.add_argument(
        "--prepared-dataset-manifest",
        "--prepared_dataset_manifest",
        default=None,
    )
    parser.add_argument(
        "--results-directory",
        "--results_directory",
        default=str(Path(__file__).resolve().parent),
    )
    parser.add_argument(
        "--local-staging-directory",
        "--local_staging_directory",
        default=str(Path(__file__).resolve().parent / "prepared_datasets"),
        help="Local directory used while reading one prepared configuration.",
    )
    parser.add_argument(
        "--force-reprocess",
        "--force_reprocess",
        action="store_true",
        help=(
            "Recompute and append a new result row even if a successful one is "
            "already recorded for this configuration_md5/algorithm."
        ),
    )
    return parser


def parse_script_arguments_from_list(arguments: list[str]) -> argparse.Namespace:
    return build_argument_parser().parse_args(arguments)


def parse_script_arguments() -> argparse.Namespace:
    return build_argument_parser().parse_args()


def main() -> None:
    execution_provenance.guard_environment_or_exit()
    try:
        exit_code = run_configuration(parse_script_arguments())
    except PreparedDatasetStorageError as error:
        print(f"Prepared storage failure: {error}")
        raise SystemExit(PREPARED_STORAGE_FAILURE_EXIT_CODE) from error
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
