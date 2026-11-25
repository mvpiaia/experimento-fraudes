"""Detectores de anomalia sobre as configuracoes nativas sem reamostragem.

Resposta ao item G3 da revisao: o titulo e o resumo prometem identificacao de
anomalias com estrategias supervisionadas e nao supervisionadas, mas todos os
experimentos eram de classificacao com rotulos completos. Este executor
acrescenta o Isolation Forest (Liu et al., 2008) como linha de base nao
supervisionada e o DevNet (Pang et al., 2019) como detector fracamente
supervisionado, treinado com um orcamento de k fraudes rotuladas.

Nenhum arquivo novo e preparado: os detectores leem as mesmas dobras que o
CatBoost e o XGBoost, no braco sem reamostragem. RUS e SMOTE poriam ate 50% de
"anomalias" no treino e destruiriam a premissa de raridade; o braco `none`
preserva a prevalencia natural. Como as dobras sao as mesmas, o partition_id
tambem e, e o resultado se pareia dobra a dobra com o boosting.

Os hiperparametros sao fixos nos valores dos autores. Escolhe-los por metrica
de validacao exigiria rotulos e tiraria do detector o carater nao
supervisionado. O unico uso de rotulos e o limiar por F2 na validacao, o mesmo
de todos os blocos.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import execution_provenance
from catboost_xgboost_specific import (
    FoldData,
    prepare_fold_data,
    result_csv_path,
    validate_native_manifest_configuration,
)
from decision_threshold import (
    DEFAULT_THRESHOLD_STRATEGY,
    REQUESTABLE_THRESHOLD_STRATEGIES,
)
from devnet_model import (
    DEFAULT_DEVNET_SETTINGS,
    DevNetSettings,
    draw_labeled_anomalies,
    labeled_anomaly_count,
    score_devnet,
    train_devnet,
)
from experiment_datasets import BLOCK_ANOMALY, get_dataset_definition
from label_free_encoding import LabelFreeEncoder
from lr_nb_knn_specific import (
    LABEL_BUDGET_NOT_APPLICABLE,
    PREPARED_STORAGE_FAILURE_EXIT_CODE,
    PreparedDatasetStorageError,
    _parse_bool,
    _parse_nullable_value,
    _score_split,
    apply_decision_thresholds_by_unit,
    is_experiment_already_recorded,
    is_resampled,
    record_algorithm_failure_to_csv,
    record_algorithm_result_to_csv,
    stage_prepared_dataset_files,
)
from model_metadata import (
    count_trainable_parameters,
    describe_model_hyperparameters,
    summarize_training,
)
from execution_failures import check_loaded_folds
from prepared_dataset_storage import (
    CATEGORICAL_HANDLING_MODE_NATIVE,
    DEFAULT_PREPARED_DATA_DIRECTORY,
    DEFAULT_RANDOM_SEED,
    build_prepared_dataset_md5,
    load_prepared_data_splits,
    normalize_categorical_handling_mode,
)


ALGORITHM_ALL = "all"
ALGORITHM_ISOLATION_FOREST = "isolation-forest"
ALGORITHM_DEVNET = "devnet"
SUPPORTED_ALGORITHMS = (ALGORITHM_ISOLATION_FOREST, ALGORITHM_DEVNET)
# Algoritmos que recebem uma fracao das fraudes do treino rotulada.
LABEL_BUDGET_ALGORITHMS = frozenset({ALGORITHM_DEVNET})
DEFAULT_LABEL_DRAW_SEED = 0
RANDOM_SEED = 42
# O Isolation Forest nao recebe rotulo algum.
UNSUPERVISED_LABEL_BUDGET = 0
UNSUPERVISED_LABEL_FRACTION = 0.0
# Valores de Liu et al. (2008): 100 arvores bastam para o comprimento medio de
# caminho convergir, e subamostras de 256 reduzem os efeitos de swamping e
# masking. contamination nao altera score_samples, entao nao afeta metrica
# alguma; fica no padrao.
ISOLATION_FOREST_PARAMETERS = {
    "n_estimators": 100,
    "max_samples": 256,
    "max_features": 1.0,
    "contamination": "auto",
    "random_state": RANDOM_SEED,
    "n_jobs": 1,
}
SCORE_COLUMN = "anomaly_score"


@dataclass(frozen=True)
class AnomalyConfiguration:
    database_type: str
    dataset_filename: str
    prediction_column_name: str
    auxiliary_data_is_test_data: bool
    categorical_encoding_strategy: str | None
    fill_na_strategy: str | None
    undersampling_technique: str | None
    undersampling_parameter_values: str | None
    test_strategy: str
    threshold_strategy: str
    random_seed: int
    configuration_md5: str
    # Parametros de EXECUCAO do DevNet: nao mudam os arquivos preparados nem o
    # configuration_md5, mas entram na linha de resultado e na chave de
    # retomada, como a ponderacao de classe no boosting. A semente do sorteio
    # e separada da semente de preparacao, que fixa a particao.
    label_fractions: tuple[float, ...] = ()
    label_draw_seeds: tuple[int, ...] = (DEFAULT_LABEL_DRAW_SEED,)


@dataclass(frozen=True)
class ExecutionArm:
    """Uma execucao: algoritmo, fracao de fraudes rotuladas e semente do sorteio.

    Todos os bracos de uma configuracao rodam sobre a MESMA copia local das
    dobras: o DevNet tem 3 fracoes x 5 sorteios, e copiar a configuracao do HD
    a cada um custaria mais que o treino da rede. O numero de fraudes
    rotuladas so e conhecido na dobra, ceil(fracao x fraudes do treino).
    """

    algorithm: str
    label_fraction: float
    # Vazia no Isolation Forest, que nao sorteia rotulo.
    execution_seed: int | str = ""

    @property
    def description(self) -> str:
        if self.algorithm in LABEL_BUDGET_ALGORITHMS:
            return (
                f"{self.algorithm}[label_fraction={self.label_fraction:g}, "
                f"draw_seed={self.execution_seed}]"
            )
        return self.algorithm


def normalize_algorithm_selection(algorithm: str) -> list[str]:
    normalized_algorithm = str(algorithm).strip().lower()
    if normalized_algorithm == ALGORITHM_ALL:
        return list(SUPPORTED_ALGORITHMS)
    if normalized_algorithm in SUPPORTED_ALGORITHMS:
        return [normalized_algorithm]
    supported_values = ", ".join((ALGORITHM_ALL, *SUPPORTED_ALGORITHMS))
    raise ValueError(f"algorithm must be one of: {supported_values}")


def validate_not_resampled(undersampling_technique: str | None) -> None:
    if is_resampled(undersampling_technique):
        raise ValueError(
            "Anomaly detection runs only on configurations without resampling: "
            f"got undersampling_technique={undersampling_technique!r}"
        )


def anomaly_split_scores(
    target: pd.Series | None,
    anomaly_scores: np.ndarray | None,
    class_values: list,
    split_name: str,
) -> dict | None:
    """Scores de um split no formato dos demais blocos.

    O escore de anomalia ocupa a coluna da classe positiva; a coluna negativa
    so completa a matriz que _score_split espera. Daqui em diante o fluxo e o
    de qualquer classificador: limiar por F2 na validacao e as mesmas metricas.
    AUC-ROC e AUC-PR dependem so da ordem dos escores, nao da escala.
    """
    if target is None or anomaly_scores is None:
        return None
    ordered_classes = sorted(class_values)
    return _score_split(
        target,
        np.column_stack([-anomaly_scores, anomaly_scores]),
        ordered_classes,
        ordered_classes,
        split_name,
        SCORE_COLUMN,
    )


def create_isolation_forest() -> IsolationForest:
    return IsolationForest(**ISOLATION_FOREST_PARAMETERS)


def execute_isolation_forest_fold(
    fold_data: FoldData, monetary_columns: tuple[str, ...] = ()
) -> dict:
    # O ajuste do codificador sem rotulo faz parte do ajuste do detector.
    fit_started_at = time.perf_counter()
    encoder = LabelFreeEncoder.fit(
        fold_data.train_features,
        fold_data.categorical_columns,
        monetary_columns=monetary_columns,
    )
    model = create_isolation_forest()
    # Treino sem rotulo: o alvo do treino nao e passado ao estimador.
    model.fit(encoder.transform(fold_data.train_features))
    fit_time = time.perf_counter() - fit_started_at

    def score(features: pd.DataFrame | None) -> np.ndarray | None:
        if features is None:
            return None
        # score_samples e maior para pontos normais; o negativo e o escore de anomalia.
        return -model.score_samples(encoder.transform(features))

    scoring_started_at = time.perf_counter()
    split_scores = {}
    for split_name, features, target in (
        ("train_data", fold_data.train_features, fold_data.train_target),
        ("validation_data", fold_data.validation_features, fold_data.validation_target),
        ("test_data", fold_data.test_features, fold_data.test_target),
    ):
        split_result = anomaly_split_scores(
            target, score(features), fold_data.class_values, split_name
        )
        if split_result is not None:
            split_scores[split_name] = split_result
    scoring_time = time.perf_counter() - scoring_started_at

    return {
        # Tempos da dobra (item G18).
        "fit_time": round(fit_time, 6),
        "scoring_time": round(scoring_time, 6),
        "split_sizes": fold_data.split_sizes,
        "fold_index": fold_data.fold_index,
        "split_md5": fold_data.split_md5,
        "partition_md5": fold_data.partition_md5,
        "train_population_md5": fold_data.train_population_md5,
        "unit_partition_md5": fold_data.unit_partition_md5,
        "train_unit_population_md5": fold_data.train_unit_population_md5,
        "evaluation_units": fold_data.evaluation_units,
        "class_values": fold_data.class_values,
        "split_scores": split_scores,
        "label_budget": UNSUPERVISED_LABEL_BUDGET,
        "label_fraction": UNSUPERVISED_LABEL_FRACTION,
        "model_hyperparameters": describe_model_hyperparameters(model),
        "training_summary": summarize_training(model),
    }


def execute_devnet_fold(
    fold_data: FoldData,
    label_fraction: float,
    label_draw_seed: int = DEFAULT_LABEL_DRAW_SEED,
    settings: DevNetSettings = DEFAULT_DEVNET_SETTINGS,
    monetary_columns: tuple[str, ...] = (),
) -> dict:
    # A rede nao aceita ausentes e depende da escala: imputacao pela mediana e
    # padronizacao, ambas ajustadas so nos atributos do treino. O indicador de
    # ausencia preserva a informacao de que o valor nao foi observado.
    fit_started_at = time.perf_counter()
    encoder = LabelFreeEncoder.fit(
        fold_data.train_features,
        fold_data.categorical_columns,
        impute_missing=True,
        standardize=True,
        monetary_columns=monetary_columns,
        missing_indicators=True,
    )
    train_matrix = encoder.transform(fold_data.train_features).to_numpy()
    positive_class = sorted(fold_data.class_values)[-1]
    is_positive = (fold_data.train_target == positive_class).to_numpy()
    label_budget = labeled_anomaly_count(int(is_positive.sum()), label_fraction)
    labeled_indices = draw_labeled_anomalies(is_positive, label_budget, label_draw_seed)
    # O unico rotulo que o modelo ve e a marca das k fraudes sorteadas.
    model = train_devnet(train_matrix, labeled_indices, settings)
    fit_time = time.perf_counter() - fit_started_at

    def score(features: pd.DataFrame | None) -> np.ndarray | None:
        if features is None:
            return None
        return score_devnet(model, encoder.transform(features).to_numpy())

    scoring_started_at = time.perf_counter()
    split_scores = {}
    for split_name, features, target in (
        ("train_data", fold_data.train_features, fold_data.train_target),
        ("validation_data", fold_data.validation_features, fold_data.validation_target),
        ("test_data", fold_data.test_features, fold_data.test_target),
    ):
        split_result = anomaly_split_scores(
            target, score(features), fold_data.class_values, split_name
        )
        if split_result is not None:
            split_scores[split_name] = split_result
    scoring_time = time.perf_counter() - scoring_started_at

    return {
        # Tempos da dobra (item G18).
        "fit_time": round(fit_time, 6),
        "scoring_time": round(scoring_time, 6),
        "split_sizes": fold_data.split_sizes,
        "fold_index": fold_data.fold_index,
        "split_md5": fold_data.split_md5,
        "partition_md5": fold_data.partition_md5,
        "train_population_md5": fold_data.train_population_md5,
        "unit_partition_md5": fold_data.unit_partition_md5,
        "train_unit_population_md5": fold_data.train_unit_population_md5,
        "evaluation_units": fold_data.evaluation_units,
        "class_values": fold_data.class_values,
        "split_scores": split_scores,
        "label_budget": label_budget,
        "label_fraction": label_fraction,
        "execution_seed": label_draw_seed,
        "model_hyperparameters": settings.as_dict(),
        "training_summary": {
            "input_size": train_matrix.shape[1],
            "missing_indicators": len(encoder.missing_indicator_columns),
            "trainable_parameters": count_trainable_parameters(model),
            "training_steps": settings.epochs * settings.batches_per_epoch,
            "labeled_anomalies": len(labeled_indices),
            # Fraudes do treino que ficaram sem rotulo: a contaminacao do
            # conjunto tratado como normal.
            "unlabeled_anomalies": int(is_positive.sum()) - len(labeled_indices),
        },
    }


def execute_anomaly_algorithm(
    algorithm: str,
    test_data_information: dict,
    prediction_column_name: str,
    undersampling_technique: str | None = None,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    label_fraction: float | None = None,
    label_draw_seed: int = DEFAULT_LABEL_DRAW_SEED,
    monetary_columns: tuple[str, ...] = (),
) -> dict:
    if algorithm == ALGORITHM_DEVNET:
        if label_fraction is None:
            raise ValueError("devnet requires a label fraction")
        fold_executor = partial(
            execute_devnet_fold,
            label_fraction=label_fraction,
            label_draw_seed=label_draw_seed,
            monetary_columns=monetary_columns,
        )
    else:
        fold_executor = partial(
            {ALGORITHM_ISOLATION_FOREST: execute_isolation_forest_fold}[algorithm],
            monetary_columns=monetary_columns,
        )

    execution_started_at = time.perf_counter()
    results = []
    for split_information in test_data_information.get("test_info", []):
        fold_data = prepare_fold_data(split_information, prediction_column_name)
        results.append(fold_executor(fold_data))
    # Como no boosting: no k-fold o limiar de cada dobra vem das outras.
    results = apply_decision_thresholds_by_unit(results, threshold_strategy)

    return {
        "algorithm": algorithm,
        "train_test_strategy": test_data_information.get("train_test_strategy"),
        "undersampling_technique": undersampling_technique,
        "results": results,
        "algorithm_time": round(time.perf_counter() - execution_started_at, 6),
    }


def build_anomaly_configuration(args: argparse.Namespace) -> AnomalyConfiguration:
    validate_not_resampled(args.undersampling_technique)
    label_fractions = tuple(args.label_fractions or ())
    if any(not 0.0 < fraction <= 1.0 for fraction in label_fractions):
        raise ValueError(f"label fractions must be in (0, 1], got {label_fractions}")
    if ALGORITHM_DEVNET in normalize_algorithm_selection(args.algorithm) and not label_fractions:
        raise ValueError("devnet requires --label-fractions")
    expected_configuration_md5 = build_prepared_dataset_md5(
        args.database_type,
        args.categorical_encoding_strategy,
        args.fill_na_strategy,
        args.test_strategy,
        args.prediction_column_name,
        args.undersampling_technique,
        args.undersampling_parameter_values,
        args.auxiliary_data_is_test_data,
        normalize_categorical_handling_mode(args.categorical_handling_mode),
        args.random_seed,
    )
    requested_configuration_md5 = (
        args.prepared_dataset_md5 or expected_configuration_md5
    ).lower()
    if requested_configuration_md5 != expected_configuration_md5:
        raise ValueError(
            "configuration_md5 does not match the informed dataset preparation "
            "parameters"
        )
    return AnomalyConfiguration(
        database_type=args.database_type,
        dataset_filename=args.dataset_filename,
        prediction_column_name=args.prediction_column_name,
        auxiliary_data_is_test_data=args.auxiliary_data_is_test_data,
        categorical_encoding_strategy=args.categorical_encoding_strategy,
        fill_na_strategy=args.fill_na_strategy,
        undersampling_technique=args.undersampling_technique,
        undersampling_parameter_values=args.undersampling_parameter_values,
        test_strategy=args.test_strategy,
        threshold_strategy=args.threshold_strategy,
        random_seed=args.random_seed,
        configuration_md5=requested_configuration_md5,
        label_fractions=label_fractions,
        label_draw_seeds=tuple(args.label_draw_seeds),
    )


def build_execution_arms(
    configuration: AnomalyConfiguration, algorithms: list[str]
) -> list[ExecutionArm]:
    arms = []
    for algorithm in algorithms:
        if algorithm not in LABEL_BUDGET_ALGORITHMS:
            arms.append(ExecutionArm(algorithm, UNSUPERVISED_LABEL_FRACTION))
            continue
        for label_fraction in configuration.label_fractions:
            for draw_seed in configuration.label_draw_seeds:
                arms.append(ExecutionArm(algorithm, label_fraction, draw_seed))
    return arms


def record_failure(
    results_directory: str | Path,
    configuration: AnomalyConfiguration,
    arm: ExecutionArm,
    elapsed_time: float,
    error: BaseException | None = None,
) -> None:
    record_algorithm_failure_to_csv(
        result_csv_path(results_directory, arm.algorithm),
        configuration.dataset_filename,
        arm.algorithm,
        configuration.categorical_encoding_strategy,
        configuration.fill_na_strategy,
        configuration.test_strategy,
        undersampling_technique=configuration.undersampling_technique,
        random_seed=configuration.random_seed,
        threshold_strategy=configuration.threshold_strategy,
        elapsed_time=elapsed_time,
        configuration_md5=configuration.configuration_md5,
        database_type=configuration.database_type,
        # Na falha o numero de fraudes rotuladas pode nao ter sido calculado;
        # a fracao identifica o braco.
        label_budget=(
            LABEL_BUDGET_NOT_APPLICABLE
            if arm.algorithm in LABEL_BUDGET_ALGORITHMS
            else UNSUPERVISED_LABEL_BUDGET
        ),
        label_fraction=arm.label_fraction,
        execution_metadata={"execution_seed": arm.execution_seed},
        error=error,
    )


def record_success(
    results_directory: str | Path,
    configuration: AnomalyConfiguration,
    execution_result: dict,
    elapsed_time: float,
) -> None:
    record_algorithm_result_to_csv(
        result_csv_path(results_directory, execution_result["algorithm"]),
        configuration.dataset_filename,
        execution_result,
        configuration.categorical_encoding_strategy,
        configuration.fill_na_strategy,
        elapsed_time,
        algorithm_time=execution_result.get("algorithm_time"),
        random_seed=configuration.random_seed,
        threshold_strategy=configuration.threshold_strategy,
        configuration_md5=configuration.configuration_md5,
        database_type=configuration.database_type,
    )


def pending_arms(
    results_directory: str | Path,
    configuration: AnomalyConfiguration,
    arms: list[ExecutionArm],
) -> list[ExecutionArm]:
    return [
        arm
        for arm in arms
        if not is_experiment_already_recorded(
            results_directory,
            configuration.configuration_md5,
            arm.algorithm,
            threshold_strategy=configuration.threshold_strategy,
            label_fraction=arm.label_fraction,
            execution_seed=arm.execution_seed,
        )
    ]


def run_configuration(args: argparse.Namespace) -> int:
    execution_started_at = time.perf_counter()
    results_directory = Path(args.results_directory)
    configuration = build_anomaly_configuration(args)
    if BLOCK_ANOMALY not in get_dataset_definition(configuration.database_type).blocks:
        # As bases de ablacao do boosting compartilham o manifesto nativo, mas
        # nao fazem parte da deteccao de anomalia.
        print(
            "Skipping configuration outside the anomaly block: "
            f"configuration_md5={configuration.configuration_md5}, "
            f"database_type={configuration.database_type}"
        )
        return 0
    arms = build_execution_arms(
        configuration, normalize_algorithm_selection(args.algorithm)
    )
    pending = (
        arms
        if args.force_reprocess
        else pending_arms(results_directory, configuration, arms)
    )
    if not pending:
        print(
            "Skipping experiment already recorded: "
            f"configuration_md5={configuration.configuration_md5}, "
            f"arms={','.join(arm.description for arm in arms)}"
        )
        return 0

    prepared_dataset_manifest = args.prepared_dataset_manifest or (
        Path(args.prepared_data_directory) / "prepared_dataset_configurations.tsv"
    )
    failed_arms = []
    try:
        validate_native_manifest_configuration(
            prepared_dataset_manifest, configuration.configuration_md5
        )
        with stage_prepared_dataset_files(
            args.prepared_data_directory,
            prepared_dataset_manifest,
            configuration.configuration_md5,
            Path(args.local_staging_directory),
        ) as staged_data_directory:
            test_data_information = load_prepared_data_splits(
                staged_data_directory,
                prepared_dataset_manifest,
                configuration.configuration_md5,
                configuration.test_strategy,
            )
            check_loaded_folds(test_data_information, configuration.prediction_column_name)
            if (
                normalize_categorical_handling_mode(
                    test_data_information.get("categorical_handling_mode")
                )
                != CATEGORICAL_HANDLING_MODE_NATIVE
            ):
                raise ValueError("Loaded prepared data is not native categorical data")

            for arm in pending:
                arm_started_at = time.perf_counter()
                try:
                    execution_result = execute_anomaly_algorithm(
                        arm.algorithm,
                        test_data_information,
                        configuration.prediction_column_name,
                        configuration.undersampling_technique,
                        configuration.threshold_strategy,
                        label_fraction=arm.label_fraction,
                        label_draw_seed=arm.execution_seed or DEFAULT_LABEL_DRAW_SEED,
                        monetary_columns=get_dataset_definition(
                            configuration.database_type
                        ).monetary_columns,
                    )
                    record_success(
                        results_directory,
                        configuration,
                        execution_result,
                        round(time.perf_counter() - arm_started_at, 6),
                    )
                except Exception as error:
                    failed_arms.append(arm)
                    print(
                        "Error running anomaly detection experiment: "
                        f"configuration_md5={configuration.configuration_md5}, "
                        f"arm={arm.description}: {error}"
                    )
                    record_failure(
                        results_directory,
                        configuration,
                        arm,
                        round(time.perf_counter() - arm_started_at, 6),
                        error,
                    )
    except PreparedDatasetStorageError:
        raise
    except Exception as error:
        print(
            "Error loading anomaly detection configuration: "
            f"configuration_md5={configuration.configuration_md5}: {error}"
        )
        for arm in pending:
            failed_arms.append(arm)
            record_failure(
                results_directory,
                configuration,
                arm,
                round(time.perf_counter() - execution_started_at, 6),
                error,
            )

    return 1 if failed_arms else 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run anomaly detectors on native, non-resampled prepared splits."
    )
    parser.add_argument(
        "--algorithm",
        default=ALGORITHM_ALL,
        choices=[ALGORITHM_ALL, *SUPPORTED_ALGORITHMS],
    )
    parser.add_argument("--dataset-id", "--database-type", dest="database_type", required=True)
    parser.add_argument("--dataset-filename", required=True)
    parser.add_argument("--prediction-column-name", required=True)
    parser.add_argument(
        "--auxiliary-data-is-test-data", default=True, type=_parse_bool
    )
    parser.add_argument(
        "--categorical-encoding-strategy", default=None, type=_parse_nullable_value
    )
    parser.add_argument(
        "--categorical-handling-mode",
        default=CATEGORICAL_HANDLING_MODE_NATIVE,
        type=_parse_nullable_value,
    )
    parser.add_argument("--fill-na-strategy", default=None, type=_parse_nullable_value)
    parser.add_argument("--test-strategy", required=True)
    parser.add_argument(
        "--threshold-strategy",
        default=DEFAULT_THRESHOLD_STRATEGY,
        choices=list(REQUESTABLE_THRESHOLD_STRATEGIES),
    )
    parser.add_argument(
        "--random-seed",
        default=DEFAULT_RANDOM_SEED,
        type=int,
        help="Seed used when preparing this configuration; part of its identity.",
    )
    parser.add_argument(
        "--undersampling-technique", default=None, type=_parse_nullable_value
    )
    parser.add_argument(
        "--undersampling-parameter-values", default=None, type=_parse_nullable_value
    )
    parser.add_argument(
        "--prepared-data-directory", default=str(DEFAULT_PREPARED_DATA_DIRECTORY)
    )
    parser.add_argument("--prepared-dataset-md5", default=None)
    parser.add_argument("--prepared-dataset-manifest", default=None)
    parser.add_argument(
        "--results-directory", default=str(Path(__file__).resolve().parent)
    )
    parser.add_argument(
        "--local-staging-directory",
        default=str(Path(__file__).resolve().parent / "prepared_datasets"),
    )
    parser.add_argument(
        "--label-fractions",
        nargs="+",
        default=None,
        type=float,
        help=(
            "DevNet: fractions of the training frauds kept labeled, in (0, 1] "
            "(required for devnet). Each fold labels ceil(fraction x frauds)."
        ),
    )
    parser.add_argument(
        "--label-draw-seeds",
        nargs="+",
        default=[DEFAULT_LABEL_DRAW_SEED],
        type=int,
        help=(
            "DevNet: seeds of the labeled-fraud draw, recorded as execution_seed. "
            "Independent of --random-seed, which fixes the partition."
        ),
    )
    parser.add_argument("--force-reprocess", action="store_true")
    return parser


def main() -> None:
    execution_provenance.guard_environment_or_exit()
    try:
        exit_code = run_configuration(build_argument_parser().parse_args())
    except PreparedDatasetStorageError as error:
        print(f"Prepared storage failure: {error}")
        raise SystemExit(PREPARED_STORAGE_FAILURE_EXIT_CODE) from error
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
