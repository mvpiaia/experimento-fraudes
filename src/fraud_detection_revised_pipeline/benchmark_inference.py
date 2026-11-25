"""Benchmark de inferencia por algoritmo e base (item G18 da revisao).

O revisor pediu o tempo de inferencia por 1.000 transacoes, que a grade nao
mede. A grade roda em tres maquinas, e tempos de maquinas diferentes nao se
comparam; por isso a inferencia e medida aqui, numa etapa propria, numa
maquina so (o Acer), para todos os algoritmos e bases.

Para cada algoritmo e base, o benchmark escolhe uma configuracao de referencia
do manifesto (semente 42, ``stratified-holdout``, sem reamostragem, imputacao
pela mediana quando ela existe), ajusta o modelo uma vez na primeira dobra e
mede sobre o teste:

* lote: ``inference_ms_per_1000``, a mediana de cinco passadas cronometradas
  depois de uma passada de aquecimento;
* unitaria: ``latency_p50_ms`` e ``latency_p95_ms`` sobre uma amostra de
  transacoes pontuadas uma a uma, com os limites do intervalo de confianca de
  95% do p95 livre de distribuicao (estatisticas de ordem da binomial).

O tamanho padrao da amostra, 1.000, vem da precisao do p95: o intervalo do
quantil p tem meia-largura d com n ~ z^2 p (1 - p) / d^2; com p = 0,95,
z = 1,96 e d = 0,015, n ~ 811 (Conover, Practical Nonparametric Statistics;
Hahn e Meeker, Statistical Intervals). Com 1.000, o IC do p95 fica entre os
quantis 93,7 e 96,4.

A latencia inclui a transformacao que o executor aplica no momento de pontuar
(padronizacao dos classicos, conversao de categorias do boosting, codificacao
sem rotulo e padronizacao dos detectores de anomalia). As transformacoes da
preparacao (imputacao, codificacao supervisionada, log1p e PCA) ja estao
gravadas nos arquivos preparados e nao entram: o benchmark mede o modelo e a
transformacao em tempo de pontuacao, nao a preparacao completa de uma
transacao bruta.

Os Transformers ainda nao tem adaptador: o executor deles monta o modelo
dentro da dobra e nao o devolve.

Uso:
    python3 benchmark_inference.py --prepared-data-directory <dir> \\
        [--prepared-dataset-manifest <tsv>] [--results-directory <dir>] \\
        [--algorithm all|<algoritmo> ...] [--dataset <dataset_id> ...]
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

import execution_provenance
from catboost_xgboost_specific import (
    build_xgboost_category_domains,
    create_catboost_classifier,
    create_xgboost_classifier,
    prepare_catboost_dataframe,
    prepare_fold_data,
    prepare_xgboost_dataframe,
)
from anomaly_detection_specific import create_isolation_forest
from devnet_model import (
    DEFAULT_DEVNET_SETTINGS,
    draw_labeled_anomalies,
    labeled_anomaly_count,
    score_devnet,
    train_devnet,
)
from experiment_datasets import (
    BLOCK_ANOMALY,
    BLOCK_BOOSTING,
    BLOCK_CLASSIC,
    get_dataset_definition,
)
from label_free_encoding import LabelFreeEncoder
import lr_nb_knn_specific as pipeline
from prepared_dataset_storage import (
    CATEGORICAL_HANDLING_MODE_ENCODED,
    CATEGORICAL_HANDLING_MODE_NATIVE,
    load_prepared_data_splits,
    normalize_manifest_dataset_identity,
)

RESULT_FILENAME = "resultados_benchmark_inferencia.csv"
REFERENCE_TEST_STRATEGY = "stratified-holdout"
REFERENCE_RANDOM_SEED = "42"
REFERENCE_FILL_NA_STRATEGY = "median"
NO_RESAMPLING_VALUES = frozenset({"", "none", "nan"})
DEFAULT_BATCH_REPEATS = 5
# O lote e cronometrado numa amostra fixa do teste: no KNN, pontuar o teste
# inteiro das bases maiores leva mais de uma hora por passada.
DEFAULT_BATCH_SAMPLE_SIZE = 10_000
BATCH_SAMPLE_SEED = 42
DEFAULT_LATENCY_SAMPLE_SIZE = 1000
LATENCY_SAMPLE_SEED = 42
P95 = 0.95
Z_95 = 1.959963984540054
# O DevNet precisa de fraudes rotuladas para ajustar; a fracao nao altera o
# custo de pontuar.
DEVNET_REFERENCE_LABEL_FRACTION = 0.10

BENCHMARK_COLUMNS = (
    "algorithm",
    "block",
    "dataset_id",
    "configuration_md5",
    "categorical_encoding_strategy",
    "fill_na_strategy",
    "test_rows",
    "feature_count",
    "fit_time",
    "batch_repeats",
    "batch_sample_size",
    "inference_ms_per_1000",
    "latency_sample_size",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p95_ci_low_ms",
    "latency_p95_ci_high_ms",
    "threads",
    *execution_provenance.PROVENANCE_COLUMNS,
)


@dataclass(frozen=True)
class FittedScorer:
    """Modelo ajustado e a funcao que pontua um dataframe de atributos."""

    score: Callable[[pd.DataFrame], np.ndarray]
    features: pd.DataFrame
    fit_time: float
    threads: str


# ---------------------------------------------------------------------------
# Adaptadores: espelham o ajuste e a pontuacao de cada executor.
# ---------------------------------------------------------------------------


def _classic_scorer(algorithm: str, fold: dict, prediction_column: str) -> FittedScorer:
    executor = {
        "logistic-regression": pipeline.execute_logistic_regression,
        "naive-bayes": pipeline.execute_naive_bayes,
        "knn": pipeline.execute_knn,
        "linear-regression": pipeline.execute_linear_regression,
    }[algorithm]
    started_at = time.perf_counter()
    fitted = executor(
        {"test_info": [fold]}, prediction_column, execution_device="cpu"
    )["results"][0]
    fit_time = time.perf_counter() - started_at
    model = fitted["model"]
    scaler = fitted.get("feature_scaler")
    columns = list(fitted["feature_columns"])

    def score(features: pd.DataFrame) -> np.ndarray:
        matrix = features[columns].astype(float)
        if scaler is not None:
            matrix = scaler.transform(matrix)
        if algorithm == "linear-regression":
            return np.asarray(model.predict(matrix))
        return model.predict_proba(matrix)[:, 1]

    test = fold["test_data"]
    return FittedScorer(score, test.drop(columns=[prediction_column]), fit_time, "1")


def _native_fold(fold: dict, prediction_column: str):
    return prepare_fold_data(fold, prediction_column)


def _boosting_scorer(algorithm: str, fold: dict, prediction_column: str) -> FittedScorer:
    data = _native_fold(fold, prediction_column)
    categorical = list(data.categorical_columns)
    started_at = time.perf_counter()
    if algorithm == "catboost":
        model = create_catboost_classifier(categorical)
        model.fit(prepare_catboost_dataframe(data.train_features, categorical), data.train_target)

        def transform(features):
            return prepare_catboost_dataframe(features, categorical)
    else:
        domains = build_xgboost_category_domains(data.train_features, categorical)
        model = create_xgboost_classifier(
            imbalance_ratio=pipeline.compute_class_imbalance_ratio(
                data.train_target, data.class_values
            )
        )
        model.fit(prepare_xgboost_dataframe(data.train_features, categorical, domains), data.train_target)

        def transform(features):
            return prepare_xgboost_dataframe(features, categorical, domains)

    fit_time = time.perf_counter() - started_at

    def score(features: pd.DataFrame) -> np.ndarray:
        return model.predict_proba(transform(features))[:, 1]

    return FittedScorer(score, data.test_features, fit_time, "1")


def _anomaly_scorer(algorithm: str, fold: dict, prediction_column: str, dataset_id: str) -> FittedScorer:
    data = _native_fold(fold, prediction_column)
    monetary_columns = get_dataset_definition(dataset_id).monetary_columns
    started_at = time.perf_counter()
    if algorithm == "isolation-forest":
        encoder = LabelFreeEncoder.fit(
            data.train_features, data.categorical_columns, monetary_columns=monetary_columns
        )
        model = create_isolation_forest()
        model.fit(encoder.transform(data.train_features))

        def score(features: pd.DataFrame) -> np.ndarray:
            return -model.score_samples(encoder.transform(features))
    else:
        encoder = LabelFreeEncoder.fit(
            data.train_features,
            data.categorical_columns,
            impute_missing=True,
            standardize=True,
            monetary_columns=monetary_columns,
            missing_indicators=True,
        )
        matrix = encoder.transform(data.train_features).to_numpy()
        positive_class = sorted(data.class_values)[-1]
        is_positive = (data.train_target == positive_class).to_numpy()
        labeled = draw_labeled_anomalies(
            is_positive,
            labeled_anomaly_count(int(is_positive.sum()), DEVNET_REFERENCE_LABEL_FRACTION),
            0,
        )
        model = train_devnet(matrix, labeled, DEFAULT_DEVNET_SETTINGS)

        def score(features: pd.DataFrame) -> np.ndarray:
            return score_devnet(model, encoder.transform(features).to_numpy())

    fit_time = time.perf_counter() - started_at
    return FittedScorer(score, data.test_features, fit_time, "1")


ALGORITHM_BLOCKS = {
    "logistic-regression": BLOCK_CLASSIC,
    "naive-bayes": BLOCK_CLASSIC,
    "knn": BLOCK_CLASSIC,
    "linear-regression": BLOCK_CLASSIC,
    "catboost": BLOCK_BOOSTING,
    "xgboost": BLOCK_BOOSTING,
    "isolation-forest": BLOCK_ANOMALY,
    "devnet": BLOCK_ANOMALY,
}
BLOCK_HANDLING_MODES = {
    BLOCK_CLASSIC: CATEGORICAL_HANDLING_MODE_ENCODED,
    BLOCK_BOOSTING: CATEGORICAL_HANDLING_MODE_NATIVE,
    BLOCK_ANOMALY: CATEGORICAL_HANDLING_MODE_NATIVE,
}


def fit_scorer(algorithm: str, fold: dict, prediction_column: str, dataset_id: str) -> FittedScorer:
    block = ALGORITHM_BLOCKS[algorithm]
    if block == BLOCK_CLASSIC:
        return _classic_scorer(algorithm, fold, prediction_column)
    if block == BLOCK_BOOSTING:
        return _boosting_scorer(algorithm, fold, prediction_column)
    return _anomaly_scorer(algorithm, fold, prediction_column, dataset_id)


# ---------------------------------------------------------------------------
# Medicao
# ---------------------------------------------------------------------------


def quantile_confidence_ranks(sample_size: int, quantile: float = P95, z: float = Z_95) -> tuple[int, int]:
    """Postos (1-based) dos limites do IC de 95% de um quantil empirico.

    Aproximacao normal da binomial das estatisticas de ordem: os limites sao
    as estatisticas de ordem nos postos n p -/+ z sqrt(n p (1 - p)).
    """
    center = sample_size * quantile
    spread = z * math.sqrt(sample_size * quantile * (1 - quantile))
    low = max(1, math.floor(center - spread))
    high = min(sample_size, math.ceil(center + spread) + 1)
    return low, high


def batch_sample(features: pd.DataFrame, sample_size: int) -> pd.DataFrame:
    """Amostra fixa do teste (semente BATCH_SAMPLE_SEED) para o tempo em lote; 0 usa o teste inteiro."""
    if sample_size <= 0 or sample_size >= len(features):
        return features
    rng = np.random.default_rng(BATCH_SAMPLE_SEED)
    positions = np.sort(rng.choice(len(features), size=sample_size, replace=False))
    return features.iloc[positions]


def measure_batch(score, features: pd.DataFrame, repeats: int) -> float:
    """Milissegundos por 1.000 transacoes: mediana das passadas apos aquecimento."""
    score(features)
    durations = []
    for _ in range(repeats):
        started_at = time.perf_counter()
        score(features)
        durations.append(time.perf_counter() - started_at)
    return statistics.median(durations) / len(features) * 1_000_000


def measure_latency(score, features: pd.DataFrame, sample_size: int) -> dict:
    """p50, p95 e IC do p95 de transacoes pontuadas uma a uma, em ms."""
    rng = np.random.default_rng(LATENCY_SAMPLE_SEED)
    size = min(sample_size, len(features))
    positions = np.sort(rng.choice(len(features), size=size, replace=False))
    score(features.iloc[[int(positions[0])]])
    latencies = []
    for position in positions:
        row = features.iloc[[int(position)]]
        started_at = time.perf_counter()
        score(row)
        latencies.append((time.perf_counter() - started_at) * 1000)
    ordered = np.sort(np.asarray(latencies))
    low, high = quantile_confidence_ranks(size)
    return {
        "latency_sample_size": size,
        "latency_p50_ms": float(np.quantile(ordered, 0.5)),
        "latency_p95_ms": float(np.quantile(ordered, P95)),
        "latency_p95_ci_low_ms": float(ordered[low - 1]),
        "latency_p95_ci_high_ms": float(ordered[high - 1]),
    }


def _is_without_resampling(values: pd.Series) -> pd.Series:
    return values.fillna("").astype(str).str.strip().str.lower().isin(NO_RESAMPLING_VALUES)


def select_reference_configuration(manifest: pd.DataFrame, dataset_id: str, block: str) -> pd.Series | None:
    """Configuracao de referencia de uma base num bloco, ou None se nao houver."""
    rows = manifest[
        (manifest["dataset_id"] == dataset_id)
        & (manifest["categorical_handling_mode"] == BLOCK_HANDLING_MODES[block])
        & (manifest["test_strategy"] == REFERENCE_TEST_STRATEGY)
        & (manifest["random_seed"].astype(str) == REFERENCE_RANDOM_SEED)
        & _is_without_resampling(manifest["undersampling_technique"])
    ]
    if rows.empty:
        return None
    rows = rows.assign(
        _prefers_median=(rows["fill_na_strategy"] != REFERENCE_FILL_NA_STRATEGY).astype(int)
    ).sort_values(
        ["_prefers_median", "fill_na_strategy", "categorical_encoding_strategy", "configuration_md5"],
        na_position="last",
    )
    return rows.iloc[0]


def benchmark_algorithm(
    algorithm: str,
    reference: pd.Series,
    prepared_directory: Path,
    manifest_path: Path,
    batch_repeats: int,
    latency_sample_size: int,
    batch_sample_size: int = DEFAULT_BATCH_SAMPLE_SIZE,
) -> dict:
    information = load_prepared_data_splits(
        prepared_directory, manifest_path, reference["configuration_md5"], REFERENCE_TEST_STRATEGY
    )
    fold = information["test_info"][0]
    prediction_column = reference["prediction_column_name"]
    scorer = fit_scorer(algorithm, fold, prediction_column, reference["dataset_id"])
    return {
        "algorithm": algorithm,
        "block": ALGORITHM_BLOCKS[algorithm],
        "dataset_id": reference["dataset_id"],
        "configuration_md5": reference["configuration_md5"],
        "categorical_encoding_strategy": reference.get("categorical_encoding_strategy", ""),
        "fill_na_strategy": reference.get("fill_na_strategy", ""),
        "test_rows": len(scorer.features),
        "feature_count": scorer.features.shape[1],
        "fit_time": round(scorer.fit_time, 6),
        "batch_repeats": batch_repeats,
        "batch_sample_size": len(batch_sample(scorer.features, batch_sample_size)),
        "inference_ms_per_1000": measure_batch(
            scorer.score, batch_sample(scorer.features, batch_sample_size), batch_repeats
        ),
        **measure_latency(scorer.score, scorer.features, latency_sample_size),
        "threads": scorer.threads,
        **execution_provenance.collect_provenance(),
    }


def run_benchmark(args: argparse.Namespace) -> int:
    prepared_directory = Path(args.prepared_data_directory)
    manifest_path = Path(
        args.prepared_dataset_manifest
        or prepared_directory / "prepared_dataset_configurations.tsv"
    )
    manifest = normalize_manifest_dataset_identity(
        pd.read_csv(manifest_path, sep="\t", dtype=str)
    )
    manifest = manifest.drop_duplicates("configuration_md5")
    algorithms = (
        list(ALGORITHM_BLOCKS) if "all" in args.algorithm else list(args.algorithm)
    )
    datasets = args.dataset or sorted(manifest["dataset_id"].dropna().unique())
    output = Path(args.results_directory) / RESULT_FILENAME
    output.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    for dataset_id in datasets:
        definition = get_dataset_definition(dataset_id)
        for algorithm in algorithms:
            block = ALGORITHM_BLOCKS[algorithm]
            if block not in definition.blocks:
                continue
            reference = select_reference_configuration(manifest, dataset_id, block)
            if reference is None:
                print(f"No reference configuration: dataset={dataset_id}, algorithm={algorithm}")
                continue
            try:
                row = benchmark_algorithm(
                    algorithm, reference, prepared_directory, manifest_path,
                    args.batch_repeats, args.latency_sample_size, args.batch_sample_size,
                )
            except Exception as error:  # um algoritmo falho nao derruba os demais
                failures += 1
                print(f"Benchmark failed: dataset={dataset_id}, algorithm={algorithm}: {error}")
                continue
            pd.DataFrame([row], columns=BENCHMARK_COLUMNS).to_csv(
                output, mode="a", header=not output.exists(), index=False
            )
            print(
                f"{algorithm} {dataset_id}: {row['inference_ms_per_1000']:.3f} ms/1000, "
                f"p95 {row['latency_p95_ms']:.3f} ms"
            )
    return 1 if failures else 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inference benchmark on one machine (G18).")
    parser.add_argument("--prepared-data-directory", required=True)
    parser.add_argument("--prepared-dataset-manifest", default=None)
    parser.add_argument("--results-directory", default=str(Path(__file__).resolve().parent))
    parser.add_argument(
        "--algorithm", nargs="+", default=["all"], choices=["all", *ALGORITHM_BLOCKS]
    )
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--batch-repeats", type=int, default=DEFAULT_BATCH_REPEATS)
    parser.add_argument("--latency-sample-size", type=int, default=DEFAULT_LATENCY_SAMPLE_SIZE)
    parser.add_argument("--batch-sample-size", type=int, default=DEFAULT_BATCH_SAMPLE_SIZE,
                        help="Transacoes do teste no tempo em lote (amostra fixa); 0 usa o teste inteiro.")
    return parser


def main() -> None:
    execution_provenance.guard_environment_or_exit()
    raise SystemExit(run_benchmark(build_argument_parser().parse_args()))


if __name__ == "__main__":
    main()
