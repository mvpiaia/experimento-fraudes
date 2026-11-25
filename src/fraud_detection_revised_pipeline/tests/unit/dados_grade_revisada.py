"""Resultados sintéticos no esquema da grade revisada, para os testes da discussão.

Gera ``resultados_<algoritmo>.csv`` com linhas de validação e de teste por
dobra, nas colunas que os executores revisados gravam. As métricas são
coerentes com uma matriz de confusão (100 positivos em 10 mil), para que as
análises que reconstroem a matriz funcionem.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

POSITIVES = 100
TEST_SIZE = 10_000

CLASSIC = ("logistic-regression", "naive-bayes", "knn")
BOOSTING = ("catboost", "xgboost")
ANOMALY = ("isolation-forest", "devnet")
FT = ("ft-transformer-mlp", "ft-transformer-lstm")
TABBERT = ("tabbert-mlp", "tabbert-lstm")

SOURCES = ("ulb", "sparkov", "ieee-cis-completa", "ieee-cis-sem-identidade", "home-credit-completa")
TEMPORAL_SOURCES = {"sparkov", "ieee-cis-completa", "ieee-cis-sem-identidade"}
ENCODINGS = ("smoothed-mean:k-fold", "smoothed-mean:permutation-prefix", "log-odds:k-fold", "log-odds:permutation-prefix")


def _metrics(rng: np.random.Generator, quality: float) -> dict:
    true_positive = int(np.clip(rng.normal(60 + 30 * quality, 5), 5, 99))
    false_positive = int(np.clip(rng.normal(80 - 50 * quality, 10), 1, 500))
    false_negative = POSITIVES - true_positive
    negatives = TEST_SIZE - POSITIVES
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / POSITIVES
    return {
        "f1": 2 * precision * recall / (precision + recall),
        "f_beta_2": 5 * true_positive / (5 * true_positive + 4 * false_negative + false_positive),
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive / negatives,
        "accuracy": (true_positive + negatives - false_positive) / TEST_SIZE,
        "roc_auc": float(np.clip(0.85 + 0.1 * quality + rng.normal(0, 0.01), 0.5, 0.999)),
        "auc_pr": float(np.clip(precision * 0.9, 0.01, 0.99)),
        "test_df_size": TEST_SIZE,
        "fit_time": 1.0,
        "scoring_time": 0.1,
    }


def _protocols(source: str, transformer: bool = False) -> list[str]:
    protocols = ["stratified-holdout"] if transformer else ["stratified-holdout", "stratified-k-fold"]
    if source in TEMPORAL_SOURCES:
        protocols.append("temporal-holdout")
    return protocols


def _emit(rows, rng, quality, folds, **axes):
    md5 = "-".join(str(value) for value in axes.values() if value is not None)
    for fold in range(folds):
        base = {
            "configuration_md5": axes.get("configuration_md5", md5),
            "dataset_id": axes["database_type"],
            "evaluation_unit": "row",
            "execution_status": "success",
            "fold_index": fold,
            "partition_id": f"p{fold}",
            "random_seed": 42,
            "threshold_strategy": "validation-f-beta-2",
            **{key: value for key, value in axes.items() if key != "configuration_md5"},
        }
        rows.append({**base, "split": "validation_data", **_metrics(rng, quality)})
        rows.append({**base, "split": "test_data", **_metrics(rng, quality)})


def build(directory: Path, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    directory.mkdir(parents=True, exist_ok=True)
    per_algorithm: dict[str, list[dict]] = {}
    quality = {alg: q for alg, q in zip(CLASSIC + BOOSTING + ANOMALY + FT + TABBERT,
                                         (0.3, 0.1, 0.5, 0.8, 0.7, 0.2, 0.4, 0.6, 0.5, 0.65, 0.6))}
    for source in SOURCES:
        for protocol in _protocols(source):
            folds = 5 if protocol == "stratified-k-fold" else 1
            for algorithm, encoding, fill, technique in itertools.product(
                CLASSIC, ENCODINGS[:2], ("mean", "median"), ("none", "random-undersampling", "RUS+SMOTE", "tomek-links")
            ):
                rows = per_algorithm.setdefault(algorithm, [])
                bonus = 0.05 if technique == "tomek-links" and algorithm == "knn" else 0.0
                _emit(rows, rng, quality[algorithm] + bonus, folds, database_type=source, algorithm=algorithm,
                      train_validation_test_strategy=protocol, categorical_encoding_strategy=encoding,
                      fill_na_strategy=fill, undersampling_technique=technique, class_weight_intensity="none")
                if algorithm == "logistic-regression" and technique == "none":
                    for weight in ("sqrt-balanced", "balanced"):
                        _emit(rows, rng, quality[algorithm] + 0.03, folds, database_type=source, algorithm=algorithm,
                              train_validation_test_strategy=protocol, categorical_encoding_strategy=encoding,
                              fill_na_strategy=fill, undersampling_technique=technique, class_weight_intensity=weight,
                              configuration_md5=f"{source}-{protocol}-{encoding}-{fill}-none")
            for algorithm, fill, technique in itertools.product(BOOSTING, ("native", "mean"), ("none", "random-undersampling")):
                identity = 0.01 if source == "ieee-cis-completa" else 0.0
                _emit(per_algorithm.setdefault(algorithm, []), rng, quality[algorithm] + identity, folds,
                      database_type=source, algorithm=algorithm, train_validation_test_strategy=protocol,
                      fill_na_strategy=fill, undersampling_technique=technique, class_weight_intensity="none",
                      configuration_md5=f"{source}-{protocol}-{fill}-{technique}-boost")
            _emit(per_algorithm.setdefault("isolation-forest", []), rng, quality["isolation-forest"], folds,
                  database_type=source, algorithm="isolation-forest", train_validation_test_strategy=protocol,
                  undersampling_technique="none", label_fraction=0.0)
            for fraction, draw in itertools.product((0.01, 0.05, 0.10), range(5)):
                _emit(per_algorithm.setdefault("devnet", []), rng, quality["devnet"] + 2 * fraction, folds,
                      database_type=source, algorithm="devnet", train_validation_test_strategy=protocol,
                      undersampling_technique="none", label_fraction=fraction, execution_seed=draw)
        for protocol in _protocols(source, transformer=True):
            for algorithm, normalization, technique in itertools.product(FT, ("quantile", "standard"), ("none", "random-undersampling")):
                _emit(per_algorithm.setdefault(algorithm, []), rng, quality[algorithm], 1, database_type=source,
                      algorithm=algorithm, train_validation_test_strategy=protocol, numeric_normalization=normalization,
                      undersampling_technique=technique)
            for algorithm, regime in itertools.product(TABBERT, ("supervised-only", "mlm-before-finetune", "warmup-mlm-finetune")):
                _emit(per_algorithm.setdefault(algorithm, []), rng, quality[algorithm], 1, database_type=source,
                      algorithm=algorithm, train_validation_test_strategy=protocol, regime=regime,
                      undersampling_technique="none")
    for algorithm, rows in per_algorithm.items():
        pd.DataFrame(rows).to_csv(directory / f"resultados_{algorithm}.csv", index=False)
    return directory
