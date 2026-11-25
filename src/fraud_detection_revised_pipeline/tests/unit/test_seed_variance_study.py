"""Estudo de variancia por semente (item G13 da revisao).

Protege o corte reduzido da grade e a agregacao em dois niveis do resumo.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_grid  # noqa: E402
import seed_variance_analysis  # noqa: E402


FILTER_VARIABLES = ("PROTOCOL_FILTER", "UNDERSAMPLING_FILTER", "FILL_NA_FILTER")


@pytest.fixture
def clean_filters(monkeypatch):
    for variable in (*FILTER_VARIABLES, "DATASET_FILTER", "SEED_LIST"):
        monkeypatch.delenv(variable, raising=False)
    return monkeypatch


# ---------------------------------------------------------------------------
# Filtros da grade
# ---------------------------------------------------------------------------


def test_grid_is_unchanged_without_filters(clean_filters):
    entries = experiment_grid.build_classic_grid()
    assert experiment_grid.apply_axis_filters(entries) == entries
    assert {entry.test_strategy for entry in entries} > {"stratified-holdout"}


def test_protocol_and_undersampling_filters_are_strict(clean_filters):
    clean_filters.setenv("PROTOCOL_FILTER", "stratified-holdout")
    clean_filters.setenv("UNDERSAMPLING_FILTER", "none")

    for builder in (experiment_grid.build_classic_grid, experiment_grid.build_boosting_grid):
        entries = builder()
        assert entries
        assert {entry.test_strategy for entry in entries} == {"stratified-holdout"}
        assert {entry.undersampling_technique for entry in entries} == {"none"}


def test_fill_filter_keeps_datasets_without_missing_values(clean_filters):
    clean_filters.setenv("FILL_NA_FILTER", "median")
    entries = experiment_grid.build_boosting_grid()

    fill_by_dataset = {}
    for entry in entries:
        fill_by_dataset.setdefault(entry.database_type, set()).add(entry.fill_na_strategy)
    # A ULB nao tem ausentes: sua unica entrada, none, permanece.
    assert fill_by_dataset["ulb"] == {"none"}
    assert fill_by_dataset["ieee-cis-completa"] == {"median"}


def test_seed_list_multiplies_the_reduced_grid(clean_filters):
    clean_filters.setenv("DATASET_FILTER", "eqm-ref-18")
    clean_filters.setenv("PROTOCOL_FILTER", "stratified-holdout")
    clean_filters.setenv("UNDERSAMPLING_FILTER", "none")
    clean_filters.setenv("SEED_LIST", "42 7 2026")

    entries = experiment_grid.build_boosting_grid()

    assert sorted(entry.random_seed for entry in entries) == [7, 42, 2026]


# ---------------------------------------------------------------------------
# Resumo
# ---------------------------------------------------------------------------


def _row(seed, auc, fold=1, status="success", split="test", algorithm="catboost", **extra):
    return {
        "database_type": "eqm-ref-18",
        "algorithm": algorithm,
        "random_seed": str(seed),
        "fold_index": str(fold),
        "split": split,
        "execution_status": status,
        "roc_auc": str(auc),
        "auc_pr": "0.5",
        **extra,
    }


def test_folds_are_averaged_per_seed_before_the_dispersion():
    results = pd.DataFrame(
        [
            # Semente 42 com duas dobras: media 0,90.
            _row(42, 0.80, fold=1),
            _row(42, 1.00, fold=2),
            _row(7, 0.70),
        ]
    )

    summary = seed_variance_analysis.summarize_seed_variance(results)
    auc = summary[summary["metric"] == "roc_auc"].iloc[0]

    assert auc["seed_count"] == 2
    assert auc["mean"] == pytest.approx(0.80)
    assert auc["min"] == pytest.approx(0.70)
    assert auc["max"] == pytest.approx(0.90)
    assert auc["range"] == pytest.approx(0.20)


def test_failures_and_non_test_splits_are_excluded():
    results = pd.DataFrame(
        [
            _row(42, 0.90),
            _row(7, 0.80),
            _row(13, -1, status="failure"),
            _row(42, 0.10, split="validation"),
        ]
    )
    auc = seed_variance_analysis.summarize_seed_variance(results)
    auc = auc[auc["metric"] == "roc_auc"].iloc[0]
    assert auc["seed_count"] == 2
    assert auc["mean"] == pytest.approx(0.85)


def test_arms_are_summarized_separately_and_columns_are_complete():
    results = pd.DataFrame(
        [
            _row(42, 0.9, algorithm="devnet", label_fraction="0.01", execution_seed="0"),
            _row(42, 0.7, algorithm="devnet", label_fraction="0.01", execution_seed="1"),
            _row(7, 0.6, algorithm="devnet", label_fraction="0.01", execution_seed="0"),
            _row(42, 0.95, algorithm="devnet", label_fraction="0.05", execution_seed="0"),
        ]
    )
    summary = seed_variance_analysis.summarize_seed_variance(results)
    auc = summary[summary["metric"] == "roc_auc"].set_index("label_fraction")

    # Os dois sorteios da semente 42 viram 0,80 antes da dispersao.
    assert auc.loc["0.01", "mean"] == pytest.approx((0.80 + 0.60) / 2)
    assert auc.loc["0.05", "seed_count"] == 1
    for column in seed_variance_analysis.SUMMARY_STATISTICS:
        assert column in summary.columns


def test_study_results_are_read_from_every_result_file(tmp_path):
    pd.DataFrame([_row(42, 0.9)]).to_csv(tmp_path / "resultados_catboost.csv", index=False)
    pd.DataFrame([_row(42, 0.8, algorithm="knn")]).to_csv(
        tmp_path / "resultados_knn.csv", index=False
    )
    loaded = seed_variance_analysis.load_study_results(tmp_path)
    assert set(loaded["algorithm"]) == {"catboost", "knn"}

    with pytest.raises(FileNotFoundError):
        seed_variance_analysis.load_study_results(tmp_path / "vazio")
