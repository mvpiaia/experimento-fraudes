"""Item G17: grade prevista de todos os blocos e contagem de execucoes (R4)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_grid  # noqa: E402
import preparation_sanity_checks as sanity  # noqa: E402

from test_g15_verifier import CapturingReport  # noqa: E402


def _run(md5, technique="none", protocol="stratified-holdout", dataset="ieee-cis-completa",
         algorithm="knn", **extra):
    row = {
        "bloco": "classic", "dataset_id": dataset, "configuration_md5": md5, "algorithm": algorithm,
        "class_weight_intensity": "none", "regime": "", "label_fraction": "", "label_draw_seed": "",
        "undersampling_technique": technique, "test_strategy": protocol, "random_seed": "42",
        "linhas_teste_previstas": 5 if protocol == "stratified-k-fold" else 1,
    }
    row.update(extra)
    return row


def _success(md5, folds=(1,), algorithm="knn", **extra):
    rows = []
    for fold in folds:
        for split in ("train", "validation", "test"):
            rows.append({
                "configuration_md5": md5, "algorithm": algorithm, "class_weight_intensity": "none",
                "execution_status": "success", "split": split, "evaluation_unit": "row",
                "fold_index": str(fold), **extra,
            })
    return rows


def _totals_close(counts: pd.DataFrame) -> bool:
    states = sanity.RUN_STATES
    units = counts[[f"unidades_{state}" for state in states]].sum(axis=1)
    lines = counts[[f"linhas_{state}" for state in states]].sum(axis=1)
    return bool((units == counts["unidades_previstas"]).all() and (lines == counts["linhas_previstas"]).all())


def test_anomaly_grid_has_one_forest_and_one_devnet_per_fraction(monkeypatch):
    monkeypatch.setenv("LABEL_FRACTIONS", "0.01 0.05 0.10")
    monkeypatch.setenv("LABEL_DRAW_SEEDS", "0")
    runs = sanity.expected_runs((experiment_grid.BLOCK_ANOMALY,))
    per_configuration = runs.groupby("configuration_md5")["algorithm"].value_counts().unstack()
    assert (per_configuration["isolation-forest"] == 1).all()
    assert (per_configuration["devnet"] == 3).all()
    assert set(runs.loc[runs["algorithm"] == "devnet", "label_fraction"]) == {"0.01", "0.05", "0.1"}


def test_seed_list_multiplies_every_block(monkeypatch):
    monkeypatch.setenv("LABEL_DRAW_SEEDS", "0 1")
    single = sanity.expected_runs((experiment_grid.BLOCK_ANOMALY,))
    monkeypatch.setenv("SEED_LIST", "42 7")
    double = sanity.expected_runs((experiment_grid.BLOCK_ANOMALY,))
    assert len(double) == 2 * len(single)
    assert set(single["label_draw_seed"]) == {"", "0", "1"}


def test_only_blocks_present_in_results_are_expected():
    classic = sanity.expected_runs((experiment_grid.BLOCK_CLASSIC,))
    md5 = classic["configuration_md5"].iloc[0]
    report = CapturingReport()
    counts, _failures = sanity.check_run_counts(pd.DataFrame(_success(md5)), report)
    assert set(counts["bloco"]) == {experiment_grid.BLOCK_CLASSIC}


def test_kfold_with_missing_fold_is_incomplete():
    expected = pd.DataFrame([_run("a" * 32, protocol="stratified-k-fold")])
    results = pd.DataFrame(_success("a" * 32, folds=(1, 2, 3, 4)))
    report = CapturingReport()
    counts, failures = sanity.check_run_counts(results, report, expected)
    row = counts.iloc[0]
    assert (row["unidades_incompleta"], row["linhas_previstas"], row["linhas_sucesso"]) == (1, 5, 4)
    assert row["linhas_incompleta"] == 1
    assert failures["estado"].tolist() == ["incompleta"]
    assert _totals_close(counts)


def test_failures_concentrated_in_one_technique_are_visible():
    expected = pd.DataFrame([
        _run("a" * 32, technique="tomek-links", dataset="home-credit-completa"),
        _run("b" * 32, technique="RUS+SMOTE", dataset="home-credit-completa"),
    ])
    results = pd.DataFrame([
        *_success("b" * 32),
        {"configuration_md5": "a" * 32, "algorithm": "knn", "execution_status": "failure",
         "failure_category": "memoria", "failure_message": "Unable to allocate"},
    ])
    report = CapturingReport()
    counts, failures = sanity.check_run_counts(results, report, expected)
    by_technique = counts.set_index("undersampling_technique")["unidades_falha"]
    assert by_technique.to_dict() == {"RUS+SMOTE": 0, "tomek-links": 1}
    assert failures[["undersampling_technique", "motivo", "mensagem"]].values.tolist() == [
        ["tomek-links", "memoria", "Unable to allocate"]
    ]
    assert _totals_close(counts)


def test_absence_is_explained_by_the_execution_log():
    expected = pd.DataFrame([
        _run("a" * 32, algorithm="catboost", bloco="boosting", class_weight_intensity="balanced"),
    ])
    log = pd.DataFrame([
        {"configuration_md5": "a" * 32, "algoritmo": "catboost xgboost",
         "class_weight_intensity": "balanced", "label_fraction": "", "regime": "",
         "tentativa": "1", "codigo_saida": "137", "categoria_saida": "memoria"},
    ])
    report = CapturingReport()
    _counts, failures = sanity.check_run_counts(
        pd.DataFrame(columns=["configuration_md5", "algorithm"]), report, expected, execution_log=log
    )
    assert failures[["estado", "motivo", "tentativas"]].values.tolist() == [["ausente", "memoria", 1]]


def test_retried_run_ends_as_success():
    expected = pd.DataFrame([_run("a" * 32)])
    results = pd.DataFrame([
        {"configuration_md5": "a" * 32, "algorithm": "knn", "execution_status": "failure",
         "failure_category": "armazenamento"},
        *_success("a" * 32),
    ])
    report = CapturingReport()
    counts, failures = sanity.check_run_counts(results, report, expected)
    assert counts["unidades_sucesso"].iloc[0] == 1 and failures.empty
    assert report.outcomes[-1][2]


def test_devnet_draw_seeds_are_distinct_runs():
    expected = pd.DataFrame([
        _run("a" * 32, algorithm="devnet", bloco="anomaly", label_fraction="0.05", label_draw_seed=seed)
        for seed in ("0", "1")
    ])
    results = pd.DataFrame(
        _success("a" * 32, algorithm="devnet", label_fraction="0.05", execution_seed="0")
    )
    report = CapturingReport()
    _counts, failures = sanity.check_run_counts(results, report, expected)
    assert failures["label_draw_seed"].tolist() == ["1"]


def test_legacy_results_are_read_with_unknown_reason():
    expected = pd.DataFrame([_run("a" * 32)])
    results = pd.DataFrame([
        {"configuration_md5": "a" * 32, "algorithm": "knn", "execution_status": "failure"},
    ])
    report = CapturingReport()
    _counts, failures = sanity.check_run_counts(results, report, expected)
    assert failures["motivo"].tolist() == ["desconhecida"]
    assert "legado" in report.outcomes[-1][3]


def test_kfold_folds_are_evaluated_on_their_validation_split():
    """No k-fold o teste da dobra e vazio: a validacao da dobra e a avaliacao."""
    md5 = "a" * 32
    expected = pd.DataFrame([_run(md5, protocol="stratified-k-fold")])
    results = pd.DataFrame([
        {"configuration_md5": md5, "algorithm": "knn", "class_weight_intensity": "none",
         "execution_status": "success", "split": split, "evaluation_unit": unit,
         "fold_index": str(fold), "train_validation_test_strategy": "stratified-k-fold"}
        for fold in range(1, 6)
        for split, unit in (("train", "row"), ("validation", "row"), ("validation", "client"))
    ])
    counts, failures = sanity.check_run_counts(results, CapturingReport(), expected)
    row = counts.iloc[0]
    assert (row["unidades_sucesso"], row["linhas_sucesso"]) == (1, 5)
    assert failures.empty


def test_holdout_validation_rows_do_not_count_as_evaluation():
    md5 = "b" * 32
    expected = pd.DataFrame([_run(md5)])
    results = pd.DataFrame([
        {"configuration_md5": md5, "algorithm": "knn", "class_weight_intensity": "none",
         "execution_status": "success", "split": split, "evaluation_unit": "row",
         "fold_index": "1", "train_validation_test_strategy": "stratified-holdout"}
        for split in ("train", "validation")
    ])
    counts, _failures = sanity.check_run_counts(results, CapturingReport(), expected)
    assert counts.iloc[0]["unidades_sucesso"] == 0
