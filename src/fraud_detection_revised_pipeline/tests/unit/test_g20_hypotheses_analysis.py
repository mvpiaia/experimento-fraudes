"""Analise das hipoteses H1-H3 e das questoes Q1-Q3 (item G20)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import analise_hipoteses as analysis  # noqa: E402


POSITIVES = 100
TEST_SIZE = 10_000


def _metrics(true_positive: int, false_positive: int) -> dict:
    """Metricas coerentes com uma matriz de confusao de 100 positivos em 10 mil."""
    negatives = TEST_SIZE - POSITIVES
    false_negative = POSITIVES - true_positive
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    return {
        "f_beta_2": 5 * true_positive / (5 * true_positive + 4 * false_negative + false_positive),
        "precision": precision,
        "recall": true_positive / POSITIVES,
        "false_positive_rate": false_positive / negatives,
        "auc_pr": precision,
        "roc_auc": 0.9,
        "test_df_size": TEST_SIZE,
        "fit_time": 1.0,
        "scoring_time": 0.1,
    }


def _rows(
    algorithm: str,
    dataset: str,
    *,
    validation: float,
    test: tuple[int, int],
    protocol: str = analysis.TEMPORAL,
    technique: str = "none",
    md5: str | None = None,
    seed: int = 42,
    split_suffix: str = "_data",
) -> list[dict]:
    base = {
        "configuration_md5": md5 or f"{algorithm}-{dataset}-{protocol}-{technique}",
        "dataset_id": dataset,
        "database_type": dataset,
        "algorithm": algorithm,
        "train_validation_test_strategy": protocol,
        "undersampling_technique": technique,
        "random_seed": seed,
        "execution_status": "success",
        "evaluation_unit": "row",
    }
    validation_row = {**base, "split": f"validation{split_suffix}", **_metrics(*test), "f_beta_2": validation}
    test_row = {**base, "split": f"test{split_suffix}", **_metrics(*test)}
    return [validation_row, test_row]


def _normalized(rows: list[dict]) -> pd.DataFrame:
    return analysis.normalize_results(pd.DataFrame(rows))


def _leaders_grid(leaders: dict[str, str], seeds=(42,)) -> list[dict]:
    """Tres algoritmos nas tres fontes; o lider de cada fonte tem 90 acertos."""
    rows = []
    for seed in seeds:
        for source in analysis.FRAUD_SOURCES:
            for algorithm, hits in (("knn", 50), ("catboost", 50), ("xgboost", 40)):
                true_positive = 90 if leaders[source] == algorithm else hits
                rows += _rows(algorithm, source, validation=true_positive / 100,
                              test=(true_positive - (seed % 3), 30), seed=seed)
    return rows


def test_selection_uses_validation_and_reads_test():
    rows = (
        _rows("knn", "ulb", validation=0.6, test=(40, 30), md5="a")
        + _rows("knn", "ulb", validation=0.5, test=(95, 10), md5="b")
    )
    executions, _ = analysis.build_executions(_normalized(rows))
    chosen = analysis.select_representatives(executions, ["algorithm"]).iloc[0]
    assert chosen["configuration_md5"] == "a"
    assert chosen["f_beta_2"] == pytest.approx(_metrics(40, 30)["f_beta_2"])


def test_normalization_strips_split_suffix_and_resolves_aliases():
    rows = _rows("knn", "eqm-ref-18", validation=0.5, test=(50, 10))
    data = _normalized(rows)
    assert set(data["split"]) == {"validation", "test"}
    assert set(data["database_type"]) == {"ulb"}


def test_h2_refuted_and_h3_impaired_when_leader_changes():
    rows = _leaders_grid({"ulb": "knn", "sparkov": "catboost", "ieee-cis-completa": "catboost"})
    _, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=200)
    assert summary["H2"]["outcome"] == analysis.OUTCOME_REFUTED
    assert summary["H3"]["outcome"] == analysis.OUTCOME_IMPAIRED


def test_h2_corroborated_with_paired_seeds():
    rows = _leaders_grid({source: "catboost" for source in analysis.FRAUD_SOURCES}, seeds=(1, 2, 3))
    tables, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=200)
    assert summary["H2"]["outcome"] == analysis.OUTCOME_CORROBORATED
    assert summary["uncertainty_units"] == ["seed"]
    mean = tables["h2_ranking_medio"]
    assert mean.iloc[0]["algorithm"] == "catboost"
    assert mean.iloc[0]["mean_rank_f_beta_2"] == 1.0
    assert summary["H2"]["friedman_f_beta_2"]["n_sources"] == 3
    assert summary["H2"]["friedman_f_beta_2"]["k_algorithms"] == 3


def test_single_seed_uses_confusion_bootstrap():
    rows = _leaders_grid({source: "catboost" for source in analysis.FRAUD_SOURCES})
    _, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=500)
    assert summary["uncertainty_units"] == ["confusion"]
    detail = summary["H2"]["leader_vs_runner_up"][0]
    assert detail["ci_low"] < detail["difference"] < detail["ci_high"]
    # 90 contra 50 acertos em 100 positivos: diferenca grande, IC acima de zero.
    assert summary["H2"]["outcome"] == analysis.OUTCOME_CORROBORATED


def test_estimate_positives_recovers_test_positives():
    rows = _rows("knn", "ulb", validation=0.5, test=(37, 120))
    _, per_seed_test = analysis.build_executions(_normalized(rows))
    positives = analysis.estimate_positives(per_seed_test)
    assert list(positives.values()) == [POSITIVES]


def test_h1_refuted_when_rus_smote_is_better_beyond_margin():
    rows = []
    for seed in (1, 2, 3):
        for source in analysis.FRAUD_SOURCES:
            for algorithm in analysis.CLASSIC_ALGORITHMS:
                rows += _rows(algorithm, source, validation=0.4, test=(40 + seed, 30), seed=seed)
                rows += _rows(algorithm, source, validation=0.8, test=(90 + seed, 30), seed=seed,
                              technique=analysis.SYNTHETIC_TECHNIQUE)
    tables, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=200)
    assert summary["H1"]["outcome"] == analysis.OUTCOME_REFUTED
    assert summary["H1"]["compared_pairs"] == summary["H1"]["expected_pairs"]
    assert (tables["h1_pares"]["uncertainty_unit"] == "seed").all()


def test_h1_inconclusive_when_pairs_are_missing():
    rows = _rows("knn", "ulb", validation=0.5, test=(50, 10))
    _, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=50)
    assert summary["H1"]["outcome"] == analysis.OUTCOME_INCONCLUSIVE


def test_home_credit_is_ranked_apart():
    rows = _leaders_grid({source: "catboost" for source in analysis.FRAUD_SOURCES})
    for algorithm in ("knn", "catboost", "xgboost"):
        rows += _rows(algorithm, analysis.HOME_CREDIT, validation=0.5, test=(60, 20),
                      protocol=analysis.STRATIFIED)
    tables, _ = analysis.run_analysis(_normalized(rows), bootstrap_samples=50)
    assert analysis.HOME_CREDIT not in set(tables["h2_postos"]["dataset_id"])
    assert set(tables["h2_home_credit"]["dataset_id"]) == {analysis.HOME_CREDIT}


def test_algorithm_missing_a_source_is_excluded_from_ranking():
    rows = _leaders_grid({source: "catboost" for source in analysis.FRAUD_SOURCES})
    rows += _rows("tabbert-mlp", "sparkov", validation=0.99, test=(99, 1))
    _, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=50)
    assert "tabbert-mlp" in summary["H2"]["algorithms_excluded_missing_sources"]
    assert "tabbert-mlp" not in summary["H2"]["algorithms_ranked"]


def test_questions_are_exploratory_and_q1_compares_protocols():
    rows = _leaders_grid({source: "catboost" for source in analysis.FRAUD_SOURCES})
    for source in analysis.FRAUD_SOURCES:
        rows += _rows("catboost", source, validation=0.9, test=(95, 10), protocol=analysis.STRATIFIED)
    tables, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=50)
    for question in ("Q1", "Q2", "Q3"):
        assert summary[question]["kind"] == "exploratory"
        assert "outcome" not in summary[question]
    q1 = tables["q1_temporal"]
    assert set(q1["algorithm"]) == {"catboost"}
    assert (q1["f_beta_2_temporal_minus_stratified"] < 0).all()


def test_outputs_are_strict_json(tmp_path):
    rows = _leaders_grid({"ulb": "knn", "sparkov": "catboost", "ieee-cis-completa": "catboost"})
    tables, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=50)
    analysis.write_outputs(tables, summary, tmp_path)
    loaded = json.loads((tmp_path / "resumo_hipoteses.json").read_text(encoding="utf-8"))
    assert loaded["H2"]["outcome"] == analysis.OUTCOME_REFUTED
    assert (tmp_path / "h1_pares.csv").exists()


def test_empty_results_do_not_fail(tmp_path):
    tables, summary = analysis.run_analysis(analysis.normalize_results(pd.DataFrame()), bootstrap_samples=10)
    assert summary["H1"]["outcome"] == analysis.OUTCOME_INCONCLUSIVE
    assert summary["H2"]["outcome"] == analysis.OUTCOME_INCONCLUSIVE
    analysis.write_outputs(tables, summary, tmp_path)


def test_h1_margin_sensitivity_reports_each_margin():
    table = pd.DataFrame([
        {"algorithm": "knn", "dataset_id": "ulb", "status": "compared", "ci_low": -0.015, "ci_high": 0.0},
        {"algorithm": "knn", "dataset_id": "sparkov", "status": "compared", "ci_low": -0.004, "ci_high": 0.01},
    ])
    assert analysis.h1_outcome(table, 0.005) == analysis.OUTCOME_INCONCLUSIVE
    assert analysis.h1_outcome(table, 0.010) == analysis.OUTCOME_INCONCLUSIVE
    assert analysis.h1_outcome(table, 0.020) == analysis.OUTCOME_CORROBORATED

    sensitivity = analysis.h1_margin_sensitivity(table)
    assert sorted(sensitivity["margin"].unique()) == list(analysis.H1_SENSITIVITY_MARGINS)
    assert sensitivity["primary"].sum() == len(table)
    ulb = sensitivity[sensitivity["dataset_id"].eq("ulb")].set_index("margin")["pair_outcome"]
    assert ulb[0.005] == analysis.OUTCOME_INCONCLUSIVE
    assert ulb[0.020] == analysis.OUTCOME_CORROBORATED
    sparkov = sensitivity[sensitivity["dataset_id"].eq("sparkov")].set_index("margin")["pair_outcome"]
    assert set(sparkov) == {analysis.OUTCOME_CORROBORATED}


def test_h1_summary_records_margin_sensitivity():
    rows = []
    for source in analysis.FRAUD_SOURCES:
        rows += _rows("knn", source, validation=0.6, test=(60, 30), technique="none")
        rows += _rows("knn", source, validation=0.5, test=(90, 30), technique=analysis.SYNTHETIC_TECHNIQUE)
    tables, summary = analysis.run_analysis(_normalized(rows), bootstrap_samples=200)
    sensitivity = summary["H1"]["margin_sensitivity"]
    assert set(sensitivity) == {"0.005", "0.010", "0.020"}
    assert sensitivity["0.010"] == summary["H1"]["outcome"]
    assert "margin_sensitivity_stable" in summary["H1"]
    assert not tables["h1_sensibilidade_margem"].empty
