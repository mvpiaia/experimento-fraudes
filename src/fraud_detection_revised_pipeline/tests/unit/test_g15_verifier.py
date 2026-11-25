"""Itens G15 e G17: verificador de preparacoes (P10-P15) e de resultados (R1-R4).

Cada verificacao tem um caso que passa, sobre preparacoes reais de dados
sinteticos, e um caso que falha, com o defeito que ela existe para pegar.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import missing_screening  # noqa: E402
import preparation_sanity_checks as sanity  # noqa: E402

from test_g14_registry_and_categorical import _prepare, raw_data_root  # noqa: E402,F401


DATASET = "ieee-cis-transacional"
DROP = missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY
CONFIGURATIONS = (("woe", "median"), ("woe", "mean"), ("woe", DROP), ("ordered-target", "median"))


class CapturingReport:
    def __init__(self) -> None:
        self.outcomes: list[tuple[str, str, bool, str]] = []

    def add(self, check_id: str, scope: str, passed: bool, detail: str) -> None:
        self.outcomes.append((check_id, scope, passed, detail))

    def failed(self, check_id: str) -> list[str]:
        return [detail for cid, _scope, passed, detail in self.outcomes if cid == check_id and not passed]

    def passed(self, check_id: str) -> bool:
        ran = [outcome for outcome in self.outcomes if outcome[0] == check_id]
        return bool(ran) and all(outcome[2] for outcome in ran)


def _manifest_rows(md5, created, encoding, fill_na) -> list[dict]:
    """As linhas que prepare_grid_common.sh gravaria para uma configuracao."""
    return [
        {
            "database_type": DATASET,
            "dataset_id": DATASET,
            "prediction_column_name": "isFraud",
            "auxiliary_data_is_test_data": "false",
            "categorical_encoding_strategy": encoding,
            "fill_na_strategy": fill_na,
            "undersampling_technique": "none",
            "undersampling_parameter_values": "none",
            "test_strategy": "stratified-holdout",
            "random_seed": "42",
            "configuration_md5": md5,
            "dataset_md5": item.dataset_md5,
            "content_md5": item.content_md5,
            "fold_index": str(item.fold_index),
            "split": item.split_label,
            "categorical_handling_mode": "encoded",
            "preparation_report_md5": item.preparation_report_md5,
            "missing_screening_threshold": item.missing_screening_threshold,
            "missing_screening_rule": item.missing_screening_rule,
            "screened_out_columns": item.screened_out_columns,
        }
        for item in created
    ]


@pytest.fixture()
def prepared(monkeypatch, raw_data_root):  # noqa: F811
    # Os faltantes sinteticos da IEEE-CIS sao todos categoricos, e ai media e
    # mediana coincidem (a categoria ausente vira "__missing__"): o P11
    # acusaria, com razao, um eixo sem efeito. Faltantes numericos o ativam.
    transaction_path = raw_data_root / "ref-22" / "train_transaction.csv"
    transaction = pd.read_csv(transaction_path)
    transaction.loc[transaction.index % 5 == 1, "TransactionAmt"] = float("nan")
    transaction.loc[transaction.index % 3 == 0, "TransactionAmt"] *= 40
    transaction.to_csv(transaction_path, index=False)
    rows = []
    output = None
    for encoding, fill_na in CONFIGURATIONS:
        md5, created, output = _prepare(
            monkeypatch, raw_data_root, DATASET, mode="encoded", encoding=encoding, fill_na=fill_na
        )
        rows.extend(_manifest_rows(md5, created, encoding, fill_na))
    manifest = pd.DataFrame(rows)
    reports = sanity.load_fold_reports(manifest, output)
    return manifest, output, reports


def _md5_of(manifest, encoding, fill_na) -> str:
    rows = manifest[
        (manifest["categorical_encoding_strategy"] == encoding) & (manifest["fill_na_strategy"] == fill_na)
    ]
    return rows["configuration_md5"].iloc[0]


def _run_preparation_checks(manifest, directory, reports) -> CapturingReport:
    report = CapturingReport()
    sanity.check_preparation_reports(manifest, directory, reports, report)
    sanity.check_axis_effects(manifest, reports, report)
    sanity.check_missing_screening(manifest, directory, reports, report)
    sanity.check_pca_record(manifest, reports, report)
    sanity.check_configuration_identity(manifest, report)
    return report


# --- Preparacoes -----------------------------------------------------------------


def test_correct_preparations_pass_every_check(prepared):
    manifest, directory, reports = prepared
    report = _run_preparation_checks(manifest, directory, reports)
    for check_id in ("P10", "P11", "P12", "P13", "P14", "P15"):
        assert report.passed(check_id), (check_id, report.failed(check_id))


def test_encodings_produce_distinct_categorical_blocks(prepared):
    manifest, _directory, reports = prepared
    woe = reports[(_md5_of(manifest, "woe", "median"), 1)]["encoded_categorical_md5"]
    ordered = reports[(_md5_of(manifest, "ordered-target", "median"), 1)]["encoded_categorical_md5"]
    assert set(woe) == {"train", "validation", "test"}
    assert woe["train"] != ordered["train"]


def test_p10_detects_a_missing_report(prepared):
    manifest, directory, reports = prepared
    md5 = _md5_of(manifest, "woe", "mean")
    next(directory.glob(f"{md5}.fold1.preparation.json")).unlink()
    report = CapturingReport()
    sanity.check_preparation_reports(manifest, directory, reports, report)
    assert any("ausente" in detail for detail in report.failed("P10"))


def test_p10_detects_counts_that_differ_from_the_files(prepared):
    manifest, directory, reports = prepared
    md5 = _md5_of(manifest, "woe", "median")
    tampered = copy.deepcopy(reports)
    tampered[(md5, 1)]["splits"]["train"]["output"]["rows"] += 1
    report = CapturingReport()
    sanity.check_preparation_reports(manifest, directory, tampered, report)
    assert any("linhas/positivos" in detail for detail in report.failed("P10"))


def test_p11_detects_an_inert_imputation_axis(prepared):
    manifest, _directory, reports = prepared
    tampered = manifest.copy()
    median_train = tampered[
        (tampered["configuration_md5"] == _md5_of(manifest, "woe", "median")) & (tampered["split"] == "train")
    ]["content_md5"].iloc[0]
    mean_train = (tampered["configuration_md5"] == _md5_of(manifest, "woe", "mean")) & (tampered["split"] == "train")
    tampered.loc[mean_train, "content_md5"] = median_train
    report = CapturingReport()
    sanity.check_axis_effects(tampered, reports, report)
    assert report.failed("P11")
    assert report.passed("P12")


def test_p12_detects_an_inert_encoding_axis(prepared):
    manifest, _directory, reports = prepared
    tampered = copy.deepcopy(reports)
    woe_hash = tampered[(_md5_of(manifest, "woe", "median"), 1)]["encoded_categorical_md5"]["train"]
    tampered[(_md5_of(manifest, "ordered-target", "median"), 1)]["encoded_categorical_md5"]["train"] = woe_hash
    report = CapturingReport()
    sanity.check_axis_effects(manifest, tampered, report)
    assert report.failed("P12")


def test_p13_detects_a_threshold_below_the_largest_admissible(prepared):
    manifest, directory, reports = prepared
    tampered = copy.deepcopy(reports)
    screening = tampered[(_md5_of(manifest, "woe", DROP), 1)]["missing_screening"]
    # Todas as fraudes preservadas em todos os limiares: so 100 e o maior.
    screening["positives_by_threshold"] = {str(t): screening["positives_before"] for t in range(0, 101, 5)}
    screening["threshold_pct"] = 50
    report = CapturingReport()
    sanity.check_missing_screening(manifest, directory, tampered, report)
    assert any("maior admissivel 100" in detail for detail in report.failed("P13"))


def test_p13_detects_a_screened_column_left_in_the_files(prepared):
    manifest, directory, reports = prepared
    tampered = copy.deepcopy(reports)
    drop_md5 = _md5_of(manifest, "woe", DROP)
    tampered[(drop_md5, 1)]["missing_screening"]["screened_out_columns"] = ["isFraud"]
    tampered_manifest = manifest.copy()
    tampered_manifest.loc[tampered_manifest["configuration_md5"] == drop_md5, "screened_out_columns"] = "isFraud"
    report = CapturingReport()
    sanity.check_missing_screening(tampered_manifest, directory, tampered, report)
    assert any("colunas triadas presentes" in detail for detail in report.failed("P13"))


def test_p14_detects_pca_without_components(prepared):
    manifest, _directory, reports = prepared
    tampered = copy.deepcopy(reports)
    tampered[(_md5_of(manifest, "woe", "median"), 1)]["pca"] = None
    report = CapturingReport()
    sanity.check_pca_record(manifest, tampered, report)
    assert any("PCA sem componentes" in detail for detail in report.failed("P14"))


def test_p15_detects_a_parameter_outside_the_identity(prepared):
    manifest, _directory, _reports = prepared
    tampered = manifest.copy()
    tampered.loc[tampered["configuration_md5"] == _md5_of(manifest, "woe", "mean"), "fill_na_strategy"] = "mode"
    report = CapturingReport()
    sanity.check_configuration_identity(tampered, report)
    assert report.failed("P15")


# --- Resultados --------------------------------------------------------------------


def _result_row(configuration_md5, fill_na, f1, train_md5, **extra) -> dict:
    row = {
        "configuration_md5": configuration_md5,
        "dataset_id": DATASET,
        "algorithm": "logistic-regression",
        "fold_index": "1",
        "split": "test",
        "execution_status": "success",
        "categorical_encoding_strategy": "woe",
        "fill_na_strategy": fill_na,
        "undersampling_technique": "none",
        "random_seed": "42",
        "class_weight_intensity": "none",
        "train_dataset_md5": train_md5,
        "train_df_size": "10",
    }
    for metric in sanity.R2_METRICS:
        row[metric] = str(f1)
    row.update(extra)
    return row


def test_r2_detects_identical_metrics_with_different_training():
    results = pd.DataFrame([
        _result_row("a" * 32, "mean", 0.264, "t1"),
        _result_row("b" * 32, "median", 0.264, "t2"),
    ])
    report = CapturingReport()
    summary = sanity.check_one_axis_pairs(results, report, legacy=False)
    assert report.failed("R2")
    assert summary["identical_pct"].tolist() == [100.0]


def test_r2_accepts_different_metrics():
    results = pd.DataFrame([
        _result_row("a" * 32, "mean", 0.264, "t1"),
        _result_row("b" * 32, "median", 0.265, "t2"),
    ])
    report = CapturingReport()
    sanity.check_one_axis_pairs(results, report, legacy=False)
    assert report.passed("R2")


def test_r2_legacy_fails_only_when_every_pair_is_identical():
    def legacy_rows(f1_median):
        rows = []
        for fold in range(2):
            for fill_na, f1 in (("mean", 0.3 + fold), ("median", f1_median + fold)):
                row = _result_row(fill_na * 4, fill_na, f1, "")
                del row["fold_index"], row["train_dataset_md5"]
                rows.append(row)
        return pd.DataFrame(rows)

    inert = CapturingReport()
    sanity.check_one_axis_pairs(legacy_rows(0.3), inert, legacy=True)
    assert inert.failed("R2")
    active = CapturingReport()
    sanity.check_one_axis_pairs(legacy_rows(0.31), active, legacy=True)
    assert active.passed("R2")


def test_r2_skips_axes_that_cannot_act():
    # A ULB nao tem faltantes: imputacoes iguais ali sao esperadas.
    results = pd.DataFrame([
        _result_row("a" * 32, "mean", 0.2, "t1", dataset_id="ulb"),
        _result_row("b" * 32, "median", 0.2, "t2", dataset_id="ulb"),
    ])
    assert sanity.one_axis_pairs(results, legacy=False).empty


def test_r1_detects_orphans_and_divergent_parameters(prepared):
    manifest, _directory, _reports = prepared
    md5 = _md5_of(manifest, "woe", "mean")
    results = pd.DataFrame([
        _result_row(md5, "median", 0.2, "t"),
        _result_row("f" * 32, "mean", 0.2, "t"),
    ])
    report = CapturingReport()
    sanity.check_results_traceability(results, manifest, report)
    detail = report.failed("R1")[0]
    assert "1 configuracoes fora do manifesto" in detail
    assert "fill_na_strategy: median x mean" in detail


def test_r3_detects_a_training_size_mismatch(prepared):
    manifest, _directory, reports = prepared
    md5 = _md5_of(manifest, "woe", "median")
    expected_rows = reports[(md5, 1)]["splits"]["train"]["output"]["rows"]
    matching = CapturingReport()
    sanity.check_training_sizes(pd.DataFrame([_result_row(md5, "median", 0.2, "t", train_df_size=str(expected_rows))]), reports, matching)
    assert matching.passed("R3")
    mismatching = CapturingReport()
    sanity.check_training_sizes(pd.DataFrame([_result_row(md5, "median", 0.2, "t", train_df_size="1")]), reports, mismatching)
    assert mismatching.failed("R3")


def test_r4_counts_missing_runs():
    expected = pd.DataFrame([
        {"bloco": "classic", "dataset_id": DATASET, "configuration_md5": md5, "algorithm": "knn",
         "class_weight_intensity": "none", "regime": "", "label_fraction": "", "label_draw_seed": "",
         "undersampling_technique": "none", "test_strategy": "stratified-holdout", "random_seed": "42",
         "linhas_teste_previstas": 1}
        for md5 in ("a" * 32, "b" * 32, "c" * 32)
    ])
    results = pd.DataFrame([
        {"configuration_md5": "a" * 32, "algorithm": "knn", "class_weight_intensity": "none",
         "execution_status": "success", "split": "test", "fold_index": "1"},
        {"configuration_md5": "b" * 32, "algorithm": "knn", "class_weight_intensity": "none",
         "execution_status": "failure", "failure_category": "memoria"},
    ])
    report = CapturingReport()
    counts, failures = sanity.check_run_counts(results, report, expected)
    assert counts[["unidades_previstas", "unidades_sucesso", "unidades_falha", "unidades_ausente"]].iloc[0].tolist() == [3, 1, 1, 1]
    assert "1 ausentes" in report.failed("R4")[0]
    assert sorted(failures["motivo"]) == ["memoria", "nao-executada"]


def test_expected_runs_match_the_grid_identities():
    runs = sanity.expected_runs()
    assert not runs.empty
    assert runs["configuration_md5"].str.fullmatch(r"[0-9a-f]{32}").all()


def test_appendix_writes_every_table_even_without_data(tmp_path, prepared):
    manifest, _directory, reports = prepared
    written = sanity.write_appendix(tmp_path / "apendice", sanity.appendix_tables(manifest, reports))
    assert {path.name for path in written} == {
        "appendix_missing_values.csv", "appendix_row_removal.csv", "appendix_pca_components.csv",
        "appendix_axis_effects.csv", "appendix_run_counts.csv", "appendix_run_failures.csv",
        "appendix_pairing.csv", "appendix_inert_axes.csv",
        "appendix_summary.md",
    }
    removal = pd.read_csv(tmp_path / "apendice" / "appendix_row_removal.csv")
    assert len(removal) == 1 and removal["threshold_pct"].iloc[0] >= 0
    assert "Composição do `configuration_md5`" in (tmp_path / "apendice" / "appendix_summary.md").read_text()


def test_appendix_skips_the_empty_test_split_of_a_k_fold(prepared):
    manifest, _directory, reports = prepared
    # No k-fold o teste da dobra e vazio e o relatorio so traz a etapa de saida.
    md5, fold = next(iter(reports))
    report = reports[(md5, fold)]
    empty = {"output": {"rows": 0, "positives": 0, "missing_cells": 0, "rows_with_missing": 0}}
    k_fold_reports = {**reports, (md5, fold): {**report, "splits": {**report["splits"], "test": empty}}}
    missing = sanity.appendix_tables(manifest, k_fold_reports)["appendix_missing_values.csv"]
    selected = missing[(missing["configuration_md5"] == md5) & (missing["fold_index"] == fold)]
    assert "test" not in set(selected["split"])
    assert {"train", "validation"} <= set(selected["split"])
