"""Item G19: o script da discussao nao gera as figuras retiradas da dissertacao."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import gerar_discussao_resultados as discussion  # noqa: E402


REMOVED_FIGURES = {
    "eqm_ref18_atributos_classes.png",
    "eqm_ref18_algoritmos_classicos.png",
    "eqm_ref18_particionamento.png",
    "eqm_ref18_preprocessamento.png",
    "eqm_ref18_limiar_05.png",
    "eqm_ref18_distribuicao_metricas.png",
    "pca_efeito_eqm_ref18.png",
    "comparacao_bases_pca.png",
}
REMOVED_SUMMARY_KEYS = {"classic_ref18_by_algorithm", "classic_ref18_best_f1", "pca_eqm18_pca_minus_eqm18"}
DATABASES = ["eqm-ref-18", "eqm-ref-19", "eqm-ref-20", "eqm-ref-22"]
STRATEGIES = ["random-split", "stratified-train-test-split", "stratified-k-fold", "stratified-train-validation-test"]
TEST_SIZES = ["0.2:0.2", "0.3:0.3", "0.4:0.4"]


def _rows(algorithms, databases, techniques, encodings, rng):
    """Resultados no esquema de 22/09 (sem split), com metricas aleatorias."""
    rows = []
    for database, algorithm, technique, size, encoding, fill_na, strategy in itertools.product(
        databases, algorithms, techniques, TEST_SIZES, encodings, ["mean", "median"], STRATEGIES
    ):
        md5 = f"{database}-{technique}-{size}-{encoding}-{fill_na}-{strategy}"
        for _fold in range(discussion.expected_folds(strategy)):
            metrics = rng.uniform(0.05, 0.95, size=6)
            rows.append({
                "configuration_md5": md5, "database_type": database, "algorithm": algorithm,
                "execution_status": "success", "undersampling_technique": technique,
                "test_data_size": size, "categorical_encoding_strategy": encoding,
                "fill_na_strategy": fill_na, "train_validation_test_strategy": strategy,
                **dict(zip(["f1", "f_beta_2", "accuracy", "precision", "recall", "roc_auc"], metrics)),
            })
    return rows


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    rng = np.random.default_rng(19)
    results = tmp_path_factory.mktemp("resultados")
    figures = tmp_path_factory.mktemp("figuras")
    classic = _rows(
        discussion.CLASSIC_ORDER, [*DATABASES, "eqm-ref-18-pca"], ["RUS+SMOTE", "tomek-links"],
        ["ordered-target", "woe"], rng,
    )
    pd.DataFrame(classic).to_csv(results / "resultados_combinados.csv", index=False)
    for algorithm in discussion.BOOST_ORDER:
        boost = _rows([algorithm], DATABASES, ["random-undersampling"], [None], rng)
        pd.DataFrame(boost).to_csv(results / f"resultas_{algorithm}.csv", index=False)

    summary = figures / "resumo.json"
    argv = [
        "gerar_discussao_resultados.py", "--results-directory", str(results),
        "--figures-directory", str(figures), "--summary-output", str(summary),
    ]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(sys, "argv", argv)
    # main() redefine globais do modulo; registra-las aqui faz o undo restaura-las.
    for name in ("SOURCE", "FIGURES", "SUMMARY", "CLASSIC_ORDER", "EXPECTED_EXECUTIONS"):
        monkeypatch.setattr(discussion, name, getattr(discussion, name, None), raising=False)
    monkeypatch.setattr(discussion, "PAIR_COUNTS", [])
    try:
        discussion.main()
    finally:
        monkeypatch.undo()
    return figures, json.loads(summary.read_text(encoding="utf-8"))


def test_removed_figures_are_not_written(generated):
    figures, _summary = generated
    written = {path.name for path in figures.glob("*.png")}
    assert "resultados_classicos_algoritmo_base.png" in written
    assert not written & REMOVED_FIGURES


def test_removed_summary_keys_are_gone(generated):
    _figures, summary = generated
    assert not set(summary) & REMOVED_SUMMARY_KEYS


def test_legacy_ulb_pca_rows_are_dropped_on_load(generated):
    _figures, summary = generated
    assert "eqm-ref-18-pca" not in summary["coverage"]["classic_databases"]
    assert "eqm-ref-18-pca" in discussion.REMOVED_DATABASES


def test_no_database_order_lists_the_removed_base():
    for order in (discussion.DATABASE_ORDER_CLASSIC, discussion.ALL_DATABASE_ORDER, discussion.MAJORITY_CLASS_BASELINE_ACCURACY):
        assert "eqm-ref-18-pca" not in order
