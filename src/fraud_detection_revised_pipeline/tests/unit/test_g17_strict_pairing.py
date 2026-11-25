"""Item G17: pareamento estrito no script da discussao (gerar_discussao_resultados)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_grid  # noqa: E402
import gerar_discussao_resultados as discussion  # noqa: E402


@pytest.fixture(autouse=True)
def clean_pair_counts(monkeypatch):
    monkeypatch.setattr(discussion, "PAIR_COUNTS", [])
    monkeypatch.setattr(discussion, "EXPECTED_EXECUTIONS", None)


def _execution(encoding, technique, algorithm="knn", f1=0.5, status="success",
               protocol="stratified-holdout", folds=None, partition="p", database="ieee-cis-completa"):
    """Linhas de resultado (treino, validacao e teste) de uma execucao."""
    folds = folds or ([1, 2, 3, 4, 5] if protocol == "stratified-k-fold" else [1])
    rows = []
    for fold in folds:
        for split, value in (("train", 0.99), ("validation", 0.9), ("test", f1)):
            rows.append({
                "configuration_md5": f"{database}-{encoding}-{technique}-{protocol}",
                "database_type": database, "algorithm": algorithm,
                "categorical_encoding_strategy": encoding, "fill_na_strategy": "median",
                "undersampling_technique": technique, "train_validation_test_strategy": protocol,
                "random_seed": 42, "class_weight_intensity": "none",
                "threshold_strategy": "validation-f2", "regime": None, "label_fraction": -1,
                "execution_seed": None, "execution_status": status, "split": split,
                "evaluation_unit": "row", "fold_index": fold,
                "partition_id": f"{partition}{fold}",
                **{metric: (value if status == "success" else -1) for metric in discussion.METRICS},
            })
    return rows


def _expected(*combinations):
    return pd.DataFrame([
        {"database_type": "ieee-cis-completa", "algorithm": algorithm,
         "categorical_encoding_strategy": encoding, "fill_na_strategy": "median",
         "undersampling_technique": technique, "train_validation_test_strategy": "stratified-holdout",
         "random_seed": "42", "class_weight_intensity": "none", "regime": "",
         "label_fraction": "", "label_draw_seed": ""}
        for algorithm, encoding, technique in combinations
    ])


def _counts(name):
    rows = [row for row in discussion.PAIR_COUNTS if row["comparacao"] == name]
    return pd.DataFrame(rows).drop(columns=["comparacao", "fator", "positivo", "negativo"]).sum(numeric_only=True)


def test_only_test_rows_enter_the_execution_mean():
    executions = discussion.aggregate_folds(pd.DataFrame(_execution("woe", "none", f1=0.4)))
    assert len(executions) == 1
    assert executions["f1"].iloc[0] == pytest.approx(0.4)
    assert bool(executions["complete"].iloc[0])


def test_resampling_does_not_collapse_encoding_pairs():
    raw = pd.DataFrame([
        *_execution("ordered-target", "tomek-links", f1=0.6),
        *_execution("woe", "tomek-links", f1=0.5),
        *_execution("ordered-target", "RUS+SMOTE", f1=0.3),
        *_execution("woe", "RUS+SMOTE", f1=0.4),
    ])
    expected = _expected(*[("knn", e, t) for e in ("ordered-target", "woe") for t in ("tomek-links", "RUS+SMOTE")])
    deltas = discussion.paired_comparison(
        discussion.aggregate_folds(raw), "categorical_encoding_strategy", "ordered-target", "woe",
        name="cod", expected=expected,
    )
    # Dois pares, um por tecnica, e nao uma media das duas.
    assert len(deltas) == 2
    assert sorted(deltas["f1"].round(6)) == [-0.1, 0.1]
    counts = _counts("cod")
    assert (counts["previstos"], counts["realizados"]) == (2, 2)


def test_failed_side_is_discarded_and_counted():
    raw = pd.DataFrame([
        *_execution("ordered-target", "tomek-links", status="failure"),
        *_execution("woe", "tomek-links"),
        *_execution("ordered-target", "RUS+SMOTE", f1=0.3),
        *_execution("woe", "RUS+SMOTE", f1=0.4),
    ])
    expected = _expected(*[("knn", e, t) for e in ("ordered-target", "woe") for t in ("tomek-links", "RUS+SMOTE")])
    deltas = discussion.paired_comparison(
        discussion.aggregate_folds(raw), "categorical_encoding_strategy", "ordered-target", "woe",
        name="cod", expected=expected,
    )
    assert deltas["undersampling_technique"].tolist() == ["RUS+SMOTE"]
    counts = _counts("cod")
    assert (counts["previstos"], counts["realizados"], counts["descartados_falha-positivo"]) == (2, 1, 1)


def test_pair_absent_on_both_sides_still_counts_as_planned():
    raw = pd.DataFrame([*_execution("ordered-target", "RUS+SMOTE"), *_execution("woe", "RUS+SMOTE")])
    expected = _expected(*[("knn", e, t) for e in ("ordered-target", "woe") for t in ("tomek-links", "RUS+SMOTE")])
    discussion.paired_comparison(
        discussion.aggregate_folds(raw), "categorical_encoding_strategy", "ordered-target", "woe",
        name="cod", expected=expected,
    )
    counts = _counts("cod")
    assert (counts["previstos"], counts["realizados"], counts["descartados_falha-ambos"]) == (2, 1, 1)


def test_incomplete_kfold_and_divergent_partition_are_discarded():
    raw = pd.DataFrame([
        *_execution("ordered-target", "tomek-links", protocol="stratified-k-fold", folds=[1, 2, 3, 4]),
        *_execution("woe", "tomek-links", protocol="stratified-k-fold"),
        *_execution("ordered-target", "RUS+SMOTE", partition="a"),
        *_execution("woe", "RUS+SMOTE", partition="b"),
    ])
    discussion.paired_comparison(
        discussion.aggregate_folds(raw), "categorical_encoding_strategy", "ordered-target", "woe", name="cod",
    )
    counts = _counts("cod")
    assert counts["descartados_incompleto"] == 1
    assert counts["descartados_particao-divergente"] == 1
    assert counts["realizados"] == 0


def test_duplicate_execution_key_is_refused():
    executions = discussion.aggregate_folds(pd.DataFrame([
        *_execution("ordered-target", "tomek-links"), *_execution("woe", "tomek-links"),
    ]))
    duplicated = pd.concat([executions, executions.assign(configuration_md5="outra")], ignore_index=True)
    with pytest.raises(ValueError, match="mesma chave de par"):
        discussion.paired_comparison(duplicated, "categorical_encoding_strategy", "ordered-target", "woe")


def test_cell_members_have_the_same_n_on_both_levels():
    raw = pd.DataFrame([
        *_execution("ordered-target", "tomek-links", status="failure"),
        *_execution("woe", "tomek-links"),
        *_execution("ordered-target", "RUS+SMOTE"),
        *_execution("woe", "RUS+SMOTE"),
    ])
    members = discussion.paired_members(
        discussion.aggregate_folds(raw), "undersampling_technique", ["RUS+SMOTE", "tomek-links"], "fig",
    )
    # Tomek so tem a celula WoE; o par de reamostragem completo e o de WoE.
    assert members.groupby("undersampling_technique").size().to_dict() == {"RUS+SMOTE": 1, "tomek-links": 1}
    assert set(members["categorical_encoding_strategy"]) == {"woe"}


def test_classic_order_follows_the_revised_grid():
    assert discussion.CLASSIC_ORDER == list(experiment_grid.CLASSIC_ALGORITHMS)
    assert "linear-regression" not in discussion.CLASSIC_ORDER
