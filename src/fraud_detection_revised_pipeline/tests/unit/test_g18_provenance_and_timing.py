"""Proveniencia, ambiente travado e tempos por dobra (item G18 da revisao).

Protege o que torna os resultados da reexecucao em tres maquinas rastreaveis:
o mesmo ambiente produz o mesmo hash nas rodas CUDA e CPU, um ambiente
divergente e registrado sem bloquear a execucao, cada linha diz commit,
ambiente e maquina
sem que isso altere a retomada, e os tempos de ajuste e de pontuacao de cada
dobra cabem no tempo total do algoritmo.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import anomaly_detection_specific as anomaly  # noqa: E402
import execution_provenance as provenance  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
import preparation_timing  # noqa: E402
from test_anomaly_detection import TARGET, _test_data_information  # noqa: E402
from test_g14_registry_and_categorical import _prepare, raw_data_root  # noqa: E402,F401
from test_model_metadata import _execution_result, _record  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_provenance():
    provenance.collect_provenance.cache_clear()
    yield
    provenance.collect_provenance.cache_clear()


def _fake_versions(monkeypatch, versions):
    monkeypatch.setattr(provenance, "installed_version", lambda name: versions.get(name))


# --- 1. Ambiente travado ------------------------------------------------------


def test_cuda_and_cpu_wheels_share_the_environment_hash(monkeypatch):
    _fake_versions(monkeypatch, {"torch": "2.13.0+cu126", "numpy": "2.5.0"})
    acer = provenance.environment_md5(["torch", "numpy"])
    _fake_versions(monkeypatch, {"torch": "2.13.0+cpu", "numpy": "2.5.0"})
    assert provenance.environment_md5(["numpy", "torch"]) == acer
    _fake_versions(monkeypatch, {"torch": "2.12.0+cpu", "numpy": "2.5.0"})
    assert provenance.environment_md5(["torch", "numpy"]) != acer


def _lock(tmp_path, body, python=None):
    lock = tmp_path / "requirements.lock.txt"
    python_line = f"# python=={python}\n" if python else ""
    lock.write_text(f"{python_line}{body}\n", encoding="utf-8")
    return lock


def test_divergent_version_is_named_but_does_not_stop_the_run(tmp_path, monkeypatch):
    _fake_versions(monkeypatch, {"scikit-learn": "1.8.0", "numpy": "2.5.0"})
    lock = _lock(tmp_path, "scikit-learn==1.9.0\nnumpy==2.5.0")
    assert (
        "scikit-learn: locked 1.9.0, installed 1.8.0"
        in provenance.environment_divergences(lock)
    )
    assert provenance.enforce_locked_environment(lock) is False


def test_missing_package_is_a_divergence_but_other_python_is_ignored(tmp_path, monkeypatch):
    _fake_versions(monkeypatch, {})
    lock = _lock(tmp_path, "xgboost==3.4.1", python="3.9")
    divergences = provenance.environment_divergences(lock)
    assert not any(line.startswith("python:") for line in divergences)
    assert "xgboost: locked 3.4.1, not installed" in divergences


def test_unlocked_environment_is_always_allowed_and_recorded(tmp_path, monkeypatch):
    _fake_versions(monkeypatch, {"numpy": "1.0.0"})
    lock = _lock(tmp_path, "numpy==2.5.0")
    monkeypatch.setattr(provenance, "LOCK_FILE", lock)
    monkeypatch.setenv(provenance.ALLOW_UNLOCKED_ENVIRONMENT, "false")
    assert provenance.enforce_locked_environment(lock) is False
    assert provenance.collect_provenance()["environment_locked"] is False


def test_dirty_tree_is_never_an_environment_guard(tmp_path, monkeypatch):
    _fake_versions(monkeypatch, {"numpy": "2.5.0"})
    lock = _lock(tmp_path, "numpy==2.5.0")
    monkeypatch.setattr(provenance, "code_version", lambda: ("abc", True))
    monkeypatch.setenv(provenance.REQUIRE_CLEAN_TREE, "true")
    assert provenance.enforce_locked_environment(lock) is True


def test_lock_file_shipped_with_the_pipeline_matches_the_requirements():
    locked, python = provenance.read_lock_file()
    assert python is None
    assert set(locked) == {
        provenance.canonical_package_name(name) for name in provenance.requirement_names()
    }
    assert all("+" not in locked_version for locked_version in locked.values())


# --- 2. Proveniencia nas linhas -------------------------------------------------


def test_success_and_failure_rows_carry_the_provenance(tmp_path):
    recorded = _record(tmp_path, _execution_result())
    for column in provenance.PROVENANCE_COLUMNS:
        assert column in recorded.columns
    assert recorded.loc[0, "code_commit"] not in ("", provenance.UNKNOWN)
    assert recorded.loc[0, "execution_host"]

    destination = tmp_path / "resultados_knn.csv"
    pipeline.record_algorithm_failure_to_csv(
        str(destination), "", "knn", "none", "mean", "stratified-holdout",
        configuration_md5="abc", database_type="eqm-ref-18",
    )
    failure = pd.read_csv(destination, dtype=str, keep_default_na=False)
    assert failure.loc[0, "execution_status"] == "failure"
    assert failure.loc[0, "cpu_model"]
    assert failure.loc[0, "environment_md5"] == recorded.loc[0, "environment_md5"]


def test_code_version_is_unknown_outside_git(monkeypatch):
    monkeypatch.setattr(provenance, "_git", lambda *arguments: None)
    monkeypatch.delenv(provenance.CODE_COMMIT_OVERRIDE, raising=False)
    assert provenance.code_version() == (provenance.UNKNOWN, provenance.UNKNOWN)
    monkeypatch.setenv(provenance.CODE_COMMIT_OVERRIDE, "deadbeef")
    assert provenance.code_version()[0] == "deadbeef"


def test_configuration_recorded_on_another_machine_is_skipped(tmp_path):
    _record(tmp_path, _execution_result())
    path = tmp_path / "resultados_logistic-regression.csv"
    recorded = pd.read_csv(path, dtype=str, keep_default_na=False)
    recorded["execution_host"] = "dell-latitude-3540"
    recorded["cpu_model"] = "Intel Core i5"
    recorded.to_csv(path, index=False)
    assert pipeline.is_experiment_already_recorded(tmp_path, "abc", "logistic-regression")


def test_provenance_never_enters_pairing_or_arm_keys():
    import preparation_sanity_checks
    import seed_variance_analysis

    for column in provenance.PROVENANCE_COLUMNS:
        assert column not in preparation_sanity_checks.R2_KEY_COLUMNS
        assert column not in seed_variance_analysis.ARM_COLUMNS


# --- 3. Tempos por dobra e da preparacao ------------------------------------------


def _assert_fold_times(execution):
    folds = execution["results"]
    assert folds
    for fold in folds:
        assert fold["fit_time"] > 0
        assert fold["scoring_time"] > 0
    assert sum(fold["fit_time"] + fold["scoring_time"] for fold in folds) <= execution[
        "algorithm_time"
    ]


def test_anomaly_folds_record_fit_and_scoring_times():
    execution = anomaly.execute_anomaly_algorithm(
        anomaly.ALGORITHM_ISOLATION_FOREST,
        _test_data_information(np.random.default_rng(0)),
        TARGET,
    )
    _assert_fold_times(execution)


def _imbalanced_split(rows, seed):
    generator = np.random.default_rng(seed)
    target = (generator.random(rows) < 0.08).astype(int)
    return pd.DataFrame(
        {"x1": target * 1.5 + generator.normal(size=rows), "x2": generator.normal(size=rows), "alvo": target}
    )


@pytest.mark.parametrize(
    "executor",
    ["execute_logistic_regression", "execute_naive_bayes", "execute_knn"],
)
def test_classic_folds_record_fit_and_scoring_times(executor):
    information = {
        "test_info": [
            {
                "train_data": _imbalanced_split(400, seed),
                "validation_data": _imbalanced_split(200, seed + 100),
                "test_data": _imbalanced_split(200, seed + 200),
            }
            for seed in range(2)
        ]
    }
    started_at = time.perf_counter()
    execution = getattr(pipeline, executor)(information, "alvo")
    elapsed = time.perf_counter() - started_at
    for fold in execution["results"]:
        assert fold["fit_time"] > 0
        assert fold["scoring_time"] > 0
    assert sum(fold["fit_time"] + fold["scoring_time"] for fold in execution["results"]) <= elapsed


def test_recorded_rows_carry_the_fold_times(tmp_path):
    recorded = _record(tmp_path, _execution_result(fit_time=1.5, scoring_time=0.25))
    assert recorded.loc[0, "fit_time"] == "1.5"
    assert recorded.loc[0, "scoring_time"] == "0.25"
    assert recorded.loc[0, "pretraining_time"] == ""


def test_boosting_fold_records_fit_and_scoring_times():
    pytest.importorskip("catboost")
    import catboost_xgboost_specific as boosting
    from prepared_dataset_storage import FIRST_FOLD_INDEX

    rng = np.random.default_rng(0)
    features = pd.DataFrame({"amount": rng.normal(size=60)})
    target = pd.Series((features["amount"] > 0).astype(int))
    fold_data = boosting.FoldData(
        train_features=features, train_target=target,
        validation_features=None, validation_target=None,
        test_features=features, test_target=target,
        categorical_columns=[], class_values=[0, 1],
        split_sizes={"train_data": 60, "validation_data": 0, "test_data": 60},
        fold_index=FIRST_FOLD_INDEX, split_md5={}, partition_md5={},
    )
    fold = boosting.execute_xgboost_fold(fold_data)
    assert fold["fit_time"] > 0
    assert fold["scoring_time"] > 0


def test_preparation_writes_the_time_of_each_executed_step(raw_data_root, monkeypatch):
    configuration_md5, _created, output = _prepare(
        monkeypatch,
        raw_data_root,
        "ieee-cis-transacional",
        mode="encoded",
        encoding="smoothed-mean:k-fold",
        fill_na="mean",
    )
    timing = json.loads(
        preparation_timing.preparation_timing_path(output, configuration_md5).read_text()
    )
    steps = timing["step_seconds"]
    for step in ("structural_load", "partition", "monetary_transform", "imputation",
                 "encoding", "standardization", "writing"):
        assert steps[step] >= 0
    # Etapas que nao rodaram ficam ausentes, e nao com zero.
    assert "screening" not in steps
    assert "resampling" not in steps
    assert "transformer_preprocessing" not in steps
    assert timing["total_seconds"] >= sum(steps.values())
    assert timing["fold_count"] >= 1
    assert timing["execution_host"]
