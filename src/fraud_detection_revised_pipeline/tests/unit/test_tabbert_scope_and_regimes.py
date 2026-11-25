"""Grade e regimes do TabBERT (item C3.19 da revisao).

Protege tres decisoes: os tres regimes (S, PM->S e S->PM->S) tem o mesmo
total de epocas supervisionadas, o antigo mlm-after-warmup (com um terco a
mais delas) e recusado, e a grade completa (as duas
cabecas e as duas estrategias de cardinalidade em todas as bases) volta a
rodar, porque a medicao no Acer mostrou que ela cabe em poucas horas.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_grid  # noqa: E402
import grid_plan  # noqa: E402
import run_transformer_prepared_experiments as runner  # noqa: E402
import transformer_models  # noqa: E402

def test_regimes_are_the_same_in_grid_and_model():
    assert experiment_grid.TABBERT_REGIMES == transformer_models.TABBERT_TRAINING_REGIMES
    assert experiment_grid.TABBERT_REGIMES == (
        "supervised-only", "mlm-before-finetune", "warmup-mlm-finetune"
    )


def test_stages_of_each_regime():
    assert transformer_models.tabbert_training_stages("supervised-only") == ("supervised",)
    assert transformer_models.tabbert_training_stages("mlm-before-finetune") == (
        "mlm",
        "supervised",
    )
    assert transformer_models.tabbert_training_stages("warmup-mlm-finetune") == (
        "supervised-warmup",
        "mlm",
        "supervised",
    )


def test_old_warmup_regime_is_refused(monkeypatch):
    with pytest.raises(ValueError, match="mlm-after-warmup"):
        transformer_models.tabbert_training_stages("mlm-after-warmup")
    monkeypatch.setattr(
        sys, "argv",
        ["runner", "--model", "tabbert-mlp", "--configuration-md5", "0" * 32,
         "--tabbert-regime", "mlm-after-warmup"],
    )
    with pytest.raises(SystemExit):
        runner.parse_arguments()


@pytest.mark.parametrize("regime", ["supervised-only", "mlm-before-finetune", "warmup-mlm-finetune"])
def test_every_regime_trains_the_same_supervised_epochs(monkeypatch, regime):
    calls = []
    monkeypatch.setattr(
        runner, "train_tabbert_supervised",
        lambda model, data, args, settings, epochs: calls.append(("supervised", epochs)),
    )
    monkeypatch.setattr(
        runner, "train_tabbert_mlm",
        lambda model, data, args, settings: calls.append(("mlm", args.mlm_epochs)),
    )
    args = SimpleNamespace(tabbert_regime=regime, epochs=10, mlm_epochs=3)
    stage_epochs, pretraining_time = runner.train_tabbert_regime(
        None, None, args, SimpleNamespace(device="cpu")
    )
    supervised = sum(epochs for stage, epochs in calls if stage == "supervised")
    assert supervised == 10
    if regime == "supervised-only":
        assert calls == [("supervised", 10)]
        assert pretraining_time is None
        assert stage_epochs == {"supervised": 10}
    elif regime == "mlm-before-finetune":
        assert calls == [("mlm", 3), ("supervised", 10)]
        assert pretraining_time is not None
        assert stage_epochs == {"mlm": 3, "supervised": 10}
    else:
        # Sugestao do revisor: um terco antes do pre-treino, o restante depois.
        assert calls == [("supervised", 3), ("mlm", 3), ("supervised", 7)]
        assert pretraining_time is not None
        assert stage_epochs == {"supervised-warmup": 3, "mlm": 3, "supervised": 7}


def test_every_tabbert_dataset_gets_both_cardinalities():
    entries = experiment_grid.build_tabbert_grid()
    strategies = Counter((entry.dataset_id, entry.high_cardinality_strategy) for entry in entries)
    for dataset_id in {entry.dataset_id for entry in entries}:
        assert strategies[(dataset_id, "minimum-frequency")] == 1
        assert strategies[(dataset_id, "hashing")] == 1


def test_grid_counts_twelve_configurations_and_seventy_two_runs():
    entries = experiment_grid.build_tabbert_grid()
    assert len(entries) == 12
    runs = experiment_grid.run_entries_for(experiment_grid.BLOCK_TABBERT, entries)
    assert len(runs) == 6
    assert all(run_entries == entries for _label, run_entries in runs)
    assert sum(len(run_entries) for _label, run_entries in runs) == 72


def test_plan_has_both_heads_on_every_tabbert_dataset():
    import pandas as pd

    manifest = pd.DataFrame(
        [
            {"configuration_md5": md5, "dataset_id": dataset_id, "fold_index": 0,
             "categorical_handling_mode": "sequence-tokenized",
             "undersampling_technique": "none"}
            for md5, dataset_id in (("a" * 32, "sparkov"), ("b" * 32, "ieee-cis-completa"))
        ]
    )
    pairs = {(pair.algorithm, pair.dataset_id) for pair in grid_plan.build_pairs(manifest, {})}
    for dataset_id in ("sparkov", "ieee-cis-completa"):
        assert ("tabbert-mlp", dataset_id) in pairs
        assert ("tabbert-lstm", dataset_id) in pairs


def test_runner_reads_the_regimes_from_the_grid():
    output = subprocess.run(
        [sys.executable, "experiment_grid.py", "--block", "tabbert", "--format", "tabbert-regimes"],
        cwd=PIPELINE_DIRECTORY, capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert output == "supervised-only mlm-before-finetune warmup-mlm-finetune"


def test_tabbert_estimate_uses_the_acer_measurement():
    import estimate_execution_time as estimate

    seconds = estimate.seconds_per_fold("tabbert-mlp [supervised-only]", "sparkov")
    assert seconds == pytest.approx(96.0 * 10 + 464.0)
    pretrained = estimate.seconds_per_fold("tabbert-mlp [mlm-before-finetune]", "sparkov")
    assert pretrained - seconds == pytest.approx(96.0 * 3)
    warmup = estimate.seconds_per_fold("tabbert-mlp [warmup-mlm-finetune]", "sparkov")
    assert warmup == pytest.approx(pretrained)
