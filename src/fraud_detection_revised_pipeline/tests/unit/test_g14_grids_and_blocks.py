"""Item G14: bases de ablacao e os blocos em que cada base entra."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import anomaly_detection_specific as anomaly  # noqa: E402
import experiment_datasets  # noqa: E402
import experiment_grid  # noqa: E402


ABLATION = "ieee-cis-completa-sem-alta-cardinalidade"
SPARKOV_VARIANT = "sparkov-sem-titular"

GRIDS = {
    "classic": experiment_grid.build_classic_grid,
    "boosting": experiment_grid.build_boosting_grid,
    "ft-transformer": experiment_grid.build_ft_transformer_grid,
    "tabbert": experiment_grid.build_tabbert_grid,
}


def _datasets(builder) -> set[str]:
    return {entry.dataset_id for entry in builder()}


def test_ablation_is_only_in_the_boosting_grid():
    assert ABLATION in _datasets(GRIDS["boosting"])
    for block in ("classic", "ft-transformer", "tabbert"):
        assert ABLATION not in _datasets(GRIDS[block])


def test_sparkov_variant_is_in_every_grid():
    for block, builder in GRIDS.items():
        assert SPARKOV_VARIANT in _datasets(builder), block


def _axes(entries, dataset_id):
    return {
        (
            entry.fill_na_strategy,
            entry.undersampling_technique,
            entry.test_strategy,
            entry.random_seed,
            entry.categorical_handling_mode,
        )
        for entry in entries
        if entry.dataset_id == dataset_id
    }


def test_ablation_has_the_same_axes_as_its_base_for_pairing():
    entries = experiment_grid.build_boosting_grid()
    assert _axes(entries, ABLATION) == _axes(entries, "ieee-cis-completa")
    assert _axes(entries, SPARKOV_VARIANT) == _axes(entries, "sparkov")


def test_grids_without_variants_keep_their_previous_bases():
    classic = _datasets(GRIDS["classic"])
    assert "ulb" in classic and "ieee-cis-transacional" in classic
    assert "ulb" not in _datasets(GRIDS["tabbert"])


def test_dataset_filter_still_selects_one_base_inside_its_blocks(monkeypatch):
    monkeypatch.setenv("DATASET_FILTER", ABLATION)
    assert _datasets(GRIDS["boosting"]) == {ABLATION}
    assert _datasets(GRIDS["classic"]) == set()


def test_ablation_is_not_eligible_for_sequence_models():
    assert ABLATION not in experiment_datasets.datasets_supporting_sequence_models()
    assert SPARKOV_VARIANT in experiment_datasets.datasets_supporting_sequence_models()


def test_anomaly_runner_skips_a_base_outside_its_block(tmp_path, capsys):
    parser = anomaly.build_argument_parser()
    arguments = parser.parse_args(
        [
            "--algorithm", "isolation-forest",
            "--database-type", ABLATION,
            "--dataset-filename", "train_transaction.csv",
            "--prediction-column-name", "isFraud",
            "--categorical-encoding-strategy", "none",
            "--fill-na-strategy", "none",
            "--undersampling-technique", "none",
            "--undersampling-parameter-values", "none",
            "--test-strategy", "stratified-holdout",
            "--prepared-data-directory", str(tmp_path / "prepared"),
            "--results-directory", str(tmp_path / "results"),
            "--local-staging-directory", str(tmp_path / "staging"),
        ]
    )
    assert anomaly.run_configuration(arguments) == 0
    assert "outside the anomaly block" in capsys.readouterr().out
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize(
    "dataset_id",
    [
        dataset_id
        for dataset_id, definition in experiment_datasets.DATASET_DEFINITIONS.items()
        if experiment_datasets.BLOCK_ANOMALY in definition.blocks
    ],
)
def test_every_other_base_is_inside_the_anomaly_block(dataset_id):
    assert dataset_id != ABLATION

