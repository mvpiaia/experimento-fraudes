"""Avaliacao por cliente na Home Credit (item G7 da revisao).

Protege a cadeia inteira: o catalogo declara a unidade, a particao preserva a
prevalencia por cliente sem dividir clientes, o identificador fica fora dos
atributos mas chega a execucao por um arquivo lateral, e os escores das
linhas de um cliente sao agregados pela media antes do limiar e das metricas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import anomaly_detection_specific as anomaly  # noqa: E402
import data_partitioning  # noqa: E402
import experiment_datasets  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
import run_transformer_prepared_experiments as transformers  # noqa: E402
from experiment_datasets import STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD, DatasetDefinition  # noqa: E402
from prepared_dataset_storage import (  # noqa: E402
    build_prepared_dataset_md5,
    evaluation_unit_path,
    load_prepared_data_splits,
    save_prepared_data_splits,
)


UNIT = "client_id"
TARGET = "label"


def _client_frame(client_count=400, positive_rate=0.1, seed=0) -> pd.DataFrame:
    """Varias linhas por cliente, com o rotulo do cliente repetido em todas."""
    rng = np.random.default_rng(seed)
    rows = []
    for client in range(client_count):
        label = int(rng.random() < positive_rate)
        for _ in range(rng.integers(1, 8)):
            rows.append(
                {UNIT: 1000 + client, "feature": rng.normal(3.0 * label, 1.0), TARGET: label}
            )
    return pd.DataFrame(rows)


def _definition(**overrides) -> DatasetDefinition:
    arguments = dict(
        dataset_id="home-credit-completa",
        dataset_name="Home-Credit-completa",
        legacy_aliases=("eqm-ref-19",),
        source_files="memoria",
        prediction_column=TARGET,
        identifier_columns_to_drop=(UNIT,),
        group_column=UNIT,
        evaluation_unit_column=UNIT,
        protocols=(STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD),
    )
    arguments.update(overrides)
    return DatasetDefinition(**arguments)


# ---------------------------------------------------------------------------
# Catalogo
# ---------------------------------------------------------------------------


def test_home_credit_declares_client_unit_and_control_domain():
    joined = experiment_datasets.get_dataset_definition("eqm-ref-19")
    application = experiment_datasets.get_dataset_definition("eqm-ref-19-application")

    assert joined.evaluation_unit_column == "SK_ID_CURR"
    # Uma linha por cliente, mas com a mesma unidade da completa: as duas sao
    # pareadas por cliente (item G16).
    assert application.evaluation_unit_column == "SK_ID_CURR"
    assert application.paired_dataset_id == "home-credit-completa"
    assert joined.domain == application.domain == experiment_datasets.DOMAIN_OUT_OF_DOMAIN_CONTROL


def test_fraud_datasets_have_fraud_domain_and_no_unit():
    for dataset_id, definition in experiment_datasets.DATASET_DEFINITIONS.items():
        if dataset_id.startswith("home-credit"):
            continue
        assert definition.domain == experiment_datasets.DOMAIN_FRAUD
        assert definition.evaluation_unit_column is None


# ---------------------------------------------------------------------------
# Particao
# ---------------------------------------------------------------------------


def test_grouped_holdout_preserves_client_prevalence_without_splitting_clients():
    frame = _client_frame()
    folds = data_partitioning.build_partitions(
        frame, None, _definition(), STRATIFIED_HOLDOUT, random_seed=42
    )["test_info"]
    fold = folds[0]

    units = {
        split: set(fold["evaluation_units"][split])
        for split in ("validation_data", "test_data")
    }
    assert not units["validation_data"] & units["test_data"]

    client_labels = frame.groupby(UNIT)[TARGET].first()
    overall_rate = client_labels.mean()
    for split, clients in units.items():
        rate = client_labels.loc[sorted(clients)].mean()
        # Estratificado: a diferenca para a proporcao exata e de no maximo um cliente.
        assert abs(rate - overall_rate) <= 1.0 / len(clients) + 1e-9, split


def test_units_are_aligned_and_identifier_leaves_the_features():
    frame = _client_frame()
    fold = data_partitioning.build_partitions(
        frame, None, _definition(), STRATIFIED_HOLDOUT, random_seed=42
    )["test_info"][0]

    for split in ("validation_data", "test_data"):
        assert UNIT not in fold[split].columns
        assert len(fold["evaluation_units"][split]) == len(fold[split])
        # Mesmo cliente, mesmo rotulo: a ordem das unidades e a das linhas.
        by_unit = pd.DataFrame(
            {"unit": fold["evaluation_units"][split], "label": fold[split][TARGET]}
        )
        assert (by_unit.groupby("unit")["label"].nunique() == 1).all()


def test_groups_with_mixed_labels_keep_the_unstratified_split():
    frame = _client_frame()
    frame.loc[frame.index[::3], TARGET] = 1 - frame.loc[frame.index[::3], TARGET]
    labels = frame[TARGET]
    assert data_partitioning._constant_group_labels(labels, frame[UNIT]) is None


# ---------------------------------------------------------------------------
# Arquivo lateral e staging
# ---------------------------------------------------------------------------


def _prepared_configuration(tmp_path, frame=None):
    definition = _definition()
    split_information = data_partitioning.build_partitions(
        frame if frame is not None else _client_frame(), None, definition,
        STRATIFIED_HOLDOUT, random_seed=42,
    )
    configuration_md5 = build_prepared_dataset_md5(
        "eqm-ref-19", None, None, STRATIFIED_HOLDOUT, TARGET, None, None, True, "native", 42
    )
    prepared = tmp_path / "prepared"
    created = save_prepared_data_splits(split_information, prepared, configuration_md5)
    manifest = pd.DataFrame(
        [
            {
                "database_type": "eqm-ref-19",
                "configuration_md5": configuration_md5,
                "dataset_md5": created_file.dataset_md5,
                "content_md5": created_file.content_md5,
                "fold_index": created_file.fold_index,
                "split": created_file.split_label,
                "categorical_handling_mode": "native",
                "categorical_encoding_strategy": "none",
            }
            for created_file in created
        ]
    )
    manifest_path = tmp_path / "manifest.tsv"
    manifest.to_csv(manifest_path, sep="\t", index=False)
    return configuration_md5, prepared, manifest_path, created


def test_companion_files_exist_only_for_held_out_splits(tmp_path):
    _, prepared, manifest_path, created = _prepared_configuration(tmp_path)

    by_split = {created_file.split_label: created_file for created_file in created}
    assert not evaluation_unit_path(prepared, by_split["train"].dataset_md5).exists()
    for split in ("validation", "test"):
        units = pd.read_csv(evaluation_unit_path(prepared, by_split[split].dataset_md5))
        features = pd.read_csv(by_split[split].path)
        assert len(units) == len(features)
        assert UNIT not in features.columns


def test_staging_copies_units_and_loading_returns_them(tmp_path):
    configuration_md5, prepared, manifest_path, _ = _prepared_configuration(tmp_path)

    with pipeline.stage_prepared_dataset_files(
        prepared, manifest_path, configuration_md5, tmp_path / "staging"
    ) as staged:
        loaded = load_prepared_data_splits(staged, manifest_path, configuration_md5, STRATIFIED_HOLDOUT)

    fold = loaded["test_info"][0]
    assert set(fold["evaluation_units"]) == {"validation_data", "test_data"}
    assert len(fold["evaluation_units"]["test_data"]) == len(fold["test_data"])


def test_missing_companion_file_is_a_storage_failure(tmp_path):
    configuration_md5, prepared, manifest_path, created = _prepared_configuration(tmp_path)
    test_file = next(created_file for created_file in created if created_file.split_label == "test")
    evaluation_unit_path(prepared, test_file.dataset_md5).unlink()

    with pytest.raises(pipeline.PreparedDatasetStorageError, match="Evaluation unit file"):
        with pipeline.stage_prepared_dataset_files(
            prepared, manifest_path, configuration_md5, tmp_path / "staging"
        ):
            pass


# ---------------------------------------------------------------------------
# Agregacao, limiar e gravacao
# ---------------------------------------------------------------------------


def test_scores_are_averaged_per_client():
    split_scores = {
        "target": pd.Series([1, 1, 1, 0]),
        "positive_scores": np.array([0.2, 0.4, 0.9, 0.1]),
        "metric_scores": np.array([0.2, 0.4, 0.9, 0.1]),
        "multiclass_predictions": None,
        "probability_column": "p",
    }
    aggregated = pipeline.aggregate_split_scores_by_unit(split_scores, ["a", "a", "a", "b"])

    assert list(aggregated["target"]) == [1, 0]
    assert aggregated["positive_scores"] == pytest.approx([0.5, 0.1])


def test_client_with_conflicting_labels_is_rejected():
    split_scores = {"target": pd.Series([1, 0]), "positive_scores": np.array([0.2, 0.4])}
    with pytest.raises(ValueError, match="different labels"):
        pipeline.aggregate_split_scores_by_unit(split_scores, ["a", "a"])


def test_isolation_forest_records_row_and_client_rows(tmp_path):
    configuration_md5, prepared, manifest_path, _ = _prepared_configuration(tmp_path)
    command_line = [
        "--algorithm", "isolation-forest",
        "--database-type", "eqm-ref-19",
        "--dataset-filename", "memoria",
        "--prediction-column-name", TARGET,
        "--categorical-encoding-strategy", "none",
        "--fill-na-strategy", "none",
        "--undersampling-technique", "none",
        "--undersampling-parameter-values", "none",
        "--test-strategy", STRATIFIED_HOLDOUT,
        "--prepared-data-directory", str(prepared),
        "--prepared-dataset-md5", configuration_md5,
        "--prepared-dataset-manifest", str(manifest_path),
        "--results-directory", str(tmp_path / "results"),
        "--local-staging-directory", str(tmp_path / "staging"),
    ]

    assert anomaly.run_configuration(anomaly.build_argument_parser().parse_args(command_line)) == 0

    recorded = pd.read_csv(
        tmp_path / "results" / "resultados_isolation-forest.csv", dtype=str, keep_default_na=False
    )
    by_unit = recorded.groupby("evaluation_unit")["split"].apply(set).to_dict()
    assert by_unit["row"] == {"train", "validation", "test"}
    assert by_unit["client"] == {"validation", "test"}
    client_test = recorded[(recorded["evaluation_unit"] == "client") & (recorded["split"] == "test")]
    row_test = recorded[(recorded["evaluation_unit"] == "row") & (recorded["split"] == "test")]
    # O tamanho do teste por cliente e o numero de clientes, menor que o de linhas.
    assert int(client_test["test_df_size"].iloc[0]) < int(row_test["test_df_size"].iloc[0])


def test_client_threshold_is_chosen_on_client_validation():
    rng = np.random.default_rng(0)
    validation_units = np.repeat(np.arange(50), 3)
    validation_targets = np.repeat((np.arange(50) % 5 == 0).astype(int), 3)
    validation_scores = validation_targets * 0.6 + rng.random(150) * 0.3
    scored = {
        "validation": (validation_targets, validation_scores),
        "test": (validation_targets, validation_scores),
    }

    results = transformers.unit_level_results(
        scored,
        {"validation": validation_units, "test": validation_units},
        "validation-f2",
        {"model_hyperparameters": {}, "training_summary": {}},
    )

    assert {result["evaluation_split"] for result in results} == {"validation", "test"}
    assert all(result["evaluation_unit"] == "client" for result in results)
    assert all(result["evaluated_units"] == 50 for result in results)
    assert results[0]["threshold_strategy"] == "validation-f2"


def test_tabbert_target_units_follow_the_scoring_order():
    """Cada posicao-alvo recebe o cliente da linha que ela representa."""
    data = {
        "test": pd.DataFrame(
            {"row_id": [10, 11, 12, 13], "SK_ID_CURR": [7, 7, 8, 8], "label": [1, 1, 0, 0]}
        )
    }
    windows = pd.DataFrame(
        {
            "sequence_index": [0, 0, 1, 1],
            "position_index": [0, 1, 0, 1],
            "row_id": [10, 11, 12, 13],
            "is_target": [False, True, True, True],
            "target_split": ["test"] * 4,
            "source_split": ["test"] * 4,
        }
    )

    units = transformers.tabbert_target_units(data, windows, "test", "SK_ID_CURR")

    assert list(units) == [7, 8, 8]
