from __future__ import annotations

import sys
from pathlib import Path

import pytest
import pandas as pd


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_datasets  # noqa: E402
import experiment_grid  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
from prepared_dataset_storage import (  # noqa: E402
    build_prepared_dataset_md5,
    load_prepared_data_splits,
    normalize_manifest_dataset_identity,
    save_fold_vocabulary,
    save_prepared_data_splits,
    save_window_index,
)


EXPECTED_DATASETS = {
    "ulb": ("ULB", "eqm-ref-18"),
    "sparkov": ("Sparkov", "eqm-ref-20"),
    "home-credit-completa": ("Home-Credit-completa", "eqm-ref-19"),
    "home-credit-somente-aplicacao": (
        "Home-Credit-somente-aplicação",
        "eqm-ref-19-application",
    ),
    "ieee-cis-completa": ("IEEE-CIS-completa", "eqm-ref-22"),
    "ieee-cis-sem-identidade": (
        "IEEE-CIS-sem-identidade",
        "eqm-ref-22-noidentity",
    ),
    "ieee-cis-transacional": (
        "IEEE-CIS-transacional",
        "eqm-ref-22-transaction",
    ),
}


# Bases de ablacao (item G14): nao tem apelido legado, porque nunca existiram
# como eqm-ref-NN.
ABLATION_VARIANTS = {
    "sparkov-sem-titular": "Sparkov-sem-titular",
    "ieee-cis-completa-sem-alta-cardinalidade": (
        "IEEE-CIS-completa-sem-alta-cardinalidade"
    ),
}


def test_catalog_has_unique_canonical_ids_names_and_aliases():
    assert set(experiment_datasets.DATASET_DEFINITIONS) == (
        set(EXPECTED_DATASETS) | set(ABLATION_VARIANTS)
    )
    for dataset_id, dataset_name in ABLATION_VARIANTS.items():
        definition = experiment_datasets.DATASET_DEFINITIONS[dataset_id]
        assert definition.dataset_name == dataset_name
        assert definition.legacy_aliases == ()
        assert definition.base_dataset_id in EXPECTED_DATASETS
    names = set()
    aliases = set()
    for dataset_id, (dataset_name, legacy_alias) in EXPECTED_DATASETS.items():
        definition = experiment_datasets.DATASET_DEFINITIONS[dataset_id]
        assert definition.dataset_id == dataset_id
        assert definition.dataset_name == dataset_name
        assert definition.legacy_aliases == (legacy_alias,)
        names.add(definition.dataset_name)
        aliases.update(definition.legacy_aliases)
    names.update(ABLATION_VARIANTS.values())
    assert len(names) == len(EXPECTED_DATASETS) + len(ABLATION_VARIANTS)
    assert len(aliases) == len(EXPECTED_DATASETS)
    assert aliases.isdisjoint(EXPECTED_DATASETS)
    assert aliases.isdisjoint(ABLATION_VARIANTS)


@pytest.mark.parametrize(
    ("dataset_id", "legacy_alias"),
    [(dataset_id, values[1]) for dataset_id, values in EXPECTED_DATASETS.items()],
)
def test_canonical_and_legacy_identifiers_resolve_to_the_same_definition(
    dataset_id, legacy_alias
):
    assert experiment_datasets.normalize_dataset_id(dataset_id) == dataset_id
    assert experiment_datasets.normalize_dataset_id(legacy_alias) == dataset_id
    assert experiment_datasets.get_dataset_definition(legacy_alias) is (
        experiment_datasets.get_dataset_definition(dataset_id)
    )


def test_unknown_dataset_reports_canonical_identifiers():
    with pytest.raises(ValueError, match="ulb"):
        experiment_datasets.normalize_dataset_id("unknown-dataset")


def test_configuration_hash_normalizes_legacy_alias():
    arguments = dict(
        categorical_encoding=None,
        fill_na_strategy="none",
        test_strategy="stratified-holdout",
        prediction_column="Class",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        categorical_handling_mode="encoded",
        random_seed=42,
    )
    assert build_prepared_dataset_md5(dataset="ulb", **arguments) == (
        build_prepared_dataset_md5(dataset="eqm-ref-18", **arguments)
    )


@pytest.mark.parametrize(
    ("builder", "legacy_alias", "dataset_id"),
    (
        (experiment_grid.build_classic_grid, "eqm-ref-18", "ulb"),
        (experiment_grid.build_boosting_grid, "eqm-ref-18", "ulb"),
        (experiment_grid.build_ft_transformer_grid, "eqm-ref-18", "ulb"),
        (experiment_grid.build_tabbert_grid, "eqm-ref-20", "sparkov"),
    ),
)
def test_all_preparation_grids_accept_alias_filter_and_emit_canonical_identity(
    monkeypatch, builder, legacy_alias, dataset_id
):
    monkeypatch.setenv("DATASET_FILTER", legacy_alias)
    entries = builder()
    assert entries
    assert {entry.dataset_id for entry in entries} == {dataset_id}
    assert {entry.database_type for entry in entries} == {dataset_id}
    assert {entry.dataset_name for entry in entries} == {
        experiment_datasets.get_dataset_definition(dataset_id).dataset_name
    }


def test_legacy_manifest_is_normalized_in_memory_without_mutating_source():
    legacy = pd.DataFrame(
        {
            "database_type": ["eqm-ref-22"],
            "configuration_md5": ["a" * 32],
            "dataset_md5": ["b" * 32],
            "fold_index": ["1"],
            "split": ["test"],
        }
    )
    normalized = normalize_manifest_dataset_identity(legacy)
    assert legacy.loc[0, "database_type"] == "eqm-ref-22"
    assert normalized.loc[0, "database_type"] == "ieee-cis-completa"
    assert normalized.loc[0, "dataset_id"] == "ieee-cis-completa"
    assert normalized.loc[0, "dataset_name"] == "IEEE-CIS-completa"


def test_legacy_manifest_uses_persisted_hash_paths_without_rewriting(tmp_path):
    configuration_md5 = "a" * 32
    saved = save_prepared_data_splits(
        {
            "test_info": [
                {
                    "train_data": pd.DataFrame({"feature": [1], "Class": [0]}),
                    "validation_data": pd.DataFrame({"feature": [2], "Class": [1]}),
                    "test_data": pd.DataFrame({"feature": [3], "Class": [0]}),
                }
            ]
        },
        tmp_path,
        configuration_md5,
    )
    manifest_path = tmp_path / "legacy.tsv"
    pd.DataFrame(
        [
            {
                "database_type": "eqm-ref-18",
                "configuration_md5": configuration_md5,
                "dataset_md5": item.dataset_md5,
                "fold_index": item.fold_index,
                "split": item.split_label,
            }
            for item in saved
        ]
    ).to_csv(manifest_path, sep="\t", index=False)
    original_manifest = manifest_path.read_bytes()

    loaded = load_prepared_data_splits(
        tmp_path,
        manifest_path,
        configuration_md5,
        "stratified-holdout",
    )

    assert manifest_path.read_bytes() == original_manifest
    assert all(item.path.is_file() for item in saved)
    assert set(loaded["test_info"][0]["split_md5"]) == {
        "train_data",
        "validation_data",
        "test_data",
    }


def test_conflicting_manifest_identity_is_rejected():
    conflicting = pd.DataFrame(
        {
            "dataset_id": ["ulb"],
            "database_type": ["eqm-ref-20"],
            "dataset_name": ["ULB"],
        }
    )
    with pytest.raises(ValueError, match="Conflicting dataset identity"):
        normalize_manifest_dataset_identity(conflicting)


def test_prepared_artifact_names_remain_hash_based(tmp_path):
    configuration_md5 = "a" * 32
    files = save_prepared_data_splits(
        {
            "test_info": [
                {
                    "train_data": pd.DataFrame({"feature": [1], "target": [0]}),
                    "validation_data": pd.DataFrame({"feature": [2], "target": [1]}),
                    "test_data": pd.DataFrame({"feature": [3], "target": [0]}),
                    "evaluation_units": {
                        "validation_data": pd.Series(["client-validation"]),
                        "test_data": pd.Series(["client-test"]),
                    },
                }
            ]
        },
        tmp_path,
        configuration_md5,
    )
    assert all(file.path.name == f"{file.dataset_md5}.csv" for file in files)
    assert sorted(path.name for path in tmp_path.glob("*.groups.csv")) == sorted(
        f"{file.dataset_md5}.groups.csv"
        for file in files
        if file.split_label in {"validation", "test"}
    )
    _, vocabulary_path = save_fold_vocabulary(tmp_path, configuration_md5, 1, {})
    window_path = save_window_index(
        tmp_path, configuration_md5, 1, pd.DataFrame({"row_id": [0]})
    )
    assert vocabulary_path.name == f"{configuration_md5}_fold-1_vocabulary.json"
    assert window_path.name == f"{configuration_md5}_fold-1_windows.csv"


@pytest.mark.parametrize(
    "algorithm",
    ("logistic-regression", "catboost", "ft-transformer-mlp", "tabbert-mlp"),
)
@pytest.mark.parametrize("execution_status", ("success", "failure"))
def test_all_result_families_write_canonical_dataset_identity(
    tmp_path, algorithm, execution_status
):
    destination = tmp_path / f"resultados_{algorithm}.csv"
    if execution_status == "success":
        execution = {
            "algorithm": algorithm,
            "undersampling_technique": "none",
            "train_test_strategy": "stratified-holdout",
            "results": [
                {
                    "fold_index": 1,
                    "split_sizes": {},
                    "split_results": {
                        "test_data": {
                            "metrics": {
                                "f1": 1.0,
                                "f_beta_2": 1.0,
                                "accuracy": 1.0,
                                "precision": 1.0,
                                "recall": 1.0,
                                "roc_auc": 1.0,
                            }
                        }
                    },
                }
            ],
        }
        pipeline.record_algorithm_result_to_csv(
            destination,
            "",
            execution,
            "none",
            "none",
            0.0,
            configuration_md5="a" * 32,
            database_type="eqm-ref-18",
        )
    else:
        pipeline.record_algorithm_failure_to_csv(
            destination,
            "",
            algorithm,
            "none",
            "none",
            "stratified-holdout",
            configuration_md5="a" * 32,
            database_type="eqm-ref-18",
        )
    row = pd.read_csv(destination, dtype=str, keep_default_na=False).iloc[0]
    assert row["dataset_id"] == "ulb"
    assert row["dataset_name"] == "ULB"
    assert row["database_type"] == "ulb"
    assert row["execution_status"] == execution_status
