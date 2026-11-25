"""Item G14: catalogo (valores monetarios, codigos, variantes) e carga."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_datasets  # noqa: E402
from data_partitioning import build_partitions  # noqa: E402


ROW_COUNT = 40


def _write_sparkov_files(root: Path) -> None:
    (root / "ref-20").mkdir(parents=True)

    def frame(offset: int) -> pd.DataFrame:
        rows = np.arange(ROW_COUNT) + offset
        return pd.DataFrame(
            {
                "Unnamed: 0": rows,
                "trans_date_trans_time": pd.date_range(
                    "2019-01-01", periods=ROW_COUNT, freq="h"
                ).astype(str),
                "cc_num": 1000 + rows % 8,
                "merchant": [f"m{value % 5}" for value in rows],
                "category": [f"c{value % 3}" for value in rows],
                "amt": 5.0 + rows,
                "first": "A",
                "last": "B",
                "gender": ["F", "M"] * (ROW_COUNT // 2),
                "street": [f"street {value}" for value in rows],
                "city": [f"city {value % 8}" for value in rows],
                "state": "SP",
                "zip": 10000 + rows % 8,
                "lat": 10.0 + rows % 8,
                "long": 20.0 + rows % 8,
                "city_pop": 500 + rows % 8,
                "job": [f"job {value % 8}" for value in rows],
                "dob": "1980-01-01",
                "trans_num": [f"t{value}" for value in rows],
                "unix_time": 1_500_000_000 + rows * 3600,
                "merch_lat": 11.0 + rows % 5,
                "merch_long": 21.0 + rows % 5,
                "is_fraud": [1 if value % 10 == 0 else 0 for value in rows],
                "always_the_same": 7,
            }
        )

    frame(0).to_csv(root / "ref-20" / "fraudTrain.csv", index=False)
    frame(1000).to_csv(root / "ref-20" / "fraudTest.csv", index=False)


def _write_ieee_files(root: Path) -> None:
    (root / "ref-22").mkdir(parents=True)
    rows = np.arange(ROW_COUNT)
    transaction = pd.DataFrame(
        {
            "TransactionID": 2_000_000 + rows,
            "isFraud": [1 if value % 7 == 0 else 0 for value in rows],
            "TransactionDT": 86_400 + rows * 600,
            "TransactionAmt": 10.0 + rows,
            "ProductCD": ["W", "C"] * (ROW_COUNT // 2),
            "card1": 1000 + rows % 6,
            "card2": [100.0 + value % 4 if value % 9 else np.nan for value in rows],
            "card3": 150.0,
            "card5": 200.0 + rows % 3,
            "addr1": 300.0 + rows % 5,
            "addr2": 87.0,
            "M1": "T",
            "M2": ["T", np.nan] * (ROW_COUNT // 2),
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": 2_000_000 + rows,
            "DeviceInfo": [f"device {value % 9}" for value in rows],
            "id_31": [f"chrome {value % 4}" for value in rows],
            "id_33": [f"{value % 3}x{value % 5}" for value in rows],
            "id_12": "Found",
        }
    )
    transaction.to_csv(root / "ref-22" / "train_transaction.csv", index=False)
    identity.to_csv(root / "ref-22" / "train_identity.csv", index=False)


@pytest.fixture()
def raw_data_root(tmp_path, monkeypatch):
    _write_sparkov_files(tmp_path)
    _write_ieee_files(tmp_path)
    monkeypatch.setattr(experiment_datasets, "CURRENT_PATH", "")
    monkeypatch.setattr(experiment_datasets, "RELATIVE_DATA_PATH", str(tmp_path))
    return tmp_path


# --- catalogo --------------------------------------------------------------


def test_monetary_columns_are_declared_per_base():
    definitions = experiment_datasets.DATASET_DEFINITIONS
    assert definitions["ulb"].monetary_columns == ("Amount",)
    assert definitions["sparkov"].monetary_columns == ("amt",)
    assert definitions["sparkov-sem-titular"].monetary_columns == ("amt",)
    for dataset_id in (
        "ieee-cis-completa",
        "ieee-cis-sem-identidade",
        "ieee-cis-transacional",
        "ieee-cis-completa-sem-alta-cardinalidade",
    ):
        assert definitions[dataset_id].monetary_columns == ("TransactionAmt",)
    assert definitions["home-credit-completa"].monetary_columns == ()
    assert definitions["home-credit-somente-aplicacao"].monetary_columns == ()


def test_ieee_code_columns_are_declared_and_card1_leaves_the_ablation():
    definitions = experiment_datasets.DATASET_DEFINITIONS
    expected = ("card1", "card2", "card3", "card5", "addr1", "addr2")
    for dataset_id in (
        "ieee-cis-completa",
        "ieee-cis-sem-identidade",
        "ieee-cis-transacional",
    ):
        assert definitions[dataset_id].categorical_code_columns == expected
    assert definitions[
        "ieee-cis-completa-sem-alta-cardinalidade"
    ].categorical_code_columns == ("card2", "card3", "card5", "addr1", "addr2")


def test_sparkov_variant_drops_holder_attributes_but_keeps_cc_num_as_group():
    variant = experiment_datasets.DATASET_DEFINITIONS["sparkov-sem-titular"]
    base = experiment_datasets.DATASET_DEFINITIONS["sparkov"]
    assert variant.base_dataset_id == "sparkov"
    assert variant.ablated_columns == (
        "zip",
        "lat",
        "long",
        "city",
        "city_pop",
        "job",
    )
    assert variant.group_column == "cc_num"
    assert variant.sequence_keys == ("cc_num", "unix_time")
    assert variant.protocols == base.protocols
    assert variant.blocks == experiment_datasets.ALL_BLOCKS
    dropped = experiment_datasets.columns_to_drop_after_partitioning(variant)
    assert {"cc_num", "zip", "lat", "long", "city", "city_pop", "job"} <= set(dropped)
    assert "customer_merchant_km" not in dropped
    assert "zip" not in experiment_datasets.columns_to_drop_after_partitioning(base)


def test_ablation_variant_is_boosting_only_and_inherits_the_base():
    variant = experiment_datasets.DATASET_DEFINITIONS[
        "ieee-cis-completa-sem-alta-cardinalidade"
    ]
    base = experiment_datasets.DATASET_DEFINITIONS["ieee-cis-completa"]
    assert variant.blocks == (experiment_datasets.BLOCK_BOOSTING,)
    assert variant.ablated_columns == ("DeviceInfo", "id_31", "id_33", "card1")
    assert variant.group_column == base.group_column
    assert variant.sequence_keys == base.sequence_keys
    assert variant.protocols == base.protocols
    assert experiment_datasets.identifier_columns_to_drop(variant) == (
        experiment_datasets.identifier_columns_to_drop(base)
    )
    assert variant.dataset_id in experiment_datasets.IEEE_CIS_DATASET_IDS


def test_bases_without_declared_blocks_enter_every_block():
    for dataset_id, definition in experiment_datasets.DATASET_DEFINITIONS.items():
        if dataset_id == "ieee-cis-completa-sem-alta-cardinalidade":
            continue
        assert definition.blocks == experiment_datasets.ALL_BLOCKS


# --- codigos categoricos ----------------------------------------------------


def test_codes_become_integer_text_and_missing_stays_missing(raw_data_root):
    train, _auxiliary, _definition = experiment_datasets.load_raw_dataset(
        "ieee-cis-completa"
    )
    assert train["addr1"].iloc[0] == "300"
    assert train["card2"].iloc[0] != "100.0"
    assert pd.isna(train["card2"].iloc[0]) or train["card2"].iloc[0].isdigit()
    assert train["card2"].isna().any()
    assert not train["card2"].dropna().str.contains(r"\.").any()
    for column in experiment_datasets.IEEE_CIS_CODE_COLUMNS:
        if column in train.columns:
            assert train[column].dtype == object


def test_fractional_code_is_rejected():
    frame = pd.DataFrame({"card1": [1.5, 2.0]})
    with pytest.raises(ValueError, match="card1"):
        experiment_datasets._convert_code_columns_to_text(frame, ("card1",))


def test_approximate_holder_id_does_not_change_with_text_codes(raw_data_root):
    train, _auxiliary, _definition = experiment_datasets.load_raw_dataset(
        "ieee-cis-transacional"
    )
    raw = pd.read_csv(raw_data_root / "ref-22" / "train_transaction.csv")
    expected = experiment_datasets._build_approximate_holder_id(raw)
    assert train["approximate_holder_id"].tolist() == expected.tolist()


# --- colunas constantes ------------------------------------------------------


def test_constant_columns_are_removed_but_missing_counts_as_a_value(raw_data_root):
    train, _auxiliary, _definition = experiment_datasets.load_raw_dataset(
        "ieee-cis-completa"
    )
    provenance = train.attrs["column_provenance"]
    assert {"M1", "id_12", "card3", "addr2"} <= set(provenance["constant_columns"])
    assert "M1" not in train.columns
    assert "M2" in train.columns
    assert "M2" not in provenance["constant_columns"]


def test_constant_in_main_but_varying_in_auxiliary_is_kept(raw_data_root):
    test_path = raw_data_root / "ref-20" / "fraudTest.csv"
    test_frame = pd.read_csv(test_path)
    test_frame["always_the_same"] = 8
    test_frame.to_csv(test_path, index=False)
    train, auxiliary, _definition = experiment_datasets.load_raw_dataset("sparkov")
    assert "always_the_same" in train.columns
    assert "always_the_same" in auxiliary.columns


def test_constant_in_both_files_is_removed_from_both(raw_data_root):
    train, auxiliary, _definition = experiment_datasets.load_raw_dataset("sparkov")
    assert "always_the_same" not in train.columns
    assert "always_the_same" not in auxiliary.columns
    assert "always_the_same" in train.attrs["column_provenance"]["constant_columns"]


def test_label_and_structural_columns_are_never_treated_as_constant():
    frame = pd.DataFrame({"isFraud": [0, 0, 0], "card1": [5, 5, 5], "x": [1, 1, 1]})
    constants = experiment_datasets.find_constant_columns(
        [frame], {"isFraud", "card1"}
    )
    assert constants == ["x"]


def test_provenance_separates_source_and_derived_columns(raw_data_root):
    train, _auxiliary, _definition = experiment_datasets.load_raw_dataset("sparkov")
    provenance = train.attrs["column_provenance"]
    assert set(provenance["derived_columns"]) == set(
        experiment_datasets.SPARKOV_DERIVED_COLUMNS
    )
    assert "amt" in provenance["source_columns"]
    assert "customer_merchant_km" not in provenance["source_columns"]


# --- variantes: mesmas linhas, mesmas particoes -----------------------------


def _partition_ids(dataset_id: str, protocol: str, seed: int = 42) -> list[str]:
    train, auxiliary, definition = experiment_datasets.load_raw_dataset(dataset_id)
    partitions = build_partitions(train, auxiliary, definition, protocol, seed)
    fold = partitions["test_info"][0]
    return [fold["partition_md5"].get(name) for name in ("train_data", "test_data")]


def test_ablation_partitions_the_same_rows_as_its_base(raw_data_root):
    base = _partition_ids("ieee-cis-completa", "stratified-holdout")
    variant = _partition_ids(
        "ieee-cis-completa-sem-alta-cardinalidade", "stratified-holdout"
    )
    assert base == variant


@pytest.mark.parametrize(
    "protocol", ["stratified-holdout", "stratified-k-fold", "temporal-holdout"]
)
def test_identity_ablation_partitions_like_its_base_and_loses_identity(
    raw_data_root, protocol
):
    """Item G16: o par completa x sem-identidade avalia as mesmas linhas."""
    assert _partition_ids("ieee-cis-sem-identidade", protocol) == _partition_ids(
        "ieee-cis-completa", protocol
    )
    train, auxiliary, definition = experiment_datasets.load_raw_dataset(
        "ieee-cis-sem-identidade"
    )
    assert definition.base_dataset_id == "ieee-cis-completa"
    fold = build_partitions(train, auxiliary, definition, protocol, 42)["test_info"][0]
    for split_name in ("train_data", "validation_data", "test_data"):
        if fold[split_name] is None:
            continue
        leaked = set(fold[split_name].columns) & set(
            experiment_datasets.IEEE_CIS_IDENTITY_ATTRIBUTES
        )
        assert not leaked


def test_sparkov_variant_partitions_like_sparkov_and_loses_holder_columns(
    raw_data_root,
):
    base = _partition_ids("sparkov", "stratified-holdout")
    variant = _partition_ids("sparkov-sem-titular", "stratified-holdout")
    assert base == variant

    train, auxiliary, definition = experiment_datasets.load_raw_dataset(
        "sparkov-sem-titular"
    )
    fold = build_partitions(train, auxiliary, definition, "stratified-holdout", 42)[
        "test_info"
    ][0]
    prepared_columns = set(fold["train_data"].columns)
    for removed in ("zip", "lat", "long", "city", "city_pop", "job", "cc_num"):
        assert removed not in prepared_columns
    assert "customer_merchant_km" in prepared_columns
    assert "amt" in prepared_columns


def test_sparkov_variant_never_splits_a_holder_across_partitions(raw_data_root):
    train, auxiliary, definition = experiment_datasets.load_raw_dataset(
        "sparkov-sem-titular"
    )
    # O cc_num sai dos arquivos, entao o agrupamento e checado antes de gravar.
    partitions = build_partitions(
        train,
        auxiliary,
        definition,
        "stratified-holdout",
        42,
        retained_columns=("cc_num",),
    )
    fold = partitions["test_info"][0]
    holders = {
        name: set(fold[name]["cc_num"])
        for name in ("train_data", "validation_data", "test_data")
        if fold.get(name) is not None
    }
    names = list(holders)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            assert holders[names[left]].isdisjoint(holders[names[right]])
