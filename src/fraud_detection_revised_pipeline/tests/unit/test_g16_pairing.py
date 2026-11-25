"""Item G16: pareamento das bases de ablacao na remocao de incompletos.

A IEEE-CIS-sem-identidade era uma base propria. Com outras colunas, tinha
outro partition_md5 e, na remocao de incompletos, escolhia outro limiar e
removia outras linhas do treino: a ablacao deixava de comparar a mesma
populacao. Agora ela e variante de colunas da IEEE-CIS-completa, e a decisao
de remocao e tomada com as colunas da base de origem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_datasets  # noqa: E402
import experiment_grid  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
import data_partitioning  # noqa: E402
import missing_screening  # noqa: E402
import preparation_report  # noqa: E402
import preparation_sanity_checks as sanity  # noqa: E402
import validate_prepared_datasets as validator  # noqa: E402

from test_g14_catalog_and_loading import _write_sparkov_files  # noqa: E402
from test_g14_registry_and_categorical import _prepare  # noqa: E402
from test_g15_verifier import CapturingReport  # noqa: E402


DROP = missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY
ROW_COUNT = 400


def _write_sparse_identity_files(root: Path) -> None:
    """IEEE-CIS em que a identidade muda o limiar e as linhas removidas.

    Sem a identidade, so V1 tem faltantes (10%) e o limiar e 100. Com ela,
    id_01 (50%) nao cabe no piso de fraudes e sai, mas id_02 (5%) fica: a
    base completa remove tambem as linhas sem id_02.
    """
    (root / "ref-22").mkdir(parents=True)
    rows = np.arange(ROW_COUNT)
    transaction = pd.DataFrame(
        {
            "TransactionID": 3_000_000 + rows,
            "isFraud": (rows % 7 == 0).astype(int),
            "TransactionDT": 86_400 + rows * 600,
            "TransactionAmt": 10.0 + rows,
            "ProductCD": np.where(rows % 2 == 0, "W", "C"),
            "card1": 1000 + rows % 6,
            "addr1": 300.0 + rows % 5,
            "V1": np.where(rows % 10 == 0, np.nan, rows % 13 + 0.5),
        }
    )
    identity = pd.DataFrame(
        {
            "TransactionID": 3_000_000 + rows,
            "id_01": np.where(rows % 2 == 0, np.nan, rows % 11 + 0.25),
            "id_02": np.where(rows % 20 == 1, np.nan, rows % 17 + 0.75),
            "DeviceType": np.where(rows % 3 == 0, "mobile", "desktop"),
        }
    )
    transaction.to_csv(root / "ref-22" / "train_transaction.csv", index=False)
    identity.to_csv(root / "ref-22" / "train_identity.csv", index=False)


@pytest.fixture()
def raw_data_root(tmp_path, monkeypatch):
    _write_sparkov_files(tmp_path)
    _write_sparse_identity_files(tmp_path)
    monkeypatch.setattr(pipeline, "CURRENT_PATH", "")
    monkeypatch.setattr(pipeline, "RELATIVE_DATA_PATH", str(tmp_path))
    monkeypatch.setattr(experiment_datasets, "CURRENT_PATH", "")
    monkeypatch.setattr(experiment_datasets, "RELATIVE_DATA_PATH", str(tmp_path))
    return tmp_path


def _reports(output: Path, md5: str) -> dict[int, dict]:
    paths = output.glob(f"{md5}.fold*{preparation_report.PREPARATION_REPORT_FILE_SUFFIX}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    return {report["fold_index"]: report for report in reports}


def _prepare_reports(monkeypatch, root, dataset, fill_na, strategy="stratified-holdout"):
    md5, created, output = _prepare(
        monkeypatch, root, dataset, mode="encoded", strategy=strategy,
        encoding="woe", fill_na=fill_na,
    )
    return created, _reports(output, md5)


def test_variant_alone_would_choose_another_threshold(raw_data_root):
    """Sem a decisao na base de origem, o par removeria linhas diferentes."""
    thresholds = {}
    for dataset_id in ("ieee-cis-completa", "ieee-cis-sem-identidade"):
        train, _auxiliary, definition = experiment_datasets.load_raw_dataset(dataset_id)
        features = train.drop(
            columns=[
                column
                for column in experiment_datasets.columns_to_drop_after_partitioning(
                    definition
                )
                if column in train.columns
            ]
        )
        thresholds[dataset_id] = missing_screening.select_missing_screening(
            features, "isFraud"
        ).threshold_pct
    assert thresholds["ieee-cis-completa"] != thresholds["ieee-cis-sem-identidade"]


@pytest.mark.parametrize("strategy", ["stratified-holdout", "stratified-k-fold"])
def test_variant_removes_the_same_training_rows_as_its_base(
    monkeypatch, raw_data_root, strategy
):
    _created, base = _prepare_reports(
        monkeypatch, raw_data_root, "ieee-cis-completa", DROP, strategy
    )
    _created, variant = _prepare_reports(
        monkeypatch, raw_data_root, "ieee-cis-sem-identidade", DROP, strategy
    )
    assert base.keys() == variant.keys()
    for fold_index, base_report in base.items():
        variant_report = variant[fold_index]
        base_screening = base_report["missing_screening"]
        variant_screening = variant_report["missing_screening"]
        assert variant_screening["threshold_pct"] == base_screening["threshold_pct"]
        assert variant_screening["rows_after"] == base_screening["rows_after"]
        assert variant_screening["source_dataset_id"] == "ieee-cis-completa"
        assert base_screening["source_dataset_id"] == "ieee-cis-completa"
        assert variant_report["train_population_md5"] == base_report["train_population_md5"]
        # A identidade sai da variante, inclusive da lista de colunas triadas.
        identity = set(experiment_datasets.IEEE_CIS_IDENTITY_ATTRIBUTES)
        assert not identity & set(variant_report["input_feature_columns"])
        assert not identity & set(variant_screening["screened_out_columns"])
        assert variant_report["splits"]["train"]["after_imputation"]["missing_cells"] == 0


def test_without_removal_the_population_is_the_training_partition(
    monkeypatch, raw_data_root
):
    created, reports = _prepare_reports(
        monkeypatch, raw_data_root, "ieee-cis-completa", "mean"
    )
    for fold_index, report in reports.items():
        train_partition = {
            item.partition_md5
            for item in created
            if item.fold_index == fold_index and item.split_label == "train"
        }
        assert train_partition == {report["train_population_md5"]}


def test_removal_changes_the_training_population(monkeypatch, raw_data_root):
    created, reports = _prepare_reports(
        monkeypatch, raw_data_root, "ieee-cis-completa", DROP
    )
    for fold_index, report in reports.items():
        train_partition = next(
            item.partition_md5
            for item in created
            if item.fold_index == fold_index and item.split_label == "train"
        )
        assert report["missing_screening"]["rows_after"] < report["missing_screening"]["rows_before"]
        assert report["train_population_md5"] != train_partition


# --- par da Home Credit: mesmos clientes em cada split ----------------------


CLIENT_COUNT = 300


def _write_home_credit_files(root: Path) -> None:
    """Clientes com 0 a 4 pedidos anteriores; os sem historico sao 1 em 5."""
    (root / "ref-19").mkdir(parents=True, exist_ok=True)
    clients = np.arange(100_000, 100_000 + CLIENT_COUNT)
    generator = np.random.default_rng(7)
    application = pd.DataFrame(
        {
            "SK_ID_CURR": clients,
            "TARGET": (generator.random(CLIENT_COUNT) < 0.12).astype(int),
            "AMT_INCOME_TOTAL": generator.normal(5.0, 1.0, CLIENT_COUNT),
            "NAME_CONTRACT_TYPE": generator.choice(["Cash loans", "Revolving loans"], CLIENT_COUNT),
        }
    )
    previous_counts = np.where(np.arange(CLIENT_COUNT) % 5 == 0, 0, 1 + np.arange(CLIENT_COUNT) % 4)
    owners = np.repeat(clients, previous_counts)
    previous = pd.DataFrame(
        {
            "SK_ID_PREV": np.arange(len(owners)) + 2_000_000,
            "SK_ID_CURR": owners,
            "DAYS_DECISION": -np.arange(len(owners)) % 700 - 1,
            "AMT_APPLICATION": generator.normal(10.0, 2.0, len(owners)),
        }
    )
    application.to_csv(root / "ref-19" / "application_data.csv", index=False)
    previous.to_csv(root / "ref-19" / "previous_application.csv", index=False)


@pytest.fixture()
def home_credit_root(tmp_path, monkeypatch):
    _write_home_credit_files(tmp_path)
    monkeypatch.setattr(experiment_datasets, "CURRENT_PATH", "")
    monkeypatch.setattr(experiment_datasets, "RELATIVE_DATA_PATH", str(tmp_path))
    return tmp_path


def _client_folds(dataset_id: str, protocol: str) -> list[dict]:
    train, auxiliary, definition = experiment_datasets.load_raw_dataset(dataset_id)
    return data_partitioning.build_partitions(train, auxiliary, definition, protocol, 42)[
        "test_info"
    ]


def test_clients_without_history_stay_in_the_joined_dataset(home_credit_root):
    joined, _auxiliary, _definition = experiment_datasets.load_raw_dataset(
        "home-credit-completa"
    )
    application, _auxiliary, _definition = experiment_datasets.load_raw_dataset(
        "home-credit-somente-aplicacao"
    )
    assert set(joined["SK_ID_CURR"]) == set(application["SK_ID_CURR"])
    without_history = joined[joined["SK_ID_PREV"].isna()]
    assert len(without_history) == CLIENT_COUNT // 5
    assert without_history["SK_ID_CURR"].is_unique


@pytest.mark.parametrize("protocol", ["stratified-holdout", "stratified-k-fold"])
def test_both_home_credit_datasets_put_each_client_in_the_same_split(
    home_credit_root, protocol
):
    joined = _client_folds("home-credit-completa", protocol)
    application = _client_folds("home-credit-somente-aplicacao", protocol)
    assert len(joined) == len(application)
    for joined_fold, application_fold in zip(joined, application):
        assert joined_fold["unit_partition_md5"] == application_fold["unit_partition_md5"]
        # As linhas brutas diferem; so as unidades coincidem.
        assert joined_fold["partition_md5"] != application_fold["partition_md5"]
        for split_name, units in joined_fold["evaluation_units"].items():
            assert set(units) == set(application_fold["evaluation_units"][split_name])


def test_k_fold_never_puts_a_client_in_two_validation_folds(home_credit_root):
    folds = _client_folds("home-credit-completa", "stratified-k-fold")
    seen: set = set()
    for fold in folds:
        clients = set(fold["evaluation_units"]["validation_data"])
        assert not clients & seen
        assert not clients & set(fold["evaluation_units"]["train_data"])
        seen |= clients
    assert len(seen) == CLIENT_COUNT


# --- nomes distintos: remocao, tratamento nativo e ausencia de faltantes ----


def test_native_grid_names_native_handling_and_keeps_none_for_complete_bases():
    entries = experiment_grid.GRID_BUILDERS["boosting"]()
    by_dataset: dict[str, set] = {}
    for entry in entries:
        by_dataset.setdefault(entry.database_type, set()).add(entry.fill_na_strategy)
    assert by_dataset["ieee-cis-completa"] == {"native", "mean", "median", "mode"}
    assert by_dataset["ulb"] == {"none"}
    classic = experiment_grid.GRID_BUILDERS["classic"]()
    assert "native" not in {entry.fill_na_strategy for entry in classic}


def test_native_label_is_rejected_in_the_classic_block(monkeypatch, raw_data_root):
    with pytest.raises(ValueError, match="native boosting block"):
        _prepare(
            monkeypatch, raw_data_root, "ieee-cis-completa", mode="encoded",
            encoding="woe", fill_na="native",
        )


def test_native_label_leaves_missing_values_to_the_estimator(monkeypatch, raw_data_root):
    md5, created, output = _prepare(
        monkeypatch, raw_data_root, "ieee-cis-completa", fill_na="native"
    )
    for report in _reports(output, md5).values():
        train = report["splits"]["train"]
        # Numericos ausentes chegam ao estimador como ausentes.
        assert train["output"]["missing_cells"] > 0
        assert report["missing_screening"] is None


# --- verificador: P9 por base de origem e por unidade, P16 e R5 --------------


def _manifest_row(dataset, split, fill_na="median", partition="p", unit="", population="",
                  fold=1, configuration=None):
    return {
        "database_type": dataset,
        "dataset_id": dataset,
        "test_strategy": "stratified-holdout",
        "random_seed": "42",
        "auxiliary_data_is_test_data": "false",
        "fold_index": str(fold),
        "split": split,
        "fill_na_strategy": fill_na,
        "categorical_handling_mode": "encoded",
        "configuration_md5": configuration or f"{dataset}-{fill_na}",
        "partition_md5": partition,
        "unit_partition_md5": unit,
        "train_population_md5": population,
    }


def test_p9_requires_the_variant_to_share_the_base_partition():
    shared = pd.DataFrame(
        [_manifest_row(dataset, "test", partition="same")
         for dataset in ("ieee-cis-completa", "ieee-cis-sem-identidade")]
    )
    report = CapturingReport()
    validator.validate_partition_pairing(shared, report)
    assert report.passed("P9")

    diverging = shared.assign(partition_md5=["one", "other"])
    report = CapturingReport()
    validator.validate_partition_pairing(diverging, report)
    assert report.failed("P9")


def test_p9_requires_the_home_credit_pair_to_share_the_clients():
    rows = [
        _manifest_row("home-credit-completa", "test", partition="joined", unit="clients"),
        _manifest_row("home-credit-somente-aplicacao", "test", partition="application", unit="clients"),
    ]
    report = CapturingReport()
    validator.validate_partition_pairing(pd.DataFrame(rows), report)
    assert report.passed("P9")

    rows[1]["unit_partition_md5"] = "other-clients"
    report = CapturingReport()
    validator.validate_partition_pairing(pd.DataFrame(rows), report)
    assert any("clientes diferentes" in detail for detail in report.failed("P9"))


def test_p16_requires_the_same_training_rows_in_paired_removals():
    rows = [
        _manifest_row("ieee-cis-completa", "train", DROP, partition="train", population="kept"),
        _manifest_row("ieee-cis-sem-identidade", "train", DROP, partition="train", population="kept"),
        _manifest_row("ieee-cis-completa", "train", "mean", partition="train", population="train"),
    ]
    report = CapturingReport()
    sanity.check_training_population(pd.DataFrame(rows), report)
    assert report.passed("P16")

    rows[1]["train_population_md5"] = "kept-differently"
    report = CapturingReport()
    sanity.check_training_population(pd.DataFrame(rows), report)
    assert report.failed("P16")

    rows[1]["train_population_md5"] = "kept"
    rows[2]["train_population_md5"] = "not-the-partition"
    report = CapturingReport()
    sanity.check_training_population(pd.DataFrame(rows), report)
    assert report.failed("P16")


def _result_row(dataset, fill_na="median", partition="p", population="t", unit=""):
    return {
        "dataset_id": dataset,
        "execution_status": "success",
        "train_validation_test_strategy": "stratified-holdout",
        "random_seed": "42",
        "fold_index": "1",
        "split": "test",
        "fill_na_strategy": fill_na,
        "partition_id": partition,
        "train_population_id": population,
        "unit_partition_id": unit,
    }


def test_r5_accepts_paired_results():
    results = pd.DataFrame(
        [
            _result_row("ieee-cis-completa", DROP, population="kept"),
            _result_row("ieee-cis-sem-identidade", DROP, population="kept"),
            _result_row("home-credit-completa", partition="joined", unit="clients"),
            _result_row("home-credit-somente-aplicacao", partition="application", unit="clients"),
        ]
    )
    report = CapturingReport()
    sanity.check_result_pairing(results, report)
    assert report.passed("R5")


@pytest.mark.parametrize(
    "change, message",
    [
        ({"partition_id": "other"}, "particoes diferentes"),
        ({"train_population_id": "other"}, "treinos diferentes"),
        ({"train_population_id": ""}, "sem train_population_id"),
    ],
)
def test_r5_rejects_an_unpaired_ablation(change, message):
    variant = {**_result_row("ieee-cis-sem-identidade", DROP, population="kept"), **change}
    results = pd.DataFrame([_result_row("ieee-cis-completa", DROP, population="kept"), variant])
    report = CapturingReport()
    sanity.check_result_pairing(results, report)
    assert any(message in detail for detail in report.failed("R5"))


def test_r5_rejects_home_credit_results_on_different_clients():
    results = pd.DataFrame(
        [
            _result_row("home-credit-completa", unit="clients"),
            _result_row("home-credit-somente-aplicacao", unit="other-clients"),
        ]
    )
    report = CapturingReport()
    sanity.check_result_pairing(results, report)
    assert any("clientes diferentes" in detail for detail in report.failed("R5"))


# --- braco de remocao do par da Home Credit: mesmos clientes no treino -------


@pytest.fixture()
def sparse_home_credit_root(tmp_path, monkeypatch):
    """Home Credit com faltantes na aplicacao e nos pedidos anteriores."""
    _write_home_credit_files(tmp_path)
    application_path = tmp_path / "ref-19" / "application_data.csv"
    application = pd.read_csv(application_path)
    positions = np.arange(len(application))
    application["EXT_SOURCE_1"] = np.where(positions % 9 == 4, np.nan, positions % 7 + 0.5)
    application.to_csv(application_path, index=False)
    previous_path = tmp_path / "ref-19" / "previous_application.csv"
    previous = pd.read_csv(previous_path)
    previous["AMT_DOWN_PAYMENT"] = np.where(
        np.arange(len(previous)) % 6 == 0, np.nan, 1.0 + np.arange(len(previous)) % 5
    )
    previous.to_csv(previous_path, index=False)
    for module in (pipeline, experiment_datasets):
        monkeypatch.setattr(module, "CURRENT_PATH", "")
        monkeypatch.setattr(module, "RELATIVE_DATA_PATH", str(tmp_path))
    return tmp_path


def _prepare_home_credit(monkeypatch, root, dataset, fill_na, strategy="stratified-holdout"):
    md5, created, output = _prepare(
        monkeypatch, root, dataset, mode="encoded", strategy=strategy,
        encoding="woe", fill_na=fill_na, label="TARGET",
    )
    return created, _reports(output, md5)


@pytest.mark.parametrize("strategy", ["stratified-holdout", "stratified-k-fold"])
def test_home_credit_removal_arm_trains_on_the_same_clients(
    monkeypatch, sparse_home_credit_root, strategy
):
    _created, joined = _prepare_home_credit(
        monkeypatch, sparse_home_credit_root, "home-credit-completa", DROP, strategy
    )
    created, application = _prepare_home_credit(
        monkeypatch, sparse_home_credit_root, "home-credit-somente-aplicacao", DROP, strategy
    )
    assert joined.keys() == application.keys()
    for fold_index, joined_report in joined.items():
        application_report = application[fold_index]
        assert (
            application_report["train_unit_population_md5"]
            == joined_report["train_unit_population_md5"]
        )
        assert (
            application_report["train_unit_population_size"]
            == joined_report["train_unit_population_size"]
        )
        screening = application_report["missing_screening"]
        assert screening["source_dataset_id"] == "home-credit-completa"
        assert screening["threshold_pct"] == joined_report["missing_screening"]["threshold_pct"]
        train = application_report["splits"]["train"]
        assert train["after_imputation"]["missing_cells"] == 0
        # Um cliente por linha: o treino tem tantas linhas quanto clientes.
        assert train["after_imputation"]["rows"] == application_report["train_unit_population_size"]
        assert train["after_imputation"]["rows"] < train["input"]["rows"]
        recorded = {
            item.train_unit_population_md5 for item in created if item.fold_index == fold_index
        }
        assert recorded == {application_report["train_unit_population_md5"]}


def test_home_credit_without_removal_trains_on_the_partition_clients(
    monkeypatch, sparse_home_credit_root
):
    created, reports = _prepare_home_credit(
        monkeypatch, sparse_home_credit_root, "home-credit-somente-aplicacao", "mean"
    )
    for fold_index, report in reports.items():
        train_clients = next(
            item.unit_partition_md5
            for item in created
            if item.fold_index == fold_index and item.split_label == "train"
        )
        assert report["train_unit_population_md5"] == train_clients


def test_paired_removal_requires_the_same_clients_per_split():
    units = pd.Series([1, 2, 3])
    fold = {
        "train_data": pd.DataFrame({"a": [1.0, 2.0, 3.0]}),
        "evaluation_units": {"train_data": units},
        "unit_partition_md5": {"train_data": "clients"},
    }
    paired = {**fold, "unit_partition_md5": {"train_data": "other-clients"}}
    with pytest.raises(ValueError, match="same evaluation units"):
        missing_screening.adopt_paired_removal(fold, paired)


def test_p16_requires_the_same_training_clients_in_the_home_credit_removal():
    rows = [
        _manifest_row("home-credit-completa", "train", DROP, unit="partition"),
        _manifest_row("home-credit-somente-aplicacao", "train", DROP, unit="partition"),
        _manifest_row("home-credit-somente-aplicacao", "train", "mean", unit="partition"),
    ]
    clients = ["kept", "kept", "partition"]
    manifest = pd.DataFrame(rows).assign(
        train_population_md5=["joined-kept", "application-kept", "application"],
        partition_md5=["joined", "application", "application"],
        train_unit_population_md5=clients,
    )
    # Sem remocao, o treino efetivo da somente-aplicacao e o da particao.
    manifest.loc[2, "train_population_md5"] = "application"
    report = CapturingReport()
    sanity.check_training_population(manifest, report)
    assert report.passed("P16")
    assert not report.failed("P16")

    manifest.loc[1, "train_unit_population_md5"] = "kept-differently"
    report = CapturingReport()
    sanity.check_training_population(manifest, report)
    assert any("clientes no treino" in detail for detail in report.failed("P16"))


def test_r5_rejects_home_credit_removal_on_different_training_clients():
    rows = [
        {**_result_row("home-credit-completa", DROP, unit="clients"),
         "train_unit_population_id": "kept"},
        {**_result_row("home-credit-somente-aplicacao", DROP, unit="clients"),
         "train_unit_population_id": "kept"},
    ]
    report = CapturingReport()
    sanity.check_result_pairing(pd.DataFrame(rows), report)
    assert report.passed("R5") and not report.failed("R5")

    rows[1]["train_unit_population_id"] = "other"
    report = CapturingReport()
    sanity.check_result_pairing(pd.DataFrame(rows), report)
    assert any("clientes de treino diferentes" in detail for detail in report.failed("R5"))


def test_pairing_table_marks_paired_and_unpaired_groups():
    rows = [
        _manifest_row("ieee-cis-completa", "test", DROP, partition="test"),
        _manifest_row("ieee-cis-sem-identidade", "test", DROP, partition="test"),
        _manifest_row("ieee-cis-completa", "train", DROP, partition="train", population="kept"),
        _manifest_row("ieee-cis-sem-identidade", "train", DROP, partition="train", population="kept"),
        _manifest_row("home-credit-completa", "test", DROP, partition="joined", unit="test-clients"),
        _manifest_row("home-credit-somente-aplicacao", "test", DROP, partition="app", unit="test-clients"),
        _manifest_row("home-credit-completa", "train", DROP, partition="joined", unit="clients"),
        _manifest_row("home-credit-somente-aplicacao", "train", DROP, partition="app", unit="clients"),
    ]
    manifest = pd.DataFrame(rows).assign(
        train_unit_population_md5=["", "", "", "", "", "", "kept", "other"]
    )
    table = sanity.pairing_table(manifest, {}).set_index("pair_type")
    assert table.loc["ablacao", "paired"] == "sim"
    assert table.loc["ablacao", "same_train_with_removal"] == "sim"
    assert table.loc["ablacao", "same_train_without_removal"] == "n/a"
    assert table.loc["unidade", "same_test"] == "sim"
    assert table.loc["unidade", "same_train_with_removal"] == "nao"
    assert table.loc["unidade", "paired"] == "nao"
    assert "appendix_pairing.csv" in sanity.appendix_tables(manifest, {})
