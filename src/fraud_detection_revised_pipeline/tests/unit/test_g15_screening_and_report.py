"""Item G15: triagem do drop-incomplete-rows, relatorio de preparacao e protecoes.

Os resultados de 22/09 mostraram metricas identicas bit a bit entre
imputacoes e entre codificacoes, porque uma PCA anterior a divisao descartava
as categoricas e completava os faltantes em silencio. Estes testes protegem as
tres correcoes: a PCA e os executores recusam entradas invalidas, a remocao de
registros nao esvazia o treino, e cada dobra registra o que aconteceu com ela.
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

import lr_nb_knn_specific as pipeline  # noqa: E402
import missing_screening  # noqa: E402
import preparation_report  # noqa: E402
import prepared_dataset_storage as storage  # noqa: E402

from test_g14_registry_and_categorical import _prepare, raw_data_root  # noqa: E402,F401


DROP = missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY


def _md5(fill_na_strategy: str) -> str:
    return storage.build_prepared_dataset_md5(
        "ieee-cis-completa", "woe", fill_na_strategy, "stratified-holdout",
        "isFraud", None, None, False, "encoded",
    )


# --- 1. Protecoes de entrada --------------------------------------------------


def test_pca_rejects_non_numeric_columns():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"], "label": [0, 1, 0]})
    with pytest.raises(ValueError, match="non-numeric columns: \\['b'\\]"):
        pipeline._apply_pca(train, None, None, "label")


def test_pca_rejects_missing_values():
    train = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [1.0, 2.0, 4.0], "label": [0, 1, 0]})
    with pytest.raises(ValueError, match="missing or non-finite values: \\['a'\\]"):
        pipeline._apply_pca(train, None, None, "label")


def test_pca_rejects_missing_values_in_held_out_data():
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 5.0], "b": [1.0, 2.0, 4.0, 3.0], "label": [0, 1, 0, 1]})
    test = pd.DataFrame({"a": [1.0, np.nan], "b": [1.0, 2.0], "label": [0, 1]})
    with pytest.raises(ValueError, match="missing or non-finite"):
        pipeline._apply_pca(train, None, test, "label")


def test_pca_returns_its_summary():
    rng = np.random.default_rng(0)
    train = pd.DataFrame(rng.normal(size=(40, 5)), columns=list("abcde"))
    train["label"] = [0, 1] * 20
    *_frames, summary = pipeline._apply_pca(train, None, None, "label")
    assert summary["pca_input_columns"] == 5
    assert 1 <= summary["pca_components"] <= 5
    assert summary["pca_explained_variance"] >= 0.95


def test_classic_executor_rejects_residual_missing_values():
    data = pd.DataFrame({"a": [1.0, np.nan], "label": [0, 1]})
    with pytest.raises(ValueError, match="missing or non-finite values: \\['a'\\]"):
        pipeline._linear_regression_features_and_target(data, "label")


def test_classic_executor_rejects_residual_categorical_columns():
    data = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"], "label": [0, 1]})
    with pytest.raises(ValueError, match="non-numeric columns: \\['b'\\]"):
        pipeline._linear_regression_features_and_target(data, "label")


# --- 2. Triagem do drop-incomplete-rows ---------------------------------------


def _screening_frame() -> pd.DataFrame:
    """20 linhas, 4 fraudes.

    ``sparse`` falta em 60% das linhas, incluindo 2 das 4 fraudes;
    ``mild`` falta em 10% (2 linhas legitimas); ``full`` nunca falta.
    """
    label = [1, 1, 1, 1] + [0] * 16
    sparse = [np.nan, np.nan, 1.0, 1.0] + [np.nan] * 10 + [1.0] * 6
    mild = [1.0] * 18 + [np.nan, np.nan]
    return pd.DataFrame({"sparse": sparse, "mild": mild, "full": range(20), "label": label})


def test_largest_threshold_that_keeps_enough_frauds_is_chosen():
    screening = missing_screening.select_missing_screening(_screening_frame(), "label")
    # Manter "sparse" (limiar >= 60%) preservaria 2 de 4 fraudes (50%); de 10%
    # a 55% so "sparse" sai e todas as fraudes ficam. O maior admissivel e 55.
    assert screening.threshold_pct == 55
    assert screening.screened_out_columns == ("sparse",)
    assert screening.positives_after == screening.positives_before == 4
    assert screening.rows_after == 18


def test_threshold_zero_is_the_fallback():
    frame = pd.DataFrame({"a": [np.nan, 1.0, 2.0, 3.0], "label": [1, 0, 0, 0]})
    screening = missing_screening.select_missing_screening(frame, "label")
    assert screening.threshold_pct == 20
    assert screening.screened_out_columns == ("a",)
    # "a" falta so na unica fraude (2,5% das linhas): de 5% para cima a
    # coluna fica e a fraude sai; so o limiar 0 a preserva.
    rare_gap_in_the_fraud = pd.DataFrame(
        {"a": [np.nan] + [1.0] * 39, "b": range(40), "label": [1] + [0] * 39}
    )
    fallback = missing_screening.select_missing_screening(rare_gap_in_the_fraud, "label")
    assert fallback.threshold_pct == 0
    assert fallback.screened_out_columns == ("a",)
    assert fallback.rows_after == 40


def test_threshold_is_chosen_on_training_data_only():
    fold = {
        "train_data": _screening_frame(),
        # Validacao e teste sem nenhum valor em "mild": nao podem mudar a escolha.
        "validation_data": _screening_frame().assign(mild=np.nan),
        "test_data": _screening_frame().assign(mild=np.nan),
    }
    imputed = pipeline._impute_data_splits({"test_info": [fold]}, DROP, "label")
    imputed_fold = imputed["test_info"][0]
    assert imputed_fold["missing_screening"].threshold_pct == 55
    for split_name in ("train_data", "validation_data", "test_data"):
        assert list(imputed_fold[split_name].columns) == ["mild", "full", "label"]
    assert len(imputed_fold["train_data"]) == 18
    assert len(imputed_fold["validation_data"]) == 20
    assert not imputed_fold["test_data"].isna().any().any()


@pytest.mark.parametrize("strategy", ["mean", "median", "mode"])
def test_imputation_strategies_are_not_screened(strategy):
    imputed = pipeline._impute_data_splits(
        {"test_info": [{"train_data": _screening_frame()}]}, strategy, "label"
    )
    fold = imputed["test_info"][0]
    assert "missing_screening" not in fold
    assert list(fold["train_data"].columns) == ["sparse", "mild", "full", "label"]
    assert len(fold["train_data"]) == 20


def test_screening_rule_enters_only_the_drop_identity(monkeypatch):
    before = {strategy: _md5(strategy) for strategy in ("mean", "median", "mode", DROP)}
    monkeypatch.setattr(storage, "DROP_SCREENING_RULE", "min-fraud-retention=0.9;step=5")
    after = {strategy: _md5(strategy) for strategy in ("mean", "median", "mode", DROP)}
    assert {s: after[s] for s in ("mean", "median", "mode")} == {
        s: before[s] for s in ("mean", "median", "mode")
    }
    assert after[DROP] != before[DROP]


def test_g14_preparations_are_not_reused(monkeypatch):
    current = _md5("mean")
    monkeypatch.setattr(storage, "PREPROCESSING_REVISION", "g14")
    assert _md5("mean") != current


# --- 3. Relatorio de preparacao ------------------------------------------------


def _reports(output: Path, md5: str) -> list[dict]:
    paths = sorted(output.glob(f"{md5}.fold*{preparation_report.PREPARATION_REPORT_FILE_SUFFIX}"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _saved(created, fold_index: int, split_label: str) -> pd.DataFrame:
    match = next(
        item for item in created
        if item.fold_index == fold_index and item.split_label == split_label
    )
    return pd.read_csv(match.path)


@pytest.mark.parametrize("fill_na", ["median", DROP])
def test_classic_report_matches_the_saved_files(monkeypatch, raw_data_root, fill_na):
    md5, created, output = _prepare(
        monkeypatch, raw_data_root, "ieee-cis-transacional", mode="encoded",
        encoding="woe", fill_na=fill_na,
    )
    reports = _reports(output, md5)
    assert len(reports) == len({item.fold_index for item in created})
    for report in reports:
        fold_index = report["fold_index"]
        train = _saved(created, fold_index, "train")
        splits = report["splits"]
        assert splits["train"]["input"]["missing_cells"] > 0
        for split_label in ("train", "validation", "test"):
            assert splits[split_label]["after_imputation"]["missing_cells"] == 0
            assert splits[split_label]["output"]["missing_cells"] == 0
        assert splits["train"]["output"]["rows"] == len(train)
        assert splits["train"]["output"]["positives"] == int((train["isFraud"] == 1).sum())
        # IEEE-CIS recebe PCA no bloco classico.
        assert report["pca"]["pca_components"] == len(train.columns) - 1
        assert report["output_feature_columns"] == [c for c in train.columns if c != "isFraud"]
        manifest_fields = {
            item.preparation_report_md5 for item in created if item.fold_index == fold_index
        }
        assert manifest_fields == {
            storage.compute_file_content_md5(
                preparation_report.preparation_report_path(output, md5, fold_index)
            )
        }


def test_drop_report_and_manifest_fields_follow_the_rule(monkeypatch, raw_data_root):
    md5, created, output = _prepare(
        monkeypatch, raw_data_root, "ieee-cis-transacional", mode="encoded",
        encoding="woe", fill_na=DROP,
    )
    for report in _reports(output, md5):
        screening = report["missing_screening"]
        assert screening["rule"] == missing_screening.DROP_SCREENING_RULE
        assert (
            screening["positives_after"]
            >= missing_screening.DROP_SCREENING_MIN_FRAUD_RETENTION * screening["positives_before"]
        )
        train_after_removal = report["splits"]["train"]["after_imputation"]
        assert train_after_removal["rows"] == screening["rows_after"]
        rows = [item for item in created if item.fold_index == report["fold_index"]]
        assert {item.missing_screening_threshold for item in rows} == {
            str(screening["threshold_pct"])
        }
        assert {item.screened_out_columns for item in rows} == {
            ",".join(screening["screened_out_columns"])
        }


def test_native_none_report_matches_the_saved_files(monkeypatch, raw_data_root):
    md5, created, output = _prepare(monkeypatch, raw_data_root, "ieee-cis-transacional")
    for report in _reports(output, md5):
        train = report["splits"]["train"]
        saved_train = _saved(created, report["fold_index"], "train")
        assert train["input"]["missing_cells"] > 0
        # No nativo, categorica ausente vira a categoria explicita
        # "__missing__"; numerico ausente continua ausente para o estimador.
        assert train["output"]["missing_cells"] == int(
            saved_train.drop(columns=["isFraud"]).isna().to_numpy().sum()
        )
        assert report["missing_screening"] is None
        assert report["pca"] is None


def test_preparation_fails_on_divergent_schema():
    fold = {
        "train_data": pd.DataFrame({"a": [1.0], "label": [0]}),
        "test_data": pd.DataFrame({"b": [1.0], "label": [0]}),
    }
    with pytest.raises(ValueError, match="test_data schema differs"):
        preparation_report.require_common_schema(fold, 1)


def test_staging_requires_the_declared_report(tmp_path):
    md5 = "0" * 32
    dataset_md5 = "1" * 32
    (tmp_path / f"{dataset_md5}.csv").write_text("a,label\n1,0\n", encoding="utf-8")
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [{
            "configuration_md5": md5, "dataset_md5": dataset_md5, "fold_index": "1",
            "dataset_id": "ieee-cis-completa",
            "split": "train", "preparation_report_md5": "2" * 32,
        }]
    ).to_csv(manifest, sep="\t", index=False)
    with pytest.raises(pipeline.PreparedDatasetStorageError, match="was not found"):
        with pipeline.stage_prepared_dataset_files(tmp_path, manifest, md5, tmp_path / "stage"):
            pass
    preparation_report.preparation_report_path(tmp_path, md5, 1).write_text("{}\n")
    with pytest.raises(pipeline.PreparedDatasetStorageError, match="does not match"):
        with pipeline.stage_prepared_dataset_files(tmp_path, manifest, md5, tmp_path / "stage"):
            pass


# --- 4. Manifesto --------------------------------------------------------------


def test_manifest_header_ends_with_the_report_columns():
    script = (PIPELINE_DIRECTORY / "prepare_grid_common.sh").read_text(encoding="utf-8")
    header = script.split('manifest_header="$(printf')[1].split(')"')[0]
    columns = [line.strip().strip('\\').strip().strip('"') for line in header.splitlines()]
    columns = [column for column in columns if column and column != "'%s\\t'"]
    assert columns[-8:] == [
        "feature_schema_md5",
        "preparation_report_md5",
        "missing_screening_threshold",
        "missing_screening_rule",
        "screened_out_columns",
        # Chaves de pareamento do item G16.
        "train_population_md5",
        "unit_partition_md5",
        "train_unit_population_md5",
    ]
    # Manifesto com outro cabecalho e recusado em vez de misturar esquemas.
    assert "Existing manifest has a different header" in script
