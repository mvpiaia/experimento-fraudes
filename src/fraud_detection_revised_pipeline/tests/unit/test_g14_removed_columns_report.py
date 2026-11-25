"""Item G14: relatorio de colunas removidas por base e por motivo."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import experiment_datasets  # noqa: E402
import report_removed_columns as report  # noqa: E402

from test_g14_catalog_and_loading import (  # noqa: E402
    _write_ieee_files,
    _write_sparkov_files,
)


@pytest.fixture()
def raw_data_root(tmp_path, monkeypatch):
    _write_sparkov_files(tmp_path)
    _write_ieee_files(tmp_path)
    monkeypatch.setattr(experiment_datasets, "CURRENT_PATH", "")
    monkeypatch.setattr(experiment_datasets, "RELATIVE_DATA_PATH", str(tmp_path))
    return tmp_path


def test_variant_removes_the_ablated_columns_from_the_base_features(raw_data_root):
    result = report.build_report(["sparkov", "sparkov-sem-titular"])
    base, variant = result["sparkov"], result["sparkov-sem-titular"]
    assert base["ablated_columns"] == []
    assert variant["ablated_columns"] == [
        "zip", "lat", "long", "city", "city_pop", "job"
    ]
    assert variant["feature_count"] == base["feature_count"] - 6
    assert variant["constant_columns"] == base["constant_columns"]
    assert "customer_merchant_km" in variant["kept_columns"]
    assert "cc_num" in variant["identifier_columns"]


def test_ablation_of_the_ieee_base_drops_card1_from_the_code_columns(raw_data_root):
    result = report.build_report(
        ["ieee-cis-completa", "ieee-cis-completa-sem-alta-cardinalidade"]
    )
    variant = result["ieee-cis-completa-sem-alta-cardinalidade"]
    assert "card1" not in variant["categorical_code_columns"]
    assert {"card2", "addr1"} <= set(variant["categorical_code_columns"])
    assert variant["ablated_columns"] == ["DeviceInfo", "id_31", "id_33", "card1"]
    assert "M1" in variant["constant_columns"]


def test_counts_add_up(raw_data_root):
    item = report.build_report(["ieee-cis-completa"])["ieee-cis-completa"]
    assert item["feature_count"] == item["categorical_count"] + item["numeric_count"]
    assert item["monetary_columns"] == ["TransactionAmt"]


def test_markdown_lists_every_base_and_reason(raw_data_root):
    text = report.format_markdown(report.build_report(["sparkov-sem-titular"]))
    assert "`sparkov-sem-titular`" in text
    assert "Ablação" in text and "Constantes" in text
    assert "`zip`" in text
