"""Validacao ``--mode partition`` sobre os partidores que devolvem posicoes."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import validate_prepared_datasets as validator  # noqa: E402


def test_partition_audit_materializes_fold_positions(monkeypatch):
    rows = 400
    generator = np.random.default_rng(0)
    data = pd.DataFrame({
        "Time": np.arange(rows, dtype=float),
        "V1": generator.normal(size=rows),
        "Amount": generator.uniform(size=rows),
        "Class": (np.arange(rows) % 10 == 0).astype(int),
    })
    monkeypatch.setattr(validator.pipeline, "load_data", lambda _dataset: (data, None, None))
    report = validator.ValidationReport()
    validator.validate_partitioning("ulb", report, 42)
    checks = {outcome.check_id for outcome in report.outcomes}
    assert {"D2", "D3", "D4"} <= checks
    assert not report.failures
