"""Staging dos arquivos laterais dos Transformers (vocabulario e janelas)."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import run_transformer_prepared_experiments as runner  # noqa: E402
PreparedDatasetStorageError = runner.PreparedDatasetStorageError


def _rows(vocabulary: str, windows: str) -> pd.DataFrame:
    return pd.DataFrame({"vocabulary_path": [vocabulary], "window_index_path": [windows]})


def test_side_files_are_found_by_name_after_the_preparation_scratch_is_gone(tmp_path):
    prepared = tmp_path / "prepared"
    staged = tmp_path / "staged"
    prepared.mkdir()
    staged.mkdir()
    (prepared / "abc_fold-1_vocabulary.json").write_text("{}", encoding="utf-8")
    (prepared / "abc_fold-1_windows.csv").write_text("w\n", encoding="utf-8")
    scratch = "/home/worker/preparation-scratch/abc"

    runner.stage_transformer_artifacts(
        _rows(f"{scratch}/abc_fold-1_vocabulary.json", f"{scratch}/abc_fold-1_windows.csv"),
        staged,
        prepared,
    )

    assert (staged / "abc_fold-1_vocabulary.json").read_text(encoding="utf-8") == "{}"
    assert (staged / "abc_fold-1_windows.csv").is_file()


def test_missing_side_file_is_still_a_storage_failure(tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    with pytest.raises(PreparedDatasetStorageError, match="abc_fold-1_vocabulary.json"):
        runner.stage_transformer_artifacts(
            _rows("/home/worker/preparation-scratch/abc/abc_fold-1_vocabulary.json", ""),
            staged,
            tmp_path / "prepared",
        )
