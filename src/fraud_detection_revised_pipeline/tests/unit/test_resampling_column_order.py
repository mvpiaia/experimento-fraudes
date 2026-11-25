"""A reamostragem preserva a ordem das colunas (fix-resampling-column-order).

No bloco nativo, a carga acrescenta atributos derivados depois do alvo
(``hour_of_day`` na ULB, os derivados da IEEE-CIS). Se o treino reamostrado
mandasse o alvo para o fim, a validacao e o teste ficariam com outra ordem e o
``require_common_schema`` recusaria toda configuracao nativa com RUS.
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

import lr_nb_knn_specific as pipeline  # noqa: E402
import prepare_lr_nb_knn_datasets as preparation  # noqa: E402

from test_g14_catalog_and_loading import _write_ieee_files  # noqa: E402


def _training_frame(target_last: bool = False) -> pd.DataFrame:
    generator = np.random.default_rng(7)
    row_count = 60
    frame = pd.DataFrame(
        {
            "a": generator.normal(size=row_count),
            "Class": [1] * 12 + [0] * (row_count - 12),
            "derivado": generator.normal(size=row_count),
        }
    )
    if target_last:
        frame = frame[["a", "derivado", "Class"]]
    return frame


@pytest.mark.parametrize(
    ("technique", "parameters"),
    [
        ("random-undersampling", "sampling_strategy=auto:replacement=false"),
        ("RUS+SMOTE", "rus_sampling_strategy=0.5:smote_sampling_strategy=1.0"),
        ("tomek-links", None),
    ],
)
def test_resampling_keeps_the_input_column_order(technique, parameters):
    resampled = pipeline.resample_dataframe(
        _training_frame(), "Class", technique, parameters
    )

    assert list(resampled.columns) == ["a", "Class", "derivado"]


def test_resampling_with_the_target_last_is_unchanged():
    # Bloco classico: o alvo ja e a ultima coluna, e o resultado e o mesmo de
    # antes da correcao (features reamostradas seguidas do alvo).
    frame = _training_frame(target_last=True)
    resampled = pipeline.resample_dataframe(
        frame, "Class", "random-undersampling", "sampling_strategy=auto:replacement=false"
    )

    from imblearn.under_sampling import RandomUnderSampler

    features, target = RandomUnderSampler(
        sampling_strategy="auto", replacement=False, random_state=pipeline.RANDOM_STATE
    ).fit_resample(frame.drop(columns=["Class"]), frame["Class"])
    expected = features.reset_index(drop=True)
    expected["Class"] = target.reset_index(drop=True)
    pd.testing.assert_frame_equal(resampled, expected)


def test_native_preparation_with_undersampling_passes_the_schema_check(
    tmp_path, monkeypatch
):
    _write_ieee_files(tmp_path)
    monkeypatch.setattr(pipeline, "CURRENT_PATH", "")
    monkeypatch.setattr(pipeline, "RELATIVE_DATA_PATH", str(tmp_path))
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--dataset", "ieee-cis-completa",
            "--test-strategy", "stratified-holdout",
            "--prediction-column", "isFraud",
            "--fill-na-strategy", "none",
            "--categorical-handling-mode", "native",
            "--undersampling-technique", "random-undersampling",
            "--undersampling-parameters", "sampling_strategy=auto:replacement=false",
            "--output-directory", str(output),
            "--auxiliary-data-is-test-data", "false",
        ],
    )

    _, created = preparation.prepare_datasets(preparation.parse_arguments())

    headers = {
        created_file.split_label: list(pd.read_csv(created_file.path, nrows=0).columns)
        for created_file in created
    }
    assert headers["train"] == headers["validation"] == headers["test"]
