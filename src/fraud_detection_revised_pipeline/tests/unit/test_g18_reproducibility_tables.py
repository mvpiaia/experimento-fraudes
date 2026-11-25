"""Tabelas de reprodutibilidade geradas dos resultados (item G18 da revisao).

Protege a regra de que o texto nao transcreve numeros a mao: a taxa
automatica do CatBoost aparece como faixa por base, os bracos de execucao
ficam a parte, tempos de maquinas diferentes nunca se agregam, o benchmark de
inferencia de mais de uma maquina e recusado e a saida e deterministica.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import build_reproducibility_tables as tables  # noqa: E402


def _row(dataset, fold, host, learning_rate, fit_time, algorithm="catboost", **extra):
    return {
        "configuration_md5": f"{dataset}-cfg",
        "algorithm": algorithm,
        "regime": "",
        "fold_index": str(fold),
        "split": "test",
        "evaluation_unit": "row",
        "execution_status": "success",
        "dataset_id": dataset,
        "class_weight_intensity": "sqrt",
        "label_fraction": "-1",
        "execution_seed": "",
        "threshold_strategy": "validation-f2",
        "model_hyperparameters": json.dumps(
            {"iterations": 1000, "depth": 6, "learning_rate": learning_rate,
             "auto_class_weights": "SqrtBalanced", "verbose": False}
        ),
        "fit_time": str(fit_time),
        "scoring_time": "0.5",
        "execution_host": host,
        "cpu_model": "AMD Ryzen 7 7735HS" if host == "acer" else "Intel Core i5",
        "gpu_model": "none",
        **extra,
    }


def _write_results(directory: Path, rows) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / "resultados_catboost.csv", index=False)


def test_data_dependent_default_is_shown_as_a_range_per_dataset():
    results = pd.DataFrame(
        [_row("ulb", 1, "acer", 0.05, 10), _row("sparkov", 1, "acer", 0.12, 20)]
    )
    table = tables.hyperparameter_table(results).set_index("name")
    assert table.loc["depth", "value"] == "6"
    assert table.loc["learning_rate", "value"] == "0.05–0.12"
    assert "sparkov: 0.12" in table.loc["learning_rate", "by_dataset"]
    # Bracos de execucao e parametros de registro.
    assert table.loc["auto_class_weights", "kind"] == "arm"
    assert table.loc["class_weight_intensity", "kind"] == "arm"
    assert "verbose" not in table.index


def test_times_of_different_machines_are_never_aggregated():
    results = pd.DataFrame(
        [
            _row("ulb", 1, "acer", 0.05, 10),
            _row("ulb", 2, "acer", 0.05, 12),
            _row("ulb", 3, "e470", 0.05, 40),
        ]
    )
    timing = tables.fold_timing_table(results)
    assert len(timing) == 2
    by_host = timing.set_index("execution_host")
    assert by_host.loc["acer", "fit_median_s"] == pytest.approx(11)
    assert by_host.loc["e470", "fit_median_s"] == pytest.approx(40)


def test_split_rows_of_the_same_fold_count_once():
    results = pd.DataFrame(
        [_row("ulb", 1, "acer", 0.05, 10), _row("ulb", 1, "acer", 0.05, 10, split="validation")]
    )
    assert tables.fold_timing_table(results).loc[0, "folds"] == 1


def test_benchmark_from_more_than_one_machine_is_rejected(tmp_path):
    path = tmp_path / "bench.csv"
    pd.DataFrame(
        [
            {"algorithm": "knn", "dataset_id": "ulb", "execution_host": "acer"},
            {"algorithm": "knn", "dataset_id": "sparkov", "execution_host": "dell"},
        ]
    ).to_csv(path, index=False)
    with pytest.raises(tables.MixedMachinesError, match="more than one machine"):
        tables.benchmark_table(path)


def test_outputs_are_identical_across_runs(tmp_path):
    _write_results(
        tmp_path / "results",
        [_row("ulb", 1, "acer", 0.05, 10), _row("sparkov", 1, "dell", 0.12, 30)],
    )
    outputs = []
    for run in ("a", "b"):
        args = tables.build_argument_parser().parse_args(
            [
                "--results-directory", str(tmp_path / "results"),
                "--latex-directory", str(tmp_path / run),
                "--markdown-output", str(tmp_path / f"{run}.md"),
            ]
        )
        tables.build_tables(args)
        outputs.append(
            [(tmp_path / run / name).read_text() for name in sorted(p.name for p in (tmp_path / run).iterdir())]
            + [(tmp_path / f"{run}.md").read_text()]
        )
    assert outputs[0] == outputs[1]
    latex = (tmp_path / "a" / "tabela_hiperparametros.tex").read_text()
    assert r"\begin{longtable}" in latex
    assert r"learning\_rate" in latex
    assert "Commit:" in (tmp_path / "a.md").read_text()


def test_preparation_times_come_from_the_timing_files(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    for md5, total in (("c1", 4.0), ("c2", 6.0)):
        (prepared / f"{md5}.preparation_timing.json").write_text(
            json.dumps({"configuration_md5": md5, "total_seconds": total,
                        "execution_host": "acer", "cpu_model": "AMD", "gpu_model": "rtx"})
        )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(
        [
            {"configuration_md5": "c1", "dataset_id": "ulb", "categorical_handling_mode": "encoded"},
            {"configuration_md5": "c2", "dataset_id": "ulb", "categorical_handling_mode": "encoded"},
        ]
    ).to_csv(manifest, sep="\t", index=False)
    table = tables.preparation_timing_table(prepared, manifest)
    assert table.loc[0, "configurations"] == 2
    assert table.loc[0, "median_s"] == pytest.approx(5.0)
