"""Preparacao em paralelo por partes e execucao de um processo do plano.

Protege o roteiro da reexecucao: cada parte prepara num manifesto proprio
(que comeca com o que ja foi preparado), a juncao das partes nao perde nem
duplica linhas e recusa conflitos, e run_lane.sh escolhe os runners pelo tipo
de processo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]

HEADER = "\t".join(
    ["database_type", "dataset_filename", "prediction_column_name", "auxiliary_data_is_test_data",
     "categorical_encoding_strategy", "fill_na_strategy", "undersampling_technique",
     "undersampling_parameter_values", "test_strategy", "random_seed", "configuration_md5",
     "dataset_md5", "content_md5", "fold_index", "split"]
)


def _row(md5: str, split: str = "test", dataset: str = "ulb") -> str:
    fields = [dataset, "f.csv", "Class", "false", "none", "none", "none", "none",
              "temporal-holdout", "42", md5, md5[:31] + split[0], "c" * 32, "1", split]
    return "\t".join(fields)


def _bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script], cwd=PIPELINE_DIRECTORY, capture_output=True, text=True,
        env={**os.environ, **(env or {})},
    )


def test_merge_joins_partitions_without_duplicates_and_keeps_a_backup(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    (data / "prepared_dataset_configurations.tsv").write_text(HEADER + "\n" + _row("a" * 32) + "\n")
    first, second = tmp_path / "p1.tsv", tmp_path / "p2.tsv"
    # As duas partes comecaram com a configuracao "a" ja preparada.
    first.write_text("\n".join([HEADER, _row("a" * 32), _row("b" * 32)]) + "\n")
    second.write_text("\n".join([HEADER, _row("a" * 32), _row("c" * 32), _row("c" * 32, "train")]) + "\n")
    result = _bash(f'./merge_partition_manifests.sh "{data}" "{first}" "{second}"')
    assert result.returncode == 0, result.stderr
    lines = (data / "prepared_dataset_configurations.tsv").read_text().splitlines()
    assert lines[0] == HEADER
    assert sorted(line.split("\t")[10][0] + line.split("\t")[14] for line in lines[1:]) == [
        "atest", "btest", "ctest", "ctrain"
    ]
    assert len(list(data.glob("prepared_dataset_configurations.2*.tsv"))) == 1
    assert "3 configuracoes" in result.stdout


def test_merge_refuses_conflicting_rows_and_different_headers(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    first, second = tmp_path / "p1.tsv", tmp_path / "p2.tsv"
    first.write_text(HEADER + "\n" + _row("a" * 32) + "\n")
    conflicting = _row("a" * 32).replace("c" * 32, "d" * 32)
    second.write_text(HEADER + "\n" + conflicting + "\n")
    result = _bash(f'./merge_partition_manifests.sh "{data}" "{first}" "{second}"')
    assert result.returncode == 6 and "conflicting rows" in result.stderr
    assert not (data / "prepared_dataset_configurations.tsv").exists()
    second.write_text("outro\tcabecalho\n")
    assert _bash(f'./merge_partition_manifests.sh "{data}" "{first}" "{second}"').returncode == 4


def test_partition_starts_from_the_main_manifest_and_uses_its_own(tmp_path):
    data = tmp_path / "prepared"
    data.mkdir()
    (data / "prepared_dataset_configurations.tsv").write_text(HEADER + "\n" + _row("a" * 32) + "\n")
    # Script de preparacao falso: registra o manifesto e a base que recebeu.
    fake_dir = tmp_path / "pipeline"
    fake_dir.mkdir()
    for name in ("prepare_partition.sh",):
        (fake_dir / name).write_text((PIPELINE_DIRECTORY / name).read_text())
        (fake_dir / name).chmod(0o755)
    log = tmp_path / "calls.txt"
    fake = fake_dir / "prepare_lr_nb_knn_specific_experiment_datasets.sh"
    fake.write_text(f'#!/usr/bin/env bash\necho "$DATASET_FILTER $CONFIGURATION_MANIFEST $1" >> "{log}"\n')
    fake.chmod(0o755)
    partition = tmp_path / "parte_classic.tsv"
    result = subprocess.run(
        [str(fake_dir / "prepare_partition.sh"), str(data), str(partition), "classic", "ulb", "sparkov"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert partition.read_text() == (data / "prepared_dataset_configurations.tsv").read_text()
    assert log.read_text().splitlines() == [
        f"ulb {partition} {data}", f"sparkov {partition} {data}"
    ]
    bad = subprocess.run(
        [str(fake_dir / "prepare_partition.sh"), str(data), str(partition), "anomaly", "ulb"],
        capture_output=True, text=True,
    )
    assert bad.returncode == 2


def test_run_lane_picks_the_runners_by_lane_kind(tmp_path):
    plan = tmp_path / "plan"
    for lane in ("acer-cpu1", "acer-gpu1"):
        (plan / "lanes" / lane).mkdir(parents=True)
    cpu = _bash(f'./run_lane.sh /dados "{plan}" acer-cpu1', {"LANE_DRY_RUN": "true"})
    gpu = _bash(f'./run_lane.sh /dados "{plan}" acer-gpu1', {"LANE_DRY_RUN": "true"})
    assert cpu.returncode == 0 and gpu.returncode == 0
    for runner in ("run_lr_nb_knn_specific", "run_catboost_xgboost_native", "run_anomaly_detection"):
        assert runner in cpu.stdout and runner not in gpu.stdout
    for runner in ("run_ft_transformer", "run_tabbert"):
        assert runner in gpu.stdout and runner not in cpu.stdout
    assert f"GRID_PLAN={plan} GRID_LANE=acer-cpu1" in cpu.stdout
    missing = _bash(f'./run_lane.sh /dados "{plan}" dell-cpu1')
    assert missing.returncode == 5 and "acer-cpu1" in missing.stderr
