"""Contratos dos scripts de preparacao, execucao e retomada por maquina."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]


def _run(script: str, *arguments: str, **environment: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(PIPELINE_DIRECTORY / script), *arguments],
        cwd=PIPELINE_DIRECTORY,
        capture_output=True,
        text=True,
        env={**os.environ, "DRY_RUN": "true", **environment},
    )


def test_preparation_retry_and_force_are_explicit(tmp_path):
    retry = _run("run_acer_preparation.sh", str(tmp_path / "data"), "retry")
    force = _run("run_acer_preparation.sh", str(tmp_path / "data"), "force")
    assert retry.returncode == 0, retry.stderr
    assert force.returncode == 0, force.stderr
    assert "prepare_partition.sh" in retry.stdout
    assert "validate_prepared_datasets.py" in retry.stdout


def test_machine_runner_maps_modes_to_force_reprocess_and_uses_only_plan_lanes(tmp_path):
    plan = tmp_path / "plan"
    (plan / "lanes" / "acer-cpu1").mkdir(parents=True)
    common = {
        "EXECUTION_HOST": "acer",
        "VENV_DIRECTORY": str(tmp_path / "venv"),
        "LOG_DIRECTORY": str(tmp_path / "logs"),
        "TMUX_COMMAND": "true",
    }
    retry = _run("run_machine_lanes.sh", "/data", str(plan), "retry", **common)
    force = _run("run_machine_lanes.sh", "/data", str(plan), "force", **common)
    assert retry.returncode == force.returncode == 0
    assert "FORCE_REPROCESS=false" in retry.stdout
    assert "FORCE_REPROCESS=true" in force.stdout
    assert "acer-cpu1" in force.stdout


def test_machine_runner_refuses_grid_reducing_variables(tmp_path):
    plan = tmp_path / "plan"
    (plan / "lanes" / "acer-cpu1").mkdir(parents=True)
    common = {
        "EXECUTION_HOST": "acer",
        "VENV_DIRECTORY": str(tmp_path / "venv"),
        "LOG_DIRECTORY": str(tmp_path / "logs"),
        "TMUX_COMMAND": "true",
    }
    reduced = _run("run_machine_lanes.sh", "/data", str(plan), "retry",
                   LABEL_FRACTIONS="0.05", **common)
    assert reduced.returncode == 6
    assert "LABEL_FRACTIONS=0.05" in reduced.stderr
    weighted = _run("run_machine_lanes.sh", "/data", str(plan), "retry",
                    CLASS_WEIGHT_INTENSITIES="none", **common)
    assert weighted.returncode == 6
    assert "CLASS_WEIGHT_INTENSITIES=none" in weighted.stderr
    assert "acer-cpu1" not in reduced.stdout
    allowed = _run("run_machine_lanes.sh", "/data", str(plan), "retry",
                   LABEL_FRACTIONS="0.05", ALLOW_GRID_REDUCTION="true", **common)
    assert allowed.returncode == 0, allowed.stderr


def test_machine_runner_reads_the_tmux_global_environment(tmp_path):
    plan = tmp_path / "plan"
    (plan / "lanes" / "acer-cpu1").mkdir(parents=True)
    fake_tmux = tmp_path / "tmux"
    fake_tmux.write_text(
        '#!/usr/bin/env bash\n[[ "$1" == show-environment ]] && echo LABEL_DRAW_SEEDS=0\nexit 0\n',
        encoding="utf-8",
    )
    fake_tmux.chmod(0o755)
    result = _run("run_machine_lanes.sh", "/data", str(plan), "retry",
                  EXECUTION_HOST="acer", VENV_DIRECTORY=str(tmp_path / "venv"),
                  LOG_DIRECTORY=str(tmp_path / "logs"), TMUX_COMMAND=str(fake_tmux))
    assert result.returncode == 6
    assert "LABEL_DRAW_SEEDS=0" in result.stderr


def test_execution_orchestrators_never_call_preparation_scripts():
    for name in (
        "build_and_stage_execution_plan.sh",
        "run_machine_lanes.sh",
        "collect_and_validate_results.sh",
    ):
        text = (PIPELINE_DIRECTORY / name).read_text(encoding="utf-8")
        assert "prepare_partition.sh" not in text
        assert "FORCE_RECREATE_PREPARED_DATASETS" not in text


def test_plan_is_validated_before_allocation_and_staging():
    text = (PIPELINE_DIRECTORY / "build_and_stage_execution_plan.sh").read_text(
        encoding="utf-8"
    )
    validation = text.index("validate_prepared_datasets.py")
    allocation = text.index("grid_plan.py")
    staging = text.index("stage_machine_configurations.sh")
    assert validation < allocation < staging
