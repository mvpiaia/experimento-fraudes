"""Divisao da grade entre o Acer, o E470 e o Dell (item G18 da revisao).

Protege o que torna a reexecucao em tres maquinas confiavel: cada maquina
le o proprio diretorio, sem tentar remontar disco onde nao ha; o plano e
completo, disjunto, deterministico e manda a GPU so para o Acer; a copia
conferida remove um arquivo corrompido e nao recopia o que ja chegou; o
runner pula o que e de outra maquina; e a juncao recusa duplicatas e lista
lacunas.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import calibrate_machine  # noqa: E402
import grid_plan  # noqa: E402
import merge_machine_results as merging  # noqa: E402


def _bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=PIPELINE_DIRECTORY,
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "ALLOW_UNLOCKED_ENVIRONMENT": "true",
            "PYTHON_COMMAND": sys.executable,
            **(env or {}),
        },
    )


# --- 2. Armazenamento por maquina -------------------------------------------------


def test_machine_directory_without_uuid_is_local(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "machine_storage.conf"
    config.write_text(f"MACHINE_PREPARED_DATA_DIRECTORY={prepared}\n")
    result = _bash(
        f'source run_storage_common.sh; configure_prepared_storage "{prepared}" '
        '&& echo "local=${STORAGE_IS_LOCAL} uuid=${STORAGE_DEVICE_UUID}" '
        "&& is_expected_storage_mounted && echo mounted; recover_external_storage; echo recover=$?",
        {"MACHINE_STORAGE_CONFIG": str(config), "STORAGE_DEVICE_UUID": ""},
    )
    assert "local=true uuid=" in result.stdout
    assert "mounted" in result.stdout
    assert "recover=1" in result.stdout
    assert "there is no disk to remount" in result.stderr


def test_machine_uuid_keeps_the_storage_recovery(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    config = tmp_path / "machine_storage.conf"
    config.write_text(
        f"MACHINE_PREPARED_DATA_DIRECTORY={prepared}\nMACHINE_STORAGE_DEVICE_UUID=abcd-1234\n"
    )
    result = _bash(
        f'source run_storage_common.sh; configure_prepared_storage "{prepared}" '
        '&& echo "local=${STORAGE_IS_LOCAL} path=${STORAGE_DEVICE_PATH}"',
        {"MACHINE_STORAGE_CONFIG": str(config), "STORAGE_DEVICE_UUID": ""},
    )
    assert "local=false path=/dev/disk/by-uuid/abcd-1234" in result.stdout


def test_unknown_directory_is_still_rejected(tmp_path):
    result = _bash(
        f'source run_storage_common.sh; configure_prepared_storage "{tmp_path}"; echo code=$?',
        {"MACHINE_STORAGE_CONFIG": str(tmp_path / "absent.conf")},
    )
    assert "code=2" in result.stdout


# --- 3. Calibracao -------------------------------------------------------------------


def test_memory_line_separates_fixed_and_proportional_memory():
    small = {"peak_rss_increase_bytes": 100 + 2 * 1000, "data_csv_bytes": 1000}
    large = {"peak_rss_increase_bytes": 100 + 2 * 4000, "data_csv_bytes": 4000}
    fixed, per_byte = calibrate_machine.memory_line(small, large)
    assert per_byte == pytest.approx(2.0)
    assert fixed == pytest.approx(100.0)


def test_calibration_writes_one_entry_per_family(tmp_path):
    path = calibrate_machine.calibrate(tmp_path, tmp_path / "calibration", families=["linear"])
    calibration = json.loads(path.read_text())
    family = calibration["families"]["linear"]
    assert family["seconds"] > 0
    assert family["memory_per_csv_byte"] >= 0
    assert len(family["measurements"]) == 2
    assert calibration["execution_host"]
    assert "gpu_model" in calibration
    assert calibration["disk_free_bytes"] > 0


# --- 4. Alocacao ---------------------------------------------------------------------


def _calibration(host, gpu, factor, memory_gb=16, disk=10**12):
    families = {
        family: {"seconds": 10.0 * factor, "memory_fixed_bytes": 100 * 2**20, "memory_per_csv_byte": 2.0}
        for family in calibrate_machine.FAMILIES
    }
    return {"execution_host": host, "gpu_model": gpu, "memory_total_gb": memory_gb,
            "disk_free_bytes": disk, "families": families}


def _plan_inputs(tmp_path, machines):
    prepared = tmp_path / "prepared"
    prepared.mkdir(exist_ok=True)
    rows = []
    configurations = [
        ("c-enc-ulb", "ulb", "encoded", "none"),
        ("c-enc-ulb-rus", "ulb", "encoded", "rus"),
        ("c-nat-ulb", "ulb", "native", "none"),
        ("c-enc-ieee", "ieee-cis-completa", "encoded", "none"),
        ("c-nat-ieee", "ieee-cis-completa", "native", "none"),
        ("c-ft-ulb", "ulb", "transformer-indexed", "none"),
    ]
    for md5, dataset, mode, resampling in configurations:
        for split in ("train", "test"):
            dataset_md5 = f"{md5}-{split}"
            content = f"{md5},{split}\n" * 50
            (prepared / f"{dataset_md5}.csv").write_text(content)
            rows.append(
                {"database_type": dataset, "configuration_md5": md5, "dataset_md5": dataset_md5,
                 "content_md5": hashlib.md5(content.encode()).hexdigest(), "fold_index": "1",
                 "split": split, "categorical_handling_mode": mode, "undersampling_technique": resampling}
            )
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(rows).to_csv(manifest, sep="\t", index=False)
    calibration_directory = tmp_path / "calibration"
    calibration_directory.mkdir(exist_ok=True)
    for calibration in machines:
        (calibration_directory / f"calibration_{calibration['execution_host']}.json").write_text(
            json.dumps(calibration)
        )
    return manifest, prepared, calibration_directory


def _allocate(tmp_path, machines, plan_name="plan"):
    manifest, prepared, calibration_directory = _plan_inputs(tmp_path, machines)
    args = grid_plan.build_argument_parser().parse_args(
        ["allocate", "--manifest", str(manifest), "--prepared-data-directory", str(prepared),
         "--calibration-directory", str(calibration_directory), "--reference-host", "acer",
         "--plan-directory", str(tmp_path / plan_name)]
    )
    grid_plan.run_allocate(args)
    return grid_plan.read_plan(tmp_path / plan_name)


THREE_MACHINES = [
    _calibration("acer", "NVIDIA GeForce RTX 4050 Laptop GPU", 1.0, memory_gb=30),
    _calibration("e470", "none", 3.0),
    _calibration("dell", "none", 4.0),
]


def test_plan_is_complete_disjoint_and_shards_configurations(tmp_path):
    plan = _allocate(tmp_path, THREE_MACHINES)
    pairs = [
        (pair["algorithm"], pair["dataset_id"], pair["configurations"][0])
        for machine in plan["machines"].values()
        for pair in machine["pairs"]
    ]
    assert len(pairs) == len(set(pairs))
    assert ("knn", "ulb", "c-enc-ulb") in pairs
    assert ("devnet", "ieee-cis-completa", "c-nat-ieee") in pairs
    for machine in plan["machines"].values():
        for pair in machine["pairs"]:
            assert len(pair["configurations"]) == 1
            if pair["algorithm"] == "isolation-forest":
                # Deteccao de anomalia so sobre configuracoes sem reamostragem.
                assert all("rus" not in md5 for md5 in pair["configurations"])


def test_configurations_of_same_algorithm_and_dataset_can_use_different_hosts(tmp_path):
    calibration = _calibration("acer", "none", 1.0, memory_gb=30)
    machines = [
        grid_plan.Machine("acer", calibration, is_reference=True),
        grid_plan.Machine(
            "dell", {**calibration, "execution_host": "dell"}, is_reference=False
        ),
    ]
    pairs = [
        grid_plan.Pair("knn", "ulb", ["c1"], 10.0, {"c1": 0}),
        grid_plan.Pair("knn", "ulb", ["c2"], 10.0, {"c2": 0}),
    ]
    grid_plan.allocate(pairs, machines, {"c1": [], "c2": []})
    assert {machine.host for machine in machines if machine.pairs} == {"acer", "dell"}


def test_lane_manifest_preserves_methodological_identity_columns(tmp_path):
    manifest, prepared, calibration = _plan_inputs(tmp_path, THREE_MACHINES)
    source = pd.read_csv(manifest, sep="\t", dtype=str)
    for column in (
        "partition_md5",
        "unit_partition_md5",
        "train_population_md5",
        "train_unit_population_md5",
    ):
        source[column] = [f"{column}-{index}" for index in range(len(source))]
    source.to_csv(manifest, sep="\t", index=False)
    args = grid_plan.build_argument_parser().parse_args(
        [
            "allocate",
            "--manifest",
            str(manifest),
            "--prepared-data-directory",
            str(prepared),
            "--calibration-directory",
            str(calibration),
            "--reference-host",
            "acer",
            "--plan-directory",
            str(tmp_path / "plan"),
        ]
    )
    grid_plan.run_allocate(args)
    planned = pd.concat(
        [
            pd.read_csv(path, sep="\t", dtype=str)
            for path in (tmp_path / "plan" / "lanes").glob("*/manifest.tsv")
        ],
        ignore_index=True,
    ).drop_duplicates()
    keys = ["configuration_md5", "dataset_md5", "split"]
    expected = source.sort_values(keys).reset_index(drop=True)
    actual = planned.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns], expected)


def test_gpu_pairs_go_only_to_the_acer(tmp_path):
    plan = _allocate(tmp_path, THREE_MACHINES)
    for host, machine in plan["machines"].items():
        for pair in machine["pairs"]:
            if pair["algorithm"].startswith("ft-transformer"):
                assert host == "acer"


def test_no_gpu_machine_is_an_explicit_error(tmp_path):
    machines = [_calibration("acer", "none", 1.0), _calibration("dell", "none", 4.0)]
    with pytest.raises(grid_plan.AllocationError, match="no calibrated machine has a GPU"):
        _allocate(tmp_path, machines)


def test_memory_limit_keeps_a_pair_off_a_small_machine(tmp_path):
    tiny = _calibration("dell", "none", 0.1, memory_gb=0.0001)
    plan = _allocate(tmp_path, [THREE_MACHINES[0], tiny])
    assert plan["machines"]["dell"]["pairs"] == []


def test_plan_is_identical_across_runs(tmp_path):
    _allocate(tmp_path, THREE_MACHINES, "plan-a")
    _allocate(tmp_path, THREE_MACHINES, "plan-b")
    first, second = tmp_path / "plan-a", tmp_path / "plan-b"
    files = sorted(path.relative_to(first) for path in first.rglob("*") if path.is_file())
    assert files == sorted(path.relative_to(second) for path in second.rglob("*") if path.is_file())
    for name in files:
        assert (first / name).read_text() == (second / name).read_text()


def test_unknown_host_is_not_in_the_plan(tmp_path):
    _allocate(tmp_path, THREE_MACHINES)
    args = grid_plan.build_argument_parser().parse_args(
        ["assignments", "--plan-directory", str(tmp_path / "plan"), "--host", "outra",
         "--output", str(tmp_path / "a.tsv")]
    )
    assert grid_plan.run_assignments(args) == grid_plan.HOST_NOT_IN_PLAN_EXIT_CODE


# --- 5. Copia conferida ----------------------------------------------------------------


def _stage(tmp_path, host="e470"):
    return _bash(
        f'./stage_machine_configurations.sh "{tmp_path / "plan"}" {host} local "{tmp_path / "remote"}"',
        {"PREPARED_DATA_DIRECTORY": str(tmp_path / "prepared")},
    )


def _plan_with_e470_files(tmp_path):
    plan = _allocate(tmp_path, THREE_MACHINES)
    if not plan["machines"]["e470"]["pairs"]:
        pytest.skip("allocation gave the E470 nothing to copy")
    return plan


def test_staging_copies_and_verifies_the_planned_files(tmp_path):
    _plan_with_e470_files(tmp_path)
    result = _stage(tmp_path)
    assert result.returncode == 0, result.stderr
    names = (tmp_path / "plan" / "files_e470.txt").read_text().split()
    for name in names:
        assert (tmp_path / "remote" / name).exists()
    assert (tmp_path / "remote" / "prepared_dataset_configurations.tsv").exists()
    assert "verified" in result.stdout
    # O plano chega por ultimo, pronto para GRID_PLAN naquela maquina.
    assert (tmp_path / "remote" / "grid_plan" / "grid_plan.json").exists()
    assert (tmp_path / "remote" / "grid_plan" / "assignments_e470.tsv").exists()
    assert f"GRID_PLAN={tmp_path / 'remote' / 'grid_plan'}" in result.stdout


def test_corrupted_copy_is_removed_and_fails(tmp_path):
    _plan_with_e470_files(tmp_path)
    manifest_path = tmp_path / "plan" / "manifest_e470.tsv"
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    corrupted = manifest.loc[0, "dataset_md5"]
    manifest.loc[0, "content_md5"] = "0" * 32
    manifest.to_csv(manifest_path, sep="\t", index=False)
    result = _stage(tmp_path)
    assert result.returncode == 11
    assert not (tmp_path / "remote" / f"{corrupted}.csv").exists()
    # Sem dados conferidos, o plano nao chega a maquina.
    assert not (tmp_path / "remote" / "grid_plan").exists()


def test_resumed_staging_does_not_copy_again(tmp_path):
    _plan_with_e470_files(tmp_path)
    assert _stage(tmp_path).returncode == 0
    name = (tmp_path / "plan" / "files_e470.txt").read_text().split()[0]
    first = (tmp_path / "remote" / name).stat().st_ino
    assert _stage(tmp_path).returncode == 0
    assert (tmp_path / "remote" / name).stat().st_ino == first


def test_manifest_without_content_checksum_is_refused(tmp_path):
    _plan_with_e470_files(tmp_path)
    manifest_path = tmp_path / "plan" / "manifest_e470.tsv"
    pd.read_csv(manifest_path, sep="\t", dtype=str).drop(columns=["content_md5"]).to_csv(
        manifest_path, sep="\t", index=False
    )
    assert _stage(tmp_path).returncode == 6


# --- 6. Runners com plano --------------------------------------------------------------


def test_runner_skips_pairs_of_other_machines(tmp_path):
    plan = _allocate(tmp_path, THREE_MACHINES)
    own = plan["machines"]["acer"]["pairs"][0]
    other_host = next(host for host in ("e470", "dell") if plan["machines"][host]["pairs"])
    other = plan["machines"][other_host]["pairs"][0]
    script = (
        "source run_storage_common.sh; GRID_PLAN={plan}; load_grid_plan || exit 9; "
        "plan_assigns {own_alg} {own_md5} && echo own-yes; "
        "plan_assigns {other_alg} {other_md5} || echo other-no"
    ).format(
        plan=tmp_path / "plan", own_alg=own["algorithm"], own_md5=own["configurations"][0],
        other_alg=other["algorithm"], other_md5=other["configurations"][0],
    )
    # O host desta maquina de testes e substituido pelo "acer" do plano.
    plan_dir = tmp_path / "plan"
    real_host = __import__("platform").node()
    (plan_dir / f"assignments_{real_host}.tsv").write_text((plan_dir / "assignments_acer.tsv").read_text())
    data = json.loads((plan_dir / "grid_plan.json").read_text())
    data["machines"][real_host] = data["machines"]["acer"]
    (plan_dir / "grid_plan.json").write_text(json.dumps(data))
    result = _bash(script)
    assert "own-yes" in result.stdout
    assert "other-no" in result.stdout


def test_host_outside_the_plan_stops_the_runner(tmp_path):
    _allocate(tmp_path, THREE_MACHINES)
    result = _bash(f"source run_storage_common.sh; GRID_PLAN={tmp_path / 'plan'}; load_grid_plan; echo code=$?")
    assert "code=5" in result.stdout


# --- 7. Juncao ---------------------------------------------------------------------------


def _result_row(md5, algorithm, host, **extra):
    return {"configuration_md5": md5, "algorithm": algorithm, "regime": "", "fold_index": "1",
            "split": "test", "evaluation_unit": "row", "execution_status": "success",
            "class_weight_intensity": "none", "label_fraction": "-1", "execution_seed": "",
            "threshold_strategy": "validation-f2", "execution_host": host, **extra}


def _write_collection(tmp_path, rows_by_host):
    collection = tmp_path / "collection"
    for host, rows in rows_by_host.items():
        (collection / host).mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(rows)
        for algorithm, group in frame.groupby("algorithm"):
            group.to_csv(collection / host / f"resultados_{algorithm}.csv", index=False)
    return collection


def _small_plan(tmp_path):
    plan_directory = tmp_path / "plan"
    plan_directory.mkdir()
    (plan_directory / "grid_plan.json").write_text(json.dumps({
        "code_commit": "abc",
        "machines": {
            "acer": {"pairs": [{"algorithm": "knn", "dataset_id": "ulb", "configurations": ["m1"]}]},
            "dell": {"pairs": [{"algorithm": "naive-bayes", "dataset_id": "ulb", "configurations": ["m1", "m2"]}]},
        },
    }))
    return plan_directory


def test_complete_grid_is_merged(tmp_path):
    plan = _small_plan(tmp_path)
    collection = _write_collection(tmp_path, {
        "acer": [_result_row("m1", "knn", "acer")],
        "dell": [_result_row("m1", "naive-bayes", "dell"), _result_row("m2", "naive-bayes", "dell")],
    })
    report = merging.merge(plan, collection, tmp_path / "merged")
    assert report["complete"] and not report["unplanned_rows"]
    assert len(pd.read_csv(tmp_path / "merged" / "resultados_naive-bayes.csv")) == 2


def test_duplicate_rows_stop_the_merge(tmp_path):
    plan = _small_plan(tmp_path)
    collection = _write_collection(tmp_path, {
        "acer": [_result_row("m1", "knn", "acer")],
        "dell": [_result_row("m1", "knn", "dell")],
    })
    with pytest.raises(merging.DuplicateRowsError):
        merging.merge(plan, collection, tmp_path / "merged")
    assert not (tmp_path / "merged").exists()


def test_rows_from_an_unplanned_machine_and_gaps_are_reported(tmp_path):
    plan = _small_plan(tmp_path)
    collection = _write_collection(tmp_path, {
        "acer": [_result_row("m1", "knn", "acer"), _result_row("m1", "naive-bayes", "acer")],
    })
    report = merging.merge(plan, collection, tmp_path / "merged")
    assert report["unplanned_rows"] == [
        {"algorithm": "naive-bayes", "configuration_md5": "m1", "planned_host": "dell", "execution_host": "acer"}
    ]
    assert [gap["configuration_md5"] for gap in report["gaps"]] == ["m2"]
    assert not report["complete"]
