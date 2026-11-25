"""Processos paralelos por maquina e HDs da rede no plano da grade.

Protege o que leva o makespan para perto da fila da GPU: o LPT reparte os
pares entre processos (a GPU do Acer separada da CPU dele), os processos nunca
dividem par, CSV nem staging, a memoria simultanea respeita a RAM, a leitura
pela rede pesa na estimativa, so a maquina com HD proprio recebe copias, e o
backup dos resultados e seguro com o processo gravando.
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

import calibrate_machine  # noqa: E402
import grid_plan  # noqa: E402
import merge_machine_results as merging  # noqa: E402
from test_g18_grid_distribution import (  # noqa: E402
    THREE_MACHINES,
    _allocate,
    _bash,
    _calibration,
    _plan_inputs,
)


def _with_lanes(calibration: dict, cpu_lanes: int, gpu_lanes: int | None = None, **extra) -> dict:
    result = {**calibration, "cpu_lanes": cpu_lanes, **extra}
    if gpu_lanes is not None:
        result["gpu_lanes"] = gpu_lanes
    return result


LANE_MACHINES = [
    _with_lanes(THREE_MACHINES[0], 3, 1, memory_gb=30),
    _with_lanes(THREE_MACHINES[1], 2),
    _with_lanes(THREE_MACHINES[2], 2),
]


def _lane_pairs(plan: dict) -> dict[str, list[tuple[str, str, str]]]:
    return {
        lane: [
            (pair["algorithm"], pair["dataset_id"], pair["configurations"][0])
            for pair in entry["pairs"]
        ]
        for machine in plan["machines"].values()
        for lane, entry in machine["lanes"].items()
    }


def test_lanes_are_declared_by_the_calibration():
    acer = grid_plan.Machine("acer", LANE_MACHINES[0], is_reference=True)
    assert [lane.name for lane in acer.lanes] == ["acer-cpu1", "acer-cpu2", "acer-cpu3", "acer-gpu1"]
    old = grid_plan.Machine("dell", THREE_MACHINES[2], is_reference=False)
    assert [lane.name for lane in old.lanes] == ["dell-cpu1"]
    no_gpu = grid_plan.Machine("e470", _with_lanes(THREE_MACHINES[1], 2, 1), is_reference=False)
    assert [lane.kind for lane in no_gpu.lanes] == ["cpu", "cpu"]


def test_pairs_are_disjoint_between_lanes_and_gpu_pairs_stay_on_the_gpu_lane(tmp_path):
    plan = _allocate(tmp_path, LANE_MACHINES)
    by_lane = _lane_pairs(plan)
    all_pairs = [pair for pairs in by_lane.values() for pair in pairs]
    assert len(all_pairs) == len(set(all_pairs))
    host_pairs = sorted(
        (pair["algorithm"], pair["dataset_id"], pair["configurations"][0])
        for machine in plan["machines"].values()
        for pair in machine["pairs"]
    )
    assert sorted(all_pairs) == host_pairs
    for lane, pairs in by_lane.items():
        for algorithm, _dataset, _configuration in pairs:
            assert algorithm.startswith("ft-transformer") == lane.endswith("gpu1")
    # Mais processos que pares nao e erro: sobra processo vazio.
    assert sum(1 for pairs in by_lane.values() if pairs) >= 4


def test_lanes_shorten_the_makespan_down_to_the_largest_pair(tmp_path):
    single = _allocate(tmp_path, THREE_MACHINES, "single")
    lanes = _allocate(tmp_path, LANE_MACHINES, "lanes")
    makespan = lambda plan: max(machine["estimated_seconds"] for machine in plan["machines"].values())  # noqa: E731
    assert makespan(lanes) <= makespan(single)
    # Um par roda inteiro num processo: o maior par e o limite inferior do
    # makespan, e com processos suficientes o plano o atinge.
    largest_pair = max(
        pair["estimated_seconds"]
        for machine in lanes["machines"].values()
        for pair in machine["pairs"]
    )
    assert makespan(lanes) == pytest.approx(largest_pair)
    # Sem processos, as maquinas carregam mais trabalho em fila.
    total_cpu_seconds = lambda plan: sum(  # noqa: E731
        entry["estimated_seconds"]
        for machine in plan["machines"].values()
        for name, entry in machine["lanes"].items()
        if "-cpu" in name
    )
    busiest = lambda plan: sorted(  # noqa: E731
        entry["estimated_seconds"]
        for machine in plan["machines"].values()
        for name, entry in machine["lanes"].items()
        if "-cpu" in name
    )[-2]
    assert busiest(lanes) < busiest(single) or total_cpu_seconds(lanes) <= total_cpu_seconds(single)


def test_lane_artifacts_and_assignments(tmp_path):
    plan = _allocate(tmp_path, LANE_MACHINES)
    plan_directory = tmp_path / "plan"
    for machine in plan["machines"].values():
        for lane, entry in machine["lanes"].items():
            assignments = (plan_directory / "lanes" / lane / "assignments.tsv").read_text()
            expected = {md5 for pair in entry["pairs"] for md5 in pair["configurations"]}
            assert {line.split("\t")[1] for line in assignments.splitlines()} == expected
            manifest = pd.read_csv(plan_directory / "lanes" / lane / "manifest.tsv", sep="\t", dtype=str)
            assert set(manifest["configuration_md5"]) == expected
    host, lane = "acer", "acer-cpu1"
    output = tmp_path / "lane.tsv"
    args = grid_plan.build_argument_parser().parse_args(
        ["assignments", "--plan-directory", str(plan_directory), "--host", host,
         "--lane", lane, "--output", str(output)]
    )
    assert grid_plan.run_assignments(args) == 0
    assert output.read_text() == (plan_directory / "lanes" / lane / "assignments.tsv").read_text()
    args = grid_plan.build_argument_parser().parse_args(
        ["assignments", "--plan-directory", str(plan_directory), "--host", host,
         "--lane", "dell-cpu1", "--output", str(output)]
    )
    assert grid_plan.run_assignments(args) == grid_plan.HOST_NOT_IN_PLAN_EXIT_CODE


def test_concurrent_memory_counts_the_other_lanes():
    machine = grid_plan.Machine("dell", _with_lanes(THREE_MACHINES[2], 2), is_reference=False)
    first, second = machine.lanes
    first.peak_memory = 9 * 2**30
    # 16 GB de RAM: 9 GB num processo + 4 GB no outro passam de 80% (12,8 GB).
    assert grid_plan._concurrent_memory(machine, second, 4 * 2**30) == 13 * 2**30
    assert grid_plan._concurrent_memory(machine, second, 4 * 2**30) > 0.8 * machine.memory_bytes
    assert grid_plan._concurrent_memory(machine, first, 4 * 2**30) == 4 * 2**30


def test_memory_of_other_lanes_keeps_a_second_large_pair_off_the_machine(tmp_path):
    # Cada par usa ~11 GB: so cabe um por vez numa maquina de 16 GB.
    big = {"memory_fixed_bytes": 11 * 2**30, "memory_per_csv_byte": 0.0, "seconds": 10.0}
    dell = _with_lanes(_calibration("dell", "none", 1.0), 2)
    dell["families"] = {family: dict(big) for family in calibrate_machine.FAMILIES}
    acer = _with_lanes(THREE_MACHINES[0], 1, 1, memory_gb=30)
    plan = _allocate(tmp_path, [acer, dell])
    dell_lanes = plan["machines"]["dell"]["lanes"]
    busy = [lane for lane, entry in dell_lanes.items() if entry["pairs"]]
    assert len(busy) <= 1


def test_network_reads_are_charged_and_only_own_drives_receive_copies(tmp_path):
    manifest, prepared, _calibrations = _plan_inputs(tmp_path, [])
    primary = str(prepared)
    acer = _with_lanes(THREE_MACHINES[0], 1, 1, memory_gb=30, prepared_data_directory=primary)
    lenovo = _with_lanes(_calibration("e470", "none", 1.0), 1, prepared_data_directory="/mnt/network-transparent-hd/prepared")
    dell = _with_lanes(
        _calibration("dell", "none", 1.0), 1,
        prepared_data_directory=primary, storage_is_network=True,
    )
    plan = _allocate(tmp_path, [acer, lenovo, dell])
    machines = plan["machines"]
    assert machines["e470"]["copies_configurations"] is True
    assert machines["e470"]["read_directory"] == "/mnt/network-transparent-hd/prepared"
    assert machines["dell"]["copies_configurations"] is False
    assert machines["dell"]["read_directory"] == primary
    assert machines["dell"]["reads_over_network"] is True
    assert (tmp_path / "plan" / "files_dell.txt").read_text() == ""
    assert (tmp_path / "plan" / "files_e470.txt").read_text() != "" or not machines["e470"]["pairs"]
    # Mesma velocidade, mas o Dell paga a leitura: o mesmo par custa mais nele.
    raw = pd.read_csv(manifest, sep="\t", dtype=str)
    files = grid_plan._configuration_files(prepared, raw)
    pair = grid_plan.build_pairs(grid_plan.normalize_manifest_dataset_identity(raw), files)[0]
    pair_bytes = sum(path.stat().st_size for md5 in pair.configurations for path in files[md5])

    def cost_on(calibration: dict) -> float:
        reference = grid_plan.Machine("acer", acer, is_reference=True)
        for lane in reference.lanes:
            lane.load_seconds = 1e12  # ocupado: o par vai para a outra maquina
        other = grid_plan.Machine(calibration["execution_host"], calibration, is_reference=False)
        grid_plan.allocate([pair], [reference, other], files, primary, network_bytes_per_second=1.0)
        return other.lanes[0].load_seconds

    local_cost = cost_on(lenovo)
    assert cost_on(dell) == pytest.approx(local_cost + pair_bytes)


def test_calibration_records_lanes_and_network_storage(tmp_path):
    concurrent = calibrate_machine._measure_concurrently("linear", 2000, 2)
    assert concurrent["copies"] == 2 and concurrent["seconds"] > 0
    assert calibrate_machine.storage_is_network(tmp_path) is False


def test_merge_treats_each_lane_as_a_source_and_joins_companion_files(tmp_path):
    plan = _allocate(tmp_path, LANE_MACHINES)
    collection = tmp_path / "collection"
    row = {"execution_status": "success", "algorithm": "knn", "configuration_md5": "c-enc-ulb",
           "fold_index": "1", "split": "test", "execution_host": "acer"}
    for lane in ("acer-cpu1", "acer-cpu2"):
        (collection / lane).mkdir(parents=True)
        (collection / lane / "resultados_catboost_history_strata.jsonl").write_text(
            json.dumps({"lane": lane}) + "\n"
        )
    pd.DataFrame([row]).to_csv(collection / "acer-cpu1" / "resultados_knn.csv", index=False)
    report = merging.merge(tmp_path / "plan", collection, tmp_path / "merged")
    assert report["companion_files"] == ["resultados_catboost_history_strata.jsonl"]
    lines = (tmp_path / "merged" / "resultados_catboost_history_strata.jsonl").read_text().splitlines()
    assert [json.loads(line)["lane"] for line in lines] == ["acer-cpu1", "acer-cpu2"]
    # A mesma linha de sucesso em dois processos e duplicata.
    pd.DataFrame([row]).to_csv(collection / "acer-cpu2" / "resultados_knn.csv", index=False)
    with pytest.raises(merging.DuplicateRowsError):
        merging.merge(tmp_path / "plan", collection, tmp_path / "merged-again")
    assert plan  # o plano existe para a juncao


def test_runner_lane_uses_its_own_manifest_results_and_staging(tmp_path):
    _allocate(tmp_path, LANE_MACHINES)
    # O host desta maquina nao e o do plano: o Python falso fixa --host acer.
    fake_python = tmp_path / "python_acer"
    fake_python.write_text(f'#!/usr/bin/env bash\nexec {sys.executable} "$@" --host acer\n')
    fake_python.chmod(0o755)
    result = _bash(
        f"""
source run_storage_common.sh
GRID_PLAN="{tmp_path / 'plan'}"
GRID_LANE=acer-cpu2
load_grid_plan || exit 9
echo "manifest=${{CONFIGURATION_MANIFEST}}"
echo "results=${{RESULTS_DIRECTORY}}"
echo "staging=${{LOCAL_STAGING_DIRECTORY}}"
""",
        {"PYTHON_COMMAND": str(fake_python), "RESULTS_DIRECTORY": "", "LOCAL_STAGING_DIRECTORY": ""},
    )
    assert f"manifest={tmp_path / 'plan'}/lanes/acer-cpu2/manifest.tsv" in result.stdout, result.stderr
    assert "lanes/acer-cpu2/results" in result.stdout
    assert "lanes/acer-cpu2/staging" in result.stdout
    without_plan = _bash("source run_storage_common.sh; GRID_PLAN=; GRID_LANE=x; load_grid_plan; echo s=$?")
    assert "s=5" in without_plan.stdout and "requires GRID_PLAN" in without_plan.stderr


def test_results_on_a_network_mount_are_refused(tmp_path):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "findmnt").write_text("#!/usr/bin/env bash\necho cifs\n")
    (fake_bin / "findmnt").chmod(0o755)
    import os

    result = _bash(
        f"""
source run_storage_common.sh
RESULTS_DIRECTORY="{tmp_path}"
require_local_results_directory; echo status=$?
""",
        {"PATH": f"{fake_bin}:{os.environ['PATH']}"},
    )
    assert "status=1" in result.stdout
    assert "network mount (cifs)" in result.stderr


def test_backup_drops_the_line_being_written_and_publishes_atomically(tmp_path):
    lanes = tmp_path / "lanes"
    results = lanes / "dell-cpu1" / "results"
    results.mkdir(parents=True)
    (results / "resultados_xgboost.csv").write_text("a,b\n1,2\n3,")
    (results / "resultados_xgboost_history_strata.jsonl").write_text('{"x": 1}\n')
    backup = tmp_path / "backup"
    result = _bash(f'./backup_lane_results.sh "{backup}" "{lanes}"')
    assert result.returncode == 0, result.stderr
    copied = backup / "dell-cpu1"
    assert (copied / "resultados_xgboost.csv").read_text() == "a,b\n1,2\n"
    assert (copied / "resultados_xgboost_history_strata.jsonl").read_text() == '{"x": 1}\n'
    assert not list(copied.glob(".*.tmp.*"))
    # O arquivo do processo nao muda.
    assert (results / "resultados_xgboost.csv").read_text() == "a,b\n1,2\n3,"


def test_placement_ships_the_host_lanes_only_after_the_checksum(tmp_path):
    _allocate(tmp_path, LANE_MACHINES)
    result = _bash(
        f'./stage_machine_configurations.sh "{tmp_path / "plan"}" e470 local "{tmp_path / "remote"}"',
        {"PREPARED_DATA_DIRECTORY": str(tmp_path / "prepared")},
    )
    assert result.returncode == 0, result.stderr
    lanes = sorted(path.name for path in (tmp_path / "remote" / "grid_plan" / "lanes").iterdir())
    assert lanes == ["e470-cpu1", "e470-cpu2"]


def test_estimate_reports_each_lane_and_the_plan_makespan(tmp_path):
    import estimate_execution_time

    plan = _allocate(tmp_path, LANE_MACHINES)
    lines = estimate_execution_time.plan_makespan_lines(plan)
    longest = max(
        entry["estimated_seconds"]
        for machine in plan["machines"].values()
        for entry in machine["lanes"].values()
    )
    assert any(line.strip().startswith("acer-gpu1") for line in lines)
    assert lines[-1].startswith(
        f"Makespan do plano: {estimate_execution_time._format_duration(longest)}"
    )


def test_boosting_estimate_is_anchored_on_the_acer_measurement():
    import estimate_execution_time as estimate

    # A ancora reproduz a medicao na propria base medida...
    assert estimate.seconds_per_fold("xgboost", "ieee-cis-transacional") == pytest.approx(85.4)
    assert estimate.seconds_per_fold("catboost", "ieee-cis-transacional") == pytest.approx(1755.5)
    # ...e as medianas historicas so dao a proporcao entre bases.
    historical = estimate._historical_boosting_seconds
    for algorithm in ("xgboost", "catboost"):
        ratio = estimate.seconds_per_fold(algorithm, "home-credit-completa") / estimate.seconds_per_fold(
            algorithm, "ieee-cis-transacional"
        )
        expected = historical(algorithm, "home-credit-completa") / historical(algorithm, "ieee-cis-transacional")
        assert ratio == pytest.approx(expected)


def test_memory_floor_for_native_files_overrides_an_optimistic_calibration():
    machine = grid_plan.Machine("dell", THREE_MACHINES[2], is_reference=False)
    csv_bytes = 717_105_985
    for algorithm, family in (("xgboost", "boosting"), ("devnet", "anomaly"), ("knn", "knn")):
        pair = grid_plan.Pair(algorithm, "ieee-cis-transacional", ["m"], 1.0, {"m": csv_bytes})
        calibrated = 100 * 2**20 + 2.0 * csv_bytes
        expected = max(calibrated, grid_plan.MEASURED_MEMORY_PER_CSV_BYTE.get(family, 0.0) * csv_bytes)
        assert grid_plan.estimated_memory(machine, pair) == pytest.approx(expected)
    # ~8 GB para a ieee-cis-transacional: dois juntos passam de 80% de 16 GB.
    pair = grid_plan.Pair("catboost", "ieee-cis-transacional", ["m"], 1.0, {"m": csv_bytes})
    assert 2 * grid_plan.estimated_memory(machine, pair) > 0.8 * machine.memory_bytes


def test_measured_peak_replaces_the_floor_for_its_algorithm_dataset_and_protocol():
    machine = grid_plan.Machine("dell", THREE_MACHINES[2], is_reference=False)
    csv_bytes = 10 * 2**30  # cinco dobras da home-credit-completa: o piso daria ~112 GiB
    k_fold = grid_plan.Pair(
        "catboost", "home-credit-completa", ["m"], 1.0, {"m": csv_bytes}, "stratified-k-fold"
    )
    measured = grid_plan.MEASURED_PEAK_MEMORY_BYTES[("catboost", "home-credit-completa", "stratified-k-fold")]
    assert grid_plan.estimated_memory(machine, k_fold) == pytest.approx(
        measured * grid_plan.MEASURED_PEAK_MARGIN
    )
    # Sem medicao para o trio, continua valendo o piso.
    unmeasured = grid_plan.Pair(
        "devnet", "ieee-cis-transacional", ["m"], 1.0, {"m": csv_bytes}, "stratified-k-fold"
    )
    assert grid_plan.estimated_memory(machine, unmeasured) == pytest.approx(
        grid_plan.MEASURED_MEMORY_PER_CSV_BYTE["anomaly"] * csv_bytes
    )
