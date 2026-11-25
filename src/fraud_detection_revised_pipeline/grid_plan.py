"""Plano de divisao da grade entre as maquinas (item G18 da revisao).

A reexecucao roda no Acer Nitro ANV15-41 (com GPU e o HD externo), num Lenovo
ThinkPad E470 e num Dell Latitude 3490. O plano atribui cada unidade
(algoritmo, base, configuracao) a uma unica maquina. Todas as dobras, regimes
e bracos da configuracao ficam juntos, mas configuracoes da mesma base podem
ir para maquinas diferentes. O tempo total estimado (makespan) e minimizado
pela heuristica LPT: unidades em ordem decrescente de custo, cada uma
para a maquina que termina antes.

* O custo de uma unidade e dobras x bracos x ``seconds_per_fold`` (estimativas
  medidas no Acer, ``estimate_execution_time.py``), escalada pelo fator de
  velocidade da familia naquela maquina (``calibrate_machine.py``).
* Pares que exigem GPU (FT-Transformer e TabBERT) vao so para maquinas cuja
  calibracao registrou uma GPU.
* Um par nao vai para uma maquina em que a memoria estimada (fixa mais
  proporcional ao CSV da configuracao) passe de 80% da RAM, nem para uma
  maquina secundaria cujo disco nao comporte as configuracoes copiadas.

Processos (lanes): cada maquina roda varios processos ao mesmo tempo, como
declarado na calibracao (``cpu_lanes`` e ``gpu_lanes``). O LPT distribui os
pares entre os processos, e nao entre as maquinas, e a GPU do Acer e um recurso
separado da CPU dele. A memoria de um par novo soma-se ao maior par de cada
outro processo da mesma maquina. Uma calibracao sem processos declarados vale
por um processo de CPU (mais um de GPU, se houver GPU), como os planos antigos.

HDs: cada maquina le do diretorio registrado na calibracao. O Acer e o Lenovo
leem do HD local de cada um; o Dell le pela rede (calibracao com
``storage_is_network``) e paga, na estimativa, a leitura de cada configuracao a
``--network-bytes-per-second``. As configuracoes de uma maquina que le de outro
diretorio que nao o da preparacao sao copiadas para ele
(``stage_machine_configurations.sh``).

Saidas, no diretorio do plano: ``grid_plan.json``; por maquina,
``manifest_<host>.tsv`` (manifesto filtrado), ``assignments_<host>.tsv``
(algoritmo e configuration_md5 atribuidos) e ``files_<host>.txt`` (arquivos a
copiar, vazio quando a maquina le do diretorio da preparacao); e, por processo,
``lanes/<host>-<cpu|gpu><n>/`` com ``manifest.tsv`` e ``assignments.tsv``. O
plano e deterministico.

Uso:
    python3 grid_plan.py allocate --manifest <tsv> --prepared-data-directory <dir> \\
        --calibration-directory <dir> --reference-host <host> --plan-directory <dir>
    python3 grid_plan.py assignments --plan-directory <dir> --host <host> \
        [--lane <host>-cpu1] --output <tsv>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

import equivalent_configurations as equivalent
import execution_provenance
from estimate_execution_time import seconds_per_fold
from experiment_datasets import (
    BLOCK_ANOMALY,
    BLOCK_BOOSTING,
    BLOCK_CLASSIC,
    get_dataset_definition,
)
from experiment_grid import TABBERT_REGIMES
from prepared_dataset_storage import (
    CATEGORICAL_HANDLING_MODE_ENCODED,
    CATEGORICAL_HANDLING_MODE_NATIVE,
    CATEGORICAL_HANDLING_MODE_SEQUENCE,
    CATEGORICAL_HANDLING_MODE_TRANSFORMER,
    COMPRESSED_SPLIT_FILE_SUFFIX,
    STORAGE_METADATA_FILE_SUFFIX,
    normalize_manifest_dataset_identity,
)

PLAN_FILENAME = "grid_plan.json"
HOST_NOT_IN_PLAN_EXIT_CODE = 5
MEMORY_USE_LIMIT = 0.8
MIB = 1024 * 1024
LANES_DIRECTORY = "lanes"
# Leitura pela rede (Wi-Fi + SMB): ordem de grandeza medida entre as maquinas.
DEFAULT_NETWORK_BYTES_PER_SECOND = 10e6
DISK_USE_LIMIT = 0.9
NO_RESAMPLING_VALUES = frozenset({"", "none", "nan"})
CLASS_WEIGHT_ARMS = 3
DEVNET_ARMS = 3 * 5
GPU_FAMILY = "gpu"


@dataclass(frozen=True)
class AlgorithmSpec:
    """Onde um algoritmo roda e como seu custo e calculado."""

    handling_mode: str
    family: str
    block: str | None = None
    requires_no_resampling: bool = False


ALGORITHMS = {
    "logistic-regression": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_ENCODED, "linear", BLOCK_CLASSIC),
    "naive-bayes": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_ENCODED, "linear", BLOCK_CLASSIC),
    "knn": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_ENCODED, "knn", BLOCK_CLASSIC),
    "catboost": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_NATIVE, "boosting", BLOCK_BOOSTING),
    "xgboost": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_NATIVE, "boosting", BLOCK_BOOSTING),
    "isolation-forest": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_NATIVE, "anomaly", BLOCK_ANOMALY, True),
    "devnet": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_NATIVE, "anomaly", BLOCK_ANOMALY, True),
    "ft-transformer-mlp": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_TRANSFORMER, GPU_FAMILY),
    "ft-transformer-lstm": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_TRANSFORMER, GPU_FAMILY),
    "tabbert-mlp": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_SEQUENCE, GPU_FAMILY),
    "tabbert-lstm": AlgorithmSpec(CATEGORICAL_HANDLING_MODE_SEQUENCE, GPU_FAMILY),
}


class AllocationError(ValueError):
    """O plano nao pode ser construido com as maquinas calibradas."""


def _is_without_resampling(value) -> bool:
    return str(value if value is not None else "").strip().lower() in NO_RESAMPLING_VALUES


def arms_for(algorithm: str, without_resampling: bool) -> int:
    if algorithm in {"logistic-regression", "catboost", "xgboost"}:
        return CLASS_WEIGHT_ARMS if without_resampling else 1
    if algorithm == "devnet":
        return DEVNET_ARMS
    return 1


def configuration_seconds(algorithm: str, dataset_id: str, folds: int, without_resampling: bool) -> float:
    """Custo estimado de um algoritmo numa configuracao, na maquina de referencia."""
    if algorithm.startswith("tabbert"):
        return folds * sum(
            seconds_per_fold(f"{algorithm} [{regime}]", dataset_id) for regime in TABBERT_REGIMES
        )
    return folds * arms_for(algorithm, without_resampling) * seconds_per_fold(algorithm, dataset_id)


@dataclass
class Pair:
    algorithm: str
    dataset_id: str
    configurations: list[str]
    reference_seconds: float
    csv_bytes_by_configuration: dict[str, int]
    protocol: str = ""

    @property
    def family(self) -> str:
        return ALGORITHMS[self.algorithm].family


@dataclass
class Lane:
    """Um processo sequencial de uma maquina: um trabalhador do LPT."""

    name: str
    kind: str
    load_seconds: float = 0.0
    peak_memory: float = 0.0
    pairs: list = field(default_factory=list)


@dataclass
class Machine:
    host: str
    calibration: dict
    is_reference: bool
    configurations: set = field(default_factory=set)
    copied_bytes: int = 0
    pairs: list = field(default_factory=list)
    lanes: list = field(default_factory=list)

    def __post_init__(self):
        if not self.lanes:
            self.lanes = [
                Lane(f"{self.host}-cpu{index}", "cpu") for index in range(1, self.cpu_lanes + 1)
            ] + [
                Lane(f"{self.host}-gpu{index}", "gpu") for index in range(1, self.gpu_lanes + 1)
            ]

    @property
    def has_gpu(self) -> bool:
        return self.calibration.get("gpu_model", "none") not in ("", "none")

    @property
    def cpu_lanes(self) -> int:
        return max(1, int(self.calibration.get("cpu_lanes", 1)))

    @property
    def gpu_lanes(self) -> int:
        if not self.has_gpu:
            return 0
        return max(0, int(self.calibration.get("gpu_lanes", 1)))

    @property
    def memory_bytes(self) -> float:
        return float(self.calibration["memory_total_gb"]) * 2**30

    @property
    def read_directory(self) -> str:
        return str(self.calibration.get("prepared_data_directory", "")).rstrip("/")

    @property
    def reads_over_network(self) -> bool:
        return bool(self.calibration.get("storage_is_network", False))

    @property
    def load_seconds(self) -> float:
        """Makespan estimado da maquina: o processo que termina por ultimo."""
        return max((lane.load_seconds for lane in self.lanes), default=0.0)

    def copies_configurations(self, primary_directory: str) -> bool:
        """A maquina le de um diretorio que nao e o da preparacao: recebe copias."""
        return not self.is_reference and self.read_directory != primary_directory.rstrip("/")


def configuration_content_bytes(md5: str, paths: list[Path]) -> int:
    """Bytes de CSV da configuracao, base das estimativas de memoria.

    Soma os ``.csv`` (splits e laterais), como antes da compressao. Um split
    ``.csv.zst`` vale o tamanho descomprimido registrado no ``storage.json``.
    """
    splits = {}
    for path in paths:
        if path.name == f"{md5}{STORAGE_METADATA_FILE_SUFFIX}":
            splits = json.loads(path.read_text(encoding="utf-8")).get("splits", {})
            break
    total = 0
    for path in paths:
        if path.name.endswith(COMPRESSED_SPLIT_FILE_SUFFIX):
            entry = splits.get(path.name.removesuffix(COMPRESSED_SPLIT_FILE_SUFFIX))
            if entry is None:
                raise ValueError(f"{path.name} has no storage metadata; run compact_prepared_storage.py")
            total += int(entry["content_bytes"])
        elif path.suffix == ".csv":
            total += path.stat().st_size
    return total


def _configuration_files(prepared_directory: Path, manifest: pd.DataFrame) -> dict[str, list[Path]]:
    """Arquivos de cada configuracao: CSVs, registros de colunas, unidades,
    estratos de historico, relatorios, tempos de preparacao, vocabularios e
    janelas."""
    files: dict[str, list[Path]] = defaultdict(list)
    for md5, rows in manifest.groupby("configuration_md5"):
        names = set()
        for dataset_md5 in rows["dataset_md5"].dropna():
            names.add(f"{dataset_md5}.csv")
            names.add(f"{dataset_md5}.csv.zst")
            names.add(f"{dataset_md5}.columns.json")
            names.add(f"{dataset_md5}.groups.csv")
            names.add(f"{dataset_md5}.history.csv")
        for column in ("vocabulary_path", "window_index_path"):
            if column in rows:
                names.update(Path(value).name for value in rows[column].dropna() if value)
        for path in prepared_directory.glob(f"{md5}.*"):
            names.add(path.name)
        files[md5] = sorted(prepared_directory / name for name in names if (prepared_directory / name).exists())
    return files


def build_pairs(manifest: pd.DataFrame, files: dict[str, list[Path]]) -> list[Pair]:
    """Cria uma unidade indivisivel por algoritmo, base e configuracao."""
    configurations = manifest.drop_duplicates("configuration_md5")
    folds = manifest.groupby("configuration_md5")["fold_index"].nunique()
    pairs: list[Pair] = []
    for _, row in configurations.sort_values("configuration_md5").iterrows():
        dataset_id = row["dataset_id"]
        definition = get_dataset_definition(dataset_id)
        without_resampling = _is_without_resampling(row.get("undersampling_technique"))
        md5 = row["configuration_md5"]
        csv_bytes = configuration_content_bytes(md5, files.get(md5, []))
        for algorithm, spec in ALGORITHMS.items():
            if row["categorical_handling_mode"] != spec.handling_mode:
                continue
            if spec.block is not None and spec.block not in definition.blocks:
                continue
            if spec.requires_no_resampling and not without_resampling:
                continue
            pairs.append(
                Pair(
                    algorithm,
                    dataset_id,
                    [md5],
                    configuration_seconds(
                        algorithm, dataset_id, int(folds[md5]), without_resampling
                    ),
                    {md5: csv_bytes},
                    str(row.get("test_strategy", "")),
                )
            )
    return sorted(
        pairs,
        key=lambda pair: (pair.algorithm, pair.dataset_id, pair.configurations[0]),
    )


def speed_factor(machine: Machine, reference: Machine, family: str) -> float:
    if family == GPU_FAMILY:
        return 1.0
    return machine.calibration["families"][family]["seconds"] / reference.calibration["families"][family]["seconds"]


# Piso medido de memoria para os blocos que leem os arquivos nativos (com
# categoricas como texto): no Acer, em 31/10/2026, a ieee-cis-transacional
# (717 MB de CSV na configuracao) teve pico de 7,8 GB no XGBoost e 7,0 GB no
# CatBoost, ~11 vezes o CSV. A calibracao usa dados sinteticos so numericos e
# subestima esse pico; sem o piso, dois desses pares caberiam "no papel" numa
# maquina de 16 GB e estourariam a memoria juntos.
MEASURED_MEMORY_PER_CSV_BYTE = {"boosting": 11.2, "anomaly": 11.2}

# Picos medidos (RSS amostrado) por algoritmo, base e protocolo. Quando existe,
# substituem o piso por byte de CSV, com margem. Na Home Credit nativa o piso
# estima 106 GiB para o k-fold, que mede ~15 GiB: o piso veio das muitas
# categoricas em texto da IEEE-CIS e o k-fold soma o CSV das cinco dobras, mas
# o executor so tem uma dobra em memoria por vez. Medidos no Acer em 08/11/2026,
# na configuracao none/mean sem ponderacao de classe (a de maior CSV do protocolo).
# Anomalia (13/11/2026): mesmas configuracoes nativas, DevNet num braco so
# (fracao 0,1, sorteio 0). Classicos (13/11/2026): a codificada de maior CSV
# (039cdf3a, smoothed-mean:permutation-prefix/mode, 5,8 GiB nas cinco dobras),
# que a calibracao estimava em 21,4 GiB (LR/NB) e 16,9 GiB (KNN).
MEASURED_PEAK_MEMORY_BYTES = {
    ("xgboost", "home-credit-completa", "stratified-holdout"): 5_738 * MIB,
    ("xgboost", "home-credit-completa", "stratified-k-fold"): 13_801 * MIB,
    ("catboost", "home-credit-completa", "stratified-holdout"): 8_277 * MIB,
    ("catboost", "home-credit-completa", "stratified-k-fold"): 15_297 * MIB,
    ("isolation-forest", "home-credit-completa", "stratified-holdout"): 9_525 * MIB,
    ("isolation-forest", "home-credit-completa", "stratified-k-fold"): 17_476 * MIB,
    ("devnet", "home-credit-completa", "stratified-holdout"): 8_683 * MIB,
    ("devnet", "home-credit-completa", "stratified-k-fold"): 18_976 * MIB,
    ("logistic-regression", "home-credit-completa", "stratified-k-fold"): 10_389 * MIB,
    ("naive-bayes", "home-credit-completa", "stratified-k-fold"): 9_549 * MIB,
    ("knn", "home-credit-completa", "stratified-k-fold"): 14_495 * MIB,
}
MEASURED_PEAK_MARGIN = 1.2


def estimated_memory(machine: Machine, pair: Pair) -> float:
    if pair.family == GPU_FAMILY:
        return 0.0
    measured = MEASURED_PEAK_MEMORY_BYTES.get((pair.algorithm, pair.dataset_id, pair.protocol))
    if measured is not None:
        return measured * MEASURED_PEAK_MARGIN
    family = machine.calibration["families"][pair.family]
    largest = max(pair.csv_bytes_by_configuration.values(), default=0)
    calibrated = family["memory_fixed_bytes"] + family["memory_per_csv_byte"] * largest
    return max(calibrated, MEASURED_MEMORY_PER_CSV_BYTE.get(pair.family, 0.0) * largest)


def _concurrent_memory(machine: Machine, lane: Lane, pair_memory: float) -> float:
    """Memoria do par somada ao maior par de cada outro processo da maquina."""
    others = sum(other.peak_memory for other in machine.lanes if other is not lane)
    return pair_memory + others


def allocate(
    pairs: list[Pair],
    machines: list[Machine],
    files: dict[str, list[Path]],
    primary_directory: str = "",
    network_bytes_per_second: float = DEFAULT_NETWORK_BYTES_PER_SECOND,
) -> None:
    """LPT: pares em ordem decrescente de custo, cada um para o processo viavel
    que termina antes. Empates resolvidos pelo host e pelo nome do processo."""
    reference = next(machine for machine in machines if machine.is_reference)
    file_bytes = {md5: sum(path.stat().st_size for path in paths) for md5, paths in files.items()}
    for pair in sorted(
        pairs,
        key=lambda item: (
            -item.reference_seconds,
            item.algorithm,
            item.dataset_id,
            item.configurations[0],
        ),
    ):
        lane_kind = "gpu" if pair.family == GPU_FAMILY else "cpu"
        pair_bytes = sum(file_bytes.get(md5, 0) for md5 in pair.configurations)
        candidates = []
        for machine in sorted(machines, key=lambda item: item.host):
            if pair.family == GPU_FAMILY and not machine.has_gpu:
                continue
            pair_memory = estimated_memory(machine, pair)
            copies = machine.copies_configurations(primary_directory)
            new_bytes = sum(
                file_bytes.get(md5, 0) for md5 in set(pair.configurations) - machine.configurations
            )
            if copies and (
                machine.copied_bytes + new_bytes > DISK_USE_LIMIT * machine.calibration["disk_free_bytes"]
            ):
                continue
            seconds = pair.reference_seconds * speed_factor(machine, reference, pair.family)
            if machine.reads_over_network:
                seconds += pair_bytes / network_bytes_per_second
            for lane in machine.lanes:
                if lane.kind != lane_kind:
                    continue
                if _concurrent_memory(machine, lane, pair_memory) > MEMORY_USE_LIMIT * machine.memory_bytes:
                    continue
                candidates.append(
                    (lane.load_seconds + seconds, machine.host, lane.name, machine, lane,
                     seconds, new_bytes, pair_memory, copies)
                )
        if not candidates:
            reason = "no calibrated machine has a GPU" if pair.family == GPU_FAMILY else "no machine has enough memory or disk"
            raise AllocationError(f"Cannot place {pair.algorithm} on {pair.dataset_id}: {reason}")
        (_finish, _host, _lane_name, machine, lane, seconds, new_bytes, pair_memory, copies) = min(
            candidates, key=lambda item: item[:3]
        )
        lane.load_seconds += seconds
        lane.peak_memory = max(lane.peak_memory, pair_memory)
        lane.pairs.append((pair, seconds))
        machine.copied_bytes += new_bytes if copies else 0
        machine.configurations.update(pair.configurations)
        machine.pairs.append((pair, seconds))


def _md5_of_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _pair_entries(pairs: list) -> list[dict]:
    return [
        {
            "algorithm": pair.algorithm,
            "dataset_id": pair.dataset_id,
            "estimated_seconds": round(seconds, 1),
            "configurations": sorted(pair.configurations),
        }
        for pair, seconds in sorted(
            pairs,
            key=lambda item: (
                item[0].algorithm,
                item[0].dataset_id,
                item[0].configurations[0],
            ),
        )
    ]


def _assignment_lines(pairs: list) -> str:
    return "".join(
        f"{pair.algorithm}\t{md5}\n"
        for pair, _seconds in sorted(
            pairs,
            key=lambda item: (
                item[0].algorithm,
                item[0].dataset_id,
                item[0].configurations[0],
            ),
        )
        for md5 in sorted(pair.configurations)
    )


def _manifest_rows(raw_manifest: pd.DataFrame, pairs: list) -> pd.DataFrame:
    configurations = {md5 for pair, _seconds in pairs for md5 in pair.configurations}
    return raw_manifest[raw_manifest["configuration_md5"].isin(sorted(configurations))]


def write_plan(
    plan_directory: Path,
    raw_manifest: pd.DataFrame,
    machines: list[Machine],
    files: dict[str, list[Path]],
    calibration_paths: list[Path],
    primary_directory: str = "",
    equivalent_configurations: dict[str, str] | None = None,
) -> Path:
    plan_directory.mkdir(parents=True, exist_ok=True)
    commit, dirty = execution_provenance.code_version()
    plan = {
        "code_commit": commit,
        "code_tree_dirty": dirty,
        "reference_host": next(machine.host for machine in machines if machine.is_reference),
        "prepared_data_directory": primary_directory,
        "calibrations": {path.name: _md5_of_file(path) for path in sorted(calibration_paths)},
        "equivalent_configurations": dict(sorted((equivalent_configurations or {}).items())),
        "machines": {},
    }
    for machine in sorted(machines, key=lambda item: item.host):
        copies = machine.copies_configurations(primary_directory)
        plan["machines"][machine.host] = {
            "is_reference": machine.is_reference,
            "estimated_seconds": round(machine.load_seconds, 1),
            "copied_bytes": machine.copied_bytes,
            "read_directory": machine.read_directory if copies else primary_directory,
            "copies_configurations": copies,
            "reads_over_network": machine.reads_over_network,
            "pairs": _pair_entries(machine.pairs),
            "lanes": {
                lane.name: {
                    "kind": lane.kind,
                    "estimated_seconds": round(lane.load_seconds, 1),
                    "estimated_peak_memory_bytes": round(lane.peak_memory),
                    "pairs": _pair_entries(lane.pairs),
                }
                for lane in machine.lanes
            },
        }
        configurations = sorted(machine.configurations)
        _manifest_rows(raw_manifest, machine.pairs).to_csv(
            plan_directory / f"manifest_{machine.host}.tsv", sep="\t", index=False
        )
        (plan_directory / f"assignments_{machine.host}.tsv").write_text(
            _assignment_lines(machine.pairs), encoding="utf-8"
        )
        (plan_directory / f"files_{machine.host}.txt").write_text(
            "".join(
                f"{path.name}\n"
                for md5 in configurations
                for path in files.get(md5, [])
            )
            if copies
            else "",
            encoding="utf-8",
        )
        for lane in machine.lanes:
            lane_directory = plan_directory / LANES_DIRECTORY / lane.name
            lane_directory.mkdir(parents=True, exist_ok=True)
            _manifest_rows(raw_manifest, lane.pairs).to_csv(
                lane_directory / "manifest.tsv", sep="\t", index=False
            )
            (lane_directory / "assignments.tsv").write_text(
                _assignment_lines(lane.pairs), encoding="utf-8"
            )
    path = plan_directory / PLAN_FILENAME
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_machines(calibration_directory: Path, reference_host: str) -> tuple[list[Machine], list[Path]]:
    paths = sorted(calibration_directory.glob("calibration_*.json"))
    machines = [
        Machine(
            host=(calibration := json.loads(path.read_text(encoding="utf-8")))["execution_host"],
            calibration=calibration,
            is_reference=False,
        )
        for path in paths
    ]
    hosts = [machine.host for machine in machines]
    if reference_host not in hosts:
        raise AllocationError(f"Reference host {reference_host} has no calibration in {calibration_directory}")
    for machine in machines:
        machine.is_reference = machine.host == reference_host
    return machines, paths


def run_allocate(args: argparse.Namespace) -> int:
    raw_manifest = pd.read_csv(args.manifest, sep="\t", dtype=str)
    manifest = normalize_manifest_dataset_identity(raw_manifest.copy())
    prepared_directory = Path(args.prepared_data_directory)
    machines, calibration_paths = load_machines(Path(args.calibration_directory), args.reference_host)
    # Gemeas por eixo inerte tem as mesmas entradas do representante: so ele
    # entra no plano (equivalent_configurations).
    twins = equivalent.twin_map(equivalent.inert_high_cardinality_groups(manifest))
    manifest = manifest[~manifest["configuration_md5"].isin(twins)]
    files = _configuration_files(prepared_directory, manifest)
    pairs = build_pairs(manifest, files)
    primary_directory = str(prepared_directory).rstrip("/")
    allocate(pairs, machines, files, primary_directory, args.network_bytes_per_second)
    path = write_plan(
        Path(args.plan_directory), raw_manifest, machines, files, calibration_paths, primary_directory,
        twins,
    )
    if twins:
        print(f"{len(twins)} configuracoes gemeas por eixo inerte executadas pelo representante")
    for machine in sorted(machines, key=lambda item: item.host):
        print(
            f"{machine.host}: {len(machine.pairs)} pairs, "
            f"{machine.load_seconds / 3600:.1f} h estimated, {machine.copied_bytes / 2**30:.1f} GiB to copy"
        )
        for lane in machine.lanes:
            print(f"  {lane.name}: {len(lane.pairs)} pairs, {lane.load_seconds / 3600:.1f} h")
    print(path)
    return 0


def read_plan(plan_directory: Path) -> dict:
    return json.loads((plan_directory / PLAN_FILENAME).read_text(encoding="utf-8"))


def run_assignments(args: argparse.Namespace) -> int:
    plan_directory = Path(args.plan_directory)
    plan = read_plan(plan_directory)
    host = args.host or execution_provenance.collect_provenance()["execution_host"]
    if host not in plan["machines"]:
        print(f"Host {host} is not in the grid plan {plan_directory}.", file=sys.stderr)
        return HOST_NOT_IN_PLAN_EXIT_CODE
    source = plan_directory / f"assignments_{host}.tsv"
    if args.lane:
        if args.lane not in plan["machines"][host].get("lanes", {}):
            print(f"Lane {args.lane} is not a lane of {host} in {plan_directory}.", file=sys.stderr)
            return HOST_NOT_IN_PLAN_EXIT_CODE
        source = plan_directory / LANES_DIRECTORY / args.lane / "assignments.tsv"
    Path(args.output).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return 0


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    allocate_parser = commands.add_parser("allocate")
    allocate_parser.add_argument("--manifest", required=True)
    allocate_parser.add_argument("--prepared-data-directory", required=True)
    allocate_parser.add_argument("--calibration-directory", required=True)
    allocate_parser.add_argument("--reference-host", required=True)
    allocate_parser.add_argument("--plan-directory", required=True)
    allocate_parser.add_argument(
        "--network-bytes-per-second",
        type=float,
        default=DEFAULT_NETWORK_BYTES_PER_SECOND,
        help="Leitura estimada pela rede, para maquinas que leem os dados por CIFS.",
    )
    assignments_parser = commands.add_parser("assignments")
    assignments_parser.add_argument("--plan-directory", required=True)
    assignments_parser.add_argument("--host", default=None)
    assignments_parser.add_argument("--lane", default=None)
    assignments_parser.add_argument("--output", required=True)
    return parser


def main() -> int:
    args = build_argument_parser().parse_args()
    try:
        return run_allocate(args) if args.command == "allocate" else run_assignments(args)
    except AllocationError as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
