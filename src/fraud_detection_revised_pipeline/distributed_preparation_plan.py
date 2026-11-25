"""Planeja e verifica a preparacao distribuida entre Acer, Dell e Lenovo."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import execution_provenance
import experiment_grid
import verify_datasets
from prepared_dataset_storage import (
    DEFAULT_NUMERIC_BINS,
    build_prepared_dataset_md5,
    read_storage_metadata,
    transformer_identity_parameters,
)

PLAN_FILENAME = "preparation_plan.json"
BLOCKS = ("classic", "boosting", "ft-transformer", "tabbert")
DEFAULT_RESERVE_BYTES = 30 * 2**30
DEFAULT_MICRO_SHARD_BYTES = 8 * 2**30
# Igual a publish_prepared_configuration.NO_SPACE_EXIT_CODE: o worker para a fila.
NO_SPACE_EXIT_CODE = 10
GRID_ENVIRONMENT_VARIABLES = (
    "SEED_LIST",
    "DATASET_FILTER",
    "PROTOCOL_FILTER",
    "UNDERSAMPLING_FILTER",
    "FILL_NA_FILTER",
)


def cohort(dataset_id: str) -> str:
    if dataset_id.startswith("home-credit-"):
        return "home-credit"
    if dataset_id.startswith("ieee-cis-"):
        return "ieee-cis"
    if dataset_id.startswith("sparkov"):
        return "sparkov"
    return dataset_id


def configuration_md5(
    entry: experiment_grid.GridEntry, include_transformer_schema_revision: bool = True
) -> str:
    """Identidade da entrada, pela mesma regra de ``prepare_lr_nb_knn_datasets``.

    Os argumentos espelham os que ``prepare_grid_common.sh`` passa ao
    preparador: o FT-Transformer nao recebe --numeric-bins e herda o padrao.
    """
    sequence = entry.categorical_handling_mode == "sequence-tokenized"
    return build_prepared_dataset_md5(
        entry.dataset_id,
        entry.categorical_encoding_strategy,
        entry.fill_na_strategy,
        entry.test_strategy,
        entry.prediction_column,
        entry.undersampling_technique,
        entry.undersampling_parameter_values,
        entry.auxiliary_data_is_test_data == "true",
        entry.categorical_handling_mode,
        entry.random_seed,
        *transformer_identity_parameters(
            entry.categorical_handling_mode,
            entry.numeric_normalization,
            entry.high_cardinality_strategy,
            entry.high_cardinality_parameter,
            experiment_grid.TABBERT_NUMERIC_BINS if sequence else DEFAULT_NUMERIC_BINS,
            experiment_grid.TABBERT_WINDOW_LENGTH,
            experiment_grid.TABBERT_HISTORY_LENGTH,
            experiment_grid.TABBERT_WINDOW_STRIDE,
        ),
        include_transformer_schema_revision=include_transformer_schema_revision,
    )


def _legacy_configuration_md5(entry: experiment_grid.GridEntry) -> str:
    """Regra anterior a fix-ft-transformer-plan-md5, usada so por ``amend``.

    Omitia ``numeric_bins`` no FT-Transformer; os planos criados com ela tem
    md5s de FT que o preparador nunca produz.
    """
    sequence = entry.categorical_handling_mode == "sequence-tokenized"
    transformer = entry.categorical_handling_mode in {
        "transformer-indexed",
        "sequence-tokenized",
    }
    return build_prepared_dataset_md5(
        entry.dataset_id,
        entry.categorical_encoding_strategy,
        entry.fill_na_strategy,
        entry.test_strategy,
        entry.prediction_column,
        entry.undersampling_technique,
        entry.undersampling_parameter_values,
        entry.auxiliary_data_is_test_data == "true",
        entry.categorical_handling_mode,
        entry.random_seed,
        entry.numeric_normalization if transformer else None,
        entry.high_cardinality_strategy if transformer else None,
        entry.high_cardinality_parameter if transformer else None,
        experiment_grid.TABBERT_NUMERIC_BINS if sequence else None,
        experiment_grid.TABBERT_WINDOW_LENGTH if sequence else None,
        experiment_grid.TABBERT_HISTORY_LENGTH if sequence else None,
        experiment_grid.TABBERT_WINDOW_STRIDE if sequence else None,
    )


def grid_assignments(limit: int = 0) -> list[dict]:
    assignments = []
    for block in BLOCKS:
        for entry in experiment_grid.GRID_BUILDERS[block]():
            assignments.append(
                {
                    "block": block,
                    "dataset_id": entry.dataset_id,
                    "cohort": cohort(entry.dataset_id),
                    "configuration_md5": configuration_md5(entry),
                }
            )
    assignments.sort(
        key=lambda item: (item["cohort"], item["block"], item["dataset_id"], item["configuration_md5"])
    )
    return assignments[:limit] if limit else assignments


def source_checksums() -> dict[str, str]:
    path = verify_datasets.DEFAULT_DATASETS_DIRECTORY / verify_datasets.CHECKSUM_FILENAME
    return {relative: md5 for md5, relative in verify_datasets.read_checksums(path)}


def create_plan(args: argparse.Namespace) -> Path:
    dataset_problems = verify_datasets.verify_datasets()
    if dataset_problems:
        raise ValueError("raw dataset verification failed: " + "; ".join(dataset_problems))
    hosts = {
        "acer": args.acer_host,
        "dell": args.dell_host,
        "lenovo": args.lenovo_host,
    }
    roots = {
        args.acer_host: str(Path(args.black_root) / "preparation-runs" / args.run_id / "acer"),
        args.dell_host: str(Path(args.black_root) / "preparation-runs" / args.run_id / "dell"),
        args.lenovo_host: str(Path(args.transparent_root) / "preparation-runs" / args.run_id / "lenovo"),
    }
    cohort_role = {
        "home-credit": "acer",
        "ieee-cis": "dell",
        "sparkov": "lenovo",
        "ulb": "lenovo",
    }
    assignments = grid_assignments(args.limit)
    for item in assignments:
        item["host"] = hosts[cohort_role[item["cohort"]]]
    provenance = execution_provenance.collect_provenance()
    plan = {
        "run_id": args.run_id,
        "mode": args.mode,
        "code_commit": provenance["code_commit"],
        "code_tree_dirty": provenance["code_tree_dirty"],
        "environment_md5": provenance["environment_md5"],
        "environment_locked": provenance["environment_locked"],
        "source_checksums": source_checksums(),
        "grid_environment": {
            name: os.environ.get(name, "") for name in GRID_ENVIRONMENT_VARIABLES
        },
        "reserve_bytes": args.reserve_bytes,
        "micro_shard_bytes": args.micro_shard_bytes,
        "hosts": {
            host: {"role": role, "shard_root": roots[host]}
            for role, host in hosts.items()
        },
        "assignments": assignments,
    }
    if args.mode == "force":
        occupied = [
            root
            for root in roots.values()
            if (Path(root) / "manifests").exists()
        ]
        if occupied:
            raise ValueError(
                "force requires a new run-id; published manifests exist at "
                + ", ".join(occupied)
            )
    output = Path(args.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / PLAN_FILENAME
    if path.exists():
        raise ValueError(f"Plan already exists: {path}")
    path.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    for host in sorted(roots):
        with (output / f"preparation_assignments_{host}.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("block", "dataset_id", "cohort", "configuration_md5"),
                delimiter="\t",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(
                {key: item[key] for key in writer.fieldnames}
                for item in assignments
                if item["host"] == host
            )
    return path


def verify_worker(plan: dict, host: str, reserve_bytes: int | None = None) -> list[str]:
    problems = []
    if host not in plan["hosts"]:
        return [f"host {host} is not in the preparation plan"]
    if source_checksums() != plan["source_checksums"]:
        problems.append("source checksum registry differs from the preparation plan")
    current_grid_environment = {
        name: os.environ.get(name, "") for name in GRID_ENVIRONMENT_VARIABLES
    }
    if current_grid_environment != plan.get("grid_environment", {}):
        problems.append("grid environment differs from the preparation plan")
    problems.extend(verify_datasets.verify_datasets())
    root = Path(plan["hosts"][host]["shard_root"])
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    required = (reserve_bytes if reserve_bytes is not None else plan["reserve_bytes"]) + plan[
        "micro_shard_bytes"
    ]
    if free < required:
        problems.append(f"space: {free} bytes free, {required} required at {root}")
    if shutil.which("zstd") is None:
        problems.append("zstd is not installed; prepared splits are published as .csv.zst")
    return problems


def configuration_space_requirement(plan: dict, host: str, block: str, dataset_id: str) -> int:
    """Reserva do plano mais a maior configuracao ja publicada do mesmo bloco e base.

    Sem historico (ou sem ``storage.json``), vale o micro shard do plano.
    """
    root = Path(plan["hosts"][host]["shard_root"])
    largest = 0
    for item in plan["assignments"]:
        if item["host"] != host or item["block"] != block or item["dataset_id"] != dataset_id:
            continue
        metadata = read_storage_metadata(root / "files", item["configuration_md5"].lower())
        if metadata:
            largest = max(
                largest, sum(int(split["stored_bytes"]) for split in metadata["splits"].values())
            )
    return int(plan["reserve_bytes"]) + (largest or int(plan["micro_shard_bytes"]))


def check_configuration_space(plan: dict, host: str, block: str, dataset_id: str) -> str | None:
    """Mensagem de falta de espaco no shard do host, ou None se ha espaco."""
    root = Path(plan["hosts"][host]["shard_root"])
    root.mkdir(parents=True, exist_ok=True)
    required = configuration_space_requirement(plan, host, block, dataset_id)
    free = shutil.disk_usage(root).free
    if free < required:
        return (
            f"No space for the next configuration ({block}, {dataset_id}): "
            f"{free} bytes free, {required} required at {root}"
        )
    return None


def load_plan(directory: str | Path) -> dict:
    return json.loads((Path(directory) / PLAN_FILENAME).read_text(encoding="utf-8"))


def _identity_map(grid_environment: dict) -> tuple[set[str], dict[str, str]]:
    """md5s atuais da grade e o mapa md5 legado -> atual das entradas que mudaram."""
    saved = {name: os.environ.get(name) for name in GRID_ENVIRONMENT_VARIABLES}
    try:
        for name in GRID_ENVIRONMENT_VARIABLES:
            os.environ[name] = grid_environment.get(name, "")
        current: set[str] = set()
        legacy: dict[str, str] = {}
        for block in BLOCKS:
            for entry in experiment_grid.GRID_BUILDERS[block]():
                new = configuration_md5(entry)
                old = _legacy_configuration_md5(entry)
                current.add(new)
                if old != new:
                    legacy[old] = new
        return current, legacy
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _replace_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def amend_plan(directory: str | Path, timestamp: str | None = None) -> list[dict]:
    """Corrige no mesmo run-id os md5s que divergem do preparador.

    Recusa md5 desconhecido (a grade mudou) e md5 antigo ja publicado em algum
    shard, sem alterar nada. Guarda os arquivos anteriores com sufixo
    ``.before-amend-<data-hora>`` e registra a correcao no plano.
    """
    directory = Path(directory)
    plan = load_plan(directory)
    current, legacy = _identity_map(plan.get("grid_environment", {}))
    changes = []
    for item in plan["assignments"]:
        md5 = item["configuration_md5"].lower()
        if md5 in current:
            continue
        if md5 not in legacy:
            raise ValueError(
                f"assignment {md5} ({item['block']}, {item['dataset_id']}) is not in the "
                "current grid; the grid changed since the plan was created"
            )
        changes.append(
            {
                "old": md5,
                "new": legacy[md5],
                "block": item["block"],
                "dataset_id": item["dataset_id"],
                "host": item["host"],
            }
        )
    if not changes:
        return []
    published = [
        str(Path(host["shard_root"]) / "manifests" / f"{change['old']}.tsv")
        for change in changes
        for host in plan["hosts"].values()
        if (Path(host["shard_root"]) / "manifests" / f"{change['old']}.tsv").exists()
    ]
    if published:
        raise ValueError(
            "amend refused: configurations to be renamed are already published: "
            + ", ".join(published)
        )

    renamed = {change["old"]: change["new"] for change in changes}
    suffix = f".before-amend-{timestamp or datetime.now().strftime('%Y%m%d-%H%M%S')}"
    plan_path = directory / PLAN_FILENAME
    assignment_paths = sorted(directory.glob("preparation_assignments_*.tsv"))
    for path in (plan_path, *assignment_paths):
        shutil.copy2(path, path.with_name(path.name + suffix))

    for path in assignment_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = [name.strip() for name in reader.fieldnames or ()]
            rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
        for row in rows:
            row["configuration_md5"] = renamed.get(row["configuration_md5"].lower(), row["configuration_md5"])
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _replace_text(path, buffer.getvalue())

    for item in plan["assignments"]:
        item["configuration_md5"] = renamed.get(item["configuration_md5"].lower(), item["configuration_md5"])
    plan.setdefault("amendments", []).append(
        {
            "date": datetime.now().isoformat(timespec="seconds"),
            "code_commit": execution_provenance.collect_provenance()["code_commit"],
            "reason": "fix-ft-transformer-plan-md5: numeric_bins in FT-Transformer identity",
            "backup_suffix": suffix,
            "changes": [
                {key: change[key] for key in ("old", "new", "block", "dataset_id")}
                for change in changes
            ],
        }
    )
    _replace_text(plan_path, json.dumps(plan, indent=2, sort_keys=True))
    return changes


SUPERSEDE_REASON = (
    "hc-numeric-rates: APARTMENTS_MODE, BASEMENTAREA_MODE, RATE_DOWN_PAYMENT e "
    "RATE_INTEREST_* numericas nos Transformers da home-credit-completa"
)


def supersede_plan(directory: str | Path, timestamp: str | None = None) -> list[dict]:
    """Troca no mesmo run-id as configuracoes cuja identidade mudou por uma
    revisao de esquema (``TRANSFORMER_SCHEMA_REVISIONS``).

    Diferente de ``amend``, aceita configuracoes antigas ja publicadas: os
    dados delas foram preparados com o esquema anterior e precisam ser
    refeitos. O assignment passa a apontar para o md5 novo, no mesmo host e
    bloco; o merge ignora o manifesto antigo, que nao e mais atribuido.
    Recusa md5 que nao e da grade atual nem de uma revisao, sem alterar nada.
    Guarda os arquivos anteriores com sufixo ``.before-supersede-<data-hora>``
    e registra a troca em ``supersessions``.
    """
    directory = Path(directory)
    plan = load_plan(directory)
    current, legacy = _identity_map(plan.get("grid_environment", {}))
    previous = _pre_revision_map(plan.get("grid_environment", {}))
    assigned = {item["configuration_md5"].lower() for item in plan["assignments"]}
    changes = []
    for item in plan["assignments"]:
        md5 = item["configuration_md5"].lower()
        if md5 in current:
            continue
        if md5 not in previous:
            hint = "; run amend first" if md5 in legacy else ""
            raise ValueError(
                f"assignment {md5} ({item['block']}, {item['dataset_id']}) is not superseded "
                f"by a schema revision{hint}"
            )
        if previous[md5] in assigned:
            raise ValueError(f"superseding md5 {previous[md5]} is already assigned")
        changes.append(
            {
                "old": md5,
                "new": previous[md5],
                "block": item["block"],
                "dataset_id": item["dataset_id"],
                "host": item["host"],
                "old_published": any(
                    (Path(host["shard_root"]) / "manifests" / f"{md5}.tsv").exists()
                    for host in plan["hosts"].values()
                ),
            }
        )
    if not changes:
        return []

    renamed = {change["old"]: change["new"] for change in changes}
    suffix = f".before-supersede-{timestamp or datetime.now().strftime('%Y%m%d-%H%M%S')}"
    plan_path = directory / PLAN_FILENAME
    assignment_paths = sorted(directory.glob("preparation_assignments_*.tsv"))
    for path in (plan_path, *assignment_paths):
        shutil.copy2(path, path.with_name(path.name + suffix))
    for path in assignment_paths:
        with path.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fieldnames = [name.strip() for name in reader.fieldnames or ()]
            rows = [{key.strip(): value.strip() for key, value in row.items()} for row in reader]
        for row in rows:
            row["configuration_md5"] = renamed.get(row["configuration_md5"].lower(), row["configuration_md5"])
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        _replace_text(path, buffer.getvalue())
    for item in plan["assignments"]:
        item["configuration_md5"] = renamed.get(item["configuration_md5"].lower(), item["configuration_md5"])
    plan.setdefault("supersessions", []).append(
        {
            "date": datetime.now().isoformat(timespec="seconds"),
            "code_commit": execution_provenance.collect_provenance()["code_commit"],
            "reason": SUPERSEDE_REASON,
            "backup_suffix": suffix,
            "changes": changes,
        }
    )
    _replace_text(plan_path, json.dumps(plan, indent=2, sort_keys=True))
    return changes


def _pre_revision_map(grid_environment: dict) -> dict[str, str]:
    """md5 anterior a uma revisao de esquema -> md5 atual, das entradas afetadas."""
    saved = {name: os.environ.get(name) for name in GRID_ENVIRONMENT_VARIABLES}
    try:
        for name in GRID_ENVIRONMENT_VARIABLES:
            os.environ[name] = grid_environment.get(name, "")
        mapping: dict[str, str] = {}
        for block in BLOCKS:
            for entry in experiment_grid.GRID_BUILDERS[block]():
                new = configuration_md5(entry)
                old = configuration_md5(entry, include_transformer_schema_revision=False)
                if old != new:
                    mapping[old] = new
        return mapping
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def reassign_plan(
    directory: str | Path,
    source_host: str,
    target_host: str,
    blocks: list[str],
    datasets: list[str] | None = None,
    timestamp: str | None = None,
) -> list[dict]:
    """Move no mesmo run-id os assignments de blocos inteiros entre dois hosts.

    A unidade e o par (bloco, dataset): todas as configuracoes de cada par
    selecionado mudam juntas. Recusa par com alguma configuracao ja publicada
    em qualquer shard, sem alterar nada. Guarda os arquivos anteriores com
    sufixo ``.before-reassign-<data-hora>`` e registra a mudanca no plano.
    O worker do host de origem deve estar parado e sem scratch dessas
    configuracoes.
    """
    directory = Path(directory)
    plan = load_plan(directory)
    for host in (source_host, target_host):
        if host not in plan["hosts"]:
            raise ValueError(f"host {host} is not in the preparation plan")
    if source_host == target_host:
        raise ValueError("source and target hosts are the same")
    selected = [
        item
        for item in plan["assignments"]
        if item["host"] == source_host
        and item["block"] in blocks
        and (not datasets or item["dataset_id"] in datasets)
    ]
    if not selected:
        raise ValueError(f"no assignments of {source_host} match blocks {blocks} and datasets {datasets}")
    pairs = {(item["block"], item["dataset_id"]) for item in selected}
    published = [
        str(Path(host["shard_root"]) / "manifests" / f"{item['configuration_md5']}.tsv")
        for item in plan["assignments"]
        if (item["block"], item["dataset_id"]) in pairs
        for host in plan["hosts"].values()
        if (Path(host["shard_root"]) / "manifests" / f"{item['configuration_md5']}.tsv").exists()
    ]
    if published:
        raise ValueError(
            "reassign refused: pairs to be moved have published configurations: "
            + ", ".join(sorted(published))
        )

    suffix = f".before-reassign-{timestamp or datetime.now().strftime('%Y%m%d-%H%M%S')}"
    plan_path = directory / PLAN_FILENAME
    source_path = directory / f"preparation_assignments_{source_host}.tsv"
    target_path = directory / f"preparation_assignments_{target_host}.tsv"
    for path in (plan_path, source_path, target_path):
        if path.exists():
            shutil.copy2(path, path.with_name(path.name + suffix))

    moved = {item["configuration_md5"].lower() for item in selected}
    for item in selected:
        item["host"] = target_host
    fieldnames = ("block", "dataset_id", "cohort", "configuration_md5")
    for host, path in ((source_host, source_path), (target_host, target_path)):
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {key: item[key] for key in fieldnames}
            for item in plan["assignments"]
            if item["host"] == host
        )
        _replace_text(path, buffer.getvalue())

    plan.setdefault("reassignments", []).append(
        {
            "date": datetime.now().isoformat(timespec="seconds"),
            "code_commit": execution_provenance.collect_provenance()["code_commit"],
            "source_host": source_host,
            "target_host": target_host,
            "backup_suffix": suffix,
            "pairs": [
                {
                    "block": block,
                    "dataset_id": dataset,
                    "configurations": sum(
                        1 for item in selected if (item["block"], item["dataset_id"]) == (block, dataset)
                    ),
                }
                for block, dataset in sorted(pairs)
            ],
            "configuration_md5s": sorted(moved),
        }
    )
    _replace_text(plan_path, json.dumps(plan, indent=2, sort_keys=True))
    return selected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("--run-id", required=True)
    create.add_argument("--mode", choices=("retry", "force"), default="retry")
    create.add_argument("--output-directory", required=True)
    create.add_argument("--acer-host", default="marcospiaia-Nitro-ANV15-41")
    create.add_argument("--dell-host", default="marcospiaia-Latitude-3490")
    create.add_argument("--lenovo-host", default="marcospiaia-ThinkPad-E470")
    create.add_argument("--black-root", default="/mnt/network-black-hd")
    create.add_argument("--transparent-root", default="/mnt/network-transparent-hd")
    create.add_argument("--reserve-bytes", type=int, default=DEFAULT_RESERVE_BYTES)
    create.add_argument("--micro-shard-bytes", type=int, default=DEFAULT_MICRO_SHARD_BYTES)
    create.add_argument("--limit", type=int, default=0)
    verify = commands.add_parser("verify-worker")
    verify.add_argument("--plan-directory", required=True)
    verify.add_argument("--host", default=platform.node())
    amend = commands.add_parser("amend")
    amend.add_argument("--plan-directory", required=True)
    supersede = commands.add_parser("supersede")
    supersede.add_argument("--plan-directory", required=True)
    reassign = commands.add_parser("reassign")
    reassign.add_argument("--plan-directory", required=True)
    reassign.add_argument("--from-host", required=True)
    reassign.add_argument("--to-host", required=True)
    reassign.add_argument("--block", action="append", required=True, choices=BLOCKS)
    reassign.add_argument("--dataset", action="append", default=None)
    space =commands.add_parser("space")
    space.add_argument("--plan-directory", required=True)
    space.add_argument("--host", default=platform.node())
    space.add_argument("--block", required=True)
    space.add_argument("--dataset", required=True)
    shard = commands.add_parser("shard-root")
    shard.add_argument("--plan-directory", required=True)
    shard.add_argument("--host", default=platform.node())
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "create":
            print(create_plan(args))
            return 0
        if args.command == "amend":
            changes = amend_plan(args.plan_directory)
            if not changes:
                print("Plan already matches the preparer identities; nothing changed.")
                return 0
            counts: dict[tuple[str, str], int] = {}
            for change in changes:
                key = (change["host"], change["block"])
                counts[key] = counts.get(key, 0) + 1
            print(f"Amended {len(changes)} assignments:")
            for (host, block), count in sorted(counts.items()):
                print(f"  {host}\t{block}\t{count}")
            return 0
        if args.command == "supersede":
            changes = supersede_plan(args.plan_directory)
            if not changes:
                print("No assignment is superseded by a schema revision; nothing changed.")
                return 0
            print(f"Superseded {len(changes)} assignments ({SUPERSEDE_REASON}):")
            for change in changes:
                state = "published" if change["old_published"] else "unpublished"
                print(
                    f"  {change['host']}\t{change['block']}\t{change['dataset_id']}\t"
                    f"{change['old']} -> {change['new']} ({state})"
                )
            return 0
        if args.command == "reassign":
            moved = reassign_plan(
                args.plan_directory, args.from_host, args.to_host, args.block, args.dataset
            )
            counts: dict[tuple[str, str], int] = {}
            for item in moved:
                key = (item["block"], item["dataset_id"])
                counts[key] = counts.get(key, 0) + 1
            print(f"Reassigned {len(moved)} assignments from {args.from_host} to {args.to_host}:")
            for (block, dataset), count in sorted(counts.items()):
                print(f"  {block}\t{dataset}\t{count}")
            return 0
        plan = load_plan(args.plan_directory)
        if args.command == "shard-root":
            if args.host not in plan["hosts"]:
                raise ValueError(f"host {args.host} is not in the preparation plan")
            print(plan["hosts"][args.host]["shard_root"])
            return 0
        if args.command == "space":
            if args.host not in plan["hosts"]:
                raise ValueError(f"host {args.host} is not in the preparation plan")
            problem = check_configuration_space(plan, args.host, args.block, args.dataset)
            if problem:
                print(problem, file=sys.stderr)
                return NO_SPACE_EXIT_CODE
            return 0
        problems = verify_worker(plan, args.host)
        if problems:
            print("Worker preflight failed:\n  " + "\n  ".join(problems), file=sys.stderr)
            return 1
        print(f"Worker {args.host} matches the preparation plan.")
        return 0
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
