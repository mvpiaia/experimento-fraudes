"""Reune shards previstos no plano em um diretorio preparado canonico."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

from distributed_preparation_plan import load_plan
from publish_prepared_configuration import (
    LINK_UNSUPPORTED_ERRNOS,
    configuration_files,
    read_rows,
    verify_files,
)
from verify_datasets import file_md5

MANIFEST_NAME = "prepared_dataset_configurations.tsv"


def _link_same_bytes(existing: Path | None, source: Path, temporary: Path) -> bool:
    """Liga ``temporary`` a um arquivo do canonico com os mesmos bytes de ``source``."""
    if existing is None or existing.stat().st_size != source.stat().st_size:
        return False
    if file_md5(existing) != file_md5(source):
        return False
    try:
        os.link(existing, temporary)
    except OSError as error:
        if error.errno in LINK_UNSUPPORTED_ERRNOS:
            return False
        raise
    return True


def merge(plan_directory: Path, destination: Path) -> Path:
    plan = load_plan(plan_directory)
    assignments = plan["assignments"]
    md5s = [item["configuration_md5"].lower() for item in assignments]
    if len(md5s) != len(set(md5s)):
        raise ValueError("preparation plan contains duplicate configuration assignments")

    collected: list[tuple[list[str], list[dict], Path, str]] = []
    for item in assignments:
        md5 = item["configuration_md5"].lower()
        shard = Path(plan["hosts"][item["host"]]["shard_root"])
        manifest = shard / "manifests" / f"{md5}.tsv"
        if not manifest.is_file():
            raise ValueError(f"missing published configuration: {md5} on {item['host']}")
        fields, rows = read_rows(manifest, md5)
        if not rows or {row["configuration_md5"].lower() for row in rows} != {md5}:
            raise ValueError(f"invalid individual manifest: {manifest}")
        files_dir = shard / "files"
        verify_files(files_dir, rows)
        files = configuration_files(files_dir, rows, md5)
        collected.append((fields, rows, files_dir, md5))

    destination.mkdir(parents=True, exist_ok=True)
    destination_device = destination.stat().st_dev
    required_bytes = sum(
        source.stat().st_size
        for _fields, rows, files_dir, md5 in collected
        for source in configuration_files(files_dir, rows, md5)
        if source.stat().st_dev != destination_device
        and not (destination / source.name).exists()
    )
    free = shutil.disk_usage(destination).free
    if free < required_bytes + int(plan["reserve_bytes"]):
        raise ValueError(
            f"canonical storage has {free} bytes free; "
            f"{required_bytes + int(plan['reserve_bytes'])} required"
        )

    header = collected[0][0] if collected else []
    # Splits iguais de configuracoes diferentes viram um unico inode no canonico.
    canonical_by_content: dict[str, Path] = {}
    merged_rows: list[dict] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for fields, rows, files_dir, md5 in collected:
        if fields != header:
            raise ValueError(f"manifest header differs for {md5}")
        for row in rows:
            key = (md5, row["fold_index"], row["split"])
            if key in seen_keys:
                raise ValueError(f"duplicate manifest row: {'|'.join(key)}")
            seen_keys.add(key)
            merged_rows.append(row)
        content_by_name = {
            name: row["content_md5"].lower()
            for row in rows
            for name in (f"{row['dataset_md5']}.csv", f"{row['dataset_md5']}.csv.zst")
        }
        for source in configuration_files(files_dir, rows, md5):
            target = destination / source.name
            if target.exists():
                if file_md5(source) != file_md5(target):
                    raise ValueError(f"canonical file conflict: {source.name}")
                continue
            temporary = destination / f".{source.name}.incoming"
            temporary.unlink(missing_ok=True)
            content = content_by_name.get(source.name)
            same_content = canonical_by_content.get(content) if content else None
            if source.stat().st_dev == destination_device:
                os.link(source, temporary)
            elif not _link_same_bytes(same_content, source, temporary):
                # Conteudo novo de outro disco (shard do Lenovo): copia.
                shutil.copy2(source, temporary)
            if file_md5(source) != file_md5(temporary):
                temporary.unlink(missing_ok=True)
                raise ValueError(f"copy checksum mismatch: {source.name}")
            os.replace(temporary, target)
            if content:
                canonical_by_content.setdefault(content, target)

    temporary_manifest = destination / f".{MANIFEST_NAME}.incoming"
    with temporary_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            sorted(
                merged_rows,
                key=lambda row: (
                    row["configuration_md5"],
                    int(row["fold_index"]),
                    row["split"],
                ),
            )
        )
    manifest_path = destination / MANIFEST_NAME
    os.replace(temporary_manifest, manifest_path)
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-directory", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args()
    try:
        print(merge(Path(args.plan_directory), Path(args.destination)))
        return 0
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
