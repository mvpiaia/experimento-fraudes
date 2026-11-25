"""Publica uma configuracao preparada em um shard e grava o manifesto por ultimo."""

from __future__ import annotations

import argparse
import csv
import errno
import os
import shutil
import subprocess
import sys
from pathlib import Path

from prepared_dataset_storage import (
    PreparedSplitDecompressorError,
    compress_prepared_split,
    is_compressed_split,
    prepared_split_path,
    split_content_digest,
    write_storage_metadata,
)
from verify_datasets import file_md5

# Codigo de saida quando o shard nao tem espaco: o worker para a fila.
NO_SPACE_EXIT_CODE = 10
LINK_UNSUPPORTED_ERRNOS = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK}


def read_rows(manifest: Path, configuration_md5: str) -> tuple[list[str], list[dict]]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [
            row
            for row in reader
            if row["configuration_md5"].lower() == configuration_md5.lower()
        ]
        return list(reader.fieldnames or ()), rows


def configuration_files(source: Path, rows: list[dict], configuration_md5: str) -> list[Path]:
    files: set[Path] = set(source.glob(f"{configuration_md5}*"))
    for row in rows:
        dataset_md5 = row["dataset_md5"]
        files.update(source.glob(f"{dataset_md5}.*"))
        for column in ("vocabulary_path", "window_index_path"):
            value = row.get(column, "").strip()
            if value:
                candidate = source / Path(value).name
                if candidate.exists():
                    files.add(candidate)
    return sorted(path for path in files if path.is_file())


def verify_files(directory: Path, rows: list[dict]) -> dict[str, int]:
    """Confere cada split pelo conteudo descomprimido; devolve os tamanhos do CSV."""
    sizes: dict[str, int] = {}
    for row in rows:
        dataset_md5 = row["dataset_md5"]
        split_path = prepared_split_path(directory, dataset_md5)
        columns_path = directory / f"{dataset_md5}.columns.json"
        if split_path is None or not columns_path.is_file():
            raise ValueError(f"missing CSV or column registry for {dataset_md5}")
        found, size = split_content_digest(split_path)
        expected = row["content_md5"].lower()
        if found != expected:
            raise ValueError(f"content_md5 mismatch for {split_path.name}: {found} != {expected}")
        sizes[dataset_md5.lower()] = size
    return sizes


def compress_splits(directory: Path, rows: list[dict]) -> None:
    """Troca cada <dataset_md5>.csv por .csv.zst, conferido antes de remover o CSV."""
    for row in rows:
        path = prepared_split_path(directory, row["dataset_md5"])
        if path is not None and not is_compressed_split(path):
            compress_prepared_split(path, row["content_md5"])
            path.unlink()


def content_index(shard: Path) -> dict[str, Path]:
    """content_md5 -> split ja publicado no shard, pelos manifestos individuais."""
    index: dict[str, Path] = {}
    files_directory = shard / "files"
    for manifest in sorted((shard / "manifests").glob("*.tsv")):
        with manifest.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                content = (row.get("content_md5") or "").strip().lower()
                if not content or content in index:
                    continue
                path = prepared_split_path(files_directory, row.get("dataset_md5", "").strip())
                if path is not None:
                    index[content] = path
    return index


def link_identical(existing: Path, path: Path, destination: Path) -> bool:
    """Liga ``destination`` ao arquivo ja armazenado, se os bytes forem iguais.

    Sem suporte a hard link (CIFS, outro disco), devolve False e quem chama
    grava o arquivo; a deduplicacao no Acer resolve depois.
    """
    if not existing.is_file() or existing.stat().st_size != path.stat().st_size:
        return False
    if file_md5(existing) != file_md5(path):
        return False
    temporary = destination.with_name(f".{destination.name}.link")
    temporary.unlink(missing_ok=True)
    try:
        os.link(existing, temporary)
    except OSError as error:
        if error.errno in LINK_UNSUPPORTED_ERRNOS:
            return False
        raise
    os.replace(temporary, destination)
    path.unlink()
    return True


def _rsync(files: list[Path], destination: Path) -> None:
    result = subprocess.run(
        ["rsync", "-a", *(str(path) for path in files), f"{destination}/"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if "No space left" in result.stderr:
            raise OSError(errno.ENOSPC, result.stderr.strip())
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)


def publish(source: Path, manifest: Path, shard: Path, configuration_md5: str) -> Path:
    fields, rows = read_rows(manifest, configuration_md5)
    if not rows:
        raise ValueError(f"manifest has no rows for {configuration_md5}")
    if {row["configuration_md5"].lower() for row in rows} != {configuration_md5.lower()}:
        raise ValueError("individual manifest contains another configuration")
    # Comprime no scratch: o rsync (pela rede, no Dell) leva a forma compacta.
    compress_splits(source, rows)
    sizes = verify_files(source, rows)
    write_storage_metadata(source, configuration_md5, rows, sizes)
    files = configuration_files(source, rows, configuration_md5)
    incoming = shard / ".incoming" / configuration_md5
    if incoming.exists():
        shutil.rmtree(incoming)
    incoming.mkdir(parents=True)
    _rsync(files, incoming)
    verify_files(incoming, rows)
    files_directory = shard / "files"
    manifests_directory = shard / "manifests"
    files_directory.mkdir(parents=True, exist_ok=True)
    manifests_directory.mkdir(parents=True, exist_ok=True)
    published = manifests_directory / f"{configuration_md5}.tsv"
    repairing_unpublished = not published.exists()
    content_by_name = {
        split_name: row["content_md5"].lower()
        for row in rows
        for split_name in (f"{row['dataset_md5']}.csv", f"{row['dataset_md5']}.csv.zst")
    }
    index = content_index(shard)
    for path in sorted(incoming.iterdir()):
        destination = files_directory / path.name
        if destination.exists():
            if file_md5(path) != file_md5(destination):
                if not repairing_unpublished:
                    raise ValueError(f"conflicting shard file: {path.name}")
                os.replace(path, destination)
            else:
                path.unlink()
            continue
        content = content_by_name.get(path.name)
        existing = index.get(content) if content else None
        if existing is None or not link_identical(existing, path, destination):
            os.replace(path, destination)
        if content:
            index.setdefault(content, destination)
    incoming.rmdir()
    temporary_manifest = manifests_directory / f".{configuration_md5}.tsv.tmp"
    with temporary_manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_manifest, published)
    return published


def verify_published(shard: Path, configuration_md5: str) -> Path:
    manifest = shard / "manifests" / f"{configuration_md5}.tsv"
    if not manifest.is_file():
        raise ValueError(f"configuration is not published: {configuration_md5}")
    _fields, rows = read_rows(manifest, configuration_md5)
    if not rows:
        raise ValueError(f"published manifest has no rows: {configuration_md5}")
    verify_files(shard / "files", rows)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-directory", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-directory", required=True)
    parser.add_argument("--configuration-md5", required=True)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        print(
            verify_published(Path(args.shard_directory), args.configuration_md5)
            if args.verify_only
            else publish(
                Path(args.source_directory),
                Path(args.manifest),
                Path(args.shard_directory),
                args.configuration_md5,
            )
        )
        return 0
    except (
        OSError, ValueError, subprocess.CalledProcessError, PreparedSplitDecompressorError
    ) as error:
        print(error, file=sys.stderr)
        if isinstance(error, OSError) and error.errno == errno.ENOSPC:
            return NO_SPACE_EXIT_CODE
        return 1


if __name__ == "__main__":
    sys.exit(main())
