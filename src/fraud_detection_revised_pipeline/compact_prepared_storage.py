"""Compacta no lugar um shard ou diretorio canonico preparado (compact-prepared-storage).

Uso:
  compact_prepared_storage.py <dir> --dedupe-only   liga por hard link os splits de mesmo conteudo
  compact_prepared_storage.py <dir>                 comprime cada conteudo uma vez (.csv.zst)
  compact_prepared_storage.py <dir> --decompress    volta aos .csv (rollback)

<dir> e um shard (``manifests/`` e ``files/``) ou um canonico
(``prepared_dataset_configurations.tsv`` com os arquivos ao lado). Trata uma
configuracao por vez, nunca muda o conteudo de um nome e e retomavel: um
``.csv.zst`` so aparece depois de conferido, e o CSV so e removido depois dele.
O pico de espaco e um split comprimido.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from prepared_dataset_storage import (
    COMPRESSED_SPLIT_FILE_SUFFIX,
    SPLIT_FILE_SUFFIX,
    PreparedSplitDecompressorError,
    compress_prepared_split,
    materialize_prepared_split,
    prepared_split_path,
    read_storage_metadata,
    split_content_digest,
    storage_metadata_path,
    write_storage_metadata,
)
from verify_datasets import file_md5

CANONICAL_MANIFEST_NAME = "prepared_dataset_configurations.tsv"
TEMPORARY_PATTERNS = (".*.tmp", ".*.link")


@dataclass
class Configuration:
    md5: str
    rows: list[dict]
    files: Path


@dataclass
class Summary:
    configurations: int = 0
    skipped: int = 0
    compressed: int = 0
    linked: int = 0
    decompressed: int = 0
    problems: list[str] = field(default_factory=list)


def configurations(root: Path) -> list[Configuration]:
    """Configuracoes do shard (um manifesto por configuracao) ou do canonico."""
    if (root / "manifests").is_dir():
        found = []
        for manifest in sorted((root / "manifests").glob("*.tsv")):
            found.append(Configuration(manifest.stem.lower(), _read_rows(manifest), root / "files"))
        return found
    manifest = root / CANONICAL_MANIFEST_NAME
    if not manifest.is_file():
        raise ValueError(f"{root} has neither manifests/ nor {CANONICAL_MANIFEST_NAME}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in _read_rows(manifest):
        grouped[row["configuration_md5"].strip().lower()].append(row)
    return [Configuration(md5, rows, root) for md5, rows in sorted(grouped.items())]


def _read_rows(manifest: Path) -> list[dict]:
    with manifest.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _unique_splits(rows: list[dict]) -> list[tuple[str, str]]:
    """(dataset_md5, content_md5) de cada split, sem repeticao."""
    seen: dict[str, str] = {}
    for row in rows:
        seen.setdefault(row["dataset_md5"].strip().lower(), row["content_md5"].strip().lower())
    return list(seen.items())


def _remove_temporaries(directories: set[Path]) -> None:
    for directory in directories:
        for pattern in TEMPORARY_PATTERNS:
            for path in directory.glob(pattern):
                path.unlink()


def _link_over(existing: Path, destination: Path) -> None:
    """Troca ``destination`` por um hard link para ``existing``, de forma atomica."""
    temporary = destination.with_name(f".{destination.name}.link")
    temporary.unlink(missing_ok=True)
    os.link(existing, temporary)
    os.replace(temporary, destination)


def dedupe(items: list[Configuration], summary: Summary, log=print) -> None:
    """Liga os splits de mesmo conteudo e mesma forma armazenada a um unico inode."""
    # content_md5 -> (arquivo conferido, MD5 dos bytes armazenados)
    representatives: dict[tuple[str, str], tuple[Path, str]] = {}
    for position, item in enumerate(items, start=1):
        linked = 0
        for dataset_md5, content in _unique_splits(item.rows):
            path = prepared_split_path(item.files, dataset_md5)
            if path is None:
                summary.problems.append(f"{item.md5}: {dataset_md5} missing")
                continue
            form = COMPRESSED_SPLIT_FILE_SUFFIX if path.name.endswith(COMPRESSED_SPLIT_FILE_SUFFIX) else SPLIT_FILE_SUFFIX
            key = (content, form)
            stored = file_md5(path)
            if key not in representatives:
                found = stored if form == SPLIT_FILE_SUFFIX else split_content_digest(path)[0]
                if found != content:
                    raise ValueError(f"{path} does not match content_md5 {content}: {found}")
                representatives[key] = (path, stored)
                continue
            representative, representative_stored = representatives[key]
            if path.samefile(representative):
                continue
            if stored != representative_stored:
                # Mesmo conteudo, bytes armazenados diferentes (outro zstd):
                # nao se liga sem mudar o que o nome guarda.
                if form == SPLIT_FILE_SUFFIX:
                    raise ValueError(f"{path} does not match content_md5 {content}")
                continue
            _link_over(representative, path)
            linked += 1
        summary.linked += linked
        summary.configurations += 1
        log(f"[{position}/{len(items)}] {item.md5}: {linked} linked")


def compress(items: list[Configuration], summary: Summary, log=print) -> None:
    """Comprime cada conteudo uma vez e liga os demais nomes ao mesmo .csv.zst."""
    compressed_by_content: dict[str, Path] = {}
    content_bytes: dict[str, int] = {}
    pending = []
    for item in items:
        metadata = read_storage_metadata(item.files, item.md5)
        splits = _unique_splits(item.rows)
        if metadata and all(
            metadata["splits"].get(dataset_md5, {}).get("codec") == "zstd"
            and (item.files / f"{dataset_md5}{COMPRESSED_SPLIT_FILE_SUFFIX}").is_file()
            and not (item.files / f"{dataset_md5}{SPLIT_FILE_SUFFIX}").exists()
            for dataset_md5, _content in splits
        ):
            for dataset_md5, content in splits:
                compressed_by_content.setdefault(
                    content, item.files / f"{dataset_md5}{COMPRESSED_SPLIT_FILE_SUFFIX}"
                )
                content_bytes.setdefault(content, int(metadata["splits"][dataset_md5]["content_bytes"]))
            summary.skipped += 1
        else:
            pending.append(item)

    for position, item in enumerate(pending, start=1):
        sizes: dict[str, int] = {}
        compressed = linked = 0
        for dataset_md5, content in _unique_splits(item.rows):
            plain = item.files / f"{dataset_md5}{SPLIT_FILE_SUFFIX}"
            packed = item.files / f"{dataset_md5}{COMPRESSED_SPLIT_FILE_SUFFIX}"
            if packed.is_file():
                # Um .csv.zst so aparece depois de conferido; um CSV ao lado e
                # sobra de uma interrupcao.
                plain.unlink(missing_ok=True)
                if content not in content_bytes:
                    found, content_bytes[content] = split_content_digest(packed)
                    if found != content:
                        raise ValueError(f"{packed} does not match content_md5 {content}: {found}")
                compressed_by_content.setdefault(content, packed)
            elif plain.is_file():
                existing = compressed_by_content.get(content)
                if existing is not None and existing.is_file():
                    _link_over(existing, packed)
                    linked += 1
                else:
                    packed, content_bytes[content] = compress_prepared_split(plain, content)
                    compressed_by_content[content] = packed
                    compressed += 1
                plain.unlink()
            else:
                summary.problems.append(f"{item.md5}: {dataset_md5} missing")
                continue
            sizes[dataset_md5] = content_bytes[content]
        if len(sizes) == len(_unique_splits(item.rows)):
            write_storage_metadata(item.files, item.md5, item.rows, sizes)
        summary.compressed += compressed
        summary.linked += linked
        summary.configurations += 1
        log(f"[{position}/{len(pending)}] {item.md5}: {compressed} compressed, {linked} linked")


def decompress(items: list[Configuration], summary: Summary, log=print) -> None:
    """Rollback: cada .csv.zst volta a CSV (um por conteudo, demais por link)."""
    plain_by_content: dict[str, Path] = {}
    for position, item in enumerate(items, start=1):
        restored = 0
        for dataset_md5, content in _unique_splits(item.rows):
            plain = item.files / f"{dataset_md5}{SPLIT_FILE_SUFFIX}"
            packed = item.files / f"{dataset_md5}{COMPRESSED_SPLIT_FILE_SUFFIX}"
            if plain.is_file():
                packed.unlink(missing_ok=True)
                plain_by_content.setdefault(content, plain)
                continue
            if not packed.is_file():
                summary.problems.append(f"{item.md5}: {dataset_md5} missing")
                continue
            existing = plain_by_content.get(content)
            if existing is not None and existing.is_file():
                _link_over(existing, plain)
            else:
                materialize_prepared_split(packed, plain, content)
                plain_by_content[content] = plain
            packed.unlink()
            restored += 1
        storage_metadata_path(item.files, item.md5).unlink(missing_ok=True)
        summary.decompressed += restored
        summary.configurations += 1
        log(f"[{position}/{len(items)}] {item.md5}: {restored} restored")


def run(root: Path, mode: str, log=print) -> Summary:
    items = configurations(root)
    _remove_temporaries({item.files for item in items})
    summary = Summary()
    {"dedupe": dedupe, "compress": compress, "decompress": decompress}[mode](items, summary, log)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--dedupe-only", action="store_true")
    group.add_argument("--decompress", action="store_true")
    args = parser.parse_args()
    root = Path(args.directory)
    mode = "dedupe" if args.dedupe_only else "decompress" if args.decompress else "compress"
    before = shutil.disk_usage(root).free
    try:
        summary = run(root, mode, log=lambda line: print(line, flush=True))
    except (OSError, ValueError, PreparedSplitDecompressorError) as error:
        print(f"compact_prepared_storage failed: {error}", file=sys.stderr)
        return 1
    after = shutil.disk_usage(root).free
    print(
        f"{mode}: {summary.configurations} configurations ({summary.skipped} already done), "
        f"{summary.compressed} compressed, {summary.linked} linked, "
        f"{summary.decompressed} restored; free space {before / 2**30:.1f} -> {after / 2**30:.1f} GiB"
    )
    for problem in summary.problems:
        print(f"problem: {problem}", file=sys.stderr)
    return 1 if summary.problems else 0


if __name__ == "__main__":
    sys.exit(main())
