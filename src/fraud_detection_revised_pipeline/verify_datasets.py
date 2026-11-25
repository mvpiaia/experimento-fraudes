"""Confere o MD5 das bases originais antes da preparacao (item G18).

As bases vao publicadas no repositorio publico ``experimento-fraudes``, nos
.zip originais do Kaggle, em Git LFS; os CSV sao extraidos de cada .zip na
propria pasta. Quem reproduz o trabalho precisa dos mesmos bytes; um arquivo
truncado, um ponteiro do LFS nao baixado ou uma versao diferente publicada no
Kaggle mudariam os resultados sem aviso. Esta verificacao compara cada arquivo
listado em ``datasets/CHECKSUMS.md5`` com o MD5 registrado e interrompe a
preparacao na primeira divergencia.

Uso:
    python3 verify_datasets.py [--datasets-directory <dir>]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

DEFAULT_DATASETS_DIRECTORY = Path(__file__).resolve().parent.parent / "datasets"
CHECKSUM_FILENAME = "CHECKSUMS.md5"
DATASET_VERIFICATION_EXIT_CODE = 4
# Ponteiros do Git LFS comecam assim: o arquivo nao foi baixado.
LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/"


def read_checksums(checksum_path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        expected, relative_path = line.split(maxsplit=1)
        entries.append((expected.lower(), relative_path.strip()))
    return entries


def file_md5(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_datasets(datasets_directory: Path = DEFAULT_DATASETS_DIRECTORY) -> list[str]:
    """Problemas encontrados, um por arquivo; vazio quando tudo confere."""
    checksum_path = datasets_directory / CHECKSUM_FILENAME
    if not checksum_path.exists():
        return [f"checksum file not found: {checksum_path}"]
    problems = []
    for expected, relative_path in read_checksums(checksum_path):
        path = datasets_directory / relative_path
        if not path.exists():
            problems.append(f"{relative_path}: missing (extract the .zip in its folder)")
            continue
        with path.open("rb") as handle:
            if handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
                problems.append(f"{relative_path}: Git LFS pointer, run `git lfs pull`")
                continue
        found = file_md5(path)
        if found != expected:
            problems.append(f"{relative_path}: expected MD5 {expected}, found {found}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datasets-directory", default=str(DEFAULT_DATASETS_DIRECTORY))
    args = parser.parse_args()
    problems = verify_datasets(Path(args.datasets_directory))
    if problems:
        print("Dataset verification failed:\n  " + "\n  ".join(problems), file=sys.stderr)
        return DATASET_VERIFICATION_EXIT_CODE
    print("Datasets match datasets/CHECKSUMS.md5.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
