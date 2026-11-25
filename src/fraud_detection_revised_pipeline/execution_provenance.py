"""Proveniencia de cada linha de resultado (item G18 da revisao).

Toda linha de resultado, de sucesso ou de falha, passa a dizer com que codigo,
com que ambiente e em que maquina foi produzida. A reexecucao roda dividida
entre tres maquinas (Acer Nitro ANV15-41, Lenovo ThinkPad E470 e Dell Latitude
15 3540); sem isso, nao haveria como atribuir um resultado ou um tempo a uma
maquina, nem como provar que as tres usaram as mesmas bibliotecas.

A proveniencia e coletada uma vez por processo e acrescentada a cada linha.
Ela nunca entra em chave de retomada ou de pareamento: a mesma configuracao,
rodada em outra maquina, e a mesma execucao.

O ambiente de referencia e descrito em ``requirements.lock.txt``.
``environment_md5`` e o
MD5 da lista ordenada ``nome==versao`` dos pacotes que o arquivo nomeia, com a
versao instalada e sem o sufixo local (``+cu126``, ``+cpu``): a roda CUDA do
PyTorch no Acer e a roda CPU no E470 e no Dell tem o mesmo codigo Python, e o
dispositivo ja e registrado a parte. Sem remover o sufixo, as tres maquinas
nunca teriam o mesmo hash.

Uso pelos scripts de shell:
    python3 execution_provenance.py --check        # guarda do ambiente
    python3 execution_provenance.py --write-lock   # gera o arquivo travado
    python3 execution_provenance.py --show         # imprime a proveniencia
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PIPELINE_DIRECTORY = Path(__file__).resolve().parent
REQUIREMENTS_FILE = PIPELINE_DIRECTORY / "requirements.txt"
LOCK_FILE = PIPELINE_DIRECTORY / "requirements.lock.txt"
PYTHON_LOCK_PREFIX = "# python=="

# Mantidas por compatibilidade com roteiros existentes. O ambiente destravado
# e sempre permitido e a arvore Git nunca e usada como guarda de execucao.
ALLOW_UNLOCKED_ENVIRONMENT = "ALLOW_UNLOCKED_ENVIRONMENT"
REQUIRE_CLEAN_TREE = "REQUIRE_CLEAN_TREE"
# Permite informar o commit quando o codigo roda fora de um checkout git,
# como numa copia exportada.
CODE_COMMIT_OVERRIDE = "CODE_COMMIT"

UNKNOWN = "unknown"
NO_GPU = "none"

PROVENANCE_COLUMNS = (
    "code_commit",
    "code_tree_dirty",
    "environment_md5",
    "environment_locked",
    "execution_host",
    "cpu_model",
    "cpu_logical_cores",
    "memory_total_gb",
    "gpu_model",
    "gpu_memory_gb",
    "operating_system",
)


class EnvironmentLockError(RuntimeError):
    """O ambiente instalado diverge do arquivo travado."""


def canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).strip().lower()


def normalize_version(installed_version: str) -> str:
    """Versao sem o sufixo local: '2.13.0+cu126' -> '2.13.0'."""
    return installed_version.split("+", 1)[0]


def requirement_names(requirements_path: Path = REQUIREMENTS_FILE) -> list[str]:
    names = []
    for line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        names.append(re.split(r"[<>=!~;\[ ]", line, maxsplit=1)[0])
    return names


def read_lock_file(lock_path: Path = LOCK_FILE) -> tuple[dict[str, str], str | None]:
    """Pacotes de referencia e a versao Python legada, que nao e validada."""
    packages: dict[str, str] = {}
    python_version = None
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith(PYTHON_LOCK_PREFIX):
            python_version = line[len(PYTHON_LOCK_PREFIX):].strip()
            continue
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        name, separator, locked_version = line.partition("==")
        if not separator:
            raise ValueError(f"lock file line without an exact version: {raw_line!r}")
        packages[canonical_package_name(name)] = locked_version.strip()
    return packages, python_version


def installed_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def environment_md5(package_names) -> str:
    entries = []
    for name in package_names:
        found = installed_version(name)
        entries.append(
            f"{canonical_package_name(name)}=="
            f"{normalize_version(found) if found else 'missing'}"
        )
    payload = "\n".join(sorted(entries))
    return hashlib.md5(payload.encode("utf-8"), usedforsecurity=False).hexdigest()


def environment_divergences(lock_path: Path = LOCK_FILE) -> list[str]:
    """Diferencas de pacotes instalados, apenas para registro informativo."""
    if not lock_path.exists():
        return [f"lock file not found: {lock_path}"]
    packages, _legacy_python = read_lock_file(lock_path)
    divergences = []
    for name, locked_version in sorted(packages.items()):
        found = installed_version(name)
        if found is None:
            divergences.append(f"{name}: locked {locked_version}, not installed")
        elif normalize_version(found) != normalize_version(locked_version):
            divergences.append(f"{name}: locked {locked_version}, installed {found}")
    return divergences


def write_lock_file(
    lock_path: Path = LOCK_FILE, requirements_path: Path = REQUIREMENTS_FILE
) -> Path:
    lines = [
        "# Versoes exatas do ambiente de referencia (Acer Nitro ANV15-41), geradas",
        "# por `python3 execution_provenance.py --write-lock` a partir de",
        "# requirements.txt. Instale este arquivo nas tres maquinas da reexecucao;",
        "# o sufixo local do PyTorch (+cu126, +cpu) nao entra na comparacao.",
    ]
    for name in requirement_names(requirements_path):
        found = installed_version(name)
        if found is None:
            raise EnvironmentLockError(f"{name} is not installed; cannot lock it")
        lines.append(f"{canonical_package_name(name)}=={normalize_version(found)}")
    lock_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lock_path


def _git(*arguments: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(PIPELINE_DIRECTORY), *arguments],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout


def code_version() -> tuple[str, bool | str]:
    """Commit do codigo e se a arvore tem alteracoes nao commitadas."""
    override = os.environ.get(CODE_COMMIT_OVERRIDE, "").strip()
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return (override or UNKNOWN), UNKNOWN
    # So arquivos rastreados: resultados e dados preparados gerados durante a
    # execucao nao tornam a arvore suja.
    status = _git("status", "--porcelain", "--untracked-files=no")
    return commit.strip(), bool(status and status.strip())


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def cpu_model() -> str:
    # /proc/cpuinfo, e nao lscpu, cuja saida e traduzida pelo locale.
    for line in _read_text("/proc/cpuinfo").splitlines():
        if line.lower().startswith("model name"):
            return line.split(":", 1)[1].strip()
    return platform.processor() or UNKNOWN


def memory_total_gb() -> float | str:
    match = re.search(r"^MemTotal:\s+(\d+)\s+kB", _read_text("/proc/meminfo"), re.M)
    if not match:
        return UNKNOWN
    return round(int(match.group(1)) / 1024**2, 1)


def gpu_description() -> tuple[str, float | str]:
    try:
        import torch
    except ImportError:
        return NO_GPU, ""
    if not torch.cuda.is_available():
        return NO_GPU, ""
    properties = torch.cuda.get_device_properties(0)
    return properties.name, round(properties.total_memory / 1024**3, 1)


def operating_system() -> str:
    match = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', _read_text("/etc/os-release"), re.M)
    name = match.group(1) if match else platform.system()
    return f"{name}, kernel {platform.release()}"


@lru_cache(maxsize=1)
def collect_provenance() -> dict:
    """Proveniencia do processo, calculada uma vez e reutilizada em cada linha."""
    commit, dirty = code_version()
    gpu_model, gpu_memory = gpu_description()
    locked = not environment_divergences()
    package_names = (
        list(read_lock_file()[0]) if LOCK_FILE.exists() else requirement_names()
    )
    return {
        "code_commit": commit,
        "code_tree_dirty": dirty,
        "environment_md5": environment_md5(package_names),
        "environment_locked": locked,
        "execution_host": platform.node() or UNKNOWN,
        "cpu_model": cpu_model(),
        "cpu_logical_cores": os.cpu_count() or UNKNOWN,
        "memory_total_gb": memory_total_gb(),
        "gpu_model": gpu_model,
        "gpu_memory_gb": gpu_memory,
        "operating_system": operating_system(),
    }


def enforce_locked_environment(lock_path: Path = LOCK_FILE) -> bool:
    """Informa se os pacotes coincidem, sem bloquear a execucao.

    O comportamento e sempre o equivalente a
    ``ALLOW_UNLOCKED_ENVIRONMENT=true``. Commit, arvore Git e versao do Python
    sao metadados e nunca guardas de execucao.
    """
    divergences = environment_divergences(lock_path)
    return not divergences


def guard_environment_or_exit() -> bool:
    """Conferencia informativa chamada por executores e preparadores."""
    locked = enforce_locked_environment()
    if not locked:
        print(
            "Warning: running outside the reference environment; "
            "rows will record environment_locked=false.",
            file=sys.stderr,
        )
    return locked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--write-lock", action="store_true")
    action.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.write_lock:
        print(write_lock_file())
        return 0
    if args.show:
        print(json.dumps(collect_provenance(), indent=2, ensure_ascii=False))
        return 0
    guard_environment_or_exit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
