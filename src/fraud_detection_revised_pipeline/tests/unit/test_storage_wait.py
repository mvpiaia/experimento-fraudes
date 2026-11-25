"""Espera pelos HDs da rede e repeticao do passo interrompido.

Decisao do usuario (31/10/2026): os scripts usam os HDs compartilhados pela
rede e, quando perdem o acesso, tentam de novo a cada 2 minutos, sem desistir,
ate o acesso voltar; entao o passo que travou e executado de novo. Os testes
simulam a queda com um contador de checagens e intervalo zero.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]


def _bash(script: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", script],
        cwd=PIPELINE_DIRECTORY,
        capture_output=True,
        text=True,
        env={**os.environ, "STORAGE_RETRY_SECONDS": "0", **(env or {})},
    )


# Torna o armazenamento inacessivel nas N primeiras checagens.
FLAKY_STORAGE = """
storage_checks=0
storage_directory_is_accessible() {{
  storage_checks=$((storage_checks + 1))
  echo "${{storage_checks}}" > "{counter}"
  ((storage_checks > {failures}))
}}
"""


def test_wait_reports_each_failed_check_and_the_recovery(tmp_path):
    result = _bash(
        f"""
source data_directory_common.sh
{FLAKY_STORAGE.format(counter=tmp_path / "checks", failures=3)}
wait_for_storage /mnt/network-transparent-hd "boosting/xgboost/abc"
echo "status=$? waited=${{STORAGE_WAITED}}"
"""
    )
    assert "status=0 waited=true" in result.stdout
    failed_checks = [line for line in result.stderr.splitlines() if "indisponível" in line]
    assert len(failed_checks) == 3
    assert "(tentativa 1): /mnt/network-transparent-hd — passo: boosting/xgboost/abc" in failed_checks[0]
    assert "nova tentativa em 0 s" in failed_checks[0]
    assert "acesso restabelecido a /mnt/network-transparent-hd" in result.stderr
    assert "repetindo o passo: boosting/xgboost/abc" in result.stderr


def test_wait_without_failure_is_silent(tmp_path):
    result = _bash(
        f"""
source data_directory_common.sh
{FLAKY_STORAGE.format(counter=tmp_path / "checks", failures=0)}
wait_for_storage /mnt/network-black-hd "passo"
echo "status=$? waited=${{STORAGE_WAITED}}"
"""
    )
    assert "status=0 waited=false" in result.stdout
    assert result.stderr == ""


def test_bounded_attempts_stop_with_the_storage_code(tmp_path):
    result = _bash(
        """
source data_directory_common.sh
wait_for_storage "/nonexistent/network-drive" "passo"
echo "status=$?"
""",
        {"STORAGE_RETRY_ATTEMPTS": "2"},
    )
    assert "status=75" in result.stdout
    assert "após 2 tentativas" in result.stderr and "Encerrando" in result.stderr


def test_root_filesystem_is_refused_unless_allowed(tmp_path):
    script = """
source data_directory_common.sh
storage_directory_is_accessible / && echo accessible || echo refused
"""
    assert "refused" in _bash(script).stdout
    assert "accessible" in _bash(script, {"STORAGE_ALLOW_ROOT_FILESYSTEM": "true"}).stdout


def _runner_script(tmp_path: Path, command: str, failures: int) -> str:
    return f"""
source run_storage_common.sh
PREPARED_DATA_DIRECTORY="{tmp_path}"
RESULTS_DIRECTORY="{tmp_path}"
EXECUTION_LOG_BLOCK="boosting"
STORAGE_IS_LOCAL=true
configuration_files_are_available() {{ return 0; }}
configuration_manifest_is_complete() {{ return 0; }}
report_unavailable_configuration_files() {{ echo "missing-file-report" >&2; }}
{FLAKY_STORAGE.format(counter=tmp_path / "checks", failures=failures)}
set_execution_log_context xgboost
run_with_storage_recovery 0123456789abcdef0123456789abcdef {command}
echo "status=$? exit=${{command_exit_code}} runs=$(cat "{tmp_path}/runs")"
"""


def test_step_that_lost_access_is_repeated_after_recovery(tmp_path):
    # Primeira execucao: sai com 75 e o armazenamento fica fora por 3 checagens.
    command = (
        f"bash -c 'n=$(cat {tmp_path}/runs 2>/dev/null || echo 0); n=$((n+1)); "
        f"echo $n > {tmp_path}/runs; [ $n -ge 2 ] || exit 75'"
    )
    result = _bash(_runner_script(tmp_path, command, failures=3))
    assert "status=0 exit=0 runs=2" in result.stdout
    assert "perdeu o acesso ao armazenamento (código 75)" in result.stderr
    assert "algoritmo xgboost, configuration_md5 0123456789abcdef0123456789abcdef" in result.stderr
    assert "acesso restabelecido" in result.stderr
    # Duas tentativas registradas, a primeira com 75 e a segunda com sucesso.
    log = (tmp_path / "registro_execucoes.csv").read_text().splitlines()
    assert len(log) == 3


def test_code_75_with_accessible_storage_is_repeated_once_then_fails(tmp_path):
    command = (
        f"bash -c 'n=$(cat {tmp_path}/runs 2>/dev/null || echo 0); "
        f"echo $((n+1)) > {tmp_path}/runs; exit 75'"
    )
    result = _bash(_runner_script(tmp_path, command, failures=0))
    assert "status=1 exit=75 runs=2" in result.stdout
    assert "repetindo uma vez" in result.stderr
    assert "a falha não é de acesso" in result.stderr
    assert "missing-file-report" in result.stderr
    assert "indisponível" not in result.stderr


def test_ordinary_failure_with_accessible_storage_is_not_repeated(tmp_path):
    command = (
        f"bash -c 'n=$(cat {tmp_path}/runs 2>/dev/null || echo 0); "
        f"echo $((n+1)) > {tmp_path}/runs; exit 1'"
    )
    result = _bash(_runner_script(tmp_path, command, failures=0))
    assert "status=1 exit=1 runs=1" in result.stdout


FAKE_PYTHON = r"""#!/usr/bin/env bash
# Simula o Python da preparacao: a grade tem uma configuracao, e a primeira
# preparacao cai junto com o armazenamento.
case "$1" in
  *verify_datasets.py|*execution_provenance.py) exit 0 ;;
  *experiment_grid.py)
    printf 'ulb\tULB\tref-18/creditcard.csv\tClass\tfalse\tnative\tnone\tmean\tnone\tnone\ttemporal-holdout\t42\tnone\tnone\t0\n'
    exit 0 ;;
esac
md5=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
if [[ " $* " == *" --print-configuration-md5-only "* ]]; then
  echo "configuration_md5=${md5}"; exit 0
fi
runs_file="__RUNS__"
runs=$(cat "${runs_file}" 2>/dev/null || echo 0); runs=$((runs + 1)); echo "${runs}" > "${runs_file}"
if ((runs == 1)); then
  touch "__OFFLINE__"   # o HD cai no meio da gravacao
  echo "OSError: [Errno 112] Host is down" >&2
  exit 1
fi
echo "configuration_md5=${md5}"
echo "prepared_dataset=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb|1|test|cccccccccccccccccccccccccccccccc|x.csv|dddddddddddddddddddddddddddddddd|eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee|||"
"""


def test_interrupted_preparation_is_redone_and_published_once(tmp_path):
    fake_python = tmp_path / "fake_python"
    fake_python.write_text(
        FAKE_PYTHON.replace("__RUNS__", str(tmp_path / "runs")).replace(
            "__OFFLINE__", str(tmp_path / "offline")
        )
    )
    fake_python.chmod(0o755)
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    result = _bash(
        f"""
source prepare_grid_common.sh
offline_checks=0
storage_directory_is_accessible() {{
  if [[ -f "{tmp_path}/offline" ]]; then
    offline_checks=$((offline_checks + 1))
    # Volta depois de duas checagens.
    if ((offline_checks > 2)); then rm -f "{tmp_path}/offline"; return 0; fi
    return 1
  fi
  return 0
}}
prepare_grid_run native "{prepared}"
echo "status=$?"
""",
        {
            "PYTHON_COMMAND": str(fake_python),
            "MACHINE_STORAGE_CONFIG": str(tmp_path / "none.conf"),
            "MACHINE_PREPARED_DATA_DIRECTORY": str(prepared),
        },
    )
    assert "status=0" in result.stdout, result.stderr
    assert (tmp_path / "runs").read_text().strip() == "2"
    assert "a preparação perdeu o acesso ao armazenamento (código 1)" in result.stderr
    assert "acesso restabelecido" in result.stderr
    assert "Failed configurations: 0" in result.stdout
    manifest = (prepared / "prepared_dataset_configurations.tsv").read_text().splitlines()
    assert len(manifest) == 2
    assert manifest[1].split("\t")[10] == "a" * 32


def test_write_check_creates_the_directory_only_on_a_mounted_drive(tmp_path):
    # /tmp e uma montagem propria nesta maquina; o diretorio novo e criado.
    target = tmp_path / "prepared" / "nested"
    result = _bash(f'source data_directory_common.sh; storage_directory_is_accessible "{target}" write && echo ok')
    assert "ok" in result.stdout and target.is_dir()
    # Sob o sistema de arquivos raiz (um ponto de montagem vazio), nada e criado.
    root_target = Path("/var/claude-storage-probe-should-not-exist/prepared")
    result = _bash(f'source data_directory_common.sh; storage_directory_is_accessible "{root_target}" write || echo refused')
    assert "refused" in result.stdout and not root_target.exists()
