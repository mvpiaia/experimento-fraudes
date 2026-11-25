"""Item G17: registro de execucoes gravado pelos executores shell."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
MD5 = "0123456789abcdef0123456789abcdef"


def _run(script: str, tmp_path: Path) -> pd.DataFrame:
    """Roda ``script`` com run_storage_common.sh carregado e le o registro."""
    preamble = f"""
source "{PIPELINE_DIRECTORY}/run_storage_common.sh"
RESULTS_DIRECTORY="{tmp_path}"
EXECUTION_LOG_BLOCK="classic"
# A recuperacao do HD nao e o objeto do teste.
recover_external_storage() {{ return 0; }}
wait_for_configuration_files() {{ return 0; }}
configuration_files_are_available() {{ return 0; }}
"""
    subprocess.run(["bash", "-c", preamble + script], check=True, cwd=tmp_path)
    return pd.read_csv(tmp_path / "registro_execucoes.csv", dtype=str, keep_default_na=False)


def test_process_killed_by_oom_is_logged_as_memory(tmp_path):
    log = _run(
        f"""
set_execution_log_context knn default
run_with_storage_recovery {MD5} bash -c 'exit 137' || true
""",
        tmp_path,
    )
    assert len(log) == 1
    row = log.iloc[0]
    assert (row["bloco"], row["configuration_md5"], row["algoritmo"]) == ("classic", MD5, "knn")
    assert (row["codigo_saida"], row["categoria_saida"], row["tentativa"]) == ("137", "memoria", "1")


def test_storage_retry_logs_each_attempt(tmp_path):
    # Primeira tentativa sai com o codigo de armazenamento; a segunda passa.
    log = _run(
        f"""
set_execution_log_context logistic-regression balanced
marker="{tmp_path}/tentou"
run_with_storage_recovery {MD5} bash -c '[[ -f "$0" ]] && exit 0; touch "$0"; exit 75' "$marker"
""",
        tmp_path,
    )
    assert list(log["tentativa"]) == ["1", "2"]
    assert list(log["categoria_saida"]) == ["armazenamento", "sucesso"]
    assert set(log["class_weight_intensity"]) == {"balanced"}


def test_python_error_and_label_fractions(tmp_path):
    log = _run(
        f"""
EXECUTION_LOG_BLOCK=anomaly
set_execution_log_context devnet "" "0.01 0.05 0.10"
run_with_storage_recovery {MD5} bash -c 'exit 1' || true
""",
        tmp_path,
    )
    row = log.iloc[0]
    assert (row["bloco"], row["label_fraction"], row["categoria_saida"]) == (
        "anomaly", "0.01 0.05 0.10", "erro-python"
    )
