"""Calibracao de velocidade e memoria de uma maquina (item G18).

A reexecucao roda no Acer Nitro ANV15-41, num Lenovo ThinkPad E470 e num Dell
Latitude 3490. As estimativas de ``estimate_execution_time.py`` foram
medidas no Acer; para dividir a grade, e preciso saber quanto mais lenta e
cada maquina, e a diferenca nao e a mesma em tudo: o KNN depende da memoria,
o boosting roda com uma thread, o DevNet e uma rede pequena.

A calibracao roda a mesma carga curta e fixa, sobre dados sinteticos gerados
com semente fixa, para cada familia de algoritmo, cada uma num subprocesso
proprio, e grava:

* o tempo da carga, do qual a alocacao tira o fator de velocidade em relacao
  ao Acer;
* o pico de memoria residente em dois tamanhos de carga, do qual sai uma reta
  ``memoria = fixa + por_byte x bytes de CSV``: a parte fixa e o custo das
  bibliotecas (so o PyTorch passa de 800 MB), e a parte proporcional e o que a
  alocacao aplica ao tamanho dos arquivos preparados de cada configuracao;
* a RAM, a GPU e o espaco livre no diretorio de dados preparados;
* se esse diretorio e lido pela rede (CIFS, SMB ou NFS): o plano cobra a
  leitura das maquinas que leem os dados pela rede;
* quantos processos a maquina roda ao mesmo tempo (``--cpu-lanes`` e
  ``--gpu-lanes``). Com mais de um processo de CPU, o tempo usado no fator de
  velocidade e medido com essa quantidade de copias da carga rodando juntas:
  processos simultaneos disputam nucleos, cache e temperatura, e o fator medido
  com um processo so seria otimista.

Saida: ``calibration_<host>.json`` no diretorio de calibracao.

Uso:
    python3 calibrate_machine.py --prepared-data-directory <dir> \\
        [--cpu-lanes N] [--gpu-lanes N] [--output-directory <dir>]
"""

from __future__ import annotations

import argparse
import json
import platform
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

import execution_provenance

FAMILIES = ("linear", "knn", "boosting", "anomaly")
CALIBRATION_ROWS = 20_000
# Dois tamanhos para separar a memoria fixa da proporcional aos dados.
CALIBRATION_SIZES = (10_000, 40_000)
CALIBRATION_SCORED_ROWS = 5_000
CALIBRATION_FEATURES = 30
CALIBRATION_POSITIVE_RATE = 0.01
CALIBRATION_SEED = 42
TARGET = "target"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "calibration"


def calibration_frame(rows: int = CALIBRATION_ROWS) -> pd.DataFrame:
    """Dados sinteticos fixos: 30 numericos, 1 categorico e o alvo."""
    rng = np.random.default_rng(CALIBRATION_SEED)
    target = (rng.random(rows) < CALIBRATION_POSITIVE_RATE).astype(int)
    features = rng.normal(size=(rows, CALIBRATION_FEATURES))
    features[:, 0] += 2.0 * target
    frame = pd.DataFrame(features, columns=[f"v{index}" for index in range(CALIBRATION_FEATURES)])
    frame["channel"] = rng.choice(["web", "pos", "atm", "app"], size=rows)
    frame[TARGET] = target
    return frame


def calibration_csv_bytes(frame: pd.DataFrame) -> int:
    return len(frame.to_csv(index=False).encode("utf-8"))


def _run_linear(frame):
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.preprocessing import StandardScaler

    features = frame.drop(columns=[TARGET, "channel"])
    scaled = StandardScaler().fit_transform(features)
    LogisticRegression(max_iter=2000, random_state=42).fit(scaled, frame[TARGET]).predict_proba(scaled)
    GaussianNB().fit(features, frame[TARGET]).predict_proba(features)


def _run_knn(frame):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    features = StandardScaler().fit_transform(frame.drop(columns=[TARGET, "channel"]))
    model = KNeighborsClassifier(n_neighbors=5).fit(features, frame[TARGET])
    model.predict_proba(features[:CALIBRATION_SCORED_ROWS])


def _run_boosting(frame):
    from catboost_xgboost_specific import (
        build_xgboost_category_domains,
        create_catboost_classifier,
        create_xgboost_classifier,
        prepare_catboost_dataframe,
        prepare_xgboost_dataframe,
    )

    features = frame.drop(columns=[TARGET])
    categorical = ["channel"]
    catboost = create_catboost_classifier(categorical)
    catboost.fit(prepare_catboost_dataframe(features, categorical), frame[TARGET])
    domains = build_xgboost_category_domains(features, categorical)
    xgboost = create_xgboost_classifier(imbalance_ratio=99.0)
    xgboost.fit(prepare_xgboost_dataframe(features, categorical, domains), frame[TARGET])


def _run_anomaly(frame):
    from anomaly_detection_specific import create_isolation_forest
    from devnet_model import DEFAULT_DEVNET_SETTINGS, train_devnet

    features = frame.drop(columns=[TARGET, "channel"]).to_numpy(dtype=np.float32, copy=True)
    create_isolation_forest().fit(features).score_samples(features)
    labeled = np.flatnonzero(frame[TARGET].to_numpy() == 1)[:20]
    train_devnet(features, labeled, DEFAULT_DEVNET_SETTINGS)


FAMILY_WORKLOADS = {
    "linear": _run_linear,
    "knn": _run_knn,
    "boosting": _run_boosting,
    "anomaly": _run_anomaly,
}


def _peak_rss_bytes() -> int:
    # ru_maxrss vem em kB no Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def run_family(family: str, rows: int) -> dict:
    """Roda a carga de uma familia neste processo e mede tempo e memoria."""
    # Linha de base antes dos dados e das bibliotecas da familia: numa
    # execucao real, os dados carregados e as importacoes tambem ocupam a RAM.
    baseline = _peak_rss_bytes()
    frame = calibration_frame(rows)
    data_bytes = calibration_csv_bytes(frame)
    started_at = time.perf_counter()
    FAMILY_WORKLOADS[family](frame)
    seconds = time.perf_counter() - started_at
    return {
        "rows": rows,
        "seconds": round(seconds, 6),
        "peak_rss_increase_bytes": max(0, _peak_rss_bytes() - baseline),
        "data_csv_bytes": data_bytes,
    }


def memory_line(small: dict, large: dict) -> tuple[float, float]:
    """Memoria fixa e por byte de CSV, pela reta entre os dois tamanhos."""
    byte_delta = large["data_csv_bytes"] - small["data_csv_bytes"]
    per_byte = max(
        0.0, (large["peak_rss_increase_bytes"] - small["peak_rss_increase_bytes"]) / byte_delta
    )
    fixed = max(0.0, small["peak_rss_increase_bytes"] - per_byte * small["data_csv_bytes"])
    return fixed, per_byte


def _measure_in_subprocess(family: str, rows: int) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json") as handle:
        subprocess.run(
            [sys.executable, __file__, "--run-family", family, "--rows", str(rows),
             "--family-output", handle.name],
            check=True,
        )
        return json.loads(Path(handle.name).read_text(encoding="utf-8"))


def _measure_concurrently(family: str, rows: int, copies: int) -> dict:
    """Tempo medio de ``copies`` processos rodando a mesma carga ao mesmo tempo."""
    with tempfile.TemporaryDirectory() as directory:
        outputs = [Path(directory) / f"copy{index}.json" for index in range(copies)]
        processes = [
            subprocess.Popen(
                [sys.executable, __file__, "--run-family", family, "--rows", str(rows),
                 "--family-output", str(output)]
            )
            for output in outputs
        ]
        for process in processes:
            if process.wait() != 0:
                raise subprocess.CalledProcessError(process.returncode, process.args)
        measurements = [json.loads(output.read_text(encoding="utf-8")) for output in outputs]
    return {
        "copies": copies,
        "seconds": round(sum(item["seconds"] for item in measurements) / copies, 6),
        "measurements": measurements,
    }


NETWORK_FILESYSTEMS = frozenset({"cifs", "smb3", "smbfs", "nfs", "nfs4", "fuse.sshfs"})


def storage_is_network(directory: Path) -> bool:
    """True quando o diretorio esta num sistema de arquivos de rede."""
    try:
        filesystem = subprocess.run(
            ["findmnt", "-n", "-o", "FSTYPE", "-T", str(directory)],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return False
    return filesystem in NETWORK_FILESYSTEMS


def calibrate(
    prepared_directory: Path,
    output_directory: Path,
    families=FAMILIES,
    cpu_lanes: int = 1,
    gpu_lanes: int | None = None,
) -> Path:
    results = {}
    for family in families:
        small, large = (_measure_in_subprocess(family, rows) for rows in CALIBRATION_SIZES)
        fixed, per_byte = memory_line(small, large)
        results[family] = {
            # O fator de velocidade usa a carga maior.
            "seconds": large["seconds"],
            "memory_fixed_bytes": round(fixed),
            "memory_per_csv_byte": round(per_byte, 6),
            "measurements": [small, large],
        }
        if cpu_lanes > 1:
            # Com processos simultaneos, o fator vem da carga rodando junto.
            concurrent = _measure_concurrently(family, CALIBRATION_SIZES[-1], cpu_lanes)
            results[family]["single_process_seconds"] = large["seconds"]
            results[family]["seconds"] = concurrent["seconds"]
            results[family]["concurrent"] = concurrent
        print(
            f"{family}: {results[family]['seconds']:.2f}s ({cpu_lanes} processo(s)), "
            f"{fixed / 2**20:.0f} MiB + {per_byte:.1f} x CSV"
        )
    provenance = execution_provenance.collect_provenance()
    has_gpu = provenance.get("gpu_model", "none") not in ("", "none", None)
    calibration = {
        **provenance,
        "families": results,
        "cpu_lanes": cpu_lanes,
        "gpu_lanes": (1 if has_gpu else 0) if gpu_lanes is None else (gpu_lanes if has_gpu else 0),
        "disk_free_bytes": shutil.disk_usage(prepared_directory).free,
        "prepared_data_directory": str(prepared_directory).rstrip("/"),
        "storage_is_network": storage_is_network(prepared_directory),
        "workload": {
            "rows": list(CALIBRATION_SIZES),
            "features": CALIBRATION_FEATURES,
            "positive_rate": CALIBRATION_POSITIVE_RATE,
            "seed": CALIBRATION_SEED,
        },
    }
    output_directory.mkdir(parents=True, exist_ok=True)
    path = output_directory / f"calibration_{provenance['execution_host'] or platform.node()}.json"
    path.write_text(json.dumps(calibration, indent=2, sort_keys=True), encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepared-data-directory")
    parser.add_argument("--output-directory", default=str(DEFAULT_OUTPUT_DIRECTORY))
    parser.add_argument("--family", nargs="+", default=list(FAMILIES), choices=FAMILIES)
    parser.add_argument("--cpu-lanes", type=int, default=1, help="Processos de CPU simultaneos.")
    parser.add_argument(
        "--gpu-lanes", type=int, default=None,
        help="Processos de GPU simultaneos (padrao: 1 com GPU, 0 sem).",
    )
    # Modo interno: uma familia por subprocesso, para isolar o pico de memoria.
    parser.add_argument("--run-family", choices=FAMILIES, help=argparse.SUPPRESS)
    parser.add_argument("--family-output", help=argparse.SUPPRESS)
    parser.add_argument("--rows", type=int, default=CALIBRATION_ROWS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.run_family:
        Path(args.family_output).write_text(
            json.dumps(run_family(args.run_family, args.rows)), encoding="utf-8"
        )
        return 0
    if not args.prepared_data_directory:
        parser.error("--prepared-data-directory is required")
    execution_provenance.guard_environment_or_exit()
    if args.cpu_lanes < 1:
        parser.error("--cpu-lanes must be at least 1")
    print(
        calibrate(
            Path(args.prepared_data_directory),
            Path(args.output_directory),
            args.family,
            args.cpu_lanes,
            args.gpu_lanes,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
