"""Painel de acompanhamento da execucao do plano da grade.

Mostra, por maquina do plano:

1. o que esta rodando: o par de cada lane e os demais processos (validacao,
   staging, preparacao);
2. o estado de cada lane: pares com sucesso, falhas e se o processo esta vivo;
3. o espaco livre nos discos (local e os dois HDs da rede) e a memoria;
4. o tempo gasto: por lane, por maquina e desde o inicio da fase, ao lado da
   estimativa do plano.

So usa a biblioteca padrao. A coleta de outra maquina roda este mesmo arquivo
por SSH (``python3 - collect``), sem precisar dele instalado la.

Uso:
  python3 monitor_execution.py --plan <plano>              # imprime uma vez
  python3 monitor_execution.py --plan <plano> --output f   # grava em f (atomico)
  watch -n 600 -t cat f                                    # em qualquer maquina que veja f
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

DEFAULT_ADDRESSES = {
    "marcospiaia-Nitro-ANV15-41": "192.168.0.194",
    "marcospiaia-Latitude-3490": "192.168.0.240",
    "marcospiaia-ThinkPad-E470": "192.168.0.223",
}
SHORT_NAMES = {
    "marcospiaia-Nitro-ANV15-41": "Acer",
    "marcospiaia-Latitude-3490": "Dell",
    "marcospiaia-ThinkPad-E470": "Lenovo",
}
DEFAULT_PIPELINE = "~/particular/github/unicamp/mestrado/tese/code/src/fraud_detection_revised_pipeline"
DISKS = ("/mnt/network-black-hd", "/mnt/network-transparent-hd")
EXECUTORS = (
    "lr_nb_knn_specific.py",
    "catboost_xgboost_specific.py",
    "anomaly_detection_specific.py",
    "run_transformer_prepared_experiments.py",
)
# Processos fora das lanes que vale mostrar (validacao, staging, preparacao).
OTHER_PROCESS_MARKERS = (
    "validate_prepared_datasets.py",
    "merge_distributed_preparation.py",
    "prepare_lr_nb_knn_datasets.py",
    "run_distributed_preparation_worker.sh",
    "stage_machine_configurations.sh",
    "calibrate_machine.py",
)
GIB = 1024**3


# ---------------------------------------------------------------------------
# Coleta (roda na maquina observada)
# ---------------------------------------------------------------------------


def _read(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError:
        return b""


def _processes() -> list[dict]:
    """Processos do usuario: pid, argumentos, GRID_LANE, idade e RSS."""
    uid = os.getuid()
    boot = 0.0
    for line in _read(Path("/proc/stat")).decode().splitlines():
        if line.startswith("btime "):
            boot = float(line.split()[1])
    ticks = os.sysconf("SC_CLK_TCK")
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            if entry.stat().st_uid != uid:
                continue
        except OSError:
            continue
        arguments = [part.decode(errors="replace") for part in _read(entry / "cmdline").split(b"\0") if part]
        if not arguments:
            continue
        environment = dict(
            part.decode(errors="replace").split("=", 1)
            for part in _read(entry / "environ").split(b"\0")
            if b"=" in part
        )
        stat = _read(entry / "stat").decode(errors="replace")
        started = boot + int(stat.rsplit(")", 1)[1].split()[19]) / ticks if ")" in stat else time.time()
        rss = 0
        for line in _read(entry / "status").decode(errors="replace").splitlines():
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) * 1024
        found.append({
            "pid": int(entry.name),
            "arguments": arguments,
            "lane": environment.get("GRID_LANE", ""),
            "seconds": max(0.0, time.time() - started),
            "rss": rss,
        })
    return found


def _option(arguments: list[str], name: str) -> str:
    for position, value in enumerate(arguments[:-1]):
        if value == name:
            return arguments[position + 1]
    return ""


def _describe_executor(process: dict) -> dict:
    arguments = process["arguments"]
    return {
        "algorithm": _option(arguments, "--algorithm") or _option(arguments, "--model"),
        "dataset": _option(arguments, "--dataset-id"),
        "configuration": (_option(arguments, "--prepared-dataset-md5") or _option(arguments, "--configuration-md5"))[:8],
        "seconds": process["seconds"],
        "rss": process["rss"],
    }


def _lane_results(lane_directory: Path) -> tuple[set, set]:
    """Pares (md5, algoritmo) com sucesso e com falha nos resultados da lane."""
    succeeded, failed = set(), set()
    for path in lane_directory.glob("resultados_*.csv"):
        try:
            with path.open(newline="", encoding="utf-8", errors="replace") as handle:
                for row in csv.DictReader(handle):
                    key = (row.get("configuration_md5", "").lower(), row.get("algorithm", ""))
                    if row.get("execution_status") == "success":
                        succeeded.add(key)
                    else:
                        failed.add(key)
        except (OSError, csv.Error):
            continue
    return succeeded, failed - succeeded


def _lane_times(lane_directory: Path, configurations: set | None = None) -> tuple[str, str, int]:
    """Primeiro inicio, ultimo fim e tentativas com codigo de saida != 0.

    Com ``configurations``, so conta as linhas dessas configuracoes: uma lane
    de outra fase com o mesmo nome escreve no mesmo registro, e as linhas da
    fase anterior inflariam o tempo decorrido. Filtra pelo MD5 e nao pelo
    algoritmo porque o registro grava o bloco (``catboost xgboost``) ou deixa
    o algoritmo vazio em parte das linhas.
    """
    path = lane_directory / "registro_execucoes.csv"
    first, last, nonzero = "", "", 0
    try:
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            for row in csv.DictReader(handle):
                if configurations is not None and row.get("configuration_md5", "").lower() not in configurations:
                    continue
                start, end = row.get("timestamp_inicio", ""), row.get("timestamp_fim", "")
                if start and (not first or start < first):
                    first = start
                if end and end > last:
                    last = end
                if row.get("codigo_saida", "0") not in ("0", ""):
                    nonzero += 1
    except (OSError, csv.Error):
        pass
    return first, last, nonzero


def _disk(path: str) -> dict:
    # Um CIFS travado nao pode travar o painel: df com timeout.
    try:
        output = subprocess.run(
            ["timeout", "10", "df", "-B1", "--output=size,avail", path],
            capture_output=True, text=True, check=False,
        ).stdout.split("\n")[1].split()
        return {"path": path, "size": int(output[0]), "free": int(output[1])}
    except (IndexError, ValueError, OSError):
        return {"path": path, "size": 0, "free": -1}


def _memory() -> dict:
    values = {}
    for line in _read(Path("/proc/meminfo")).decode().splitlines():
        name, _, rest = line.partition(":")
        values[name] = int(rest.split()[0]) * 1024 if rest.split() else 0
    return {"total": values.get("MemTotal", 0), "available": values.get("MemAvailable", 0)}


def collect(plan_directory: Path, pipeline_directory: Path) -> dict:
    host = socket.gethostname()
    plan = json.loads((plan_directory / "grid_plan.json").read_text(encoding="utf-8"))
    machine = plan["machines"].get(host, {})
    processes = _processes()
    running_lanes = {
        process["arguments"][-1]
        for process in processes
        if any(argument.endswith("run_lane.sh") for argument in process["arguments"])
    }
    executors = {
        process["lane"]: _describe_executor(process)
        for process in processes
        if process["lane"] and any(Path(argument).name in EXECUTORS for argument in process["arguments"])
    }
    lanes = []
    for lane in sorted(machine.get("lanes", {})):
        assignments = plan_directory / "lanes" / lane / "assignments.tsv"
        pairs = set()
        if assignments.is_file():
            for line in assignments.read_text(encoding="utf-8").splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    pairs.add((parts[1].lower(), parts[0]))
        lane_directory = pipeline_directory / "lanes" / lane / "results"
        succeeded, failed = _lane_results(lane_directory)
        first, last, nonzero = _lane_times(lane_directory, {md5 for md5, _ in pairs})
        lanes.append({
            "lane": lane,
            "pairs": len(pairs),
            "succeeded": len(succeeded & pairs),
            "failed": len(failed & pairs),
            "nonzero_attempts": nonzero,
            "running": lane in running_lanes or lane in executors,
            "current": executors.get(lane),
            "first_start": first,
            "last_end": last,
            "estimated_seconds": float(machine["lanes"][lane].get("estimated_seconds", 0.0)),
        })
    others = []
    for process in processes:
        if process["lane"]:
            continue
        text = " ".join(process["arguments"])
        marker = next((name for name in OTHER_PROCESS_MARKERS if name in text), None)
        if marker and not text.startswith(("tmux", "bash -c", "/bin/bash -c", "tee")):
            others.append({"name": marker, "seconds": process["seconds"], "rss": process["rss"]})
    tmux = subprocess.run(
        ["tmux", "list-sessions", "-F", "#{session_name}"], capture_output=True, text=True, check=False
    ).stdout.split()
    return {
        "host": host,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "lanes": lanes,
        "others": others,
        "tmux": tmux,
        "disks": [_disk(str(pipeline_directory)), *(_disk(path) for path in DISKS)],
        "memory": _memory(),
    }


# ---------------------------------------------------------------------------
# Painel (roda em quem consegue acessar todas as maquinas)
# ---------------------------------------------------------------------------


def _remote(address: str, plan: str, pipeline: str) -> dict:
    source = Path(__file__).read_text(encoding="utf-8")
    command = [
        "ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", f"marcospiaia@{address}",
        f"python3 - collect --plan {plan} --pipeline {pipeline}",
    ]
    result = subprocess.run(command, input=source, capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr.strip().splitlines() or ["sem resposta"])[-1])
    return json.loads(result.stdout)


def _duration(seconds: float) -> str:
    if seconds <= 0:
        return "-"
    minutes = int(seconds // 60)
    days, minutes = divmod(minutes, 24 * 60)
    hours, minutes = divmod(minutes, 60)
    return f"{days}d{hours:02d}h{minutes:02d}" if days else f"{hours}h{minutes:02d}"


def _since(timestamp: str) -> float:
    try:
        return time.time() - datetime.fromisoformat(timestamp).timestamp()
    except ValueError:
        return 0.0


def _between(start: str, end: str) -> float:
    try:
        return datetime.fromisoformat(end).timestamp() - datetime.fromisoformat(start).timestamp()
    except ValueError:
        return 0.0


def _gib(value: int) -> str:
    return "?" if value < 0 else f"{value / GIB:.0f}G"


def render(plan_directory: Path, states: dict) -> str:
    lines = [
        f"Plano {plan_directory}   atualizado {datetime.now():%d/%m %H:%M}",
        "",
    ]
    starts, total_pairs, total_succeeded, total_failed = [], 0, 0, 0
    for host, state in states.items():
        name = SHORT_NAMES.get(host, host)
        if isinstance(state, str):
            lines += [f"== {name} ({host}): INACESSIVEL - {state}", ""]
            continue
        memory = state["memory"]
        labels = ("local", "black-hd", "transparent-hd")
        disks = "  ".join(
            f"{label} {_gib(disk['free'])}/{_gib(disk['size'])}" for label, disk in zip(labels, state["disks"])
        )
        lines.append(
            f"== {name}   RAM livre {_gib(memory['available'])}/{_gib(memory['total'])}   disco livre: {disks}"
        )
        lines.append(
            f"   {'lane':<10}{'estado':<13}{'sucesso':>11}{'falhas':>8}  {'decorrido':>10}{'estimado':>10}  em execucao"
        )
        for lane in state["lanes"]:
            total_pairs += lane["pairs"]
            total_succeeded += lane["succeeded"]
            total_failed += lane["failed"]
            if lane["first_start"]:
                starts.append(lane["first_start"])
            if lane["pairs"] and lane["succeeded"] + lane["failed"] >= lane["pairs"] and not lane["running"]:
                status = "concluida"
            elif lane["running"]:
                status = "rodando"
            elif lane["first_start"]:
                status = "PARADA"
            else:
                status = "nao iniciada"
            if not lane["pairs"]:
                status = "sem pares"
            elapsed = (
                _since(lane["first_start"]) if lane["running"]
                else _between(lane["first_start"], lane["last_end"])
            ) if lane["first_start"] else 0.0
            current = lane["current"]
            doing = (
                f"{current['algorithm']} {current['dataset']} {current['configuration']}"
                f" ha {_duration(current['seconds'])}, {current['rss'] / GIB:.1f} GiB"
                if current else ""
            )
            short_lane = lane["lane"].removeprefix(f"{host}-")
            lines.append(
                f"   {short_lane:<10}{status:<13}{lane['succeeded']:>5}/{lane['pairs']:<5}"
                f"{lane['failed']:>8}  {_duration(elapsed):>10}{_duration(lane['estimated_seconds']):>10}  {doing}"
            )
        for other in state["others"]:
            lines.append(f"   + {other['name']} ha {_duration(other['seconds'])}, {other['rss'] / GIB:.1f} GiB")
        if state["tmux"]:
            lines.append(f"   tmux: {', '.join(state['tmux'])}")
        lines.append("")
    progress = 100 * total_succeeded / total_pairs if total_pairs else 0.0
    phase = _duration(_since(min(starts))) if starts else "-"
    lines.append(
        f"Fase: {total_succeeded}/{total_pairs} pares com sucesso ({progress:.1f}%), "
        f"{total_failed} com falha; tempo desde o inicio {phase}"
    )
    return "\n".join(lines) + "\n"


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", nargs="?", choices=["show", "collect"], default="show")
    parser.add_argument(
        "--plan", action="append", required=True,
        help="diretorio do plano; repita para mostrar varias fases no mesmo painel (no modo collect, so um)",
    )
    parser.add_argument("--pipeline", default=DEFAULT_PIPELINE)
    parser.add_argument("--output", help="grava o painel neste arquivo em vez de imprimir")
    parser.add_argument(
        "--address", action="append", default=[], metavar="HOST=IP", help="endereco SSH de um host do plano"
    )
    return parser


def main() -> int:
    arguments = build_argument_parser().parse_args()
    pipeline = Path(os.path.expanduser(arguments.pipeline))
    if arguments.mode == "collect":
        print(json.dumps(collect(Path(arguments.plan[-1]), pipeline)))
        return 0
    addresses = {**DEFAULT_ADDRESSES, **dict(item.split("=", 1) for item in arguments.address)}
    local = socket.gethostname()
    sections = []
    for plan_argument in arguments.plan:
        plan_directory = Path(plan_argument)
        plan = json.loads((plan_directory / "grid_plan.json").read_text(encoding="utf-8"))
        states: dict[str, dict | str] = {}
        for host in sorted(plan["machines"], key=lambda name: SHORT_NAMES.get(name, name)):
            try:
                states[host] = (
                    collect(plan_directory, pipeline) if host == local
                    else _remote(addresses.get(host, host), str(plan_directory), arguments.pipeline)
                )
            except Exception as error:  # noqa: BLE001 - uma maquina fora do ar nao derruba o painel
                states[host] = str(error)
        sections.append(render(plan_directory, states))
    text = ("\n" + "=" * 100 + "\n\n").join(sections)
    if arguments.output:
        output = Path(arguments.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.name}.tmp")
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, output)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
