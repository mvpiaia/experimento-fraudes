"""Juncao dos resultados das tres maquinas (item G18 da revisao).

Cada maquina (ou cada processo de uma maquina) grava os mesmos
``resultados_<algoritmo>.csv``; a coleta, ou o backup periodico dos processos,
os traz para ``<coleta>/<host ou processo>/``: cada subdiretorio e uma fonte. A
juncao concatena os arquivos por algoritmo e e estrita, porque e dela que
saem os numeros da dissertacao:

* a mesma execucao bem-sucedida (mesma chave de retomada, dobra e split) em
  duas linhas e erro: a juncao para e lista as duplicatas, em vez de escolher
  uma;
* uma linha gravada numa maquina diferente da que o plano atribuiu ao par e
  relatada;
* um par (algoritmo, configuracao) ou braco do plano sem linha de sucesso e
  listado como lacuna, por maquina.

Os arquivos companheiros (``resultados_*_details.jsonl`` dos Transformers,
``resultados_*_history_strata.jsonl`` do boosting e
``resultados_*_test_scores.jsonl`` dos escores do teste, item G20) sao
concatenados linha a linha, na ordem das fontes.

A saida, um diretorio com os ``resultados_<algoritmo>.csv`` combinados, e a
entrada do verificador do G15 (``preparation_sanity_checks.py``), das tabelas
do G18 (``build_reproducibility_tables.py``) e do dashboard.

Uso:
    python3 merge_machine_results.py --plan-directory <dir> \\
        --collection-directory <dir> --output-directory <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from grid_plan import TABBERT_REGIMES, read_plan

# A chave de retomada mais a dobra, o split e a unidade: o que identifica uma
# linha de resultado.
ROW_KEY_COLUMNS = (
    "configuration_md5",
    "algorithm",
    "regime",
    "class_weight_intensity",
    "label_fraction",
    "execution_seed",
    "threshold_strategy",
    "fold_index",
    "split",
    "evaluation_unit",
)
CLASS_WEIGHT_INTENSITIES = ("none", "sqrt-balanced", "balanced")
DEVNET_LABEL_FRACTIONS = (0.01, 0.05, 0.10)
DEVNET_DRAW_SEEDS = (0, 1, 2, 3, 4)
DUPLICATES_EXIT_CODE = 1
REPORT_FILENAME = "merge_report.json"


class DuplicateRowsError(ValueError):
    def __init__(self, duplicates: pd.DataFrame):
        self.duplicates = duplicates
        super().__init__(f"{len(duplicates)} duplicated successful rows")


def read_collection(collection_directory: Path) -> dict[str, pd.DataFrame]:
    """Linhas de cada arquivo de resultado, de todas as maquinas, com a origem."""
    frames: dict[str, list[pd.DataFrame]] = {}
    for host_directory in sorted(path for path in collection_directory.iterdir() if path.is_dir()):
        for path in sorted(host_directory.glob("resultados_*.csv")):
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame["collected_from"] = host_directory.name
            frames.setdefault(path.name, []).append(frame)
    return {name: pd.concat(parts, ignore_index=True) for name, parts in sorted(frames.items())}


COMPANION_PATTERNS = (
    "resultados_*_details.jsonl",
    "resultados_*_history_strata.jsonl",
    "resultados_*_test_scores.jsonl",
)


def merge_companion_files(collection_directory: Path, output_directory: Path) -> list[str]:
    """Concatena os .jsonl companheiros de todas as fontes, por nome de arquivo."""
    parts: dict[str, list[Path]] = {}
    for source_directory in sorted(path for path in collection_directory.iterdir() if path.is_dir()):
        for pattern in COMPANION_PATTERNS:
            for path in sorted(source_directory.glob(pattern)):
                parts.setdefault(path.name, []).append(path)
    for name, paths in sorted(parts.items()):
        with (output_directory / name).open("w", encoding="utf-8") as output:
            for path in paths:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        output.write(line + "\n")
    return sorted(parts)


def find_duplicates(results: pd.DataFrame) -> pd.DataFrame:
    success = results[results["execution_status"] == "success"]
    key = [column for column in ROW_KEY_COLUMNS if column in success.columns]
    duplicated = success[success.duplicated(key, keep=False)]
    return duplicated[key + ["execution_host", "collected_from"]].sort_values(key)


def expected_arms(algorithm: str, without_resampling: bool) -> list[dict]:
    """Bracos que o plano espera de um par (algoritmo, configuracao)."""
    if algorithm in {"logistic-regression", "catboost", "xgboost"} and without_resampling:
        return [{"class_weight_intensity": intensity} for intensity in CLASS_WEIGHT_INTENSITIES]
    if algorithm == "devnet":
        return [
            {"label_fraction": fraction, "execution_seed": seed}
            for fraction in DEVNET_LABEL_FRACTIONS
            for seed in DEVNET_DRAW_SEEDS
        ]
    if algorithm.startswith("tabbert"):
        return [{"regime": regime} for regime in TABBERT_REGIMES]
    return [{}]


def _matches(rows: pd.DataFrame, arm: dict) -> pd.DataFrame:
    for column, value in arm.items():
        if column not in rows.columns:
            return rows.iloc[0:0]
        if isinstance(value, float):
            rows = rows[pd.to_numeric(rows[column], errors="coerce").round(6) == round(value, 6)]
        else:
            rows = rows[rows[column].astype(str) == str(value)]
    return rows


def check_plan(plan_directory: Path, plan: dict, results: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """Lacunas do plano e linhas gravadas fora da maquina planejada."""
    success = results[results["execution_status"] == "success"]
    gaps, unplanned = [], []
    for host, machine in sorted(plan["machines"].items()):
        manifest_path = plan_directory / f"manifest_{host}.tsv"
        manifest = (
            pd.read_csv(manifest_path, sep="\t", dtype=str, keep_default_na=False)
            .drop_duplicates("configuration_md5")
            .set_index("configuration_md5")
            if manifest_path.exists()
            else pd.DataFrame()
        )
        for pair in machine["pairs"]:
            for md5 in pair["configurations"]:
                rows = success[
                    (success["algorithm"] == pair["algorithm"])
                    & (success["configuration_md5"].str.lower() == md5.lower())
                ]
                for row_host in sorted(set(rows.get("execution_host", pd.Series(dtype=str))) - {host, ""}):
                    unplanned.append(
                        {"algorithm": pair["algorithm"], "configuration_md5": md5,
                         "planned_host": host, "execution_host": row_host}
                    )
                resampling = str(manifest.loc[md5, "undersampling_technique"]) if md5 in manifest.index else ""
                without_resampling = resampling.strip().lower() in {"", "none", "nan"}
                for arm in expected_arms(pair["algorithm"], without_resampling):
                    if _matches(rows, arm).empty:
                        gaps.append(
                            {"host": host, "algorithm": pair["algorithm"], "dataset_id": pair["dataset_id"],
                             "configuration_md5": md5, "arm": json.dumps(arm, sort_keys=True)}
                        )
    return gaps, unplanned


def merge(plan_directory: Path, collection_directory: Path, output_directory: Path) -> dict:
    plan = read_plan(plan_directory)
    collected = read_collection(collection_directory)
    all_rows = pd.concat(collected.values(), ignore_index=True) if collected else pd.DataFrame(
        columns=["execution_status", "algorithm", "configuration_md5"]
    )
    duplicates = find_duplicates(all_rows) if not all_rows.empty else pd.DataFrame()
    if not duplicates.empty:
        raise DuplicateRowsError(duplicates)
    gaps, unplanned = check_plan(plan_directory, plan, all_rows)
    output_directory.mkdir(parents=True, exist_ok=True)
    for name, frame in collected.items():
        frame.drop(columns=["collected_from"]).to_csv(output_directory / name, index=False)
    companion_files = merge_companion_files(collection_directory, output_directory)
    report = {
        "plan_commit": plan.get("code_commit"),
        "files": sorted(collected),
        "companion_files": companion_files,
        "rows": int(len(all_rows)),
        "complete": not gaps,
        "gaps": gaps,
        "unplanned_rows": unplanned,
    }
    (output_directory / REPORT_FILENAME).write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--plan-directory", required=True)
    parser.add_argument("--collection-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    try:
        report = merge(Path(args.plan_directory), Path(args.collection_directory), Path(args.output_directory))
    except DuplicateRowsError as error:
        print("Duplicated successful rows; nothing was written:", file=sys.stderr)
        print(error.duplicates.to_string(index=False), file=sys.stderr)
        return DUPLICATES_EXIT_CODE
    print(f"Merged {report['rows']} rows from {len(report['files'])} files.")
    if report["unplanned_rows"]:
        print(f"{len(report['unplanned_rows'])} pairs ran on a machine other than the planned one.")
    if report["gaps"]:
        print(f"Incomplete grid: {len(report['gaps'])} planned arms without a successful row.")
        return 0
    print("Grid complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
