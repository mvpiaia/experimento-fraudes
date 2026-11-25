"""Tabelas de reprodutibilidade da dissertacao (item G18 da revisao).

Gera, a partir dos resultados gravados, e nunca por transcricao manual:

* hiperparametros efetivos por algoritmo, lidos de ``model_hyperparameters``
  (inclusive os padroes resolvidos pela biblioteca); um valor que muda entre
  bases aparece como faixa, com as bases; os bracos de execucao (ponderacao de
  classe, fracao de rotulos, estrategia de limiar) ficam a parte;
* tempos de ajuste e de pontuacao por dobra, por algoritmo, base e maquina
  (mediana e intervalo interquartil), nunca agregando maquinas;
* tempos de preparacao por configuracao, dos arquivos
  ``<configuration_md5>.preparation_timing.json``;
* o benchmark de inferencia, que precisa vir de uma maquina so.

As saidas sao LaTeX (apendice da dissertacao) e Markdown (validacao interna),
com os arquivos lidos e o commit. A saida e deterministica: as mesmas entradas
produzem os mesmos arquivos.

Uso:
    python3 build_reproducibility_tables.py --results-directory <dir> \\
        [--prepared-data-directory <dir> --prepared-dataset-manifest <tsv>] \\
        [--benchmark-file <csv>] --latex-directory <dir> --markdown-output <md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import execution_provenance
from preparation_timing import PREPARATION_TIMING_FILE_SUFFIX
from prepared_dataset_storage import normalize_manifest_dataset_identity

# Colunas que identificam uma dobra executada; o split fica de fora porque os
# tempos sao da dobra, repetidos em cada split.
FOLD_KEY_COLUMNS = (
    "configuration_md5",
    "algorithm",
    "regime",
    "fold_index",
    "class_weight_intensity",
    "label_fraction",
    "execution_seed",
    "threshold_strategy",
)
MACHINE_COLUMNS = ("execution_host", "cpu_model", "gpu_model")
# Hiperparametros que sao, na verdade, bracos de execucao ou derivados dos dados.
ARM_HYPERPARAMETERS = frozenset(
    {"auto_class_weights", "class_weights", "scale_pos_weight", "class_weight"}
)
DATA_DERIVED_HYPERPARAMETERS = frozenset({"cat_features"})
# Parametros de registro e de saida, sem efeito no modelo.
IGNORED_HYPERPARAMETERS = frozenset(
    {"verbose", "verbosity", "allow_writing_files", "logging_level", "train_dir"}
)
ARM_COLUMNS = ("class_weight_intensity", "label_fraction", "threshold_strategy")


class MixedMachinesError(ValueError):
    """Linhas de benchmark de mais de uma maquina."""


def _read_results(results_directory: Path) -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(
        path
        for path in results_directory.glob("resultados_*.csv")
        if path.name != "resultados_benchmark_inferencia.csv"
    )
    frames = [pd.read_csv(path, dtype=str, keep_default_na=False) for path in paths]
    if not frames:
        raise FileNotFoundError(f"No resultados_*.csv found in {results_directory}")
    results = pd.concat(frames, ignore_index=True)
    return results[results["execution_status"] == "success"], [path.name for path in paths]


def _format_value(value) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, dict)):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return str(value)


def hyperparameter_table(results: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (algoritmo, hiperparametro): valor constante ou faixa."""
    rows = []
    for algorithm, group in sorted(results.groupby("algorithm"), key=lambda item: item[0]):
        values: dict[str, dict[str, set]] = {}
        for _, row in group.drop_duplicates(list(FOLD_KEY_COLUMNS)).iterrows():
            try:
                parameters = json.loads(row.get("model_hyperparameters") or "{}")
            except json.JSONDecodeError:
                continue
            for name, value in parameters.items():
                if name in IGNORED_HYPERPARAMETERS or name in DATA_DERIVED_HYPERPARAMETERS:
                    continue
                values.setdefault(name, {}).setdefault(row["dataset_id"], set()).add(
                    _format_value(value)
                )
        for name in sorted(values):
            by_dataset = values[name]
            distinct = sorted(set().union(*by_dataset.values()))
            kind = "arm" if name in ARM_HYPERPARAMETERS else "hyperparameter"
            if len(distinct) == 1:
                shown, datasets = distinct[0], ""
            else:
                numeric = pd.to_numeric(pd.Series(distinct), errors="coerce")
                shown = (
                    f"{numeric.min():g}–{numeric.max():g}"
                    if numeric.notna().all()
                    else ", ".join(distinct)
                )
                datasets = "; ".join(
                    f"{dataset}: {', '.join(sorted(dataset_values))}"
                    for dataset, dataset_values in sorted(by_dataset.items())
                )
            rows.append(
                {"algorithm": algorithm, "kind": kind, "name": name, "value": shown, "by_dataset": datasets}
            )
        for column in ARM_COLUMNS:
            if column in group.columns:
                arms = sorted({value for value in group[column] if value not in ("", "-1")})
                if arms:
                    rows.append(
                        {"algorithm": algorithm, "kind": "arm", "name": column,
                         "value": ", ".join(arms), "by_dataset": ""}
                    )
    return pd.DataFrame(rows, columns=["algorithm", "kind", "name", "value", "by_dataset"])


def _quartiles(values: pd.Series) -> dict:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"median": None, "q1": None, "q3": None, "count": 0}
    return {
        "median": float(numeric.median()),
        "q1": float(numeric.quantile(0.25)),
        "q3": float(numeric.quantile(0.75)),
        "count": int(len(numeric)),
    }


def fold_timing_table(results: pd.DataFrame) -> pd.DataFrame:
    """Mediana e IQR de fit_time e scoring_time por algoritmo, base e maquina."""
    if "fit_time" not in results.columns:
        return pd.DataFrame()
    folds = results
    if "evaluation_unit" in folds.columns:
        # Linhas por cliente (Home Credit) repetem os tempos da dobra.
        folds = folds[folds["evaluation_unit"].isin(["", "row"])]
    folds = folds.drop_duplicates([c for c in FOLD_KEY_COLUMNS if c in folds.columns])
    rows = []
    group_columns = ["algorithm", "dataset_id", *MACHINE_COLUMNS]
    for keys, group in sorted(folds.groupby(group_columns), key=lambda item: item[0]):
        fit = _quartiles(group["fit_time"])
        scoring = _quartiles(group["scoring_time"])
        if not fit["count"]:
            continue
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "folds": fit["count"],
                "fit_median_s": fit["median"], "fit_q1_s": fit["q1"], "fit_q3_s": fit["q3"],
                "scoring_median_s": scoring["median"], "scoring_q1_s": scoring["q1"],
                "scoring_q3_s": scoring["q3"],
            }
        )
    return pd.DataFrame(rows)


def preparation_timing_table(prepared_directory: Path, manifest_path: Path) -> pd.DataFrame:
    manifest = normalize_manifest_dataset_identity(
        pd.read_csv(manifest_path, sep="\t", dtype=str)
    ).drop_duplicates("configuration_md5")
    identity = manifest.set_index("configuration_md5")[["dataset_id", "categorical_handling_mode"]]
    records = []
    for path in sorted(prepared_directory.glob(f"*{PREPARATION_TIMING_FILE_SUFFIX}")):
        timing = json.loads(path.read_text(encoding="utf-8"))
        md5 = timing["configuration_md5"]
        if md5 not in identity.index:
            continue
        records.append(
            {
                "dataset_id": identity.loc[md5, "dataset_id"],
                "categorical_handling_mode": identity.loc[md5, "categorical_handling_mode"],
                **{column: timing.get(column, "") for column in MACHINE_COLUMNS},
                "total_seconds": timing["total_seconds"],
            }
        )
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    rows = []
    group_columns = ["dataset_id", "categorical_handling_mode", *MACHINE_COLUMNS]
    for keys, group in sorted(frame.groupby(group_columns), key=lambda item: item[0]):
        total = _quartiles(group["total_seconds"])
        rows.append(
            {**dict(zip(group_columns, keys)), "configurations": total["count"],
             "median_s": total["median"], "q1_s": total["q1"], "q3_s": total["q3"]}
        )
    return pd.DataFrame(rows)


def benchmark_table(benchmark_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(benchmark_path, dtype=str, keep_default_na=False)
    hosts = sorted(set(frame["execution_host"]))
    if len(hosts) > 1:
        raise MixedMachinesError(
            "Inference benchmark rows come from more than one machine "
            f"({', '.join(hosts)}); inference times are only comparable on one machine."
        )
    columns = [
        "algorithm", "dataset_id", "inference_ms_per_1000", "latency_p50_ms",
        "latency_p95_ms", "latency_p95_ci_low_ms", "latency_p95_ci_high_ms",
        "latency_sample_size", "execution_host", "cpu_model", "gpu_model",
    ]
    return frame[columns].sort_values(["algorithm", "dataset_id"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Saidas
# ---------------------------------------------------------------------------


def _latex_escape(value) -> str:
    text = "" if value is None else (f"{value:.4g}" if isinstance(value, float) else str(value))
    for character, replacement in (
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        text = text.replace(character, replacement)
    return text


def to_latex(frame: pd.DataFrame, caption: str, label: str) -> str:
    if frame.empty:
        return f"% {label}: sem dados\n"
    header = " & ".join(_latex_escape(column) for column in frame.columns)
    lines = [
        r"\begin{longtable}{" + "l" * len(frame.columns) + "}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\hline",
        header + r" \\",
        r"\hline",
        r"\endhead",
    ]
    for _, row in frame.iterrows():
        lines.append(" & ".join(_latex_escape(value) for value in row) + r" \\")
    lines += [r"\hline", r"\end{longtable}", ""]
    return "\n".join(lines)


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_sem dados_\n"
    def cell(value):
        return "" if value is None else (f"{value:.4g}" if isinstance(value, float) else str(value)).replace("|", "\\|")
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(value) for value in row) + " |" for _, row in frame.iterrows()]
    return "\n".join(lines) + "\n"


def build_tables(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    results, files_read = _read_results(Path(args.results_directory))
    tables = {
        "hiperparametros": hyperparameter_table(results),
        "tempos_dobra": fold_timing_table(results),
    }
    if args.prepared_data_directory and args.prepared_dataset_manifest:
        tables["tempos_preparacao"] = preparation_timing_table(
            Path(args.prepared_data_directory), Path(args.prepared_dataset_manifest)
        )
        files_read.append(Path(args.prepared_dataset_manifest).name)
    if args.benchmark_file:
        tables["inferencia"] = benchmark_table(Path(args.benchmark_file))
        files_read.append(Path(args.benchmark_file).name)

    captions = {
        "hiperparametros": "Hiperparâmetros efetivos por algoritmo, lidos dos resultados.",
        "tempos_dobra": "Tempos de ajuste e de pontuação por dobra (mediana e quartis, em segundos), por algoritmo, base e máquina.",
        "tempos_preparacao": "Tempo de preparação por configuração (mediana e quartis, em segundos), por base, modo e máquina.",
        "inferencia": "Benchmark de inferência: milissegundos por 1.000 transações e latência unitária (p50, p95 e IC de 95\\% do p95).",
    }
    commit, dirty = execution_provenance.code_version()
    latex_directory = Path(args.latex_directory)
    latex_directory.mkdir(parents=True, exist_ok=True)
    markdown = [
        "# Tabelas de reprodutibilidade (G18)",
        "",
        f"Commit: `{commit}`" + (" (árvore com alterações)" if dirty is True else ""),
        "",
        "Arquivos lidos: " + ", ".join(f"`{name}`" for name in files_read),
        "",
    ]
    for name, frame in tables.items():
        (latex_directory / f"tabela_{name}.tex").write_text(
            f"% Gerado por build_reproducibility_tables.py (commit {commit}).\n"
            + to_latex(frame, captions[name], f"tab:g18-{name.replace('_', '-')}"),
            encoding="utf-8",
        )
        markdown += [f"## {captions[name]}", "", _markdown_table(frame)]
    Path(args.markdown_output).write_text("\n".join(markdown), encoding="utf-8")
    return tables


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducibility tables (G18).")
    parser.add_argument("--results-directory", required=True)
    parser.add_argument("--prepared-data-directory", default=None)
    parser.add_argument("--prepared-dataset-manifest", default=None)
    parser.add_argument("--benchmark-file", default=None)
    parser.add_argument("--latex-directory", required=True)
    parser.add_argument("--markdown-output", required=True)
    return parser


def main() -> None:
    build_tables(build_argument_parser().parse_args())


if __name__ == "__main__":
    main()
