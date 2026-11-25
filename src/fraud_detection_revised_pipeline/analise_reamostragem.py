"""Tamanho do treino após a reamostragem e razões efetivas de balanceamento (G10).

Só lê relatórios de preparação e resultados; não altera manifestos nem
executores.

* ``tamanhos``: linhas e fraudes do treino antes e depois de cada técnica de
  reamostragem, a partir dos relatórios ``<md5>.fold<k>.preparation.json``
  (``after_imputation`` contra ``output``). A remoção de registros
  incompletos fica de fora, porque já muda o treino antes da reamostragem
  (Tabela de remoção do Apêndice C). Gera o CSV por partição e a tabela do
  apêndice: por base, no holdout temporal (ou no estratificado, onde não há
  ordem temporal), as linhas após o RUS no bloco clássico e no nativo, a
  fração dos legítimos mantida e as linhas removidas pelo Tomek Links (faixa
  entre as configurações, que variam a codificação e a imputação).

* ``pesos``: diferença pareada (braço menos sem reamostragem, demais eixos
  iguais) de F2 e AUC-PR da ponderação de classe sqrt(R) e R e da
  subamostragem aleatória, na regressão logística, no CatBoost e no
  XGBoost. Num modelo ajustado por perda, o peso w nos positivos equivale,
  na perda esperada, a manter 1/w dos negativos; os braços formam, assim,
  uma escala de razões efetivas: natural, 1:sqrt(R), 1:1 sem descarte e 1:1
  com descarte (RUS). Mesmo pareamento da figura de balanceamento
  (``figuras_discussao._paired``).

Uso::

    python3 analise_reamostragem.py tamanhos \\
        --prepared-data-directory /mnt/network-black-hd/prepared \\
        --prepared-dataset-manifest .../prepared_dataset_configurations.tsv \\
        --output-directory saidas --tables-directory tabelas
    python3 analise_reamostragem.py pesos --results-directory resultados \\
        --output-directory saidas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TEMPORAL = "temporal-holdout"
STRATIFIED = "stratified-holdout"
DROP = "drop-incomplete-rows"
NO_RESAMPLING = "none"
RUS = "random-undersampling"
SYNTHETIC = "RUS+SMOTE"
TOMEK = "tomek-links"
WEIGHT_ARMS = ("sqrt-balanced", "balanced")
WEIGHTED_ALGORITHMS = ("logistic-regression", "catboost", "xgboost")
DATASET_ORDER = ("ulb", "sparkov", "sparkov-sem-titular", "ieee-cis-completa", "ieee-cis-sem-identidade",
                 "ieee-cis-transacional", "ieee-cis-completa-sem-alta-cardinalidade",
                 "home-credit-completa", "home-credit-somente-aplicacao")


# ------------------------------------------------------------------ tamanhos

def report_rows(report: dict) -> dict:
    """Linhas e fraudes do treino antes e depois da reamostragem."""
    train = report["splits"]["train"]
    before, after = train["after_imputation"], train["output"]
    return {"rows_before": before["rows"], "positives_before": before["positives"],
            "rows_after": after["rows"], "positives_after": after["positives"]}


def collect_sizes(prepared: Path, manifest: pd.DataFrame) -> pd.DataFrame:
    configurations = (
        manifest[manifest["categorical_handling_mode"].isin(["encoded", "native"])
                 & manifest["test_strategy"].isin([TEMPORAL, STRATIFIED])
                 & ~manifest["fill_na_strategy"].eq(DROP)]
        .drop_duplicates(["configuration_md5", "fold_index"])
    )
    rows = []
    for row in configurations.itertuples(index=False):
        path = prepared / f"{row.configuration_md5}.fold{row.fold_index}.preparation.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append({
            "dataset_id": row.dataset_id, "mode": row.categorical_handling_mode,
            "fill_na_strategy": row.fill_na_strategy,
            "categorical_encoding_strategy": row.categorical_encoding_strategy,
            "technique": row.undersampling_technique, "test_strategy": row.test_strategy,
            "configuration_md5": row.configuration_md5, "fold_index": int(row.fold_index),
            **report_rows(json.loads(path.read_text(encoding="utf-8"))),
        })
    return pd.DataFrame(rows)


def summarize_sizes(sizes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por base, no protocolo das hipóteses (temporal, se houver)."""
    records = []
    for dataset_id, part in sizes.groupby("dataset_id", sort=False):
        protocol = TEMPORAL if part["test_strategy"].eq(TEMPORAL).any() else STRATIFIED
        part = part[part["test_strategy"].eq(protocol)]

        def pick(mode, technique):
            return part[part["mode"].eq(mode) & part["technique"].eq(technique)]

        reference = pd.concat([pick("encoded", NO_RESAMPLING), pick("native", NO_RESAMPLING)])
        if reference.empty:
            continue
        rows_before = int(reference["rows_before"].iloc[0])
        positives = int(reference["positives_before"].iloc[0])
        negatives = rows_before - positives
        record = {"dataset_id": dataset_id, "test_strategy": protocol, "rows_before": rows_before,
                  "positives_before": positives, "ratio": negatives / positives}
        for mode in ("encoded", "native"):
            rus = pick(mode, RUS)
            if not rus.empty:
                kept = rus["rows_after"] - rus["positives_after"]
                record[f"rus_rows_{mode}"] = int(rus["rows_after"].median())
                record[f"rus_negatives_kept_{mode}"] = float((kept / negatives).median())
        tomek = pick("encoded", TOMEK)
        if not tomek.empty:
            removed = tomek["rows_before"] - tomek["rows_after"]
            record["tomek_removed_min"] = int(removed.min())
            record["tomek_removed_max"] = int(removed.max())
            record["tomek_removed_share_max"] = float((removed / (tomek["rows_before"] - tomek["positives_before"])).max())
        synthetic = pick("encoded", SYNTHETIC)
        if not synthetic.empty:
            record["smote_rows"] = int(synthetic["rows_after"].median())
        records.append(record)
    order = {dataset: position for position, dataset in enumerate(DATASET_ORDER)}
    summary = pd.DataFrame(records)
    return summary.sort_values("dataset_id", key=lambda s: s.map(order).fillna(len(order))).reset_index(drop=True)


def sizes_table(summary: pd.DataFrame) -> str:
    import estilo_figuras as estilo

    def integer(value) -> str:
        return "---" if pd.isna(value) else f"{int(value):,}".replace(",", ".")

    def percent(value, digits=2) -> str:
        return "---" if pd.isna(value) else f"{estilo.number(100 * value, digits)}\\%"

    lines = []
    for row in summary.itertuples(index=False):
        mark = "" if row.test_strategy == TEMPORAL else "$^\\dagger$"
        tomek = ("---" if pd.isna(getattr(row, "tomek_removed_min", np.nan)) else
                 integer(row.tomek_removed_min) if row.tomek_removed_min == row.tomek_removed_max else
                 f"{integer(row.tomek_removed_min)}--{integer(row.tomek_removed_max)}")
        lines.append(" & ".join([
            estilo.dataset_label(row.dataset_id) + mark, integer(row.rows_before),
            estilo.number(np.sqrt(row.ratio), 1), integer(getattr(row, "rus_rows_encoded", np.nan)),
            percent(getattr(row, "rus_negatives_kept_encoded", np.nan)),
            integer(getattr(row, "rus_rows_native", np.nan)), tomek,
        ]) + r" \\")
    return (
        "\\begin{table}[H]\n    \\centering\n    \\scriptsize\n    \\setlength{\\tabcolsep}{3pt}\n"
        "    \\caption[Tamanho do treinamento após a reamostragem]{Linhas do treinamento antes e depois da "
        "reamostragem, no \\textit{holdout} temporal ou, nas bases marcadas com $^\\dagger$, sem ordem temporal, no estratificado: "
        "razão efetiva do braço de peso $\\sqrt{R}$ (1:$\\sqrt{R}$, com $R$ a razão entre negativos e positivos); "
        "linhas após a subamostragem aleatória 1:1 nos blocos clássico e nativo e fração das transações "
        "legítimas que ela mantém; e linhas removidas pelos \\textit{Tomek Links}, na faixa entre as "
        "configurações de codificação e imputação. Exclui o braço de remoção de registros incompletos "
        "(Tabela~\\ref{tab:apendice-remocao}).}\n"
        "    \\label{tab:apendice-reamostragem}\n"
        "    \\begin{tabular}{lrrrrrr}\n        \\toprule\n"
        "        & & & \\multicolumn{2}{c}{{\\bf \\acs{rus} clássico}} & {\\bf \\acs{rus} nativo} & {\\bf Tomek} \\\\\n"
        "        \\cmidrule(lr){4-5}\n"
        "        {\\bf Base} & {\\bf Treino} & $\\sqrt{R}$ & linhas & legít. mantidos & linhas & removidas \\\\\n"
        "        \\midrule\n        " + "\n        ".join(lines)
        + "\n        \\bottomrule\n    \\end{tabular}\n\\end{table}\n"
    )


def run_sizes(args: argparse.Namespace) -> None:
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.prepared_dataset_manifest, sep="\t", dtype=str)
    sizes = collect_sizes(Path(args.prepared_data_directory), manifest)
    sizes.to_csv(out / "reamostragem_por_particao.csv", index=False)
    summary = summarize_sizes(sizes)
    summary.to_csv(out / "reamostragem_resumo.csv", index=False)
    tables = Path(args.tables_directory)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "tab_reamostragem.tex").write_text(sizes_table(summary), encoding="utf-8")
    print(summary.round(4).to_string(index=False))


# --------------------------------------------------------------------- pesos

def weight_deltas(executions: pd.DataFrame) -> pd.DataFrame:
    import figuras_discussao as figuras

    scope = executions[executions["algorithm"].isin(WEIGHTED_ALGORITHMS)].copy()
    scope["undersampling_technique"] = (
        scope["undersampling_technique"].replace("", NO_RESAMPLING).fillna(NO_RESAMPLING))
    scope["class_weight_intensity"] = scope["class_weight_intensity"].fillna("none")
    unweighted = scope[scope["class_weight_intensity"].eq("none")]
    no_resampling = scope[scope["undersampling_technique"].eq(NO_RESAMPLING)]
    deltas = [figuras._paired(no_resampling, "class_weight_intensity", arm, "none") for arm in WEIGHT_ARMS]
    deltas.append(figuras._paired(unweighted, "undersampling_technique", RUS, NO_RESAMPLING))
    return pd.concat([d for d in deltas if not d.empty], ignore_index=True)


def run_weights(args: argparse.Namespace) -> None:
    import figuras_discussao as figuras

    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    deltas = weight_deltas(figuras.load(Path(args.results_directory)))
    columns = ["delta_f_beta_2", "delta_auc_pr"]
    overall = deltas.groupby(["algorithm", "arm"])[columns].agg(["mean", "median", "size"]).round(4)
    by_dataset = deltas.groupby(["algorithm", "arm", "database_type"])[columns].mean().round(4)
    overall.to_csv(out / "pesos_geral.csv")
    by_dataset.to_csv(out / "pesos_por_base.csv")
    print(overall.to_string())
    print(by_dataset.unstack("arm").to_string())


def main() -> None:
    parser = argparse.ArgumentParser(description="Reamostragem e razões efetivas de balanceamento (G10).")
    sub = parser.add_subparsers(dest="command", required=True)
    sizes = sub.add_parser("tamanhos")
    sizes.add_argument("--prepared-data-directory", required=True)
    sizes.add_argument("--prepared-dataset-manifest", required=True)
    sizes.add_argument("--tables-directory", required=True)
    weights = sub.add_parser("pesos")
    weights.add_argument("--results-directory", required=True)
    for p in (sizes, weights):
        p.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    {"tamanhos": run_sizes, "pesos": run_weights}[args.command](args)


if __name__ == "__main__":
    main()
