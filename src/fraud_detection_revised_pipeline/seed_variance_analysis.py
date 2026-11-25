"""Dispersao das metricas entre sementes de preparacao (item G13 da revisao).

Le os resultados do estudo de variancia por semente e resume, para cada base,
algoritmo e braco de execucao, quanto cada metrica de teste varia quando so a
semente de preparacao muda.

A agregacao tem dois niveis. Primeiro, a media por semente: as dobras do
k-fold e os sorteios de rotulos do DevNet viram um unico valor por semente.
So depois se mede a dispersao entre sementes. Sem isso, uma semente com cinco
dobras pesaria cinco vezes mais que outra com uma.

Uso:
    python3 seed_variance_analysis.py --results-directory seed_variance_results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SEED_COLUMN = "random_seed"
STUDIED_METRICS = (
    "roc_auc",
    "auc_pr",
    "f1",
    "f_beta_2",
    "precision",
    "recall",
    "false_positive_rate",
)
# Colunas que identificam um braco de execucao, alem da semente. Ficam de
# fora configuration_md5 e partition_id, que mudam com a semente por
# construcao, e fold_index/execution_seed, que sao promediados por semente.
ARM_COLUMNS = (
    "database_type",
    "algorithm",
    "regime",
    # Na Home Credit ha linhas por registro e por cliente: nunca se misturam.
    "evaluation_unit",
    "class_weight_intensity",
    # A fracao, e nao label_budget: no k-fold o numero de fraudes rotuladas
    # varia entre dobras e sementes para a mesma fracao.
    "label_fraction",
    "categorical_encoding_strategy",
    "fill_na_strategy",
    "undersampling_technique",
    "train_validation_test_strategy",
)
SUMMARY_STATISTICS = ("seed_count", "mean", "std", "min", "max", "range")


def load_study_results(results_directory: str | Path) -> pd.DataFrame:
    frames = [
        pd.read_csv(path, dtype=str, keep_default_na=False)
        for path in sorted(Path(results_directory).glob("resultados_*.csv"))
    ]
    if not frames:
        raise FileNotFoundError(f"No resultados_*.csv found in {results_directory}")
    return pd.concat(frames, ignore_index=True)


def summarize_seed_variance(results: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (braco, metrica) com a dispersao entre sementes."""
    selected = results[
        results["execution_status"].str.strip().str.lower() == "success"
    ]
    if "split" in selected.columns:
        selected = selected[selected["split"].str.strip().str.lower() == "test"]

    arm_columns = [column for column in ARM_COLUMNS if column in selected.columns]
    metrics = [metric for metric in STUDIED_METRICS if metric in selected.columns]
    numeric = selected[arm_columns + [SEED_COLUMN]].copy()
    for metric in metrics:
        numeric[metric] = pd.to_numeric(selected[metric], errors="coerce")

    per_seed = numeric.groupby(arm_columns + [SEED_COLUMN], dropna=False)[metrics].mean()
    across_seeds = per_seed.groupby(level=arm_columns, dropna=False)

    rows = []
    for metric in metrics:
        statistics = across_seeds[metric].agg(
            seed_count="count", mean="mean", std="std", min="min", max="max"
        )
        statistics["range"] = statistics["max"] - statistics["min"]
        statistics.insert(0, "metric", metric)
        rows.append(statistics.reset_index())
    summary = pd.concat(rows, ignore_index=True)
    return summary[arm_columns + ["metric", *SUMMARY_STATISTICS]].sort_values(
        arm_columns + ["metric"], ignore_index=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", required=True)
    parser.add_argument(
        "--output",
        help="CSV de saida; sem valor, seed_variance_summary.csv no diretorio de resultados.",
    )
    arguments = parser.parse_args()
    summary = summarize_seed_variance(load_study_results(arguments.results_directory))
    output = Path(
        arguments.output
        or Path(arguments.results_directory) / "seed_variance_summary.csv"
    )
    summary.to_csv(output, index=False)
    print(f"Seed variance summary: {output} ({len(summary)} rows)")


if __name__ == "__main__":
    main()
