"""Tabela de metricas por estrato de historico (item C3.19 da revisao).

Le os registros de estratos (``resultados_*_history_strata.jsonl`` do boosting
e ``resultados_*_details.jsonl`` dos Transformers) e escreve, por base,
algoritmo, regime e estrato:

* o numero de transacoes e a prevalencia do estrato, iguais em todos os
  algoritmos (checagem R6 do validador);
* a media, entre as configuracoes, de F1, F2, AUC-ROC e AUC-PR;
* a diferenca de cada metrica em relacao ao estrato de historico completo.

Comparar o TabBERT com os modelos que nao usam historico, nos mesmos estratos,
separa o efeito do historico do efeito da composicao do estrato: as primeiras
transacoes de cada titular sao as mais antigas e podem ter outra prevalencia.

Uso:
    python3 history_strata_analysis.py [--results-directory DIR] [--output-directory DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from history_strata import (
    HISTORY_STRATUM_NAME,
    history_strata_paths,
    read_history_strata_records,
)

TABLE_METRICS = ("f1", "f_beta_2", "roc_auc", "auc_pr")
METRIC_LABELS = {"f1": "F1", "f_beta_2": "F2", "roc_auc": "AUC-ROC", "auc_pr": "AUC-PR"}
GROUP_COLUMNS = ("dataset_id", "algorithm", "regime", "stratum")


def strata_rows(records: list[dict], split: str = "test") -> pd.DataFrame:
    """Uma linha por registro e estrato do split pedido."""
    rows = []
    for record in records:
        if record.get("split") != split:
            continue
        strata = (record.get("history_strata") or {}).get(HISTORY_STRATUM_NAME) or {}
        for stratum, values in strata.items():
            rows.append(
                {
                    "dataset_id": record.get("dataset_id"),
                    "algorithm": record.get("algorithm"),
                    "regime": record.get("regime") or "",
                    "configuration_md5": record.get("configuration_md5"),
                    "stratum": int(stratum),
                    "transactions": int(values["transactions"]),
                    "positives": int(values["positives"]),
                    "prevalence": float(values["prevalence"]),
                    **{metric: values.get(metric, np.nan) for metric in TABLE_METRICS},
                }
            )
    return pd.DataFrame(rows)


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    """Media das metricas entre configuracoes e diferenca para o maior estrato."""
    if rows.empty:
        return pd.DataFrame()
    summary = (
        rows.groupby(list(GROUP_COLUMNS), dropna=False)
        .agg(
            configurations=("configuration_md5", "nunique"),
            transactions=("transactions", "first"),
            positives=("positives", "first"),
            prevalence=("prevalence", "first"),
            **{metric: (metric, "mean") for metric in TABLE_METRICS},
        )
        .reset_index()
    )
    reference = summary.loc[
        summary.groupby(["dataset_id", "algorithm", "regime"])["stratum"].idxmax(),
        ["dataset_id", "algorithm", "regime", *TABLE_METRICS],
    ]
    summary = summary.merge(
        reference, on=["dataset_id", "algorithm", "regime"], suffixes=("", "_reference")
    )
    for metric in TABLE_METRICS:
        summary[f"{metric}_difference"] = summary[metric] - summary[f"{metric}_reference"]
        summary = summary.drop(columns=f"{metric}_reference")
    return summary.sort_values(list(GROUP_COLUMNS)).reset_index(drop=True)


def _decimal(value: float, digits: int = 3) -> str:
    if pd.isna(value):
        return "--"
    return f"{value:.{digits}f}".replace(".", "{,}")


def latex_table(summary: pd.DataFrame) -> str:
    """Tabela LaTeX com prevalencia e metricas de cada estrato, em virgula decimal."""
    lines = [
        "\\begin{tabular}{llllrr" + "r" * len(TABLE_METRICS) + "}",
        "\\toprule",
        "Base & Algoritmo & Regime & Estrato & Transações & Prevalência & "
        + " & ".join(METRIC_LABELS[metric] for metric in TABLE_METRICS)
        + " \\\\",
        "\\midrule",
    ]
    for row in summary.itertuples(index=False):
        metrics = " & ".join(_decimal(getattr(row, metric)) for metric in TABLE_METRICS)
        lines.append(
            f"{row.dataset_id} & {row.algorithm} & {row.regime or '--'} & {row.stratum} & "
            f"{row.transactions:,}".replace(",", ".")
            + f" & {_decimal(100 * row.prevalence, 2)}\\% & {metrics} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-directory", default=str(Path(__file__).resolve().parent))
    parser.add_argument("--output-directory", default=".")
    parser.add_argument("--split", default="test", choices=("test", "validation"))
    arguments = parser.parse_args()

    records = read_history_strata_records(history_strata_paths(arguments.results_directory))
    summary = summarize(strata_rows(records, arguments.split))
    if summary.empty:
        print("Nenhum registro de estratos de historico encontrado.")
        return 1
    output = Path(arguments.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "history_strata_summary.csv", index=False)
    (output / "history_strata_table.tex").write_text(latex_table(summary), encoding="utf-8")
    print(f"{len(summary)} linhas gravadas em {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
