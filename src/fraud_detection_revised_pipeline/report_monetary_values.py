"""Medicoes que justificam o log1p dos valores monetarios (item G14).

Para cada base de fraude, sobre a base inteira carregada pelo pipeline:

* assimetria do valor bruto e com ``log1p``;
* maior |z| depois de padronizar (o quanto uma transacao se afasta da media);
* taxa de fraude por faixa de percentil do proprio valor e AUC-ROC do valor
  sozinho, para checar se "valor muito alto" caracteriza fraude.

Uso:
    python3 report_monetary_values.py --data-directory ../datasets
    python3 report_monetary_values.py --format json --output valores.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import experiment_datasets

DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "datasets"
# Uma visao por base: a IEEE-CIS tem tres visoes sobre a mesma coluna.
DATASETS = ("ulb", "sparkov", "ieee-cis-transacional")
# Faixas de percentil do proprio valor (limite superior de cada faixa).
BAND_EDGES = (0.50, 0.90, 0.99, 0.999)
BAND_LABELS = ("<p50", "p50-p90", "p90-p99", "p99-p99,9", ">p99,9")


def max_abs_z(values: pd.Series) -> float:
    deviation = values.std(ddof=0)
    if deviation == 0:
        return 0.0
    return float(((values - values.mean()).abs() / deviation).max())


def fraud_rate_by_band(values: pd.Series, labels: pd.Series) -> list[dict]:
    """Taxa de fraude em cada faixa de percentil do proprio valor."""
    cut_points = values.quantile(list(BAND_EDGES)).to_numpy()
    band = np.searchsorted(cut_points, values.to_numpy(), side="right")
    rows = []
    for position, label in enumerate(BAND_LABELS):
        selected = band == position
        count = int(selected.sum())
        frauds = int(labels.to_numpy()[selected].sum())
        rows.append(
            {
                "band": label,
                "rows": count,
                "frauds": frauds,
                "fraud_rate": frauds / count if count else None,
            }
        )
    return rows


def describe(dataset_id: str) -> dict:
    train, auxiliary, definition = experiment_datasets.load_raw_dataset(dataset_id)
    frame = train if auxiliary is None else pd.concat([train, auxiliary], ignore_index=True)
    column = definition.monetary_columns[0]
    target = definition.prediction_column
    frame = frame.dropna(subset=[column, target])
    values = frame[column].astype(float)
    labels = frame[target].astype(int)
    logged = np.log1p(values)
    return {
        "dataset_id": dataset_id,
        "column": column,
        "rows": int(len(frame)),
        "prevalence": float(labels.mean()),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "skewness_raw": float(values.skew()),
        "skewness_log1p": float(logged.skew()),
        "max_abs_z_raw": max_abs_z(values),
        "max_abs_z_log1p": max_abs_z(logged),
        "auc_value_alone": float(roc_auc_score(labels, values)),
        "fraud_rate_by_band": fraud_rate_by_band(values, labels),
    }


def format_markdown(report: list[dict]) -> str:
    lines = [
        "# Valores monetários: medições que justificam o `log1p` (item G14)",
        "",
        "| Base | Coluna | Linhas | Prevalência | Assimetria bruta | Assimetria `log1p` "
        "| Maior \\|z\\| bruto | Maior \\|z\\| `log1p` | AUC do valor |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report:
        lines.append(
            f"| `{item['dataset_id']}` | `{item['column']}` | {item['rows']:,} | "
            f"{item['prevalence']:.2%} | {item['skewness_raw']:.1f} | "
            f"{item['skewness_log1p']:.2f} | {item['max_abs_z_raw']:.1f} | "
            f"{item['max_abs_z_log1p']:.1f} | {item['auc_value_alone']:.3f} |"
        )
    lines += ["", "## Taxa de fraude por faixa de percentil do próprio valor", ""]
    header = "| Base | " + " | ".join(BAND_LABELS) + " |"
    lines += [header, "|---|" + "---:|" * len(BAND_LABELS)]
    for item in report:
        cells = " | ".join(
            "—" if band["fraud_rate"] is None else f"{band['fraud_rate']:.2%} ({band['frauds']}/{band['rows']})"
            for band in item["fraud_rate_by_band"]
        )
        lines.append(f"| `{item['dataset_id']}` | {cells} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", default=str(DEFAULT_DATA_DIRECTORY))
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output")
    arguments = parser.parse_args()

    experiment_datasets.CURRENT_PATH = ""
    experiment_datasets.RELATIVE_DATA_PATH = str(Path(arguments.data_directory).resolve())
    report = [describe(dataset_id) for dataset_id in DATASETS]
    text = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if arguments.format == "json"
        else format_markdown(report)
    )
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
