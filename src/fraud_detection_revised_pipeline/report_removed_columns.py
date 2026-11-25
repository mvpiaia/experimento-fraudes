"""Colunas removidas por base e por motivo, e a dimensionalidade (item G14).

Carrega cada base com o carregador do proprio pipeline e lista:

* ``identifier``: identificadores, agrupamento e marcas de tempo absolutas;
* ``constant``: colunas com um unico valor na base inteira (o ausente conta
  como valor);
* ``ablation``: atributos que uma base de ablacao remove da base de origem.

As variantes de ablacao carregam exatamente como a base de origem, entao suas
constantes sao as da origem e a base nao e recarregada.

Uso:
    python3 report_removed_columns.py --data-directory ../datasets
    python3 report_removed_columns.py --format json --dataset sparkov
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import experiment_datasets
from column_registry import categorical_columns_of
from experiment_datasets import (
    DATASET_DEFINITIONS,
    DatasetDefinition,
    get_dataset_definition,
    identifier_columns_to_drop,
    resolve_base_definition,
)

DEFAULT_DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "datasets"


def describe_base(definition: DatasetDefinition) -> dict:
    """Carrega uma base (nao variante) e resume suas colunas."""
    train, auxiliary, _definition = experiment_datasets.load_raw_dataset(
        definition.dataset_id
    )
    provenance = train.attrs["column_provenance"]
    identifiers = [
        column
        for column in identifier_columns_to_drop(definition)
        if column in train.columns
    ]
    kept = [
        column
        for column in train.columns
        if column not in identifiers and column != definition.prediction_column
    ]
    categorical = [
        column
        for column in categorical_columns_of(train[kept + [definition.prediction_column]],
                                             definition.prediction_column)
    ]
    summary = {
        "rows": int(len(train) + (len(auxiliary) if auxiliary is not None else 0)),
        "source_columns": provenance["source_columns"],
        "derived_columns": provenance["derived_columns"],
        "constant_columns": provenance["constant_columns"],
        "identifier_columns": identifiers,
        "kept_columns": kept,
        "categorical_columns": categorical,
    }
    del train, auxiliary
    gc.collect()
    return summary


def describe_variant(definition: DatasetDefinition, base: dict) -> dict:
    ablated = [column for column in definition.ablated_columns if column in base["kept_columns"]]
    kept = [column for column in base["kept_columns"] if column not in ablated]
    return {
        **base,
        "ablated_columns": ablated,
        "kept_columns": kept,
        "categorical_columns": [c for c in base["categorical_columns"] if c in kept],
    }


def build_report(dataset_ids: list[str]) -> dict:
    bases: dict[str, dict] = {}
    report = {}
    for dataset_id in dataset_ids:
        definition = get_dataset_definition(dataset_id)
        base_definition = resolve_base_definition(definition)
        if base_definition.dataset_id not in bases:
            bases[base_definition.dataset_id] = describe_base(base_definition)
        base = bases[base_definition.dataset_id]
        summary = (
            describe_variant(definition, base)
            if definition.base_dataset_id
            else {**base, "ablated_columns": []}
        )
        summary["monetary_columns"] = list(definition.monetary_columns)
        summary["categorical_code_columns"] = [
            column
            for column in definition.categorical_code_columns
            if column in summary["kept_columns"]
        ]
        summary["feature_count"] = len(summary["kept_columns"])
        summary["categorical_count"] = len(summary["categorical_columns"])
        summary["numeric_count"] = summary["feature_count"] - summary["categorical_count"]
        report[dataset_id] = summary
    return report


def format_markdown(report: dict) -> str:
    def cells(columns: list[str]) -> str:
        return ", ".join(f"`{c}`" for c in columns) if columns else "—"

    lines = [
        "# Colunas removidas por base (item G14)",
        "",
        "| Base | Linhas | Atributos | Categóricos | Numéricos | Constantes | Ablação |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset_id, item in report.items():
        lines.append(
            f"| `{dataset_id}` | {item['rows']:,} | {item['feature_count']} | "
            f"{item['categorical_count']} | {item['numeric_count']} | "
            f"{len(item['constant_columns'])} | {len(item['ablated_columns'])} |"
        )
    for dataset_id, item in report.items():
        lines += [
            "",
            f"## `{dataset_id}`",
            "",
            f"- **Identificadores, agrupamento e ordem** ({len(item['identifier_columns'])}): "
            f"{cells(item['identifier_columns'])}",
            f"- **Constantes** ({len(item['constant_columns'])}): "
            f"{cells(item['constant_columns'])}",
            f"- **Ablação** ({len(item['ablated_columns'])}): {cells(item['ablated_columns'])}",
            f"- **Derivadas na carga**: {cells(item['derived_columns'])}",
            f"- **Valores monetários (`log1p`)**: {cells(item['monetary_columns'])}",
            f"- **Códigos tratados como categóricos**: "
            f"{cells(item['categorical_code_columns'])}",
        ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-directory", default=str(DEFAULT_DATA_DIRECTORY))
    parser.add_argument("--dataset", action="append", help="Repetivel; padrao: todas")
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument("--output", help="Arquivo de saida; padrao: stdout")
    parser.add_argument(
        "--markdown-output",
        help="Grava tambem o relatorio em Markdown, na mesma carga das bases",
    )
    arguments = parser.parse_args()

    experiment_datasets.CURRENT_PATH = ""
    experiment_datasets.RELATIVE_DATA_PATH = str(Path(arguments.data_directory).resolve())
    dataset_ids = arguments.dataset or list(DATASET_DEFINITIONS)
    report = build_report(dataset_ids)
    text = (
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if arguments.format == "json"
        else format_markdown(report)
    )
    if arguments.markdown_output:
        Path(arguments.markdown_output).write_text(
            format_markdown(report), encoding="utf-8"
        )
    if arguments.output:
        Path(arguments.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()
