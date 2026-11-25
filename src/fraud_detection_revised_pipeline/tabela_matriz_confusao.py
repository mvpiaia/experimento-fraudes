"""Matriz de confusão das configurações representativas de H2 (item G2/G17).

Os executores calculam a matriz de confusão para obter a FPR, mas gravam só as
métricas. Como as contagens são inteiras, a matriz é reconstruída exatamente
a partir da revocação, da FPR, do tamanho do teste e do número de fraudes do
teste (estimado pela mesma função da análise de hipóteses):

    VP = round(revocação · P), FN = P − VP, FP = round(FPR · N), VN = N − FP.

A reconstrução é conferida contra a precisão gravada; a maior discrepância
vai para a saída. As configurações são as de H2: por fonte de fraude e
algoritmo supervisionado, a de maior F2 na validação, no holdout temporal.

Uso::

    python3 tabela_matriz_confusao.py --results-directory <resultados> \\
        --tables-directory <eqm_paper>/tabelas --output-directory <saidas>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import analise_hipoteses as hip


def integer(value) -> str:
    return f"{int(value):,}".replace(",", ".")


def reconstruct(executions: pd.DataFrame, per_seed_test: pd.DataFrame) -> pd.DataFrame:
    representatives = hip.representatives_by_source(executions, hip.FRAUD_SOURCES, hip.TEMPORAL)
    positives = hip.estimate_positives(per_seed_test)
    key = list(hip.KEY_COLUMNS)
    rows = []
    for _, rep in representatives.iterrows():
        match = per_seed_test.merge(rep[key].to_frame().T, on=key, how="inner")
        for _, run in match.iterrows():
            total_positive = positives.get(hip._positives_key(run))
            if total_positive is None:
                raise ValueError(f"sem estimativa de positivos para {hip._positives_key(run)}")
            total_negative = int(run["test_df_size"]) - total_positive
            tp = int(round(run["recall"] * total_positive))
            fp = int(round(run["false_positive_rate"] * total_negative))
            precision = tp / (tp + fp) if tp + fp else 0.0
            rows.append({
                "database_type": rep["database_type"], "algorithm": rep["algorithm"],
                "random_seed": int(run["random_seed"]), "test_rows": int(run["test_df_size"]),
                "tp": tp, "fn": total_positive - tp, "fp": fp, "tn": total_negative - fp,
                "precision_recorded": run["precision"], "precision_rebuilt": precision,
                "f_beta_2": run["f_beta_2"],
            })
    table = pd.DataFrame(rows)
    table["precision_gap"] = (table["precision_rebuilt"] - table["precision_recorded"]).abs()
    return table


def latex(table: pd.DataFrame) -> str:
    import estilo_figuras as estilo

    order = {a: i for i, a in enumerate(estilo.ALGORITHM_LATEX)}
    sources = list(hip.FRAUD_SOURCES)
    table = table.assign(s=table["database_type"].map(sources.index), a=table["algorithm"].map(order))
    lines, last = [], None
    for row in table.sort_values(["s", "f_beta_2"], ascending=[True, False]).itertuples(index=False):
        if row.database_type != last:
            if last is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{6}}{{l}}{{\textit{{{estilo.dataset_label(row.database_type)}}}}} \\")
            last = row.database_type
        lines.append(" & ".join([estilo.ALGORITHM_LATEX[row.algorithm], integer(row.tp), integer(row.fn),
                                 integer(row.fp), integer(row.tn), estilo.number(row.f_beta_2, 3)]) + r" \\")
    body = "\n        ".join(lines)
    return (
        "\\begin{table}[H]\n    \\centering\n    \\footnotesize\n    \\setlength{\\tabcolsep}{4pt}\n"
        "    \\caption[Matriz de confusão das configurações de H2]{Matriz de confusão no teste do \\textit{holdout} "
        "temporal das configurações de H2 (a de maior F2 na validação, por fonte e algoritmo), reconstruída da "
        "revocação, da taxa de falsos positivos, do tamanho do teste e do número de fraudes do teste: VP, fraudes "
        "sinalizadas; FN, fraudes não sinalizadas; FP, transações legítimas sinalizadas; VN, legítimas não "
        "sinalizadas. Ordenada pelo F2 em cada fonte.}\n"
        "    \\label{tab:matriz-confusao}\n"
        "    \\begin{tabular}{lrrrrr}\n        \\toprule\n"
        "        {\\bf Algoritmo} & {\\bf VP} & {\\bf FN} & {\\bf FP} & {\\bf VN} & {\\bf F2} \\\\\n"
        f"        \\midrule\n        {body}\n        \\bottomrule\n    \\end{{tabular}}\n\\end{{table}}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Matriz de confusão das configurações de H2.")
    parser.add_argument("--results-directory", required=True)
    parser.add_argument("--tables-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    data = hip.normalize_results(hip.load_results(args.results_directory))
    executions, per_seed_test = hip.build_executions(data)
    table = reconstruct(executions, per_seed_test)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "matriz_confusao_h2.csv", index=False)
    tables = Path(args.tables_directory)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "tab_matriz_confusao.tex").write_text(latex(table), encoding="utf-8")
    print(table.round(4).to_string(index=False))
    print("maior discrepancia de precisao:", float(np.round(table["precision_gap"].max(), 6)))


if __name__ == "__main__":
    main()
