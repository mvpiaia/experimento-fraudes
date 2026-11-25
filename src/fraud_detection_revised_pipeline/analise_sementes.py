"""Estudo parcial de sementes de preparação (G13): dispersão e reprodutibilidade.

Lê os resultados de ``run_seed_variance_study.sh`` (corte: ULB e
IEEE-CIS-completa, holdout estratificado, sem reamostragem, imputação pela
mediana, sementes 42, 7, 2026, 13 e 101) e:

* para cada base, semente e algoritmo supervisionado, escolhe a configuração
  de maior F2 na validação (a mesma regra de H2) e relata o F2 no teste, o
  líder de cada semente e o desvio padrão entre sementes;
* compara a semente 42 do estudo com a grade principal, execução a execução
  (mesmo ``configuration_md5``, algoritmo, braço e sorteio), para medir a
  reprodutibilidade entre máquinas.

Uso::

    python3 analise_sementes.py --study-directory <resultados do estudo> \\
        --grid-directory <resultados da grade> --tables-directory <tabelas> \\
        --output-directory <saidas>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import analise_hipoteses as hip

SEED_ORDER = (42, 7, 13, 101, 2026)
DRAW_COLUMN = "execution_seed"


def per_seed_representatives(study: pd.DataFrame) -> pd.DataFrame:
    seed_of = study.groupby("configuration_md5")["random_seed"].first()
    executions, _ = hip.build_executions(study)
    executions["seed"] = executions["configuration_md5"].map(seed_of).astype(int)
    supervised = executions[executions["algorithm"].isin(hip.SUPERVISED_ALGORITHMS)]
    representatives = hip.select_representatives(supervised, ["database_type", "algorithm", "seed"])
    return representatives.pivot_table(index=["database_type", "seed"], columns="algorithm", values="f_beta_2")


def reproducibility(study: pd.DataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in dict.fromkeys([*hip.KEY_COLUMNS, "random_seed", "fold_index", "split", DRAW_COLUMN])
            if k in study.columns and k in grid.columns]
    left = study[study["random_seed"].astype(int).eq(42) & study["split"].eq("test")].copy()
    right = grid[grid["split"].eq("test")].copy()
    for frame in (left, right):
        frame[DRAW_COLUMN] = frame[DRAW_COLUMN].astype(str)
    joined = left.merge(right, on=keys, suffixes=("_estudo", "_grade"))
    if len(joined) != len(left):
        raise ValueError(f"{len(left)} execuções da semente 42 e {len(joined)} casadas com a grade")
    rows = []
    for algorithm, part in joined.groupby("algorithm"):
        rows.append({"algorithm": algorithm, "executions": len(part),
                     **{f"max_diff_{m}": float((part[f"{m}_estudo"] - part[f"{m}_grade"]).abs().max())
                        for m in ("f_beta_2", "auc_pr", "roc_auc")}})
    return pd.DataFrame(rows)


def latex(pivot: pd.DataFrame) -> str:
    import estilo_figuras as estilo

    algorithms = [a for a in estilo.ALGORITHM_LATEX if a in pivot.columns]
    lines = []
    for dataset, part in pivot.groupby(level=0, sort=False):
        lines.append(rf"\multicolumn{{{len(algorithms) + 2}}}{{l}}{{\textit{{{estilo.dataset_label(dataset)}}}}} \\")
        part = part.droplevel(0).reindex([s for s in SEED_ORDER if s in part.droplevel(0).index])
        for seed, row in part.iterrows():
            best = row[algorithms].max()
            cells = [rf"\textbf{{{estilo.number(row[a], 3)}}}" if row[a] == best else estilo.number(row[a], 3)
                     for a in algorithms]
            leader = estilo.ALGORITHM_LATEX[row[algorithms].idxmax()]
            lines.append(" & ".join([str(seed), *cells, leader]) + r" \\")
        lines.append(" & ".join([r"\textit{desvio}", *[estilo.number(part[a].std(), 3) for a in algorithms], ""]) + r" \\")
        lines.append(r"\midrule")
    lines = lines[:-1]
    header = " & ".join([r"{\bf Semente}", *[rf"{{\bf {estilo.ALGORITHM_LATEX[a]}}}" for a in algorithms], r"{\bf Líder}"])
    body = "\n        ".join(lines)
    return (
        "\\begin{table}[H]\n    \\centering\n    \\footnotesize\n    \\setlength{\\tabcolsep}{3pt}\n"
        "    \\caption[F2 por semente de preparação]{F2 no teste do \\textit{holdout} estratificado por semente de "
        "preparação, no corte do estudo de sementes (sem reamostragem, imputação pela mediana): por base, semente e "
        "algoritmo, a configuração de maior F2 na validação, como em H2. Em negrito, o maior F2 de cada semente; "
        "na última linha de cada base, o desvio padrão entre as cinco sementes.}\n"
        "    \\label{tab:apendice-sementes}\n"
        f"    \\begin{{tabular}}{{l{'r' * len(algorithms)}l}}\n        \\toprule\n        {header} \\\\\n"
        f"        \\midrule\n        {body}\n        \\bottomrule\n    \\end{{tabular}}\n\\end{{table}}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Estudo parcial de sementes (G13).")
    parser.add_argument("--study-directory", required=True)
    parser.add_argument("--grid-directory", required=True)
    parser.add_argument("--tables-directory", required=True)
    parser.add_argument("--output-directory", required=True)
    args = parser.parse_args()
    study = hip.normalize_results(hip.load_results(args.study_directory))
    grid = hip.normalize_results(hip.load_results(args.grid_directory))
    pivot = per_seed_representatives(study)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(out / "sementes_f2_por_semente.csv")
    repro = reproducibility(study, grid)
    repro.to_csv(out / "sementes_reprodutibilidade.csv", index=False)
    tables = Path(args.tables_directory)
    tables.mkdir(parents=True, exist_ok=True)
    (tables / "tab_sementes.tex").write_text(latex(pivot), encoding="utf-8")
    print(pivot.round(3).to_string())
    print(repro.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
