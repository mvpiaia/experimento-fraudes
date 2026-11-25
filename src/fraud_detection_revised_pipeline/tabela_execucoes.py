"""Tabelas de execuções previstas e realizadas (G17), resumida e completa.

Lê o ``appendix_run_counts.csv`` gravado pela verificação R4
(``validate_prepared_datasets.py --mode results --appendix-output``) e grava:

* ``tab_execucoes_resumo.tex`` (Capítulo 4): por bloco e algoritmo, as
  execuções previstas, com sucesso, gêmeas, com falha e ausentes, e as com
  sucesso por protocolo de avaliação;
* ``tab_apendice_execucoes.tex`` (Apêndice C): por bloco, base e algoritmo,
  as mesmas contagens em execuções e em linhas de resultado (uma por dobra
  avaliada), que são as duas unidades conferidas pela verificação.

Falhas incluem as execuções incompletas. O TabBERT, trabalho futuro, é
excluído.

Uso::

    python3 tabela_execucoes.py --run-counts <dir>/appendix_run_counts.csv \\
        --tables-directory <eqm_paper>/tabelas --verification-date 22/11/2026
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

BLOCKS = {"classic": "clássico", "boosting": "nativo", "anomaly": "anomalia", "ft-transformer": "FT-Transformer"}
ALGORITHMS = ("logistic-regression", "naive-bayes", "knn", "catboost", "xgboost", "isolation-forest", "devnet",
              "ft-transformer-mlp", "ft-transformer-lstm")
PROTOCOLS = ("stratified-holdout", "temporal-holdout", "stratified-k-fold")
UNITS = ("previstas", "sucesso", "equivalente", "falha", "ausente")


# Estados de ``preparation_sanity_checks.classify_runs`` (verificação R4).
LEGEND = (
    "Uma execução é a combinação de configuração preparada, algoritmo e braço de execução (no \\acs{devnet}, "
    "também fração e semente do sorteio dos rótulos), avaliada em todas as suas dobras. Sucesso: todas as "
    "dobras previstas têm linha de resultado de teste com sucesso. Gêmea: configuração que difere de uma "
    "representante apenas no eixo inerte de alta cardinalidade; não é executada e conta com o sucesso da "
    "representante. Falha: há linha de resultado gravada com falha, cujo motivo fica registrado, ou há "
    "sucesso em só parte das dobras (incompleta). Ausente: nenhuma linha de resultado; o motivo vem do "
    "registro de tentativas do roteiro, como o encerramento do processo por falta de memória, que não "
    "deixa linha de resultado."
)


def integer(value) -> str:
    return f"{int(value):,}".replace(",", ".")


def load(path: Path) -> pd.DataFrame:
    counts = pd.read_csv(path)
    counts = counts[~counts["algorithm"].str.startswith("tabbert")].copy()
    unknown = set(counts["algorithm"]) - set(ALGORITHMS) | set(counts["bloco"]) - set(BLOCKS)
    if unknown:
        raise ValueError(f"algoritmos ou blocos sem rótulo: {sorted(unknown)}")
    for unit in ("unidades", "linhas"):
        counts[f"{unit}_falha"] = counts[f"{unit}_falha"] + counts[f"{unit}_incompleta"]
    counts["block_order"] = counts["bloco"].map(list(BLOCKS).index)
    counts["algorithm_order"] = counts["algorithm"].map(ALGORITHMS.index)
    return counts


def summary_table(counts: pd.DataFrame, date: str) -> str:
    import estilo_figuras as estilo

    keys = ["block_order", "algorithm_order", "bloco", "algorithm"]
    units = counts.groupby(keys)[[f"unidades_{u}" for u in UNITS]].sum()
    protocols = (counts.groupby(keys + ["test_strategy"])["unidades_sucesso"].sum()
                 .unstack("test_strategy").reindex(columns=list(PROTOCOLS)).fillna(0)
                 .rename(columns=lambda c: c.replace("-", "_")))
    frame = units.join(protocols).sort_index().reset_index()
    lines, last = [], None
    for row in frame.itertuples(index=False):
        if row.bloco != last:
            if last is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{Bloco {BLOCKS[row.bloco]}}}}} \\")
        last = row.bloco
        values = [getattr(row, f"unidades_{u}") for u in UNITS] + [getattr(row, p.replace("-", "_"))
                                                                   for p in PROTOCOLS]
        lines.append(" & ".join([estilo.ALGORITHM_LATEX[row.algorithm]] + [integer(v) for v in values]) + r" \\")
    totals = [frame[f"unidades_{u}"].sum() for u in UNITS] + [frame[p.replace("-", "_")].sum() for p in PROTOCOLS]
    lines += [r"\midrule", r"Total & " + " & ".join(integer(v) for v in totals) + r" \\"]
    body = "\n        ".join(lines)
    return (
        "\\begin{table}[H]\n    \\centering\n    \\footnotesize\n    \\setlength{\\tabcolsep}{4pt}\n"
        "    \\caption[Execuções previstas e realizadas por bloco e algoritmo]{Execuções previstas pela grade e "
        f"situação final na verificação de {date}, por bloco e algoritmo, e execuções com sucesso por protocolo "
        f"de avaliação. {LEGEND} A contagem por base, em execuções e em linhas de resultado, está na "
        "Tabela~\\ref{tab:apendice-execucoes}.}\n"
        "    \\label{tab:resultados-execucoes}\n"
        "    \\begin{tabular}{lrrrrrrrr}\n        \\toprule\n"
        "        & \\multicolumn{5}{c}{{\\bf Execuções}} & \\multicolumn{3}{c}{{\\bf Com sucesso, por protocolo}} \\\\\n"
        "        \\cmidrule(lr){2-6} \\cmidrule(lr){7-9}\n"
        "        {\\bf Algoritmo} &{\\bf Prev.} & {\\bf Sucesso} & {\\bf Gêm.} & {\\bf Falhas} & "
        "{\\bf Aus.} & {\\bf Estrat.} & {\\bf Temp.} & {\\bf \\textit{k-fold}} \\\\\n"
        f"        \\midrule\n        {body}\n        \\bottomrule\n    \\end{{tabular}}\n\\end{{table}}\n"
    )


def full_table(counts: pd.DataFrame, date: str) -> str:
    import estilo_figuras as estilo

    short = {"IEEE-CIS-completa-sem-alta-cardinalidade": "IEEE-CIS-completa-sem-alta-card."}
    counts = counts.assign(base=counts["dataset_id"].map(estilo.dataset_label).replace(short))
    keys = ["block_order", "bloco", "base", "algorithm_order", "algorithm"]
    columns = [f"unidades_{u}" for u in UNITS] + ["linhas_previstas", "linhas_sucesso"]
    frame = counts.groupby(keys)[columns].sum().sort_index().reset_index()
    lines, last = [], (None, None)
    for row in frame.itertuples(index=False):
        if row.bloco != last[0]:
            if last[0] is not None:
                lines.append(r"\midrule")
            lines.append(rf"\multicolumn{{9}}{{l}}{{\textit{{Bloco {BLOCKS[row.bloco]}}}}} \\")
        base = row.base if (row.bloco, row.base) != last else ""
        last = (row.bloco, row.base)
        lines.append(" & ".join([base, estilo.ALGORITHM_LATEX[row.algorithm]]
                                + [integer(getattr(row, c)) for c in columns]) + r" \\")
    lines += [r"\midrule", r"\multicolumn{2}{l}{Total} & " + " & ".join(integer(frame[c].sum()) for c in columns)
              + r" \\"]
    head = (r"& & \multicolumn{5}{c}{{\bf Execuções}} & \multicolumn{2}{c}{{\bf Linhas}} \\ "
            r"\cmidrule(lr){3-7} \cmidrule(lr){8-9} "
            r"{\bf Base} & {\bf Algoritmo} & {\bf Prev.} & {\bf Real.} & {\bf Gêm.} & "
            r"{\bf Falhas} & {\bf Aus.} & {\bf Prev.} & {\bf Real.} \\")
    body = "\n        ".join(lines)
    return (
        "{\\scriptsize\n\\setlength{\\tabcolsep}{3pt}\n\\begin{longtable}{llrrrrrrr}\n"
        "    \\caption[Execuções previstas, realizadas e com falha]{Execuções previstas pela grade, realizadas com "
        f"sucesso, gêmeas, com falha e ausentes, por bloco, base e algoritmo, na verificação de {date}, e as "
        "linhas de resultado previstas e realizadas, uma por dobra avaliada (as gêmeas não geram linha própria). "
        f"{LEGEND} A Tabela~\\ref{{tab:resultados-execucoes}} resume as "
        "contagens por bloco e algoritmo. A tabela atende também ao item G17 da revisão.}\n"
        "    \\label{tab:apendice-execucoes} \\\\\n"
        f"    \\toprule\n    {head}\n    \\midrule\n    \\endfirsthead\n"
        f"    \\toprule\n    {head}\n    \\midrule\n    \\endhead\n    \\bottomrule\n    \\endlastfoot\n"
        f"        {body}\n\\end{{longtable}}\n}}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Tabelas de execuções previstas e realizadas (G17).")
    parser.add_argument("--run-counts", required=True)
    parser.add_argument("--tables-directory", required=True)
    parser.add_argument("--verification-date", required=True)
    args = parser.parse_args()
    counts = load(Path(args.run_counts))
    out = Path(args.tables_directory)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tab_execucoes_resumo.tex").write_text(summary_table(counts, args.verification_date), encoding="utf-8")
    (out / "tab_apendice_execucoes.tex").write_text(full_table(counts, args.verification_date), encoding="utf-8")
    print(counts[[f"unidades_{u}" for u in UNITS]].sum().to_string())


if __name__ == "__main__":
    main()
