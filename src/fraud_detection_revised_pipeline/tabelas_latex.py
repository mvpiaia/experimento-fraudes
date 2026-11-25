"""Tabelas LaTeX dos Capítulos 4 e 5 da dissertação.

Lê as saídas de ``analise_hipoteses.py`` e de ``analise_ablacao_ieee.py`` e os
``resultados_<algoritmo>.csv`` e grava tabelas ``.tex`` no padrão da
dissertação (item F6 da revisão): ``booktabs``, legenda antes do ``tabular``,
vírgula decimal, três casas, sem zero negativo, nomes descritivos das bases e
siglas com ``\\acs{}`` (pacote acro).

Tabelas (prefixo ``tab_``): execuções previstas e realizadas, H1, posições de
H2 e H3, Q1, Q3 e H4. Cada uma é incluída na dissertação com ``\\input``.

Só lê arquivos; não altera manifestos nem executores.

Uso::

    python3 tabelas_latex.py --results-directory resultados \\
        --hypotheses-directory analise_hipoteses \\
        --ablation-directory analise_ablacao_ieee \\
        --output-directory <eqm_paper>/tabelas
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import estilo_figuras as estilo

H4_OUTCOMES = {
    "nao-mensuravel": "não mensurável",
    "baixo-inconclusivo": "baixo, inconclusivo",
    "relevante": "relevante",
    "inconclusivo": "inconclusivo",
}
PAIR_OUTCOMES = {**estilo.OUTCOME_LABELS, "missing": "sem par", "no-pair": "sem par"}
PROTOCOL_SHORT = {"stratified-holdout": "estrat.", "temporal-holdout": "temporal", "stratified-k-fold": "VC"}
BLOCK_LABELS = {
    "classic": "Clássicos",
    "boosting": "Aumento de gradiente",
    "anomaly": "Detecção de anomalias",
    "ft-transformer": "FT-Transformer",
    "tabbert": "TabBERT",
}


def n(value, digits: int = 3) -> str:
    return estilo.number(value, digits)


def count(value) -> str:
    """Inteiro com ponto como separador de milhar (1.410)."""
    return f"{int(value):,}".replace(",", ".")


def esc(text) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def base(value) -> str:
    return esc(estilo.dataset_label(value))


def algorithm(value) -> str:
    if pd.isna(value):
        return "---"
    return estilo.ALGORITHM_LATEX.get(value, esc(value))


def table(caption_short: str, caption: str, label: str, columns: str, header: list[str], rows: list[list[str]],
          note: str = "", colsep: str = "", size: str = "footnotesize") -> str:
    # \scriptsize em documento de 12 pt equivale a 8 pt, o mínimo legível adotado.
    lines = [
        r"\begin{table}[H]",
        r"    \centering",
        f"    \\{size}",
        *([f"    \\setlength{{\\tabcolsep}}{{{colsep}}}"] if colsep else []),
        f"    \\caption[{caption_short}]{{{caption}}}",
        f"    \\label{{{label}}}",
        f"    \\begin{{tabular}}{{{columns}}}",
        r"        \toprule",
        "        " + " & ".join(header) + r" \\",
        r"        \midrule",
    ]
    lines += ["        " + " & ".join(row) + r" \\" for row in rows]
    lines += [r"        \bottomrule", r"    \end{tabular}"]
    if note:
        lines.append(f"    \\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {note}}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------ execuções

def _key(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    def norm(value) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return ""
        text = str(value).strip()
        if text.lower() in {"", "nan", "none"}:
            return ""
        try:
            number = float(text)
        except ValueError:
            return text
        return str(int(number)) if number.is_integer() else f"{number:g}"
    return frame[columns].apply(lambda row: "|".join(norm(v) for v in row), axis=1)


def executions_table(results_directory: Path) -> str:
    import preparation_sanity_checks as sanity  # leitura da grade, sem efeitos

    expected = sanity.expected_runs()
    frames = [pd.read_csv(path, low_memory=False) for path in sorted(results_directory.glob("resultados_*.csv"))]
    results = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    for column in ("configuration_md5", "algorithm", "class_weight_intensity", "regime", "label_fraction",
                   "execution_seed", "split", "evaluation_unit", "execution_status", "failure_category"):
        if column not in results.columns:
            results[column] = np.nan
    results = results[results["split"].fillna("test").astype(str).str.startswith("test")
                      & results["evaluation_unit"].fillna("row").eq("row")].copy()
    is_draw = results["algorithm"].eq("devnet")
    results["label_draw_seed"] = results["execution_seed"].where(is_draw)
    results["class_weight_intensity"] = results["class_weight_intensity"].fillna("none")
    fraction = pd.to_numeric(results["label_fraction"], errors="coerce")
    results["label_fraction"] = fraction.where(fraction > 0)
    key_columns = list(sanity.RUN_KEY)
    results["_key"] = _key(results, key_columns)
    expected["_key"] = _key(expected, key_columns)
    status = results.groupby("_key")["execution_status"].agg(lambda s: "success" if s.eq("success").all() else "failure")
    expected["_status"] = expected["_key"].map(status).fillna("absent")
    rows = []
    for (block, alg), group in expected.groupby(["bloco", "algorithm"], sort=False):
        counts = group["_status"].value_counts()
        rows.append([BLOCK_LABELS.get(block, esc(block)), algorithm(alg), count(len(group)),
                     count(counts.get("success", 0)), count(counts.get("failure", 0)), count(counts.get("absent", 0))])
    outside = int((~results["_key"].isin(set(expected["_key"]))).sum())
    total = [r"\textbf{Total}", "", count(len(expected)), count((expected["_status"] == "success").sum()),
             count((expected["_status"] == "failure").sum()), count((expected["_status"] == "absent").sum())]
    note = ("Concluída: todas as linhas de teste da execução com sucesso. Com falha: ao menos uma linha registrada "
            "como falha (motivos no Apêndice~\\ref{cap:apendice-implementacao}). Ausente: prevista pela grade e "
            f"sem registro. Linhas de resultado fora da grade: {count(outside)}.")
    return table("Execuções previstas e realizadas", "Execuções previstas pela grade e realizadas, por bloco e algoritmo.",
                 "tab:execucoes", "llrrrr", ["Bloco", "Algoritmo", "Previstas", "Concluídas", "Com falha", "Ausentes"],
                 rows + [total], note)


# ------------------------------------------------------------ hipóteses

def h1_table(directory: Path) -> str:
    data = pd.read_csv(directory / "h1_pares.csv")
    rows = []
    for _, row in data.iterrows():
        ci = "---" if pd.isna(row.get("ci_low")) else f"[{n(row['ci_low'])}; {n(row['ci_high'])}]"
        rows.append([algorithm(row["algorithm"]), base(row["dataset_id"]), n(row.get("without_synthetic_f_beta_2")),
                     n(row.get("rus_smote_f_beta_2")), n(row.get("difference")), ci,
                     PAIR_OUTCOMES.get(str(row.get("pair_outcome")), esc(row.get("pair_outcome")))])
    return table("H1: estratégias sem dados sintéticos e RUS+SMOTE",
                 r"H1: F2 da melhor estratégia sem dados sintéticos e da \acs{rus}+\acs{smote}, diferença e IC de 95\,\%, "
                 r"por algoritmo clássico e base; margem de não inferioridade de 0,010.",
                 "tab:h1", "llrrrcl", ["Algoritmo", "Base", "Sem sint.", r"\acs{rus}+\acs{smote}", "Dif.", "IC 95\\,\\%", "Desfecho"],
                 rows, colsep="3pt", size="scriptsize")


def ranks_table(directory: Path, name: str, hypothesis: str) -> str:
    ranks = pd.read_csv(directory / f"{name}_postos.csv")
    mean = pd.read_csv(directory / f"{name}_ranking_medio.csv")
    if ranks.empty:
        return table(f"{hypothesis}: posições em F2", f"{hypothesis}: sem execuções elegíveis.", f"tab:{name}", "l",
                     ["---"], [["---"]])
    sources = list(dict.fromkeys(ranks["dataset_id"]))
    pivot_rank = ranks.pivot_table(index="algorithm", columns="dataset_id", values="rank_f_beta_2", aggfunc="first")
    pivot_f2 = ranks.pivot_table(index="algorithm", columns="dataset_id", values="f_beta_2", aggfunc="first")
    mean = mean.set_index("algorithm")
    order = mean.sort_values("mean_rank_f_beta_2").index if not mean.empty else pivot_rank.index
    rows = []
    for alg in order:
        cells = [f"{n(pivot_f2.loc[alg, s])} ({n(pivot_rank.loc[alg, s], 0)}º)" if alg in pivot_rank.index and s in pivot_rank.columns
                 and pd.notna(pivot_rank.loc[alg, s]) else "---" for s in sources]
        rows.append([algorithm(alg), *cells, n(mean.loc[alg, "mean_rank_f_beta_2"], 2) if alg in mean.index else "---"])
    return table(f"{hypothesis}: posições em F2 por fonte",
                 f"{hypothesis}: F2 no teste e posição de cada algoritmo em cada fonte de fraude (\\textit{{holdout}} temporal), "
                 "com o \\textit{ranking} médio.",
                 f"tab:{name}", "l" + "r" * (len(sources) + 1), ["Algoritmo", *[base(s) for s in sources], "Posição média"], rows)


def q1_table(directory: Path) -> str:
    data = pd.read_csv(directory / "q1_temporal.csv")
    rows = [[base(r["dataset_id"]), algorithm(r["algorithm"]), n(r["f_beta_2_stratified"]), n(r["f_beta_2_temporal"]),
             n(r["f_beta_2_temporal_minus_stratified"]), n(r["auc_pr_temporal_minus_stratified"])]
            for _, r in data.iterrows()]
    return table("Q1: holdout temporal e estratificado",
                 r"Q1: F2 no \textit{holdout} estratificado e no temporal e diferenças (temporal $-$ estratificado) de F2 e \acs{aucpr}.",
                 "tab:q1", "llrrrr", ["Base", "Algoritmo", "F2 estrat.", "F2 temporal", r"$\Delta$F2", r"$\Delta$\acs{aucpr}"], rows)


def q3_table(directory: Path) -> str:
    data = pd.read_csv(directory / "q3_familias.csv")
    rows = [[base(r["dataset_id"]), PROTOCOL_SHORT.get(r["protocol"], esc(r["protocol"])), algorithm(r["classic_leader"]), n(r["classic_f_beta_2"]),
             algorithm(r["boosting_leader"]), n(r["boosting_f_beta_2"]), n(r["boosting_minus_classic_f_beta_2"])]
            for _, r in data.iterrows()]
    return table("Q2: clássicos e aumento de gradiente",
                 "Q2: melhor algoritmo clássico e melhor algoritmo de aumento de gradiente em F2, por base e protocolo.",
                 "tab:q3", "lllrlrr", ["Base", "Protocolo", "Clássico", "F2", "Gradiente", "F2", r"$\Delta$F2"], rows,
                 note="Protocolos: estrat., \\textit{holdout} estratificado; temporal, \\textit{holdout} temporal; "
                      "VC, validação cruzada estratificada.", colsep="4pt", size="scriptsize")


def h4_table(directory: Path) -> str:
    data = pd.read_csv(directory / "h4_ablacao_geral.csv")
    rows = [[estilo.METRIC_LATEX.get(r["metrica"], esc(r["metrica"])), count(r["pares"]), n(r["diferenca_media"]),
             f"[{n(r['ic_inferior'])}; {n(r['ic_superior'])}]", n(r["margem"]),
             H4_OUTCOMES.get(r["desfecho"], esc(r["desfecho"]))] for _, r in data.iterrows()]
    summary = json.loads((directory / "resumo_h4.json").read_text(encoding="utf-8"))
    note = (f"Desfecho de H4: {estilo.OUTCOME_LABELS.get(summary.get('H4'), summary.get('H4'))}. "
            f"Pares descartados: {summary['contagem']['so_com_identidade']} só com identidade e "
            f"{summary['contagem']['so_sem_identidade']} só sem identidade.")
    return table("H4: ablação dos atributos de identidade",
                 "H4: diferença pareada média (IEEE-CIS-completa $-$ IEEE-CIS-sem-identidade), IC bootstrap de 95\\,\\% e "
                 "desfecho pela margem de equivalência fixada.",
                 "tab:h4", "lrrcrl", ["Métrica", "Pares", "Diferença", "IC 95\\,\\%", "Margem", "Desfecho"], rows, note)


def run(results: Path | None, hypotheses: Path | None, ablation: Path | None, output: Path) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    written = []
    jobs = []
    if results is not None:
        jobs.append(("tab_execucoes.tex", lambda: executions_table(results)))
    if hypotheses is not None:
        jobs += [("tab_h1.tex", lambda: h1_table(hypotheses)),
                 ("tab_h2.tex", lambda: ranks_table(hypotheses, "h2", "H2")),
                 ("tab_h3.tex", lambda: ranks_table(hypotheses, "h3", "H3")),
                 ("tab_q1.tex", lambda: q1_table(hypotheses)),
                 ("tab_q3.tex", lambda: q3_table(hypotheses))]
    if ablation is not None:
        jobs.append(("tab_h4.tex", lambda: h4_table(ablation)))
    for name, build in jobs:
        try:
            text = build()
        except (FileNotFoundError, KeyError, pd.errors.EmptyDataError) as error:
            print(f"[revisar] {name} não gerada: {error}")
            continue
        path = output / name
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-directory")
    parser.add_argument("--hypotheses-directory")
    parser.add_argument("--ablation-directory")
    parser.add_argument("--output-directory", required=True)
    a = parser.parse_args()
    as_path = lambda value: Path(value) if value else None  # noqa: E731
    written = run(as_path(a.results_directory), as_path(a.hypotheses_directory), as_path(a.ablation_directory),
                  Path(a.output_directory))
    print("\n".join(str(path) for path in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
