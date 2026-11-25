"""Tabelas do Apêndice B da dissertação (item G18) no padrão da tese.

Lê os CSVs de build_reproducibility_tables.py (hiperparametros.csv,
tempos_dobra.csv e tempos_preparacao.csv), as calibrações das máquinas, os
planos da grade e, se houver, o CSV do benchmark de inferência, e grava um
.tex por tabela. O TabBERT, trabalho futuro na dissertação, é excluído.

Uso::

    python3 tabelas_apendice_reprodutibilidade.py --tables-directory <csvs> \
        --calibration-directory <dir> --plans-directory <prepared> \
        --output-directory <eqm_paper>/tabelas [--benchmark-file <csv>]
"""

import glob
import json
import math
import os
import sys
import datetime
from pathlib import Path

import argparse

import pandas as pd

_parser = argparse.ArgumentParser(description="Tabelas do Apendice B (G18).")
_parser.add_argument("--tables-directory", required=True)
_parser.add_argument("--calibration-directory", required=True)
_parser.add_argument("--plans-directory", required=True)
_parser.add_argument("--output-directory", required=True)
_parser.add_argument("--benchmark-file", help="Um ou mais CSVs do benchmark, separados por virgula.")
ARGS = _parser.parse_args()
HERE = Path(ARGS.tables_directory)
OUT = Path(ARGS.output_directory)
OUT.mkdir(parents=True, exist_ok=True)

MACHINE = {
    "marcospiaia-Nitro-ANV15-41": "Acer",
    "marcospiaia-Latitude-3490": "Dell",
    "marcospiaia-ThinkPad-E470": "Lenovo",
    "marcospiaia-H610M-H-DDR4": "H610M",
}
ALG = {
    "logistic-regression": "Regressão logística", "naive-bayes": "Naive Bayes", "knn": r"\acs{knn}",
    "catboost": "CatBoost", "xgboost": "XGBoost", "isolation-forest": "Isolation Forest", "devnet": r"\acs{devnet}",
    "ft-transformer-mlp": r"FT-Transformer (\acs{mlp})", "ft-transformer-lstm": r"FT-Transformer (\acs{lstm})",
}
ALG_ORDER = list(ALG)
BASE = {
    "ulb": "ULB", "sparkov": "Sparkov", "sparkov-sem-titular": "Sparkov-sem-titular",
    "ieee-cis-completa": "IEEE-CIS-completa", "ieee-cis-sem-identidade": "IEEE-CIS-sem-identidade",
    "ieee-cis-transacional": "IEEE-CIS-transacional",
    "ieee-cis-completa-sem-alta-cardinalidade": "IEEE-CIS-completa-sem-alta-card.",
    "home-credit-completa": "Home-Credit-completa", "home-credit-somente-aplicacao": "Home-Credit-somente-aplic.",
}
BASE_ORDER = list(BASE)


def tt(text) -> str:
    s = str(text)
    for a, b in (("\\", r"\textbackslash{}"), ("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#"),
                 ("{", r"\{"), ("}", r"\}"), ("$", r"\$"), ("~", r"\textasciitilde{}"), ("^", r"\^{}")):
        s = s.replace(a, b)
    return r"\texttt{" + s + "}"


def num(value, digits=None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    v = float(value)
    if digits is None:
        digits = 0 if abs(v) >= 100 else (1 if abs(v) >= 10 else (2 if abs(v) >= 0.1 else 3))
    text = f"{v:,.{digits}f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return text


def decimal_values(text) -> str | None:
    """Valores não inteiros com vírgula decimal (F6); None se não for lista numérica.

    Seis algarismos significativos removem o ruído de float32 gravado pela
    biblioteca (0.300000012 vira 0,3); listas usam ponto e vírgula.
    """
    parts = [p.strip() for p in str(text).split(",")]
    try:
        values = [float(p) for p in parts]
    except ValueError:
        return None
    if not any("." in p or "e" in p.lower() for p in parts):
        return None
    def one(v: float) -> str:
        text = f"{v:.6g}"
        if "e" not in text:
            return text.replace(".", ",")
        mantissa, exponent = text.split("e")
        power = f"10^{{{int(exponent)}}}"
        return f"${power}$" if mantissa == "1" else f"${mantissa.replace('.', '{,}')} \\times {power}$"

    return "; ".join(one(v) for v in values)


def longtable(spec, caption_short, caption, label, header, rows, size=r"\scriptsize", colsep="3pt"):
    head = " & ".join(rf"{{\bf {h}}}" for h in header) + r" \\"
    body = "\n".join("        " + " & ".join(r) + r" \\" for r in rows)
    return (f"{{{size}\n\\setlength{{\\tabcolsep}}{{{colsep}}}\n\\begin{{longtable}}{{{spec}}}\n"
            f"    \\caption[{caption_short}]{{{caption}}}\n    \\label{{{label}}} \\\\\n"
            f"    \\toprule\n    {head}\n    \\midrule\n    \\endfirsthead\n"
            f"    \\toprule\n    {head}\n    \\midrule\n    \\endhead\n    \\bottomrule\n    \\endfoot\n"
            f"{body}\n\\end{{longtable}}\n}}\n")


def table(spec, caption_short, caption, label, header, rows, size=r"\footnotesize"):
    head = " & ".join(rf"{{\bf {h}}}" for h in header) + r" \\"
    body = "\n".join("        " + " & ".join(r) + r" \\" for r in rows)
    return (f"\\begin{{table}}[H]\n    \\centering\n    {size}\n    \\caption[{caption_short}]{{{caption}}}\n"
            f"    \\label{{{label}}}\n    \\begin{{tabular}}{{{spec}}}\n        \\toprule\n        {head}\n"
            f"        \\midrule\n{body}\n        \\bottomrule\n    \\end{{tabular}}\n\\end{{table}}\n")


# ---------------------------------------------------------------- hiperparâmetros
h = pd.read_csv(HERE / "hiperparametros.csv")
h = h[~h["algorithm"].str.startswith("tabbert")]
OVERRIDE = {
    ("catboost", "class_weights"): r"por partição: $[1, \sqrt{R}]$ ou $[1, R]$",
    ("catboost", "combinations_ctr"): "padrão da biblioteca",
    ("catboost", "simple_ctr"): "padrão da biblioteca",
    ("catboost", "ctr_leaf_count_limit"): "sem limite",
    ("catboost", "learning_rate"): "0,0086--0,2329, escolhida pela biblioteca por base",
    ("logistic-regression", "class_weight"): r"\texttt{None}, \texttt{balanced} ou $\{0{:}\,1, 1{:}\,\sqrt{R}\}$ por partição",
    ("xgboost", "scale_pos_weight"): r"1, $\sqrt{R}$ ou $R$ por partição (até 599,5)",
    ("ft-transformer-mlp", "training"): "AdamW, taxa $10^{-4}$, 10 épocas, entropia cruzada, \\texttt{bfloat16}, lote de 10 a 256 conforme a base",
    ("ft-transformer-lstm", "training"): "AdamW, taxa $10^{-4}$, 10 épocas, entropia cruzada, \\texttt{bfloat16}, lote de 10 a 256 conforme a base",
}
KIND = {"hyperparameter": "fixo", "arm": "braço"}
rows = []
for alg in ALG_ORDER:
    part = h[h["algorithm"].eq(alg)].sort_values(["kind", "name"], ascending=[False, True])
    first = True
    for _, r in part.iterrows():
        key = (alg, r["name"])
        if key in OVERRIDE:
            value = OVERRIDE[key]
        elif r["name"] == "architecture":
            arch = json.loads(r["value"])
            value = (f"cabeça {arch['classifier_head'].upper()}, dimensão {arch['dimension']}, "
                     f"{arch['heads']} cabeças, {arch['layers']} camadas, \\textit{{dropout}} {str(arch['dropout']).replace('.', ',')}")
        elif pd.isna(r["value"]):
            value = tt("None")
        else:
            value = decimal_values(r["value"]) or tt(r["value"])
        rows.append([ALG[alg] if first else "", KIND.get(r["kind"], r["kind"]), tt(r["name"]), value])
        first = False
(OUT / "tab_hiperparametros.tex").write_text(longtable(
    "lllp{6.0cm}", "Hiperparâmetros efetivos por algoritmo",
    "Hiperparâmetros efetivos por algoritmo, lidos dos resultados, inclusive os padrões resolvidos pela biblioteca. "
    "Fixo: igual em todas as execuções; braço: eixo de execução da grade. $R$ é a razão entre negativos e positivos "
    "no treinamento de cada partição. A origem de cada configuração está na Tabela~\\ref{tab:origem-hiperparametros}.",
    "tab:g18-hiperparametros", ["Algoritmo", "Tipo", "Nome", "Valor"], rows))

# ---------------------------------------------------------------- calibração
cal = {}
for f in glob.glob(str(Path(ARGS.calibration_directory) / "calibration_*.json")):
    j = json.load(open(f))
    cal[MACHINE[j["execution_host"]]] = j
ref = cal["Acer"]["families"]
rows = []
for m in ("Acer", "Dell", "Lenovo", "H610M"):
    j = cal[m]
    fam = j["families"]
    rows.append([m, str(j["cpu_lanes"]), str(j["gpu_lanes"])] +
                [num(fam[k]["seconds"] / ref[k]["seconds"], 2) for k in ("linear", "knn", "boosting", "anomaly")])
(OUT / "tab_calibracao.tex").write_text(table(
    "lrrrrrr", "Calibração das máquinas",
    "Fator de velocidade de cada máquina por família de algoritmo: tempo da carga de calibração, com tantas cópias "
    "simultâneas quantos são os processos de CPU da máquina, dividido pelo do Acer. Valores abaixo de 1 indicam "
    "máquina mais rápida que o Acer na família.",
    "tab:g18-calibracao", ["Máquina", "Proc. CPU", "Proc. GPU", "Linear", r"\acs{knn}", r"\textit{Boosting}", "Anomalia"], rows))

# ---------------------------------------------------------------- planos
rows = []
for plan in ("fase1", "pesada", "fase2", "pesada2", "lenovo-dell", "ldh", "medicao1", "medicao2"):
    f = str(Path(ARGS.plans_directory) / f"grid_plan-{plan}" / "grid_plan.json")
    j = json.load(open(f))
    date = datetime.datetime.fromtimestamp(os.path.getmtime(f)).strftime("%d/%m")
    lanes, longest = [], 0.0
    for host, m in j["machines"].items():
        for lane, l in m["lanes"].items():
            pairs = [p for p in l["pairs"] if not p["algorithm"].startswith("tabbert")]
            if not pairs:
                continue
            seconds = sum(p["estimated_seconds"] for p in pairs)
            longest = max(longest, seconds)
            lanes.append(f"{MACHINE[host]} {lane.rsplit('-', 1)[1]}: {len(pairs)}")
    rows.append([plan, date, "; ".join(lanes), num(longest / 3600, 1)])
(OUT / "tab_plano.tex").write_text(table(
    "llp{8.4cm}r", "Planos de divisão da grade",
    "Planos de divisão da grade, na ordem em que foram executados: número de pares (algoritmo, configuração) "
    "atribuídos a cada processo e tempo estimado do processo mais longo. Os planos \\texttt{fase1} e \\texttt{fase2} "
    "são os das duas fases da grade; os demais redistribuíram pares pesados ou ainda não executados.",
    "tab:g18-plano", ["Plano", "Data", "Processos e pares", "Tempo est. (h)"], rows))

# ---------------------------------------------------------------- tempos por dobra
d = pd.read_csv(HERE / "tempos_dobra.csv")
d = d[~d["algorithm"].str.startswith("tabbert")].copy()
d["a"] = d["algorithm"].map(ALG_ORDER.index)
d["b"] = d["dataset_id"].map(BASE_ORDER.index)
d["m"] = d["execution_host"].map(MACHINE)
d = d.sort_values(["a", "b", "m"])
rows, last = [], (None, None)
for _, r in d.iterrows():
    alg = ALG[r["algorithm"]] if r["algorithm"] != last[0] else ""
    base = BASE[r["dataset_id"]] if (r["algorithm"], r["dataset_id"]) != last else ""
    last = (r["algorithm"], r["dataset_id"])
    rows.append([alg, base, r["m"], str(int(r["folds"])),
                 f"{num(r['fit_median_s'])} [{num(r['fit_q1_s'])}; {num(r['fit_q3_s'])}]",
                 f"{num(r['scoring_median_s'])} [{num(r['scoring_q1_s'])}; {num(r['scoring_q3_s'])}]"])
(OUT / "tab_tempos_dobra.tex").write_text(longtable(
    "lllrll", "Tempos de ajuste e de pontuação por dobra",
    "Tempos de ajuste e de pontuação por dobra (mediana e quartis, em segundos), por algoritmo, base e máquina, "
    "medidos com a máquina carregada pelos demais processos.",
    "tab:g18-tempos-dobra", ["Algoritmo", "Base", "Máq.", "Dobras", "Ajuste (s)", "Pontuação (s)"], rows, colsep="2pt"))

# ---------------------------------------------------------------- tempos de preparação
p = pd.read_csv(HERE / "tempos_preparacao.csv")
p = p[~p["categorical_handling_mode"].eq("sequence-tokenized")].copy()
MODE = {"encoded": "codificado", "native": "nativo", "transformer-indexed": "FT-Transformer"}
p["b"] = p["dataset_id"].map(BASE_ORDER.index)
p = p.sort_values(["b", "categorical_handling_mode"])
rows, lastb = [], None
for _, r in p.iterrows():
    base = BASE[r["dataset_id"]] if r["dataset_id"] != lastb else ""
    lastb = r["dataset_id"]
    rows.append([base, MODE[r["categorical_handling_mode"]], MACHINE[r["execution_host"]], str(int(r["configurations"])),
                 f"{num(r['median_s'])} [{num(r['q1_s'])}; {num(r['q3_s'])}]"])
(OUT / "tab_tempos_preparacao.tex").write_text(table(
    "lllrl", "Tempos de preparação por configuração",
    "Tempo de preparação por configuração (mediana e quartis, em segundos), por base, modo de preparação e máquina.",
    "tab:g18-tempos-preparacao", ["Base", "Modo", "Máquina", "Configurações", "Preparação (s)"], rows, size=r"\scriptsize"))

# ---------------------------------------------------------------- inferência (opcional)
bench = ARGS.benchmark_file
if bench and all(Path(path).exists() for path in bench.split(",")):
    b = pd.concat([pd.read_csv(path) for path in str(bench).split(",")], ignore_index=True)
    b = b[b["algorithm"].isin(ALG_ORDER)].copy()
    b["a"] = b["algorithm"].map(ALG_ORDER.index)
    b["bb"] = b["dataset_id"].map(BASE_ORDER.index)
    b = b.sort_values(["a", "bb"])
    rows, lasta = [], None
    for _, r in b.iterrows():
        alg = ALG[r["algorithm"]] if r["algorithm"] != lasta else ""
        lasta = r["algorithm"]
        rows.append([alg, BASE[r["dataset_id"]], MACHINE.get(r["execution_host"], r["execution_host"]),
                     num(r["inference_ms_per_1000"]), num(r["latency_p50_ms"]),
                     num(r["latency_p95_ms"]), f"[{num(r['latency_p95_ci_low_ms'])}; {num(r['latency_p95_ci_high_ms'])}]"])
    (OUT / "tab_inferencia.tex").write_text(longtable(
        "lllrrrl", "Tempo de inferência por algoritmo e base",
        "Tempo de inferência no Dell e no Lenovo, de \\textit{hardware} idêntico: milissegundos por 1.000 transações "
        "em lote, numa amostra fixa de 10.000 transações do teste, e latência de uma transação "
        "(percentis 50 e 95 e intervalo de confiança de 95\\% do percentil 95, sobre 1.000 transações).",
        "tab:g18-inferencia", ["Algoritmo", "Base", "Máq.", "ms / 1.000", "p50 (ms)", "p95 (ms)", "IC do p95 (ms)"], rows,
        size=r"\scriptsize", colsep="3pt"))
print(sorted(x.name for x in OUT.iterdir()))
