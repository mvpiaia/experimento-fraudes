"""Ablação dos atributos de identidade na IEEE-CIS (hipótese H4).

Regra fixada antes da execução (Capítulo 3 da dissertação, Seção
"Processamento dos dados selecionados"):

* cada resultado da IEEE-CIS-completa é pareado com o da IEEE-CIS-sem-identidade
  por algoritmo, codificação, imputação, balanceamento, protocolo e dobra; a
  diferença é "com identidade menos sem identidade";
* margens absolutas de equivalência prática: 0,010 para F1, F2, revocação e
  precisão; 0,005 para AUC-ROC;
* desfecho de cada métrica, a partir do IC bootstrap pareado de 95 % da
  diferença média:
    - ``nao-mensuravel``: IC inteiramente dentro da margem;
    - ``baixo-inconclusivo``: só a estimativa pontual dentro da margem;
    - ``relevante``: IC inteiramente fora da margem;
    - ``inconclusivo``: os demais casos;
* H4 é corroborada se todas as métricas forem ``nao-mensuravel``, refutada se
  alguma for ``relevante`` e inconclusiva nos demais casos;
* as margens são relatadas também multiplicadas por 0,5 e por 2, como
  sensibilidade; o desfecho continua sendo o das margens fixadas;
* a comparação IEEE-CIS-sem-identidade × IEEE-CIS-transacional não é pareada
  (populações diferentes) e é relatada à parte, como efeito da seleção da
  população com identidade.

Pares incompletos (sem o correspondente na outra base) são descartados e
contados. Só lê os resultados; não altera manifestos nem executores.

Uso::

    python3 analise_ablacao_ieee.py --results-directory resultados \\
        --output-directory analise_ablacao_ieee
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analise_hipoteses

WITH_IDENTITY = "ieee-cis-completa"
WITHOUT_IDENTITY = "ieee-cis-sem-identidade"
TRANSACTIONAL = "ieee-cis-transacional"
MARGINS = {"f1": 0.010, "f_beta_2": 0.010, "recall": 0.010, "precision": 0.010, "roc_auc": 0.005}
SENSITIVITY_FACTORS = (0.5, 1.0, 2.0)
PAIR_AXES = ("algorithm", "categorical_encoding_strategy", "fill_na_strategy", "undersampling_technique",
             "class_weight_intensity", "regime", "label_fraction", "numeric_normalization",
             "train_validation_test_strategy", "random_seed", "execution_seed", "fold_index")
CONFIDENCE = 0.95
SEED = 42


def outcome(low: float, high: float, point: float, margin: float) -> str:
    if np.isnan(low) or np.isnan(high):
        return "inconclusivo"
    if -margin <= low and high <= margin:
        return "nao-mensuravel"
    if low > margin or high < -margin:
        return "relevante"
    if abs(point) <= margin:
        return "baixo-inconclusivo"
    return "inconclusivo"


def hypothesis_outcome(outcomes: list[str]) -> str:
    if outcomes and all(item == "nao-mensuravel" for item in outcomes):
        return analise_hipoteses.OUTCOME_CORROBORATED
    if any(item == "relevante" for item in outcomes):
        return analise_hipoteses.OUTCOME_REFUTED
    return analise_hipoteses.OUTCOME_INCONCLUSIVE


def test_rows(results_directory: Path) -> pd.DataFrame:
    data = analise_hipoteses.load_results(results_directory)
    data = data[data["split"].eq("test")].copy()
    for column in (*PAIR_AXES, "f1"):
        if column not in data.columns:
            data[column] = np.nan
    data["f1"] = pd.to_numeric(data["f1"], errors="coerce")
    return data


def pairs(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keyed = data.assign(**{axis: data[axis].astype(str) for axis in PAIR_AXES})
    left = keyed[keyed["database_type"].eq(WITH_IDENTITY)]
    right = keyed[keyed["database_type"].eq(WITHOUT_IDENTITY)]
    metrics = list(MARGINS)
    merged = left[[*PAIR_AXES, *metrics]].merge(right[[*PAIR_AXES, *metrics]], on=list(PAIR_AXES),
                                               suffixes=("_com", "_sem"), how="outer", indicator=True)
    counts = {
        "pares": int(merged["_merge"].eq("both").sum()),
        "so_com_identidade": int(merged["_merge"].eq("left_only").sum()),
        "so_sem_identidade": int(merged["_merge"].eq("right_only").sum()),
    }
    merged = merged[merged["_merge"].eq("both")].drop(columns="_merge")
    for metric in metrics:
        merged[f"diff_{metric}"] = merged[f"{metric}_com"] - merged[f"{metric}_sem"]
    return merged, counts


def bootstrap_ci(values: np.ndarray, samples: int, rng: np.random.Generator) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    means = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    alpha = (1 - CONFIDENCE) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def summarize(paired: pd.DataFrame, samples: int, group: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    groups = [("todos", paired)] if group is None else list(paired.groupby(group))
    rows = []
    for name, frame in groups:
        for metric, margin in MARGINS.items():
            values = frame[f"diff_{metric}"].to_numpy(dtype=float)
            point = float(np.nanmean(values)) if len(values) else float("nan")
            low, high = bootstrap_ci(values, samples, rng)
            row = {"grupo": name, "metrica": metric, "pares": int(np.sum(~np.isnan(values))),
                   "diferenca_media": point, "ic_inferior": low, "ic_superior": high, "margem": margin,
                   "desfecho": outcome(low, high, point, margin)}
            for factor in SENSITIVITY_FACTORS:
                if factor != 1.0:
                    row[f"desfecho_margem_x{factor:g}"] = outcome(low, high, point, margin * factor)
            rows.append(row)
    return pd.DataFrame(rows)


def transactional(data: pd.DataFrame) -> pd.DataFrame:
    """Médias por algoritmo nas duas populações; diferença não pareada."""
    subset = data[data["database_type"].isin([WITHOUT_IDENTITY, TRANSACTIONAL])]
    if subset.empty:
        return pd.DataFrame()
    means = subset.groupby(["algorithm", "database_type"])[list(MARGINS)].mean().unstack("database_type")
    rows = []
    for algorithm, row in means.iterrows():
        record = {"algorithm": algorithm}
        for metric in MARGINS:
            without = row.get((metric, WITHOUT_IDENTITY), np.nan)
            trans = row.get((metric, TRANSACTIONAL), np.nan)
            record.update({f"{metric}_sem_identidade": without, f"{metric}_transacional": trans,
                           f"{metric}_transacional_menos_sem_identidade": trans - without})
        rows.append(record)
    return pd.DataFrame(rows)


def run(results_directory: Path, samples: int = 2000) -> tuple[dict[str, pd.DataFrame], dict]:
    data = test_rows(results_directory)
    paired, counts = pairs(data)
    overall = summarize(paired, samples)
    by_algorithm = summarize(paired, samples, group="algorithm") if not paired.empty else pd.DataFrame()
    summary = {
        "regra": "IC bootstrap pareado de 95% da diferenca media (com - sem identidade) contra as margens fixadas",
        "margens": MARGINS,
        "contagem": counts,
        "desfechos": dict(zip(overall["metrica"], overall["desfecho"])) if not overall.empty else {},
        "H4": hypothesis_outcome(list(overall["desfecho"])) if counts["pares"] else analise_hipoteses.OUTCOME_INCONCLUSIVE,
    }
    tables = {"h4_ablacao_geral": overall, "h4_ablacao_por_algoritmo": by_algorithm,
              "h4_transacional": transactional(data)}
    return tables, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-directory", required=True)
    parser.add_argument("--output-directory", default="analise_ablacao_ieee")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    arguments = parser.parse_args()
    tables, summary = run(Path(arguments.results_directory), arguments.bootstrap_samples)
    output = Path(arguments.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "resumo_h4.json").write_text(
        json.dumps(analise_hipoteses.json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"H4: {summary['H4']}; pares: {summary['contagem']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
