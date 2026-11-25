"""Comparação dos braços ``native`` e ``median`` do DevNet.

A rede do DevNet não aceita ausentes: o executor imputa a mediana do treino.
Até 12/11/2026 isso tornava o braço ``native`` (ausentes deixados ao
estimador) idêntico ao ``median`` da preparação sempre que os ausentes da
partição eram numéricos (R2 da validação ``results``). Desde então, cada
atributo com ausentes no treino ganha um indicador ``__is_missing``, como no
FT-Transformer (``label_free_encoding.LabelFreeEncoder``), e a diferença entre
os dois braços mede o efeito de informar a rede de que o valor não foi
observado.

Regra:

* pareia cada execução ``native`` com a ``median`` de mesma base, protocolo,
  semente de preparação, dobra, fração rotulada e semente do sorteio; a
  diferença é ``native - median``;
* a linha avaliada é a do teste e, na validação cruzada estratificada, a da
  validação da dobra, que é o conjunto avaliado nela;
* o IC de 95 % da diferença média é bootstrap sobre as partições (base,
  protocolo, semente e dobra): as frações e sementes de sorteio de uma mesma
  partição compartilham os dados e não são independentes;
* conta também os pares com métricas idênticas, que deveriam desaparecer com
  o indicador.

Só lê os resultados; não altera manifestos nem executores.

Uso::

    python3 analise_ausentes_devnet.py --results-directory resultados \\
        --output-directory analise_ausentes_devnet --latex-output tabela.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import analise_hipoteses

ALGORITHM = "devnet"
NATIVE = "native"
MEDIAN = "median"
K_FOLD = "stratified-k-fold"
METRICS = ("f_beta_2", "auc_pr", "roc_auc", "recall", "precision")
PARTITION = ("database_type", "train_validation_test_strategy", "random_seed", "fold_index")
PAIR_AXES = (*PARTITION, "categorical_encoding_strategy", "undersampling_technique",
             "label_fraction", "execution_seed")
CONFIDENCE = 0.95
SEED = 42


def evaluation_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Linhas avaliadas: teste, ou validação da dobra no k-fold."""
    protocol = data["train_validation_test_strategy"].astype(str)
    split = data["split"].astype(str)
    return data[split.eq("test") | (protocol.eq(K_FOLD) & split.eq("validation"))]


def load(results_directories: list[Path]) -> pd.DataFrame:
    frames = [analise_hipoteses.load_results(directory) for directory in results_directories]
    data = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()
    for column in (*PAIR_AXES, "algorithm", "fill_na_strategy", "split", *METRICS):
        if column not in data.columns:
            data[column] = np.nan
    data = data[data["algorithm"].eq(ALGORITHM) & data["fill_na_strategy"].isin([NATIVE, MEDIAN])]
    data = evaluation_rows(data)
    for metric in METRICS:
        data[metric] = pd.to_numeric(data[metric], errors="coerce")
    return data


def pairs(data: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    keyed = data.assign(**{axis: data[axis].astype(str) for axis in PAIR_AXES})
    native = keyed[keyed["fill_na_strategy"].eq(NATIVE)][[*PAIR_AXES, *METRICS]]
    median = keyed[keyed["fill_na_strategy"].eq(MEDIAN)][[*PAIR_AXES, *METRICS]]
    merged = native.merge(median, on=list(PAIR_AXES), suffixes=("_native", "_median"),
                          how="outer", indicator=True)
    counts = {
        "pares": int(merged["_merge"].eq("both").sum()),
        "so_native": int(merged["_merge"].eq("left_only").sum()),
        "so_median": int(merged["_merge"].eq("right_only").sum()),
    }
    merged = merged[merged["_merge"].eq("both")].drop(columns="_merge").copy()
    for metric in METRICS:
        merged[f"diff_{metric}"] = merged[f"{metric}_native"] - merged[f"{metric}_median"]
    merged["identicos"] = np.all(
        [np.isclose(merged[f"{metric}_native"], merged[f"{metric}_median"], rtol=0, atol=0, equal_nan=True)
         for metric in METRICS], axis=0,
    ) if not merged.empty else pd.Series(dtype=bool)
    return merged, counts


def cluster_bootstrap_ci(frame: pd.DataFrame, column: str, samples: int,
                         rng: np.random.Generator) -> tuple[float, float]:
    """IC da média reamostrando partições inteiras."""
    clusters = [group[column].to_numpy(dtype=float) for _key, group in frame.groupby(list(PARTITION))]
    clusters = [values[~np.isnan(values)] for values in clusters]
    clusters = [values for values in clusters if len(values)]
    if len(clusters) < 2:
        return float("nan"), float("nan")
    sums = np.array([values.sum() for values in clusters])
    sizes = np.array([len(values) for values in clusters])
    draws = rng.integers(0, len(clusters), size=(samples, len(clusters)))
    means = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    alpha = (1 - CONFIDENCE) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def summarize(paired: pd.DataFrame, samples: int, group: list[str] | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    groups = [(("todos",), paired)] if not group else [
        (key if isinstance(key, tuple) else (key,), frame) for key, frame in paired.groupby(group)
    ]
    rows = []
    for key, frame in groups:
        labels = dict(zip(group, key)) if group else {"grupo": key[0]}
        for metric in METRICS:
            values = frame[f"diff_{metric}"].to_numpy(dtype=float)
            low, high = cluster_bootstrap_ci(frame, f"diff_{metric}", samples, rng)
            rows.append({
                **labels, "metrica": metric, "pares": int(np.sum(~np.isnan(values))),
                "particoes": int(frame.groupby(list(PARTITION)).ngroups),
                "pares_identicos": int(frame["identicos"].sum()),
                "media_native": float(frame[f"{metric}_native"].mean()),
                "media_median": float(frame[f"{metric}_median"].mean()),
                "diferenca_media": float(np.nanmean(values)) if len(values) else float("nan"),
                "ic_inferior": low, "ic_superior": high,
            })
    return pd.DataFrame(rows)


def _number(value: float, signed: bool = False) -> str:
    if np.isnan(value):
        return "--"
    text = f"{value:+.3f}" if signed else f"{value:.3f}"
    return text.replace(".", "{,}")


def latex_table(by_dataset: pd.DataFrame) -> str:
    """Tabela da dissertação: ΔF2 e ΔAUC-PR por base, com IC e pares idênticos."""
    lines = [
        "% Gerada por analise_ausentes_devnet.py; não editar à mão.",
        "\\begin{tabular}{lrrcc}",
        "\\toprule",
        "Base & Pares & Idênticos & $\\Delta$F2 [IC 95\\%] & $\\Delta$AUC-PR [IC 95\\%] \\\\",
        "\\midrule",
    ]
    for dataset, frame in by_dataset.groupby("database_type"):
        metric = frame.set_index("metrica")
        cells = []
        for name in ("f_beta_2", "auc_pr"):
            row = metric.loc[name]
            cells.append(f"${_number(row['diferenca_media'], True)}$ "
                         f"[${_number(row['ic_inferior'], True)}$; ${_number(row['ic_superior'], True)}$]")
        first = metric.iloc[0]
        lines.append(f"\\texttt{{{dataset}}} & {int(first['pares'])} & {int(first['pares_identicos'])} & "
                     + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def run(results_directories: list[Path], samples: int = 2000) -> tuple[dict[str, pd.DataFrame], dict]:
    paired, counts = pairs(load(results_directories))
    tables = {
        "devnet_ausentes_pares": paired,
        "devnet_ausentes_geral": summarize(paired, samples),
        "devnet_ausentes_por_base": summarize(paired, samples, ["database_type"]),
        "devnet_ausentes_por_base_e_fracao": summarize(paired, samples, ["database_type", "label_fraction"]),
    }
    summary = {
        "regra": "diferenca native - median; IC bootstrap de 95% sobre particoes",
        "contagem": counts,
        "pares_identicos": int(paired["identicos"].sum()) if not paired.empty else 0,
    }
    return tables, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-directory", required=True, nargs="+")
    parser.add_argument("--output-directory", default="analise_ausentes_devnet")
    parser.add_argument("--latex-output")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    arguments = parser.parse_args()
    tables, summary = run([Path(path) for path in arguments.results_directory], arguments.bootstrap_samples)
    output = Path(arguments.output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    (output / "resumo_devnet_ausentes.json").write_text(
        json.dumps(analise_hipoteses.json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    if arguments.latex_output:
        Path(arguments.latex_output).write_text(latex_table(tables["devnet_ausentes_por_base"]), encoding="utf-8")
    print(f"pares: {summary['contagem']}; identicos: {summary['pares_identicos']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
