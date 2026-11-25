"""Figuras do Capítulo 5 da dissertação a partir dos resultados da grade revisada.

Substitui, para a grade revisada, as figuras de ``gerar_discussao_resultados.py``
(escrito sobre a grade de 22/09) e cobre as cinco famílias de algoritmos.
Segue o padrão de ``estilo_figuras`` (itens F6 e F7 da revisão).

Figuras gravadas (prefixo ``discussao_``):

* ``desempenho_<familia>.png``: F2, AUC-PR, AUC-ROC, precisão, revocação e F1
  por base e algoritmo (média entre configurações, IC bootstrap de 95 %);
* ``balanceamento_classicos.png``: diferença pareada de cada braço de
  balanceamento ou ponderação em relação à linha de base sem reamostragem;
* ``codificacao_classicos.png``: F2 médio por célula do fatorial 2x2;
* ``imputacao.png``: F2 médio por estratégia de faltantes;
* ``protocolo.png``: diferença pareada holdout temporal − estratificado;
* ``anomalia_rotulos.png``: DevNet por fração de fraudes rotuladas, com a
  dispersão entre os sorteios de rótulos, ao lado do Isolation Forest;
* ``transformers.png``: FT-Transformer por normalização e cabeça. O TabBERT
  fica como trabalho futuro e é excluído na leitura (``analise_hipoteses``).

Também grava ``figuras_discussao_resumo.json`` com os números das figuras.

Só lê os ``resultados_<algoritmo>.csv``; não altera manifestos nem executores.

Uso::

    python3 figuras_discussao.py --results-directory resultados \\
        --figures-directory <eqm_paper>/figuras --summary-output resumo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

import analise_hipoteses
import estilo_figuras as estilo

plt = estilo.plt

FAMILIES = {
    "classicos": analise_hipoteses.CLASSIC_ALGORITHMS,
    "boosting": analise_hipoteses.BOOSTING_ALGORITHMS,
    "anomalia": ("isolation-forest", "devnet"),
    "transformers": analise_hipoteses.FT_TRANSFORMER_ALGORITHMS,
}
PANEL_METRICS = ("f_beta_2", "auc_pr", "roc_auc", "precision", "recall", "f1")
EXTRA_AXES = ("numeric_normalization", "high_cardinality_strategy", "execution_seed", "random_seed")
# Eixos do FT-Transformer que os resultados não gravam e vêm do manifesto (--manifest).
MANIFEST_AXES = ("numeric_normalization", "high_cardinality_strategy")
CARDINALITY_LABELS = {"minimum-frequency": "frequência mínima", "hashing": "hashing"}
BASELINE_ARM = "none"
# Rótulos curtos das células do fatorial de codificação, para caber no mapa de calor.
ENCODING_SHORT = {
    "smoothed-mean:k-fold": "MS | k-fold",
    "smoothed-mean:permutation-prefix": "MS | prefixo",
    "log-odds:k-fold": "LO | k-fold",
    "log-odds:permutation-prefix": "LO | prefixo",
}
PAIR_AXES = ("database_type", "algorithm", "train_validation_test_strategy",
             "categorical_encoding_strategy", "fill_na_strategy", "random_seed")


def _manifest_axes(manifest: Path) -> pd.DataFrame:
    """Eixos de ``MANIFEST_AXES`` por ``configuration_md5``, lidos do manifesto da preparação."""
    axes = pd.read_csv(manifest, sep="\t", usecols=["configuration_md5", *MANIFEST_AXES], dtype=str)
    axes = axes.replace("", np.nan).dropna(subset=list(MANIFEST_AXES), how="all")
    return axes.drop_duplicates("configuration_md5")


def load(results_directory: Path, manifest: Path | None = None) -> pd.DataFrame:
    """Uma linha por execução: média das dobras de teste, com os eixos da grade."""
    data = analise_hipoteses.load_results(results_directory)
    data = data[data["split"].eq("test")].copy()
    if manifest is not None:
        data = data.drop(columns=[c for c in MANIFEST_AXES if c in data.columns])
        data = data.merge(_manifest_axes(manifest), on="configuration_md5", how="left")
    for column in ("f1", *EXTRA_AXES):
        if column not in data.columns:
            data[column] = np.nan
    data["f1"] = pd.to_numeric(data["f1"], errors="coerce")
    keys = [*analise_hipoteses.KEY_COLUMNS, *EXTRA_AXES]
    filled = data[keys].astype(object).where(data[keys].notna(), "∅")
    data = data.assign(**{column: filled[column] for column in keys})
    executions = data.groupby(keys, as_index=False)[list(PANEL_METRICS)].mean()
    return executions.replace("∅", np.nan)


def _order(values, reference) -> list:
    present = [value for value in reference if value in set(values)]
    return present + sorted(set(values) - set(present), key=str)


def _point_panels(data, x, hue, x_order, hue_order, metrics, path, ncols=3):
    # Bases no eixo vertical: os nomes longos ficam legíveis e aparecem uma vez por linha.
    rows = int(np.ceil(len(metrics) / ncols))
    height = 0.6 + 0.26 * len(x_order) * max(1.0, len(hue_order) / 3)
    fig, axes = plt.subplots(rows, ncols, figsize=(estilo.PAGE_WIDTH_IN, height * rows), sharey=True, squeeze=False)
    plot = data.copy()
    plot["_x"] = plot[x].map(estilo.dataset_label if x == "database_type" else estilo.label)
    plot["_hue"] = plot[hue].map(estilo.label)
    x_labels = [estilo.dataset_label(v) if x == "database_type" else estilo.label(v) for v in x_order]
    hue_labels = [estilo.label(v) for v in hue_order]
    for metric, axis in zip(metrics, axes.flat):
        sns.pointplot(data=plot, y="_x", x=metric, hue="_hue", order=x_labels, hue_order=hue_labels,
                      orient="h", errorbar=("ci", 95), n_boot=1000, seed=42, dodge=0.45,
                      markers=list(estilo.MARKERS[:len(hue_labels)]),
                      linestyles="none", capsize=0.1, err_kws={"linewidth": 0.8}, markersize=4, ax=axis)
        axis.set_title(estilo.METRIC_LABELS[metric])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xlim(-0.02, 1.02)
        axis.set_xticks([0, 0.5, 1])
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    for axis in list(axes.flat)[len(metrics):]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    estilo.legend_below(fig, handles, labels)
    estilo.save(fig, path)


def performance_figures(executions: pd.DataFrame, figures: Path, summary: dict) -> None:
    databases = _order(executions["database_type"], analise_hipoteses.FRAUD_SOURCES)
    for family, algorithms in FAMILIES.items():
        data = executions[executions["algorithm"].isin(algorithms)]
        if data.empty:
            print(f"[revisar] sem resultados da família {family}")
            continue
        order = _order(data["database_type"], databases)
        hue = [algorithm for algorithm in algorithms if algorithm in set(data["algorithm"])]
        _point_panels(data, "database_type", "algorithm", order, hue, PANEL_METRICS,
                      figures / f"discussao_desempenho_{family}.png")
        means = data.groupby(["database_type", "algorithm"])[list(PANEL_METRICS)].mean()
        summary[f"desempenho_{family}"] = [
            {"database_type": index[0], "algorithm": index[1], **{k: float(v) for k, v in row.items()}}
            for index, row in means.iterrows()
        ]


def _paired(data: pd.DataFrame, column: str, level: str, baseline: str, extra: dict | None = None) -> pd.DataFrame:
    """Diferença ``level − baseline`` em execuções iguais nos demais eixos."""
    axes = [axis for axis in PAIR_AXES if axis in data.columns]
    left = data[data[column].eq(level)]
    right = data[data[column].eq(baseline)]
    for key, value in (extra or {}).items():
        left = left[left[key].eq(value)]
        right = right[right[key].eq(value)]
    keyed = lambda frame: frame.assign(**{a: frame[a].astype(str) for a in axes})  # noqa: E731
    merged = keyed(left).merge(keyed(right), on=axes, suffixes=("", "_base"))
    for metric in ("f_beta_2", "auc_pr"):
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_base"]
    merged["arm"] = level
    return merged


def balancing_figure(executions: pd.DataFrame, figures: Path, summary: dict) -> None:
    classic = executions[executions["algorithm"].isin(FAMILIES["classicos"])]
    if classic.empty:
        return
    # O braço sem reamostragem chega vazio da leitura (normalize_results preenche
    # as chaves ausentes com ""); aqui ele é a linha de base "none".
    technique = classic["undersampling_technique"].replace("", BASELINE_ARM).fillna(BASELINE_ARM)
    classic = classic.assign(undersampling_technique=technique)
    unweighted = classic[classic["class_weight_intensity"].fillna("none").eq("none")]
    deltas = [
        _paired(unweighted, "undersampling_technique", arm, BASELINE_ARM)
        for arm in ("random-undersampling", "RUS+SMOTE", "tomek-links")
    ]
    no_resampling = classic[classic["undersampling_technique"].eq(BASELINE_ARM)].copy()
    no_resampling["class_weight_intensity"] = no_resampling["class_weight_intensity"].fillna("none")
    for weight in ("sqrt-balanced", "balanced"):
        deltas.append(_paired(no_resampling, "class_weight_intensity", weight, "none"))
    data = pd.concat([d for d in deltas if not d.empty], ignore_index=True)
    if data.empty:
        print("[revisar] sem pares para o balanceamento")
        return
    arms = _order(data["arm"], ("random-undersampling", "RUS+SMOTE", "tomek-links", "sqrt-balanced", "balanced"))
    algorithms = [a for a in FAMILIES["classicos"] if a in set(data["algorithm"])]
    data["_arm"] = data["arm"].map(estilo.label)
    data["_alg"] = data["algorithm"].map(estilo.label)
    # Uma linha por algoritmo, com escala própria: as diferenças do Naive Bayes
    # ficam quase todas em ±0,01 e sumiam na escala da regressão logística e do
    # KNN, que chegam a −0,6. Cada algoritmo mostra só os braços que tem.
    colors = dict(zip(algorithms, plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    metrics = ("f_beta_2", "auc_pr")
    fig, axes = plt.subplots(len(algorithms), 2, figsize=(estilo.PAGE_WIDTH_IN, 2.2 * len(algorithms)),
                             sharey=False, squeeze=False)
    for row, algorithm in enumerate(algorithms):
        part = data[data["algorithm"].eq(algorithm)]
        order = [estilo.label(a) for a in arms if a in set(part["arm"])]
        for column, metric in enumerate(metrics):
            axis = axes[row][column]
            sns.boxplot(data=part, x="_arm", y=f"delta_{metric}", order=order, color=colors[algorithm],
                        fliersize=1.5, linewidth=0.7, ax=axis)
            axis.axhline(0, color="0.3", linewidth=0.8, linestyle="--")
            axis.set_xlabel("")
            axis.set_ylabel(estilo.label(algorithm) if column == 0 else "")
            if len(order) > 3:
                axis.tick_params(axis="x", rotation=25)
            if row == 0:
                axis.set_title(f"Diferença pareada de {estilo.METRIC_LABELS[metric]}")
    fig.tight_layout()
    estilo.save(fig, figures / "discussao_balanceamento_classicos.png")
    summary["balanceamento_classicos"] = (
        data.groupby(["algorithm", "arm"])[["delta_f_beta_2", "delta_auc_pr"]]
        .agg(["mean", "median", "size"]).round(6).reset_index().to_dict(orient="records")
    )


def _heatmap(matrix: pd.DataFrame, axis, title: str) -> None:
    annot = matrix.map(lambda v: "" if pd.isna(v) else estilo.number_plain(v, 2))
    sns.heatmap(matrix, annot=annot, fmt="", vmin=0, vmax=1, cmap="cividis", cbar=False,
                linewidths=0.4, linecolor="white", annot_kws={"size": estilo.FONT_SIZE - 2}, ax=axis)
    axis.set_title(title)
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=90)
    axis.tick_params(axis="y", rotation=0)


def encoding_figure(executions: pd.DataFrame, figures: Path, summary: dict) -> None:
    # Bases sem atributos categóricos chegam com o eixo vazio e ficam fora do fatorial.
    classic = executions[executions["algorithm"].isin(FAMILIES["classicos"])
                         & executions["categorical_encoding_strategy"].fillna("").ne("")]
    if classic.empty:
        print("[revisar] sem bases com codificação categórica")
        return
    algorithms = [a for a in FAMILIES["classicos"] if a in set(classic["algorithm"])]
    fig, axes = plt.subplots(1, len(algorithms), figsize=(estilo.PAGE_WIDTH_IN, 2.9), squeeze=False)
    records = []
    for algorithm, axis in zip(algorithms, axes.flat):
        data = classic[classic["algorithm"].eq(algorithm)]
        matrix = data.pivot_table(index="database_type", columns="categorical_encoding_strategy",
                                  values="f_beta_2", aggfunc="mean")
        matrix = matrix.reindex(index=_order(matrix.index, analise_hipoteses.FRAUD_SOURCES))
        records += [{"algorithm": algorithm, "encoding": c, "database_type": i, "f_beta_2": float(v)}
                    for i, row in matrix.iterrows() for c, v in row.items() if pd.notna(v)]
        matrix.index = [estilo.dataset_label(v) for v in matrix.index]
        matrix.columns = [ENCODING_SHORT.get(v, estilo.label(v)) for v in matrix.columns]
        _heatmap(matrix, axis, estilo.label(algorithm))
        if axis is not axes.flat[0]:
            axis.set_yticklabels([])
    # Sem título interno: a legenda da figura na dissertação já o diz, com as siglas MS e LO.
    fig.tight_layout()
    estilo.save(fig, figures / "discussao_codificacao_classicos.png")
    summary["codificacao_classicos"] = records


def imputation_figure(executions: pd.DataFrame, figures: Path, summary: dict) -> None:
    # Bases sem valores faltantes chegam com o eixo vazio e ficam fora da comparação.
    data = executions[executions["fill_na_strategy"].fillna("").ne("")
                      & executions["algorithm"].isin(FAMILIES["classicos"] + FAMILIES["boosting"])]
    if data.empty:
        return
    algorithms = [a for a in FAMILIES["classicos"] + FAMILIES["boosting"] if a in set(data["algorithm"])]
    ncols = 3
    rows = int(np.ceil(len(algorithms) / ncols))
    fig, axes = plt.subplots(rows, ncols, figsize=(estilo.PAGE_WIDTH_IN, 2.4 * rows), squeeze=False)
    records = []
    for algorithm, axis in zip(algorithms, axes.flat):
        subset = data[data["algorithm"].eq(algorithm)]
        matrix = subset.pivot_table(index="database_type", columns="fill_na_strategy", values="f_beta_2", aggfunc="mean")
        matrix = matrix.reindex(index=_order(matrix.index, analise_hipoteses.FRAUD_SOURCES))
        records += [{"algorithm": algorithm, "fill_na_strategy": c, "database_type": i, "f_beta_2": float(v)}
                    for i, row in matrix.iterrows() for c, v in row.items() if pd.notna(v)]
        matrix.index = [estilo.dataset_label(v) for v in matrix.index]
        matrix.columns = [estilo.label(v) for v in matrix.columns]
        _heatmap(matrix, axis, estilo.label(algorithm))
        if list(axes.flat).index(axis) % ncols:
            axis.set_yticklabels([])
    for axis in list(axes.flat)[len(algorithms):]:
        axis.set_visible(False)
    fig.tight_layout()
    estilo.save(fig, figures / "discussao_imputacao.png")
    summary["imputacao"] = records


def protocol_figure(executions: pd.DataFrame, figures: Path, summary: dict) -> None:
    data = executions[executions["algorithm"].isin(analise_hipoteses.SUPERVISED_ALGORITHMS)]
    axes_cols = [a for a in ("database_type", "algorithm", "undersampling_technique", "class_weight_intensity",
                             "categorical_encoding_strategy", "fill_na_strategy", "regime",
                             "numeric_normalization", "random_seed") if a in data.columns]
    keyed = data.assign(**{a: data[a].astype(str) for a in axes_cols})
    temporal = keyed[keyed["train_validation_test_strategy"].eq("temporal-holdout")]
    stratified = keyed[keyed["train_validation_test_strategy"].eq("stratified-holdout")]
    merged = temporal.merge(stratified, on=axes_cols, suffixes=("", "_estrat"))
    if merged.empty:
        print("[revisar] sem pares temporal x estratificado")
        return
    for metric in ("f_beta_2", "auc_pr"):
        merged[f"delta_{metric}"] = merged[metric] - merged[f"{metric}_estrat"]
    algorithms = [a for a in analise_hipoteses.SUPERVISED_ALGORITHMS if a in set(merged["algorithm"])]
    bases = _order(merged["database_type"], analise_hipoteses.FRAUD_SOURCES)
    merged["_base"] = merged["database_type"].map(estilo.dataset_label)
    merged["_alg"] = merged["algorithm"].map(estilo.label)
    height = 0.8 + 0.22 * len(bases) * len(algorithms) / 2
    fig, axes = plt.subplots(1, 2, figsize=(estilo.PAGE_WIDTH_IN, height), sharey=True)
    for metric, axis in zip(("f_beta_2", "auc_pr"), axes):
        sns.pointplot(data=merged, y="_base", x=f"delta_{metric}", hue="_alg", orient="h",
                      order=[estilo.dataset_label(b) for b in bases], hue_order=[estilo.label(a) for a in algorithms],
                      errorbar=("ci", 95), n_boot=1000, seed=42, dodge=0.7, linestyles="none",
                      markers=list(estilo.MARKERS[:len(algorithms)]), markersize=4, capsize=0.05,
                      err_kws={"linewidth": 0.8}, ax=axis)
        axis.axvline(0, color="0.3", linewidth=0.8, linestyle="--")
        axis.set_title(estilo.METRIC_LABELS[metric])
        axis.set_xlabel("temporal − estratificado")
        axis.set_ylabel("")
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    estilo.legend_below(fig, handles, labels)
    estilo.save(fig, figures / "discussao_protocolo.png")
    summary["protocolo"] = (merged.groupby(["database_type", "algorithm"])[["delta_f_beta_2", "delta_auc_pr"]]
                            .mean().round(6).reset_index().to_dict(orient="records"))


def anomaly_figure(executions: pd.DataFrame, figures: Path, summary: dict) -> None:
    devnet = executions[executions["algorithm"].eq("devnet") & executions["label_fraction"].notna()].copy()
    forest = executions[executions["algorithm"].eq("isolation-forest")].copy()
    if devnet.empty and forest.empty:
        print("[revisar] sem resultados de detecção de anomalias")
        return
    devnet["label_fraction"] = pd.to_numeric(devnet["label_fraction"], errors="coerce")
    devnet = devnet[devnet["label_fraction"] > 0]
    devnet["_x"] = devnet["label_fraction"].map(lambda v: f"DevNet {estilo.number_plain(100 * v, 0)}%")
    forest["_x"] = "Isolation Forest"
    data = pd.concat([forest, devnet], ignore_index=True)
    x_order = ["Isolation Forest"] + [f"DevNet {estilo.number_plain(100 * v, 0)}%" for v in sorted(devnet["label_fraction"].unique())]
    bases = _order(data["database_type"], analise_hipoteses.FRAUD_SOURCES)
    data["_base"] = data["database_type"].map(estilo.dataset_label)
    fig, axes = plt.subplots(1, 2, figsize=(estilo.PAGE_WIDTH_IN, 3.0))
    for metric, axis in zip(("f_beta_2", "auc_pr"), axes):
        # Barras de erro: desvio padrão entre os sorteios de rótulos (e entre
        # as demais configurações), para mostrar a dispersão, e não um IC.
        sns.pointplot(data=data, x="_x", y=metric, hue="_base", order=x_order,
                      hue_order=[estilo.dataset_label(b) for b in bases], errorbar="sd", dodge=0.5,
                      markers=list(estilo.MARKERS[:len(bases)]), markersize=4, linestyles="none",
                      capsize=0.05, err_kws={"linewidth": 0.8}, ax=axis)
        axis.set_title(estilo.METRIC_LABELS[metric])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_ylim(-0.02, 1.02)
        axis.tick_params(axis="x", rotation=25)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    estilo.legend_below(fig, handles, labels)
    estilo.save(fig, figures / "discussao_anomalia_rotulos.png")
    summary["anomalia_rotulos"] = (data.groupby(["database_type", "_x"])[["f_beta_2", "auc_pr"]]
                                   .agg(["mean", "std", "size"]).round(6).reset_index().to_dict(orient="records"))


def transformer_figure(executions: pd.DataFrame, figures: Path, summary: dict) -> None:
    ft = executions[executions["algorithm"].isin(analise_hipoteses.FT_TRANSFORMER_ALGORITHMS)].copy()
    if ft.empty:
        print("[revisar] sem resultados do FT-Transformer")
        return
    panels = [("Cabeça", "algorithm", analise_hipoteses.FT_TRANSFORMER_ALGORITHMS)]
    for title, column, order in (("Normalização", "numeric_normalization", ("quantile", "standard")),
                                 ("Alta cardinalidade", "high_cardinality_strategy",
                                  ("minimum-frequency", "hashing"))):
        if ft[column].notna().any():
            panels.append((title, column, order))
    bases = _order(ft["database_type"], analise_hipoteses.FRAUD_SOURCES)
    labels = [estilo.dataset_label(b) for b in bases]
    ft["_base"] = ft["database_type"].map(estilo.dataset_label)
    fig, axes = plt.subplots(1, len(panels), figsize=(estilo.PAGE_WIDTH_IN, 0.9 + 0.32 * len(bases)),
                             sharey=True, squeeze=False)
    records = {}
    for (title, hue, order), axis in zip(panels, axes.flat):
        data = ft[ft[hue].notna()].copy()
        hue_order = _order(data[hue], order)
        data["_hue"] = data[hue].map(lambda v: CARDINALITY_LABELS.get(v, estilo.label(v)))
        sns.pointplot(data=data, y="_base", x="f_beta_2", hue="_hue", order=labels, orient="h",
                      hue_order=[CARDINALITY_LABELS.get(h, estilo.label(h)) for h in hue_order],
                      errorbar=("ci", 95), n_boot=1000, seed=42,
                      dodge=0.45, linestyles="none", markers=list(estilo.MARKERS[:len(hue_order)]), markersize=4,
                      capsize=0.1, err_kws={"linewidth": 0.8}, ax=axis)
        axis.set_title(title)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xlim(-0.02, 1.02)
        axis.set_xticks([0, 0.5, 1])
        axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=1, frameon=False,
                    handletextpad=0.3)
        records[title] = data.groupby(["database_type", hue])["f_beta_2"].mean().round(6).reset_index().to_dict(orient="records")
    fig.tight_layout()
    estilo.save(fig, figures / "discussao_transformers.png")
    summary["transformers"] = records


def run(results_directory: Path, figures: Path, manifest: Path | None = None) -> dict:
    estilo.configure()
    executions = load(results_directory, manifest)
    summary: dict = {"execucoes": int(len(executions))}
    for step in (performance_figures, balancing_figure, encoding_figure, imputation_figure,
                 protocol_figure, anomaly_figure, transformer_figure):
        step(executions, figures, summary)
    return summary


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-directory", required=True)
    parser.add_argument("--figures-directory", required=True)
    parser.add_argument("--summary-output", default="figuras_discussao_resumo.json")
    parser.add_argument("--manifest", help="Manifesto da preparação: eixos do FT-Transformer ausentes nos resultados.")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    manifest = Path(arguments.manifest) if arguments.manifest else None
    summary = run(Path(arguments.results_directory), Path(arguments.figures_directory), manifest)
    Path(arguments.summary_output).write_text(
        json.dumps(analise_hipoteses.json_ready(summary), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"figuras em {arguments.figures_directory}; resumo em {arguments.summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
