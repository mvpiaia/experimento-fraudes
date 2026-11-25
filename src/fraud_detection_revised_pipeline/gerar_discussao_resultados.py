"""Figuras e resumo numerico do Capitulo 5 da dissertacao (discussao).

Le os CSVs de resultado dos algoritmos classicos e do CatBoost/XGBoost, agrega
as dobras de cada configuracao e grava as figuras usadas em
``discussao_resultados.tex`` e um resumo em JSON com os numeros citados no
texto.

ATENCAO -- este script foi escrito sobre os resultados da grade de 22/09/2026,
que o item G15 invalidou na Home-Credit-completa e na IEEE-CIS-completa, e sera
executado sobre os resultados da reexecucao. As analises supoem o esquema e os
valores daquela grade: reserva em ``test_data_size`` (0,2/0,3/0,4), politicas
``random-split``/``stratified-train-test-split``, a regressao linear,
codificacoes ``ordered-target``/``woe`` e subamostragem ``RUS+SMOTE``/
``tomek-links``. A grade revisada mudou varios desses eixos
(protocolos sem reserva, fatorial 2x2 de codificacao, braco sem reamostragem e
ponderacao de classe, AUC-PR, unidade de avaliacao por cliente). Por isso:

* ``verificar_esquema`` compara, antes de qualquer figura, as colunas e os
  valores que cada analise supoe com os resultados carregados e imprime cada
  divergencia como ``[revisar]``;
* uma analise cujas colunas nao existem nos resultados e pulada, com aviso, em
  vez de falhar no meio da execucao; divergencias de valor so geram aviso,
  porque a analise roda, mas pode produzir celulas vazias ou pares
  inexistentes.

Comparacoes pareadas (item G17): toda diferenca entre dois niveis de um eixo
passa por ``paired_comparison``. Um par e formado por duas execucoes iguais em
todos os eixos, exceto o comparado, com sucesso em todas as dobras previstas e
avaliadas nas mesmas particoes; pares incompletos sao descartados e contados.
A contagem de cada comparacao (previstos, realizados e descartados por
motivo) vai para ``pares_por_comparacao.csv``, ao lado do resumo. So entram
as linhas de teste avaliadas por registro.

Toda vez que o esquema de ``RESULT_CSV_COLUMNS`` (``lr_nb_knn_specific.py``)
ou os valores de um eixo da grade (``experiment_grid.py``) mudarem, revisar
as analises apontadas como ``[revisar]`` e as constantes de ordem e de rotulo
abaixo antes de levar uma figura ou um numero para o texto.

Uso::

    python3 gerar_discussao_resultados.py \\
        --results-directory <diretorio com resultados_<algoritmo>.csv> \\
        --figures-directory <eqm_paper>/figuras \\
        --summary-output discussao_resultados_resumo.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import wilcoxon

import estilo_figuras as estilo
import experiment_grid


# Definidos em main() a partir da linha de comando; as funcoes de figura os
# leem como globais.
SOURCE = Path(".")
FIGURES = Path("figuras")
SUMMARY = Path("discussao_resultados_resumo.json")

METRICS = ["f1", "f_beta_2", "accuracy", "precision", "recall", "roc_auc"]
METRIC_LABELS = {
    "f1": "F1",
    "f_beta_2": "F2",
    "accuracy": "Acurácia",
    "precision": "Precisão",
    "recall": "Revocação",
    "roc_auc": "AUC-ROC",
}
ALGORITHM_LABELS = {
    "linear-regression": "Regressão linear",
    "logistic-regression": "Regressão logística",
    "naive-bayes": "Naive Bayes",
    "knn": "KNN",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
}
DATABASE_ORDER_CLASSIC = [
    "eqm-ref-18",
    "eqm-ref-19",
    "eqm-ref-20",
    "eqm-ref-22",
]
DATABASE_ORDER_BOOST = [
    "eqm-ref-18",
    "eqm-ref-19",
    "eqm-ref-20",
    "eqm-ref-22",
]
# A ordem segue a grade revisada. A regressao linear saiu dela e so volta a
# ordem quando os resultados lidos a contem (grade de 22/09); ver load_results.
CLASSIC_ORDER = list(experiment_grid.CLASSIC_ALGORITHMS)
LEGACY_CLASSIC_ALGORITHMS = ["linear-regression"]
BOOST_ORDER = list(experiment_grid.BOOSTING_ALGORITHMS)
TEST_SIZE_LABELS = {
    "0.2:0.2": "20%",
    "0.3:0.3": "30%",
    "0.4:0.4": "40%",
}


def _read_results_file(algorithm: str) -> pd.DataFrame | None:
    # ``resultas_`` e o erro de digitacao historico da grade de 22/09; os
    # executores revisados gravam ``resultados_<algoritmo>.csv``.
    for prefix in ("resultados_", "resultas_"):
        path = SOURCE / f"{prefix}{algorithm}.csv"
        if path.exists():
            return pd.read_csv(path)
    return None


def _read_family(algorithms: list[str], family: str) -> pd.DataFrame:
    frames = [
        frame
        for frame in (_read_results_file(algorithm) for algorithm in algorithms)
        if frame is not None
    ]
    if not frames:
        raise SystemExit(f"nenhum resultado de {family} em {SOURCE}")
    data = pd.concat(frames, ignore_index=True)
    # Resultados por cliente (item G7) repetem a configuracao com outra
    # unidade de avaliacao; as figuras usam a avaliacao por registro.
    if "evaluation_unit" in data.columns:
        data = data[data["evaluation_unit"].fillna("row").eq("row")]
    return data.reset_index(drop=True)


REMOVED_DATABASES = ("eqm-ref-21", "eqm-ref-18-pca")


def load_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    global CLASSIC_ORDER
    # A grade de 22/09 tinha os classicos reunidos em resultados_combinados.csv.
    legacy_combined = SOURCE / "resultados_combinados.csv"
    if legacy_combined.exists():
        classic = pd.read_csv(legacy_combined)
    else:
        classic = _read_family(LEGACY_CLASSIC_ALGORITHMS + CLASSIC_ORDER, "algoritmos classicos")
    legacy_algorithms = [
        algorithm
        for algorithm in LEGACY_CLASSIC_ALGORITHMS
        if algorithm in set(classic["algorithm"].dropna())
    ]
    CLASSIC_ORDER = legacy_algorithms + list(experiment_grid.CLASSIC_ALGORITHMS)
    boost = _read_family(BOOST_ORDER, "CatBoost/XGBoost")
    # Bases que sairam do catalogo, mas ainda aparecem nos CSVs de 22/09:
    # eqm-ref-21 e uma republicacao literal da ULB, e a ULB com PCA aplicava
    # uma segunda PCA sobre componentes principais publicados (item G19). O
    # filtro garante que nenhuma analise as reintroduza.
    classic = classic[~classic["database_type"].isin(REMOVED_DATABASES)].reset_index(drop=True)
    boost = boost[~boost["database_type"].isin(REMOVED_DATABASES)].reset_index(drop=True)
    return classic, boost


# Eixos que, com o configuration_md5, identificam uma execucao. Fora o md5,
# sao tambem os eixos da chave dos pares (paired_comparison).
EXECUTION_DESCRIPTORS = [
    "database_type",
    "algorithm",
    "undersampling_technique",
    "test_data_size",
    "categorical_encoding_strategy",
    "fill_na_strategy",
    "train_validation_test_strategy",
    "random_seed",
    # Grade revisada: os bracos de ponderacao de classe reusam o
    # configuration_md5 da configuracao sem reamostragem.
    "class_weight_intensity",
    "threshold_strategy",
    "regime",
    "label_fraction",
    # Semente do sorteio dos rotulos do DevNet; ver _with_execution_columns.
    "label_draw_seed",
]
KFOLD_FOLDS = 5


def expected_folds(test_strategy) -> int:
    return KFOLD_FOLDS if "k-fold" in str(test_strategy) else 1


def _with_execution_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "execution_seed" in data.columns and "algorithm" in data.columns:
        # Para o DevNet execution_seed e a semente do sorteio dos rotulos; nos
        # Transformers e a de treino, que nao identifica um braco.
        is_draw = data["algorithm"].isin(experiment_grid.ANOMALY_LABEL_BUDGET_ALGORITHMS)
        data["label_draw_seed"] = data["execution_seed"].where(is_draw)
    if "label_fraction" in data.columns:
        # -1 (eixo nao se aplica) e 0 (Isolation Forest) nao sao bracos.
        fraction = pd.to_numeric(data["label_fraction"], errors="coerce")
        data["label_fraction"] = fraction.where(fraction > 0)
    return data


def _test_rows(data: pd.DataFrame) -> pd.DataFrame:
    """Linhas de teste avaliadas por registro.

    O esquema revisado grava treino, validacao e teste de cada dobra; uma media
    sem este filtro misturaria os tres splits. CSVs sem ``split`` (22/09) so
    tem linhas de teste.
    """
    mask = pd.Series(True, index=data.index)
    if "split" in data.columns:
        mask &= data["split"].fillna("test").astype(str).isin(["test", "test_data"])
    if "evaluation_unit" in data.columns:
        mask &= data["evaluation_unit"].fillna("row").eq("row")
    return data[mask]


def aggregate_folds(data: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por execucao com sucesso, com a media das dobras de teste.

    ``fold_rows`` conta as dobras com sucesso e ``complete`` diz se sao todas
    as previstas; ``partition_signature`` registra o partition_id de cada
    dobra, para o pareamento conferir que os dois lados viram as mesmas linhas.
    """
    rows = _with_execution_columns(_test_rows(data))
    valid = rows.loc[
        rows["execution_status"].eq("success")
        & rows[METRICS].ge(0).all(axis=1)
    ].copy()
    descriptors = _present_columns(valid, ["configuration_md5", *EXECUTION_DESCRIPTORS])
    has_folds = "fold_index" in valid.columns and valid["fold_index"].notna().any()
    aggregations = {metric: (metric, "mean") for metric in METRICS}
    # Sem fold_index (22/09), cada linha de teste e uma dobra.
    aggregations["fold_rows"] = ("fold_index", "nunique") if has_folds else ("f1", "size")
    if "partition_id" in valid.columns and has_folds:
        valid["partition_fold"] = (
            valid["fold_index"].astype(str) + ":" + valid["partition_id"].fillna("").astype(str)
        )
        aggregations["partition_signature"] = (
            "partition_fold", lambda values: "|".join(sorted(set(values)))
        )
    aggregated = valid.groupby(descriptors, dropna=False, as_index=False).agg(**aggregations)
    aggregated["complete"] = aggregated["fold_rows"] >= aggregated[
        "train_validation_test_strategy"
    ].map(expected_folds)
    return aggregated


def attempted_executions(data: pd.DataFrame) -> pd.DataFrame:
    """Execucoes tentadas, com sucesso ou nao; previstas no modo legado."""
    rows = _with_execution_columns(_test_rows(data))
    columns = _present_columns(rows, EXECUTION_DESCRIPTORS)
    return rows[columns].drop_duplicates().reset_index(drop=True)


def expected_executions_from_grid(blocks: tuple[str, ...]) -> pd.DataFrame:
    """Execucoes previstas pela grade revisada, com os nomes de coluna dos resultados."""
    import preparation_sanity_checks as sanity

    expected = sanity.expected_runs(blocks).rename(
        columns={"dataset_id": "database_type", "test_strategy": "train_validation_test_strategy"}
    )
    return expected[_present_columns(expected, EXECUTION_DESCRIPTORS)]


def _present_columns(data: pd.DataFrame, columns: list[str]) -> list[str]:
    """Colunas de uma chave de pareamento que existem nos resultados.

    ``test_data_size`` so existe na grade de 22/09; sem ele, a chave pareia
    pelos demais eixos, que na grade revisada ja identificam a configuracao.
    """
    return [column for column in columns if column in data.columns]


def configure_plotting() -> None:
    # Padrão comum das figuras da dissertação (itens F6 e F7): locale pt_BR,
    # paleta segura para daltonismo e fontes legíveis no tamanho impresso.
    estilo.configure()


def finish_figure(
    fig: plt.Figure,
    filename: str,
    rect: tuple[float, float, float, float] | None = None,
) -> None:
    fig.tight_layout(rect=rect)
    fig.savefig(FIGURES / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def metric_point_grid(
    data: pd.DataFrame,
    database_order: list[str],
    algorithm_order: list[str],
    filename: str,
) -> None:
    plot_data = data.copy()
    plot_data["algorithm_label"] = plot_data["algorithm"].map(ALGORITHM_LABELS)
    hue_order = [ALGORITHM_LABELS[value] for value in algorithm_order]
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 7.2), sharex=True)
    for metric, axis in zip(METRICS, axes.flat):
        sns.pointplot(
            data=plot_data,
            x="database_type",
            y=metric,
            hue="algorithm_label",
            order=database_order,
            hue_order=hue_order,
            errorbar=("ci", 95),
            n_boot=1000,
            seed=42,
            dodge=0.35,
            markers="o",
            linestyles="none",
            capsize=0.08,
            ax=axis,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(range(len(database_order)), [estilo.dataset_label(value) for value in database_order])
        axis.tick_params(axis="x", rotation=28)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    estilo.legend_below(fig, handles, labels)
    estilo.save(fig, FIGURES / filename)


def classic_preprocessing_heatmaps(classic: pd.DataFrame) -> None:
    data = classic.copy()
    data["fill_label"] = data["fill_na_strategy"].map(estilo.label).fillna("remoção de incompletos")
    data["encoding_label"] = data["categorical_encoding_strategy"].map(
        {"ordered-target": "OTE", "woe": "WoE"}
    )
    data["strategy"] = data["encoding_label"] + " | " + data["fill_label"]
    strategy_order = [
        f"{encoding} | {fill}"
        for encoding in ["OTE", "WoE"]
        for fill in ["remoção de incompletos", "média", "mediana", "moda"]
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 10.0))
    for metric, axis in zip(METRICS, axes.flat):
        matrix = data.pivot_table(
            index="strategy", columns="algorithm", values=metric, aggfunc="mean"
        ).reindex(index=strategy_order, columns=CLASSIC_ORDER)
        matrix.columns = [ALGORITHM_LABELS[column] for column in matrix.columns]
        sns.heatmap(
            matrix,
            ax=axis,
            cmap="viridis",
            vmin=0,
            vmax=1,
            annot=matrix.map(lambda value: "" if pd.isna(value) else estilo.number_plain(value)),
            fmt="",
            linewidths=0.4,
            cbar=False,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=25)
        axis.tick_params(axis="y", rotation=0)
    finish_figure(fig, "resultados_classicos_preprocessamento.png")


def classic_strategy_by_base_heatmaps(
    classic: pd.DataFrame,
    strategy_column: str,
    strategy_order: list[str],
    strategy_labels: dict[str, str],
    filename: str,
) -> None:
    compact_algorithm_labels = {
        "linear-regression": "Linear",  # so na grade de 22/09
        "logistic-regression": "Regressão logística",
        "naive-bayes": "Naive Bayes",
        "knn": "KNN",
    }
    # Cada celula usa so as execucoes que formam par completo no contraste da
    # figura, e as duas celulas de um par tem o mesmo n (item G17, C5.11).
    members = paired_members(
        classic,
        strategy_column,
        strategy_order,
        f"{strategy_column}: {strategy_order[0]} - {strategy_order[1]} (figura)",
    )
    grouped = (
        members.groupby(
            ["algorithm", strategy_column, "database_type"],
            dropna=False,
        )[METRICS]
        .agg(["mean", "size"])
    )
    counts = grouped[(METRICS[0], "size")].rename("n")
    grouped = grouped.xs("mean", axis=1, level=1).join(counts).reset_index()
    grouped["row_label"] = grouped.apply(
        lambda row: (
            f"{compact_algorithm_labels[row['algorithm']]} | "
            f"{strategy_labels[row[strategy_column]]}"
        ),
        axis=1,
    )
    row_order = [
        f"{compact_algorithm_labels[algorithm]} | {strategy_labels[strategy]}"
        for algorithm in CLASSIC_ORDER
        for strategy in strategy_order
    ]
    cell_counts = (
        grouped.pivot(index="row_label", columns="database_type", values="n")
        .reindex(index=row_order, columns=DATABASE_ORDER_CLASSIC)
    )

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 14.0))
    for index, (metric, axis) in enumerate(zip(METRICS, axes.flat)):
        matrix = (
            grouped.pivot(index="row_label", columns="database_type", values=metric)
            .reindex(index=row_order, columns=DATABASE_ORDER_CLASSIC)
        )
        labels = np.array([
            ["" if pd.isna(value) else f"{estilo.number_plain(value)}\nn={int(count)}" for value, count in zip(values, counts)]
            for values, counts in zip(matrix.to_numpy(), cell_counts.to_numpy())
        ])
        matrix.columns = [estilo.dataset_label(value) for value in matrix.columns]
        sns.heatmap(
            matrix,
            annot=labels,
            fmt="",
            vmin=0,
            vmax=1,
            cmap="viridis",
            cbar=False,
            linewidths=0.35,
            linecolor="white",
            annot_kws={"size": 9},
            ax=axis,
        )
        axis.set_title(METRIC_LABELS[metric], fontsize=12)
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.tick_params(axis="x", rotation=24, labelsize=10)
        axis.tick_params(axis="y", rotation=0, labelsize=10)
        if index % 2 != 0:
            axis.set_yticklabels([])
    finish_figure(fig, filename)


# Previstas da grade (revisada) ou tentadas (22/09); definido em main().
EXPECTED_EXECUTIONS: pd.DataFrame | None = None
# Uma linha por comparacao, base e algoritmo; gravado em pares_por_comparacao.csv.
PAIR_COUNTS: list[dict] = []
PAIR_DISCARD_REASONS = (
    "falha-ambos",
    "falha-positivo",
    "falha-negativo",
    "incompleto",
    "particao-divergente",
)
# Contrastes em que os dois lados, por desenho, nao usam as mesmas particoes.
FACTORS_WITHOUT_SHARED_PARTITIONS = {"train_validation_test_strategy", "test_data_size"}
_NONE = "__NONE__"


def _key_value(value) -> str:
    """Valor de chave comparavel entre resultados e grade (42 = 42.0 = "42")."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return _NONE
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "null"}:
        return _NONE
    try:
        number = float(text)
    except ValueError:
        return text
    return str(int(number)) if number.is_integer() else f"{number:g}"


def _keyed(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keyed = data.copy()
    for column in columns:
        keyed[column] = keyed[column].map(_key_value)
    return keyed


def _family_expected(data: pd.DataFrame, expected: pd.DataFrame | None) -> pd.DataFrame | None:
    """Recorta as execucoes previstas a familia de algoritmos dos dados."""
    if expected is None or expected.empty:
        return None
    present = set(data["algorithm"].dropna())
    for family in (set(LEGACY_CLASSIC_ALGORITHMS) | set(experiment_grid.CLASSIC_ALGORITHMS),
                   set(BOOST_ORDER)):
        if present & family:
            return expected[expected["algorithm"].isin(family)]
    return expected[expected["algorithm"].isin(present)]


def paired_comparison(
    data: pd.DataFrame,
    factor: str,
    positive: str,
    negative: str,
    name: str | None = None,
    expected: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Diferencas pareadas ``positive - negative`` no eixo ``factor`` (item G17).

    O par sao duas execucoes iguais em todos os eixos de EXECUTION_DESCRIPTORS,
    exceto ``factor``. Ele so e realizado se os dois lados tem sucesso em todas
    as dobras previstas e as mesmas particoes; a diferenca e entre as medias
    das dobras de cada lado. Nenhum lado e media de configuracoes distintas:
    uma chave com mais de uma execucao de um lado interrompe a analise.

    Os pares previstos vem de ``expected`` (padrao: EXPECTED_EXECUTIONS), e
    nao dos dados: um par cujos dois lados faltam conta como previsto e
    descartado. A contagem vai para PAIR_COUNTS.
    """
    name = name or f"{factor}: {positive} - {negative}"
    key = [column for column in _present_columns(data, EXECUTION_DESCRIPTORS) if column != factor]
    sides = _keyed(data[data[factor].isin([positive, negative])], [*key, factor])
    positive_key, negative_key = _key_value(positive), _key_value(negative)
    duplicated = sides.duplicated([*key, factor], keep=False)
    if duplicated.any():
        raise ValueError(
            f"{name}: {int(duplicated.sum())} execucoes com a mesma chave de par; "
            "a chave nao identifica a execucao e o par seria uma media"
        )
    positive_side = sides[sides[factor] == positive_key].set_index(key)
    negative_side = sides[sides[factor] == negative_key].set_index(key)

    expected = _family_expected(data, EXPECTED_EXECUTIONS if expected is None else expected)
    expected_key = [column for column in key if expected is not None and column in expected.columns]
    if expected is not None and factor in expected.columns:
        planned = _keyed(expected, [*expected_key, factor])
        levels = planned[planned[factor].isin([positive_key, negative_key])]
        both = levels.groupby(expected_key)[factor].nunique()
        planned_keys = both[both == 2].index
    else:
        expected_key = key
        planned_keys = positive_side.index.union(negative_side.index)

    # Os pares sao avaliados na chave da grade; colunas que a grade nao conhece
    # (limiar, por exemplo) sao constantes da execucao e entram so na diferenca.
    positive_by_plan = positive_side.reset_index().set_index(expected_key)
    negative_by_plan = negative_side.reset_index().set_index(expected_key)
    for side in (positive_by_plan, negative_by_plan):
        if side.index.duplicated().any():
            raise ValueError(
                f"{name}: a chave da grade ({', '.join(expected_key)}) nao identifica "
                "uma unica execucao por lado"
            )
    check_partition = (
        factor not in FACTORS_WITHOUT_SHARED_PARTITIONS and "partition_signature" in sides.columns
    )
    records, deltas = [], []
    for pair_key in planned_keys:
        pair_key = pair_key if isinstance(pair_key, tuple) else (pair_key,)
        has_positive = pair_key in positive_by_plan.index
        has_negative = pair_key in negative_by_plan.index
        reason = None
        if not has_positive and not has_negative:
            reason = "falha-ambos"
        elif not has_positive:
            reason = "falha-positivo"
        elif not has_negative:
            reason = "falha-negativo"
        else:
            left = positive_by_plan.loc[[pair_key]].iloc[0]
            right = negative_by_plan.loc[[pair_key]].iloc[0]
            if not (bool(left["complete"]) and bool(right["complete"])):
                reason = "incompleto"
            elif check_partition and left["partition_signature"] != right["partition_signature"]:
                reason = "particao-divergente"
        identity = dict(zip(expected_key, pair_key))
        records.append({
            "database_type": identity.get("database_type", f"{positive} | {negative}"),
            "algorithm": identity.get("algorithm", f"{positive} | {negative}"),
            "reason": reason,
        })
        if reason is None:
            deltas.append({
                **identity,
                **{metric: float(left[metric]) - float(right[metric]) for metric in METRICS},
            })

    outcomes = pd.DataFrame(records, columns=["database_type", "algorithm", "reason"])
    for (database, algorithm), group in outcomes.groupby(["database_type", "algorithm"]):
        PAIR_COUNTS.append({
            "comparacao": name,
            "fator": factor,
            "positivo": positive,
            "negativo": negative,
            "database_type": database,
            "algorithm": algorithm,
            "previstos": int(len(group)),
            "realizados": int(group["reason"].isna().sum()),
            **{
                f"descartados_{reason}": int((group["reason"] == reason).sum())
                for reason in PAIR_DISCARD_REASONS
            },
        })
    result = pd.DataFrame(deltas, columns=[*expected_key, *METRICS])
    return result.replace(_NONE, np.nan)


def pair_count_label(name: str) -> str:
    """``n = realizados/previstos`` de uma comparacao, para legendas."""
    rows = [row for row in PAIR_COUNTS if row["comparacao"] == name]
    done = sum(row["realizados"] for row in rows)
    planned = sum(row["previstos"] for row in rows)
    return f"n = {done}/{planned}"


def paired_members(data: pd.DataFrame, factor: str, levels: list[str], name: str) -> pd.DataFrame:
    """Execucoes que formam par completo no contraste ``levels[0]`` x ``levels[1]``."""
    deltas = paired_comparison(data, factor, levels[0], levels[1], name=name)
    key = [column for column in _present_columns(data, EXECUTION_DESCRIPTORS) if column != factor]
    key = [column for column in key if column in deltas.columns]
    if deltas.empty:
        return data.iloc[0:0]
    realized = set(map(tuple, _keyed(deltas, key)[key].to_numpy()))
    keyed = _keyed(data, key)
    member = keyed[key].apply(tuple, axis=1).isin(realized) & data[factor].isin(levels)
    return data[member.to_numpy()]


def balance_delta_plot(classic: pd.DataFrame) -> pd.DataFrame:
    name = "reamostragem: RUS+SMOTE - Tomek Links"
    deltas = paired_comparison(classic, "undersampling_technique", "RUS+SMOTE", "tomek-links", name=name)
    deltas["algorithm_label"] = deltas["algorithm"].map(ALGORITHM_LABELS)
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.3))
    for metric, axis in zip(METRICS, axes.flat):
        sns.boxplot(
            data=deltas,
            x="algorithm_label",
            y=metric,
            order=[ALGORITHM_LABELS[value] for value in CLASSIC_ORDER],
            color="#7EA6D8",
            showfliers=False,
            ax=axis,
        )
        axis.axhline(0, color="#9B2C2C", linewidth=1, linestyle="--")
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("")
        axis.set_ylabel(r"$\Delta$ (RUS+SMOTE $-$ Tomek Links)")
        axis.tick_params(axis="x", rotation=25)
    fig.suptitle(f"Pares completos: {pair_count_label(name)}", fontsize=9)
    finish_figure(fig, "efeito_balanceamento_classicos.png")
    return deltas


def boost_fillna_grid(
    boost: pd.DataFrame,
    output_name: str = "resultados_boost_fillna.png",
) -> None:
    data = boost.copy()
    data["fill_label"] = data["fill_na_strategy"].fillna("sem preenchimento")
    data["algorithm_label"] = data["algorithm"].map(ALGORITHM_LABELS)
    fill_order = ["sem preenchimento", "mean", "median", "mode"]
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.2), sharex=True)
    for metric, axis in zip(METRICS, axes.flat):
        sns.pointplot(
            data=data,
            x="fill_label",
            y=metric,
            hue="algorithm_label",
            order=fill_order,
            hue_order=["CatBoost", "XGBoost"],
            errorbar=("ci", 95),
            n_boot=1000,
            seed=42,
            dodge=0.25,
            capsize=0.08,
            ax=axis,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_ylim(-0.02, 1.02)
        axis.tick_params(axis="x", rotation=22)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.subplots_adjust(top=0.9)
    finish_figure(fig, output_name)


# Item G15d: ULB e Sparkov não têm faltantes, e suas linhas idênticas entre
# estratégias diluíam o efeito do preenchimento na figura agregada.
BOOST_MISSING_DATABASES = ["eqm-ref-19", "eqm-ref-22"]


def boost_fillna_grid_missing_bases(boost: pd.DataFrame) -> None:
    boost_fillna_grid(
        boost[boost["database_type"].isin(BOOST_MISSING_DATABASES)],
        "resultados_boost_fillna_g15.png",
    )


def categorical_cross_family_grid(classic: pd.DataFrame, boost: pd.DataFrame) -> None:
    # A ULB entra sem PCA nas duas familias: a variante com PCA saiu do
    # catalogo (os V1-V28 ja sao componentes principais publicados; item G19).
    comparison_order = ["eqm-ref-18", "eqm-ref-19", "eqm-ref-20", "eqm-ref-22"]
    data = pd.concat(
        [
            classic[classic["database_type"].isin(comparison_order)],
            boost[boost["database_type"].isin(comparison_order)],
        ],
        ignore_index=True,
    )
    data["comparison_base"] = data["database_type"]
    data["algorithm_label"] = data["algorithm"].map(ALGORITHM_LABELS)
    algorithms = CLASSIC_ORDER + BOOST_ORDER
    hue_order = [ALGORITHM_LABELS[value] for value in algorithms]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.2), sharex=True)
    for metric, axis in zip(METRICS, axes.flat):
        sns.pointplot(
            data=data,
            x="comparison_base",
            y=metric,
            hue="algorithm_label",
            order=comparison_order,
            hue_order=hue_order,
            errorbar=("ci", 95),
            n_boot=1000,
            seed=42,
            dodge=0.45,
            linestyles="none",
            capsize=0.06,
            ax=axis,
        )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_ylim(-0.02, 1.02)
        axis.tick_params(axis="x", rotation=20)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=6,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    finish_figure(
        fig,
        "comparacao_classicos_boost_categoricas.png",
        rect=(0.0, 0.0, 1.0, 0.91),
    )


def paired_summary(deltas: pd.DataFrame) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for metric in METRICS:
        values = deltas[metric].dropna()
        statistic, p_value = wilcoxon(values) if len(values) else (np.nan, np.nan)
        summary[metric] = {
            "n": int(len(values)),
            "mean_delta": float(values.mean()),
            "median_delta": float(values.median()),
            "positive_share": float((values > 0).mean()),
            "wilcoxon_statistic": float(statistic),
            "wilcoxon_p": float(p_value),
        }
    return summary


def build_summary(
    raw_classic: pd.DataFrame,
    raw_boost: pd.DataFrame,
    classic: pd.DataFrame,
    boost: pd.DataFrame,
    balance_deltas: pd.DataFrame | None,
) -> dict:
    # Contrastes da grade de 22/09 que podem nao existir na revisada; ausentes,
    # viram None no resumo em vez de interromper a execucao.
    encoding_deltas = (
        paired_comparison(
            classic, "categorical_encoding_strategy", "ordered-target", "woe",
            name="codificacao: ordered-target - woe (resumo)",
        )
        if has_values(classic, "categorical_encoding_strategy", ["ordered-target", "woe"])
        else None
    )
    boost_deltas = paired_comparison(boost, "algorithm", "catboost", "xgboost", name="catboost - xgboost")

    def grouped_means(data: pd.DataFrame, by: list[str]) -> list[dict]:
        return (
            data.groupby(by, dropna=False)[METRICS]
            .mean()
            .reset_index()
            .round(6)
            .to_dict(orient="records")
        )

    summary = {
        "coverage": {
            "classic_raw_rows": int(len(raw_classic)),
            "classic_success_rows": int(raw_classic["execution_status"].eq("success").sum()),
            "classic_failure_rows": int(raw_classic["execution_status"].ne("success").sum()),
            "classic_config_algorithm_units": int(len(classic)),
            "boost_raw_rows": int(len(raw_boost)),
            "boost_success_rows": int(raw_boost["execution_status"].eq("success").sum()),
            "boost_config_algorithm_units": int(len(boost)),
            "classic_databases": sorted(raw_classic["database_type"].unique().tolist()),
            "boost_databases": sorted(raw_boost["database_type"].unique().tolist()),
            # Bases que as figuras esperam e que nao vieram nos resultados.
            "missing_requested_databases": sorted(
                (set(DATABASE_ORDER_CLASSIC) - set(raw_classic["database_type"].dropna()))
                | (set(DATABASE_ORDER_BOOST) - set(raw_boost["database_type"].dropna()))
            ),
        },
        "classic_by_algorithm_database": grouped_means(classic, ["algorithm", "database_type"]),
        "boost_by_algorithm_database": grouped_means(boost, ["algorithm", "database_type"]),
        "balance_rus_smote_minus_tomek": (
            paired_summary(balance_deltas) if balance_deltas is not None else None
        ),
        "encoding_ordered_target_minus_woe": (
            paired_summary(encoding_deltas) if encoding_deltas is not None else None
        ),
        "catboost_minus_xgboost": paired_summary(boost_deltas),
        "fillna_classic": grouped_means(classic.assign(fill_na_strategy=classic["fill_na_strategy"].fillna("sem preenchimento")), ["algorithm", "fill_na_strategy"]),
        "fillna_boost": grouped_means(boost.assign(fill_na_strategy=boost["fill_na_strategy"].fillna("sem preenchimento")), ["algorithm", "fill_na_strategy"]),
    }
    return summary


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    means = rng.choice(values, size=(n_boot, len(values)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return values.mean(), low, high


PARTITION_POLICY_ORDER = [
    "random-split",
    "stratified-train-test-split",
    "stratified-k-fold",
    "stratified-train-validation-test",
]
PARTITION_POLICY_LABELS = {
    "random-split": "Aleatória",
    "stratified-train-test-split": "Estrat. treino/teste",
    "stratified-k-fold": "Estrat. 5 dobras",
    "stratified-train-validation-test": "Estrat. treino/val./teste",
}
PARTITION_SIZE_LABELS = {"0.3:0.3": "30% − 20%", "0.4:0.4": "40% − 20%"}
def paired_reserve_size_deltas(data: pd.DataFrame) -> pd.DataFrame:
    """Diferenca (30% ou 40%) menos 20% de reserva, pareada dentro de cada politica."""
    frames = []
    for size, label in PARTITION_SIZE_LABELS.items():
        delta = paired_comparison(data, "test_data_size", size, "0.2:0.2", name=f"reserva: {label}")
        delta["size_contrast"] = label
        frames.append(delta)
    return pd.concat(frames, ignore_index=True)


def _bootstrap_summary(values: pd.Series) -> tuple[float, float, float]:
    return bootstrap_mean_ci(values.to_numpy())


def partition_size_effect_grid(
    data: pd.DataFrame,
    filename: str = "particionamento_tamanho_reserva.png",
    family_label: str = "algoritmos clássicos",
) -> None:
    """Efeito pareado do tamanho da reserva (30% e 40% contra 20%) em cada política."""
    metrics = ["f1", "f_beta_2", "precision", "recall", "roc_auc"]
    colors = dict(zip(PARTITION_SIZE_LABELS.values(), sns.color_palette("colorblind")[:2]))
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.8, 3.6), sharex=True, sharey=False)
    offsets = {"30% − 20%": -0.12, "40% − 20%": 0.12}
    deltas = paired_reserve_size_deltas(data)
    for column, metric in enumerate(metrics):
        axis = axes[column]
        axis.axhline(0, color="#444444", linestyle="--", linewidth=0.9)
        for contrast, offset in offsets.items():
            for position, policy in enumerate(PARTITION_POLICY_ORDER):
                subset = deltas[
                    deltas["size_contrast"].eq(contrast)
                    & deltas["train_validation_test_strategy"].eq(policy)
                ][metric]
                mean, low, high = _bootstrap_summary(subset)
                axis.errorbar(
                    position + offset,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt="o" if contrast.startswith("30") else "s",
                    color=colors[contrast],
                    markersize=6,
                    capsize=3,
                    elinewidth=1.2,
                    label=contrast if position == 0 else None,
                )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_xticks(range(len(PARTITION_POLICY_ORDER)))
        axis.set_xticklabels(
            [PARTITION_POLICY_LABELS[policy] for policy in PARTITION_POLICY_ORDER],
            rotation=28,
            ha="right",
        )
        axis.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=2, frameon=False,
        title=f"Diferença pareada entre reservas, {family_label}", bbox_to_anchor=(0.5, 1.08),
    )
    finish_figure(fig, filename, rect=(0.0, 0.0, 1.0, 0.82))


def partition_policy_by_base_grid(
    data: pd.DataFrame,
    database_order: list[str],
    filename: str = "particionamento_politica_base.png",
) -> None:
    """Média por base e política de particionamento."""
    metrics = ["f1", "f_beta_2", "recall", "roc_auc"]
    palette = dict(zip(PARTITION_POLICY_ORDER, sns.color_palette("colorblind")[:4]))
    markers = dict(zip(PARTITION_POLICY_ORDER, ["o", "s", "D", "^"]))
    offsets = np.linspace(-0.27, 0.27, len(PARTITION_POLICY_ORDER))
    fig, axes = plt.subplots(1, len(metrics), figsize=(15.8, 3.8), sharey=False)
    for column, metric in enumerate(metrics):
        axis = axes[column]
        for offset, policy in zip(offsets, PARTITION_POLICY_ORDER):
            for position, database in enumerate(database_order):
                subset = data[
                    data["database_type"].eq(database)
                    & data["train_validation_test_strategy"].eq(policy)
                ][metric]
                mean, low, high = _bootstrap_summary(subset)
                axis.errorbar(
                    position + offset,
                    mean,
                    yerr=[[mean - low], [high - mean]],
                    fmt=markers[policy],
                    color=palette[policy],
                    markersize=5.5,
                    capsize=2.5,
                    elinewidth=1.0,
                    label=PARTITION_POLICY_LABELS[policy] if position == 0 else None,
                )
        axis.set_title(METRIC_LABELS[metric])
        axis.set_ylim(-0.02, 1.02)
        axis.set_xticks(range(len(database_order)))
        axis.set_xticklabels(database_order, rotation=28, ha="right")
        axis.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.1))
    finish_figure(fig, filename, rect=(0.0, 0.0, 1.0, 0.80))


ALL_DATABASE_ORDER = [
    "eqm-ref-18",
    "eqm-ref-19",
    "eqm-ref-20",
    "eqm-ref-22",
]
ALL_ALGORITHM_ORDER = CLASSIC_ORDER + BOOST_ORDER


def best_configuration_summary_grid(
    data: pd.DataFrame,
    algorithms: list[str],
    database_order: list[str],
    filename: str = "resumo_melhor_configuracao_base.png",
) -> pd.DataFrame:
    """Melhor configuração de cada algoritmo em cada base, por métrica.

    Para cada base, algoritmo e métrica, seleciona a configuração (já agregada
    entre dobras) com o maior valor médio daquela métrica. A seleção é feita de
    forma independente por métrica: a configuração campeã em F1 não é
    necessariamente a mesma campeã em precisão ou revocação. O gráfico compara,
    portanto, o teto de desempenho alcançável por cada algoritmo em cada base,
    não uma configuração fixa recomendada.
    """
    metrics = ["f1", "f_beta_2", "precision", "recall", "roc_auc"]
    palette = dict(zip(algorithms, sns.color_palette(n_colors=len(algorithms))))
    bar_width = 0.8 / len(algorithms)
    offsets = np.linspace(
        -0.4 + bar_width / 2, 0.4 - bar_width / 2, len(algorithms)
    )

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), sharex=True)
    axes.flat[-1].axis("off")
    best_rows = []
    for metric, axis in zip(metrics, axes.flat):
        for offset, algorithm in zip(offsets, algorithms):
            for position, database in enumerate(database_order):
                subset = data[
                    data["algorithm"].eq(algorithm) & data["database_type"].eq(database)
                ]
                if subset.empty:
                    continue
                best_index = subset[metric].idxmax()
                best_value = subset.loc[best_index, metric]
                best_rows.append(
                    {
                        "database_type": database,
                        "algorithm": algorithm,
                        "metric": metric,
                        "value": best_value,
                    }
                )
                axis.bar(
                    position + offset,
                    best_value,
                    width=bar_width,
                    color=palette[algorithm],
                    edgecolor="white",
                    linewidth=0.6,
                )
        axis.set_title(f"Melhor {METRIC_LABELS[metric]} por algoritmo")
        axis.set_xticks(range(len(database_order)))
        axis.set_xticklabels(database_order, rotation=28, ha="right")
        axis.set_ylim(0, 1.02)
        axis.grid(axis="x", visible=False)

    from matplotlib.patches import Patch

    handles = [
        Patch(facecolor=palette[algorithm], label=ALGORITHM_LABELS[algorithm])
        for algorithm in algorithms
    ]
    fig.legend(
        handles=handles, loc="upper center", ncol=len(algorithms),
        frameon=False, bbox_to_anchor=(0.5, 0.995),
    )
    finish_figure(fig, filename, rect=(0.0, 0.0, 1.0, 0.93))
    return pd.DataFrame(best_rows)


def best_f2_configuration_winner_table(data: pd.DataFrame, database_order: list[str]) -> pd.DataFrame:
    """Configuração de maior F_beta (beta=2) por base, entre os algoritmos informados.

    Diferentemente de ``best_configuration_summary_grid``, que maximiza cada
    métrica de forma independente, esta tabela seleciona uma única
    configuração por base — a de maior F_beta — e reporta as demais métricas
    dessa mesma configuração, para que os valores da linha sejam
    consistentes entre si.
    """
    columns = _present_columns(data, [
        "database_type", "algorithm", "undersampling_technique",
        "test_data_size", "train_validation_test_strategy",
        "f_beta_2", "f1", "roc_auc", "recall", "precision",
    ])
    winners = data.loc[
        data.groupby("database_type")["f_beta_2"].idxmax(), columns
    ].set_index("database_type")
    present_databases = [database for database in database_order if database in winners.index]
    return winners.loc[present_databases].reset_index()


def _unused_best_f1_configuration_winner_table(best_rows: pd.DataFrame) -> pd.DataFrame:
    """Algoritmo com a melhor configuração de F1 em cada base."""
    f1_rows = best_rows[best_rows["metric"].eq("f1")]
    winners = f1_rows.loc[f1_rows.groupby("database_type")["value"].idxmax()]
    return winners[["database_type", "algorithm", "value"]].sort_values("database_type")


# Acurácia de um classificador trivial que sempre prevê a classe majoritária
# (sempre "legítima"), calculada a partir das contagens de classe descritas na
# caracterização das bases (Seção "Caracterização das Bases de Dados
# Utilizadas" em metodologia.tex).
MAJORITY_CLASS_BASELINE_ACCURACY = {
    "eqm-ref-18": 1 - 492 / 284807,
    "eqm-ref-19": 1 - 122360 / 1413701,
    "eqm-ref-20": 1 - 9651 / 1852394,
    "eqm-ref-22": 1 - 11318 / 144233,
}


def accuracy_baseline_grid(classic: pd.DataFrame, boost: pd.DataFrame) -> pd.DataFrame:
    """Acurácia por algoritmo e base, contra a linha de base da classe majoritária.

    Painel A: acurácia média de cada algoritmo em cada base (círculo = clássico,
    triângulo = aumento de gradiente), com uma estrela preta marcando a acurácia
    de um classificador trivial que sempre prevê a classe majoritária. Painel B:
    a diferença entre a acurácia obtida e essa linha de base, por algoritmo e
    base; valores negativos indicam desempenho pior que o do classificador
    trivial em acurácia.
    """
    data = pd.concat([classic, boost], ignore_index=True)
    algorithms = ALL_ALGORITHM_ORDER
    palette = dict(zip(algorithms, sns.color_palette(n_colors=len(algorithms))))
    offsets = np.linspace(-0.32, 0.32, len(algorithms))

    fig, (axis_a, axis_b) = plt.subplots(2, 1, figsize=(11.5, 8.4), sharex=True)

    for offset, algorithm in zip(offsets, algorithms):
        is_boost = algorithm in BOOST_ORDER
        for position, database in enumerate(ALL_DATABASE_ORDER):
            subset = data[
                data["algorithm"].eq(algorithm) & data["database_type"].eq(database)
            ]["accuracy"]
            if subset.empty:
                continue
            mean, low, high = bootstrap_mean_ci(subset.to_numpy())
            axis_a.errorbar(
                position + offset,
                mean,
                yerr=[[mean - low], [high - mean]],
                fmt="^" if is_boost else "o",
                color=palette[algorithm],
                markersize=6,
                capsize=2.5,
                elinewidth=1.0,
                label=ALGORITHM_LABELS[algorithm] if position == 0 else None,
            )
            axis_b.bar(
                position + offset,
                mean - MAJORITY_CLASS_BASELINE_ACCURACY[database],
                width=0.09,
                color=palette[algorithm],
                edgecolor="white",
                linewidth=0.4,
            )
    baseline_handle = None
    for position, database in enumerate(ALL_DATABASE_ORDER):
        baseline_handle = axis_a.scatter(
            position,
            MAJORITY_CLASS_BASELINE_ACCURACY[database],
            marker="*",
            s=170,
            color="black",
            zorder=5,
            label="Sempre classe majoritária" if position == 0 else None,
        )

    axis_a.set_title("Acurácia obtida por algoritmo e base")
    axis_a.set_ylim(0.0, 1.03)
    axis_a.grid(axis="x", visible=False)
    handles, labels = axis_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.995))

    axis_b.axhline(0, color="#444444", linestyle="--", linewidth=1.0)
    axis_b.set_title("Acurácia obtida menos a acurácia da classe majoritária")
    axis_b.set_xticks(range(len(ALL_DATABASE_ORDER)))
    axis_b.set_xticklabels(ALL_DATABASE_ORDER, rotation=28, ha="right")
    axis_b.grid(axis="x", visible=False)

    finish_figure(fig, "acuracia_linha_base.png", rect=(0.0, 0.0, 1.0, 0.93))

    summary_rows = []
    for family_name, family_data in [("classic", classic), ("boost", boost)]:
        means = family_data.groupby("database_type")["accuracy"].mean()
        for database, mean in means.items():
            summary_rows.append(
                {
                    "family": family_name,
                    "database_type": database,
                    "accuracy": mean,
                    "baseline": MAJORITY_CLASS_BASELINE_ACCURACY[database],
                    "delta": mean - MAJORITY_CLASS_BASELINE_ACCURACY[database],
                }
            )
    return pd.DataFrame(summary_rows)


def accuracy_versus_recall_scatter(classic: pd.DataFrame, boost: pd.DataFrame) -> None:
    """Dispersão de acurácia contra revocação, por configuração, colorida por base."""
    data = pd.concat([classic.assign(family="Clássicos"), boost.assign(family="Aumento de gradiente")], ignore_index=True)
    palette = dict(zip(ALL_DATABASE_ORDER, sns.color_palette(n_colors=len(ALL_DATABASE_ORDER))))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True, sharey=True)
    for axis, (family_label, family_data) in zip(
        axes, data.groupby("family", sort=False)
    ):
        for database in ALL_DATABASE_ORDER:
            subset = family_data[family_data["database_type"].eq(database)]
            if subset.empty:
                continue
            axis.scatter(
                subset["recall"],
                subset["accuracy"],
                s=14,
                alpha=0.55,
                color=palette[database],
                edgecolor="none",
                label=database,
            )
        axis.set_title(family_label)
        axis.set_xlabel("Revocação")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("Acurácia")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(ALL_DATABASE_ORDER), frameon=False, bbox_to_anchor=(0.5, 1.02))
    finish_figure(fig, "acuracia_versus_revocacao.png", rect=(0.0, 0.0, 1.0, 0.88))


# Valores que as analises supoem em cada eixo, tirados da grade de 22/09.
# Divergencias nao interrompem a execucao: sao impressas como [revisar].
EXPECTED_AXIS_VALUES = {
    "classicos": {
        "database_type": DATABASE_ORDER_CLASSIC,
        "algorithm": CLASSIC_ORDER,
        "categorical_encoding_strategy": ["ordered-target", "woe"],
        "undersampling_technique": ["RUS+SMOTE", "tomek-links"],
        "train_validation_test_strategy": PARTITION_POLICY_ORDER,
        "test_data_size": list(TEST_SIZE_LABELS),
    },
    "boosting": {
        "database_type": DATABASE_ORDER_BOOST,
        "algorithm": BOOST_ORDER,
        "train_validation_test_strategy": PARTITION_POLICY_ORDER,
        "test_data_size": list(TEST_SIZE_LABELS),
    },
}


def _observed_values(data: pd.DataFrame, column: str) -> set[str]:
    return set(data[column].dropna().astype(str).unique())


def has_values(data: pd.DataFrame, column: str, values: list[str]) -> bool:
    return column in data.columns and set(values) <= _observed_values(data, column)


def verificar_esquema(families: dict[str, pd.DataFrame]) -> int:
    """Imprime cada divergencia entre o que as analises supoem e os resultados."""
    divergences = 0
    for family, data in families.items():
        # Esquema revisado (com evaluation_unit) sem split: as medias
        # misturariam treino, validacao e teste.
        if "evaluation_unit" in data.columns and "split" not in data.columns:
            print(f"[revisar] {family}: coluna split ausente no esquema revisado")
            divergences += 1
    for family, expected_by_column in EXPECTED_AXIS_VALUES.items():
        data = families[family]
        for column, expected in expected_by_column.items():
            if column not in data.columns:
                print(f"[revisar] {family}: coluna {column} ausente nos resultados")
                divergences += 1
                continue
            observed = _observed_values(data, column)
            absent = [value for value in expected if value not in observed]
            unexpected = sorted(observed - set(expected))
            if absent:
                print(f"[revisar] {family}: {column} sem os valores {absent}")
                divergences += 1
            if unexpected:
                print(f"[revisar] {family}: {column} com valores nao previstos {unexpected}")
                divergences += 1
    print(f"verificacao de esquema: {divergences} divergencia(s)")
    return divergences


def run_analysis(
    name: str,
    data: pd.DataFrame,
    analysis,
    *args,
    columns: tuple[str, ...] = (),
    values: dict[str, list[str]] | None = None,
    **kwargs,
):
    """Roda a analise ou a pula, com aviso, se os dados nao tem o que ela supoe."""
    missing_columns = [column for column in columns if column not in data.columns]
    missing_values = {
        column: expected
        for column, expected in (values or {}).items()
        if not has_values(data, column, expected)
    }
    if missing_columns or missing_values:
        print(
            f"[revisar] analise {name} pulada: "
            f"colunas ausentes {missing_columns}, valores ausentes {missing_values}"
        )
        return None
    return analysis(*args, **kwargs)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--results-directory",
        required=True,
        help="Diretorio com resultados_<algoritmo>.csv (ou o layout de 22/09).",
    )
    parser.add_argument(
        "--figures-directory",
        required=True,
        help="Destino das figuras, normalmente eqm_paper/figuras da dissertacao.",
    )
    parser.add_argument(
        "--summary-output",
        default="discussao_resultados_resumo.json",
        help="JSON com os numeros citados no texto.",
    )
    return parser.parse_args()


def main() -> None:
    global SOURCE, FIGURES, SUMMARY
    arguments = parse_arguments()
    SOURCE = Path(arguments.results_directory)
    FIGURES = Path(arguments.figures_directory)
    SUMMARY = Path(arguments.summary_output)

    FIGURES.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    global EXPECTED_EXECUTIONS
    raw_classic, raw_boost = load_results()
    EXPECTED_AXIS_VALUES["classicos"]["algorithm"] = CLASSIC_ORDER
    verificar_esquema({"classicos": raw_classic, "boosting": raw_boost})
    classic = aggregate_folds(raw_classic)
    boost = aggregate_folds(raw_boost)
    # Pares previstos: da grade revisada quando os resultados sao dela (tem
    # split); na grade de 22/09, que experiment_grid nao descreve, das
    # execucoes tentadas, inclusive as que falharam.
    if "split" in raw_classic.columns:
        EXPECTED_EXECUTIONS = expected_executions_from_grid(
            (experiment_grid.BLOCK_CLASSIC, experiment_grid.BLOCK_BOOSTING)
        )
    else:
        print("[revisar] resultados sem split: pares previstos tirados das execucoes tentadas")
        EXPECTED_EXECUTIONS = pd.concat(
            [attempted_executions(raw_classic), attempted_executions(raw_boost)], ignore_index=True
        )

    reserve = ("test_data_size",)
    ulb = {"database_type": ["eqm-ref-18"]}
    encodings = {"categorical_encoding_strategy": ["ordered-target", "woe"]}
    resampling = {"undersampling_technique": ["RUS+SMOTE", "tomek-links"]}

    metric_point_grid(
        classic,
        DATABASE_ORDER_CLASSIC,
        CLASSIC_ORDER,
        "resultados_classicos_algoritmo_base.png",
    )
    classic_preprocessing_heatmaps(classic)
    balance_deltas = run_analysis(
        "balance_delta_plot", classic, balance_delta_plot, classic, values=resampling
    )
    run_analysis(
        "classicos_codificacao_base_algoritmo",
        classic,
        classic_strategy_by_base_heatmaps,
        classic,
        "categorical_encoding_strategy",
        ["ordered-target", "woe"],
        {"ordered-target": "Alvo ordenado", "woe": "WoE"},
        "resultados_classicos_codificacao_base_algoritmo.png",
        values=encodings,
    )
    run_analysis(
        "classicos_subamostragem_base_algoritmo",
        classic,
        classic_strategy_by_base_heatmaps,
        classic,
        "undersampling_technique",
        ["RUS+SMOTE", "tomek-links"],
        {"RUS+SMOTE": "RUS+SMOTE", "tomek-links": "Tomek Links"},
        "resultados_classicos_subamostragem_base_algoritmo.png",
        values=resampling,
    )
    metric_point_grid(
        boost,
        DATABASE_ORDER_BOOST,
        BOOST_ORDER,
        "resultados_boost_algoritmo_base.png",
    )
    boost_fillna_grid(boost)
    boost_fillna_grid_missing_bases(boost)
    run_analysis(
        "categorical_cross_family_grid", classic, categorical_cross_family_grid, classic, boost, values=ulb
    )
    run_analysis("partition_size_effect_grid", classic, partition_size_effect_grid, classic, columns=reserve)
    run_analysis(
        "partition_policy_by_base_grid",
        classic,
        partition_policy_by_base_grid,
        classic,
        DATABASE_ORDER_CLASSIC,
        columns=reserve,
    )
    best_configuration_summary_grid(classic, CLASSIC_ORDER, DATABASE_ORDER_CLASSIC)
    print(best_f2_configuration_winner_table(classic, DATABASE_ORDER_CLASSIC).to_string(index=False))

    run_analysis(
        "partition_size_effect_grid_boost",
        boost,
        partition_size_effect_grid,
        boost,
        "particionamento_tamanho_reserva_boost.png",
        family_label="CatBoost e XGBoost",
        columns=reserve,
    )
    run_analysis(
        "partition_policy_by_base_grid_boost",
        boost,
        partition_policy_by_base_grid,
        boost,
        DATABASE_ORDER_BOOST,
        "particionamento_politica_base_boost.png",
        columns=reserve,
    )
    best_configuration_summary_grid(
        boost, BOOST_ORDER, DATABASE_ORDER_BOOST, "resumo_melhor_configuracao_base_boost.png",
    )
    print(best_f2_configuration_winner_table(boost, DATABASE_ORDER_BOOST).to_string(index=False))
    knn_catboost_xgboost = pd.concat(
        [classic[classic["algorithm"].eq("knn")], boost], ignore_index=True
    )
    best_configuration_summary_grid(
        knn_catboost_xgboost,
        ["knn", "catboost", "xgboost"],
        DATABASE_ORDER_BOOST,
        "resumo_melhor_configuracao_knn_catboost_xgboost.png",
    )
    accuracy_summary = accuracy_baseline_grid(classic, boost)
    print(accuracy_summary.to_string(index=False))
    accuracy_versus_recall_scatter(classic, boost)

    summary = build_summary(raw_classic, raw_boost, classic, boost, balance_deltas)
    pair_counts = pd.DataFrame(PAIR_COUNTS)
    pair_counts_path = SUMMARY.with_name("pares_por_comparacao.csv")
    pair_counts.to_csv(pair_counts_path, index=False)
    summary["pairs"] = (
        pair_counts.groupby("comparacao")[
            ["previstos", "realizados", *(f"descartados_{reason}" for reason in PAIR_DISCARD_REASONS)]
        ].sum().reset_index().to_dict(orient="records")
        if not pair_counts.empty else []
    )
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["coverage"], ensure_ascii=False, indent=2))
    print(f"summary={SUMMARY}")
    print(f"pares={pair_counts_path}")


if __name__ == "__main__":
    main()
