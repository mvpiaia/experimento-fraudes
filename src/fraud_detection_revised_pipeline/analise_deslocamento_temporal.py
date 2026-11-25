"""Diagnóstico do deslocamento temporal nas bases com ordem temporal.

Feito depois dos resultados, para interpretar a Q1 (holdout temporal contra
estratificado). Não altera hipóteses, regras de decisão, manifestos nem
executores: só lê dados preparados e bases brutas.

Duas análises:

* ``taxa``: taxa de fraude e volume por período na base bruta, com a parte de
  cada período que cai no treino, na validação e no teste do holdout
  temporal. O corte é o do próprio pipeline (``data_partitioning``), e os
  tamanhos são conferidos com os arquivos preparados na análise adversarial.
  Período: mês civil na Sparkov, blocos de 30 dias de ``TransactionDT`` na
  IEEE-CIS (a data de início não é publicada) e blocos de 6 h de ``Time`` na
  ULB, que cobre dois dias.

* ``adversarial``: validação adversarial. Treino (origem 0) e teste
  (origem 1) de uma configuração preparada são reunidos, sem o rótulo de
  fraude, e um XGBoost com os mesmos parâmetros da grade tenta prever a
  origem, em validação cruzada estratificada de 5 dobras (semente 42). AUC
  perto de 0,5 indica que treino e teste não se distinguem; perto de 1,
  deslocamento forte. O ganho médio de cada atributo indica o que mudou. A
  mesma análise no holdout estratificado serve de controle: sorteado, treino
  e teste só se distinguem pelo que o sorteio separa (na Sparkov, que agrupa
  por titular, os titulares). Cada análise roda com todos os atributos e sem
  os atributos de calendário (``CALENDAR_COLUMNS``), que mudam por construção
  num corte pelo tempo.

Uso::

    python3 analise_deslocamento_temporal.py adversarial \\
        --prepared-data-directory /mnt/network-black-hd/prepared \\
        --prepared-dataset-manifest .../prepared_dataset_configurations.tsv \\
        --output-directory saidas --n-jobs 8
    python3 analise_deslocamento_temporal.py taxa \\
        --raw-directory raw --output-directory saidas
    python3 analise_deslocamento_temporal.py q1-titular \\
        --results-directory resultados --output-directory saidas
    python3 analise_deslocamento_temporal.py ciclo-ulb \\
        --raw-directory raw --output-directory saidas --n-jobs 8
    python3 analise_deslocamento_temporal.py saidas \\
        --output-directory saidas --tables-directory tabelas \\
        --figures-directory figuras
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from catboost_xgboost_specific import (
    build_xgboost_category_domains,
    create_xgboost_classifier,
    discover_categorical_columns,
    prepare_xgboost_dataframe,
)
from data_partitioning import AUXILIARY_SOURCE_COLUMN, _temporal_holdout
from experiment_datasets import get_dataset_definition
from prepared_dataset_storage import load_prepared_data_splits

RANDOM_SEED = 42
N_SPLITS = 5
TEMPORAL = "temporal-holdout"
STRATIFIED = "stratified-holdout"
DATASETS = ("ulb", "sparkov", "sparkov-sem-titular", "ieee-cis-completa",
            "ieee-cis-sem-identidade", "ieee-cis-transacional")
# Atributos derivados do relógio da transação (experiment_datasets). Num corte
# pelo tempo eles mudam por construção; a variante sem eles mede o
# deslocamento além do calendário.
CALENDAR_COLUMNS = ("hour_of_day", "day_of_week", "is_weekend", "month", "day_of_month", "age_years")
# Abaixo desta AUC o classificador não separa treino e teste, e o ganho dos
# atributos não indica deslocamento.
MIN_INFORMATIVE_AUC = 0.6
VARIANTS = {"todos": (), "sem-calendario": CALENDAR_COLUMNS}
SPLIT_LABELS = {"train_data": "treino", "validation_data": "validação", "test_data": "teste"}
SECONDS_PER_DAY = 86_400
PERIOD_LABELS = {"ulb": "bloco de 6 h", "sparkov": "mês (mm/aa)"}


# --------------------------------------------------------------- adversarial

def select_configuration(manifest: pd.DataFrame, dataset_id: str, test_strategy: str) -> pd.Series:
    """Configuração nativa, sem reamostragem e sem imputação da preparação."""
    rows = manifest[
        (manifest["dataset_id"] == dataset_id)
        & (manifest["test_strategy"] == test_strategy)
        & (manifest["categorical_handling_mode"] == "native")
        & (manifest["undersampling_technique"] == "none")
        & (manifest["fill_na_strategy"].isin(["none", "native"]))
    ]
    configurations = rows["configuration_md5"].unique()
    if len(configurations) != 1:
        raise ValueError(
            f"{dataset_id}/{test_strategy}: esperada uma configuração, achadas {len(configurations)}"
        )
    return rows.iloc[0]


def adversarial_frame(train: pd.DataFrame, test: pd.DataFrame, target_column: str):
    """Atributos de treino e teste reunidos, com a origem como alvo (teste = 1)."""
    features = pd.concat(
        [train.drop(columns=[target_column]), test.drop(columns=[target_column])],
        ignore_index=True,
    )
    origin = np.concatenate([np.zeros(len(train), dtype=int), np.ones(len(test), dtype=int)])
    return features, origin


def adversarial_auc(features: pd.DataFrame, origin: np.ndarray, n_jobs: int = 1,
                    n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> dict:
    """AUC da origem por dobra e ganho médio normalizado de cada atributo."""
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    categorical_columns = discover_categorical_columns(features)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs, gains = [], []
    for fit_index, held_index in folds.split(features, origin):
        fit_features = features.iloc[fit_index]
        domains = build_xgboost_category_domains(fit_features, categorical_columns)
        model = create_xgboost_classifier()
        model.set_params(n_jobs=n_jobs)
        model.fit(prepare_xgboost_dataframe(fit_features, categorical_columns, domains), origin[fit_index])
        held = prepare_xgboost_dataframe(features.iloc[held_index], categorical_columns, domains)
        aucs.append(float(roc_auc_score(origin[held_index], model.predict_proba(held)[:, 1])))
        gain = pd.Series(model.get_booster().get_score(importance_type="total_gain"), dtype=float)
        gains.append(gain.reindex(features.columns).fillna(0.0) / max(gain.sum(), 1e-12))
    mean_gain = pd.concat(gains, axis=1).mean(axis=1).sort_values(ascending=False)
    return {"auc_folds": aucs, "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs, ddof=1)), "gain": mean_gain}


def run_adversarial(args: argparse.Namespace) -> None:
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.prepared_dataset_manifest, sep="\t", dtype=str)
    rows, gains = [], []
    for dataset_id in args.datasets:
        for test_strategy in (TEMPORAL, STRATIFIED):
            row = select_configuration(manifest, dataset_id, test_strategy)
            start = time.monotonic()
            # Os dois holdouts têm uma única dobra.
            fold = load_prepared_data_splits(
                args.prepared_data_directory, args.prepared_dataset_manifest,
                row["configuration_md5"], test_strategy,
            )["test_info"][0]
            target = row["prediction_column_name"]
            train, test = fold["train_data"], fold["test_data"]
            all_features, origin = adversarial_frame(train, test, target)
            for variant, excluded in VARIANTS.items():
                features = all_features.drop(columns=[c for c in excluded if c in all_features.columns])
                rows.append(_adversarial_row(dataset_id, test_strategy, variant, row, fold,
                                             train, test, target, features, origin, args, gains, start))
                pd.DataFrame(rows).to_csv(out / "adversarial.csv", index=False)
                pd.concat(gains).to_csv(out / "adversarial_ganho.csv", index=False)


def _adversarial_row(dataset_id, test_strategy, variant, row, fold, train, test, target,
                     features, origin, args, gains, start) -> dict:
    result = adversarial_auc(features, origin, n_jobs=args.n_jobs)
    elapsed = time.monotonic() - start
    print(f"{dataset_id} {test_strategy} {variant}: AUC {result['auc_mean']:.4f} "
          f"± {result['auc_std']:.4f} ({elapsed:.0f} s)", flush=True)
    gain = result["gain"].rename("gain").reset_index().rename(columns={"index": "feature"})
    gain.insert(0, "variant", variant)
    gain.insert(0, "test_strategy", test_strategy)
    gain.insert(0, "dataset_id", dataset_id)
    gains.append(gain)
    return {
        "dataset_id": dataset_id, "test_strategy": test_strategy, "variant": variant,
        "configuration_md5": row["configuration_md5"],
        "train_rows": len(train), "validation_rows": len(fold["validation_data"]),
        "test_rows": len(test), "feature_count": features.shape[1],
        "train_fraud_rate": float(train[target].mean()),
        "test_fraud_rate": float(test[target].mean()),
        "auc_mean": result["auc_mean"], "auc_std": result["auc_std"],
        "auc_folds": json.dumps(result["auc_folds"]),
        "split_md5": json.dumps(fold.get("split_md5", {}), sort_keys=True),
        "seconds": round(elapsed, 1),
    }


# --------------------------------------------------------------------- taxa

def _ordered_population(dataset_id: str, raw: Path) -> pd.DataFrame:
    """Coluna de ordem, rótulo e marca de arquivo auxiliar, na ordem de carga."""
    if dataset_id in ("sparkov", "sparkov-sem-titular"):
        columns = ["trans_date_trans_time", "is_fraud"]
        main = pd.read_csv(raw / "fraudTrain.csv", usecols=columns)
        auxiliary = pd.read_csv(raw / "fraudTest.csv", usecols=columns)
        main[AUXILIARY_SOURCE_COLUMN] = False
        auxiliary[AUXILIARY_SOURCE_COLUMN] = True
        data = pd.concat([main, auxiliary], ignore_index=True)
        data["trans_date_trans_time"] = pd.to_datetime(data["trans_date_trans_time"])
        data["period"] = data["trans_date_trans_time"].dt.to_period("M").astype(str)
        return data.rename(columns={"is_fraud": "label"})
    if dataset_id == "ulb":
        # Como na carga do pipeline, registros idênticos saem antes da partição.
        data = pd.read_csv(raw / "creditcard.csv").drop_duplicates().reset_index(drop=True)[["Time", "Class"]]
        data["period"] = (data["Time"] // (6 * 3600)).astype(int) + 1
        return data.rename(columns={"Class": "label"})
    transactions = pd.read_csv(raw / "train_transaction.csv", usecols=["TransactionID", "TransactionDT", "isFraud"])
    if dataset_id in ("ieee-cis-completa", "ieee-cis-sem-identidade"):
        identity = pd.read_csv(raw / "train_identity.csv", usecols=["TransactionID"])
        transactions = transactions.merge(identity, on="TransactionID", how="inner")
    day = (transactions["TransactionDT"] - transactions["TransactionDT"].min()) / SECONDS_PER_DAY
    block = (day // 30).astype(int)
    # A base cobre cerca de 182 dias: o resto de poucos dias entra no último
    # bloco completo, em vez de formar um bloco com taxa ruidosa.
    if day.max() % 30 < 15:
        block = block.clip(upper=max(int(day.max() // 30) - 1, 0))
    transactions["period"] = block + 1
    return transactions.rename(columns={"isFraud": "label"}).reset_index(drop=True)


def temporal_split_labels(data: pd.DataFrame, dataset_id: str) -> pd.Series:
    """Parte (treino, validação, teste) de cada linha pelo corte do pipeline."""
    folds = _temporal_holdout(data.reset_index(drop=True), get_dataset_definition(dataset_id))
    split = pd.Series(index=data.index, dtype=object)
    for split_name, positions in folds[0].items():
        if positions is not None:
            split.iloc[positions] = SPLIT_LABELS[split_name]
    return split


def run_rates(args: argparse.Namespace) -> None:
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw_directory)
    tables = []
    for dataset_id in args.datasets:
        data = _ordered_population(dataset_id, raw)
        data["split"] = temporal_split_labels(data, dataset_id)
        table = (data.groupby(["period", "split"], sort=True)["label"]
                 .agg(rows="size", frauds="sum").reset_index())
        table.insert(0, "dataset_id", dataset_id)
        tables.append(table)
        sizes = data["split"].value_counts().to_dict()
        print(f"{dataset_id}: {sizes}", flush=True)
    result = pd.concat(tables, ignore_index=True)
    result["period"] = result["period"].astype(str)
    result.to_csv(out / "taxa_por_periodo.csv", index=False)


# ------------------------------------------------------------- ciclo-ulb

ULB_BLOCK_SECONDS = 6 * 3600


def ulb_daily_cycle_comparisons(data: pd.DataFrame) -> dict:
    """Pares (origem 0, origem 1) que separam o ciclo do dia da passagem dos dias.

    A ULB cobre dois dias em oito blocos de 6 h (1 a 4 no primeiro dia, 5 a 8
    no segundo). O teste do holdout temporal é o bloco 8; o bloco 4 cobre a
    mesma faixa horária no dia anterior.
    """
    period, split = data["period"], data["split"]
    return {
        "treino x teste (holdout temporal)": (split.eq("treino"), split.eq("teste")),
        "bloco 4 x bloco 8 (mesma faixa horária, dias diferentes)": (period.eq(4), period.eq(8)),
        "dia 1 x dia 2 (blocos 1-4 x 5-8)": (period.le(4), period.ge(5)),
        "bloco 7 x bloco 8 (mesmo dia, faixas vizinhas)": (period.eq(7), period.eq(8)),
    }


def run_ulb_daily_cycle(args: argparse.Namespace) -> None:
    """Validação adversarial da ULB sem atributos de calendário, por recorte."""
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(Path(args.raw_directory) / "creditcard.csv").drop_duplicates().reset_index(drop=True)
    data["period"] = (data["Time"] // ULB_BLOCK_SECONDS).astype(int) + 1
    data["split"] = temporal_split_labels(data, "ulb")
    # Mesmos atributos da variante sem calendário: V1 a V28 e Amount. O log de
    # Amount e a padronização da grade não mudam os cortes de árvore.
    features = data[[f"V{i}" for i in range(1, 29)] + ["Amount"]]
    rows, gains = [], []
    for name, (first, second) in ulb_daily_cycle_comparisons(data).items():
        mask = first | second
        origin = second[mask].astype(int).to_numpy()
        result = adversarial_auc(features[mask].reset_index(drop=True), origin, n_jobs=args.n_jobs)
        top = result["gain"].head(3)
        print(f"{name}: AUC {result['auc_mean']:.4f} ± {result['auc_std']:.4f}; "
              + ", ".join(f"{k} {v:.0%}" for k, v in top.items()), flush=True)
        rows.append({"comparison": name, "rows_0": int(first.sum()), "rows_1": int(second.sum()),
                     "fraud_rate_0": float(data.loc[first, "Class"].mean()),
                     "fraud_rate_1": float(data.loc[second, "Class"].mean()),
                     "auc_mean": result["auc_mean"], "auc_std": result["auc_std"],
                     "auc_folds": json.dumps(result["auc_folds"])})
        gain = result["gain"].rename("gain").reset_index().rename(columns={"index": "feature"})
        gain.insert(0, "comparison", name)
        gains.append(gain)
    pd.DataFrame(rows).to_csv(out / "ciclo_diario_ulb.csv", index=False)
    pd.concat(gains).to_csv(out / "ciclo_diario_ulb_ganho.csv", index=False)


# --------------------------------------------------------------- q1-titular

def run_q1_holder(args: argparse.Namespace) -> None:
    """Q1 (temporal - estratificado) na Sparkov e na Sparkov-sem-titular.

    Mesma regra de ``analise_hipoteses.analyse_q1``: por base, algoritmo e
    protocolo, a configuração de maior F2 na validação, lida no teste.
    """
    import analise_hipoteses as hipoteses

    executions, _ = hipoteses.build_executions(hipoteses.load_results(args.results_directory))
    datasets = ("sparkov", "sparkov-sem-titular")
    by_protocol = {protocol: hipoteses.representatives_by_source(executions, datasets, protocol)
                   for protocol in (STRATIFIED, TEMPORAL)}
    merged = by_protocol[STRATIFIED].merge(by_protocol[TEMPORAL], on=["database_type", "algorithm"],
                                           suffixes=("_stratified", "_temporal"))
    for metric in ("f_beta_2", "auc_pr"):
        merged[f"{metric}_temporal_minus_stratified"] = merged[f"{metric}_temporal"] - merged[f"{metric}_stratified"]
    columns = ["database_type", "algorithm", "f_beta_2_stratified", "f_beta_2_temporal",
               "f_beta_2_temporal_minus_stratified", "auc_pr_temporal_minus_stratified"]
    table = merged[columns].sort_values(["algorithm", "database_type"])
    table.to_csv(Path(args.output_directory) / "q1_titular.csv", index=False)
    print(table.round(3).to_string(index=False))


# ------------------------------------------------------------------- saídas

ULB_SAME_HOURS = "bloco 4 x bloco 8 (mesma faixa horária, dias diferentes)"
ULB_CYCLE_CAPTION = (
    "Na linha ULB, mesma faixa horária, a comparação sem calendário é entre o último bloco de 6~h do "
    "segundo dia, que contém o teste temporal, e o bloco da mesma faixa horária do primeiro dia, com a "
    "taxa de fraude de cada bloco. "
)


def _gain_cell(top: pd.DataFrame) -> str:
    import estilo_figuras as estilo

    return ", ".join(f"\\texttt{{{f.replace('_', chr(92) + '_')}}} ({estilo.number(100 * g, 0)}\\%)"
                     for f, g in zip(top.feature, top.gain))


def _ulb_cycle_row(cycle: pd.DataFrame, cycle_gains: pd.DataFrame, estilo) -> str:
    r = cycle[cycle["comparison"] == ULB_SAME_HOURS].iloc[0]
    top = cycle_gains[cycle_gains["comparison"] == ULB_SAME_HOURS].nlargest(3, "gain")
    cells = [r"\quad ULB, mesma faixa horária", "---", estilo.number(r.auc_mean), "---", "---",
             f"{estilo.number(100 * r.fraud_rate_0, 2)}\\% / {estilo.number(100 * r.fraud_rate_1, 2)}\\%",
             _gain_cell(top)]
    return " & ".join(cells) + r" \\"

def run_outputs(args: argparse.Namespace) -> None:
    import matplotlib.pyplot as plt
    import estilo_figuras as estilo

    out = Path(args.output_directory)
    adversarial = pd.read_csv(out / "adversarial.csv")
    gains = pd.read_csv(out / "adversarial_ganho.csv")
    rates = pd.read_csv(out / "taxa_por_periodo.csv", dtype={"period": str})

    # Conferência: o corte refeito sobre a base bruta tem os tamanhos dos
    # arquivos preparados do holdout temporal.
    sizes = rates.groupby(["dataset_id", "split"])["rows"].sum().unstack()
    for row in adversarial[(adversarial["test_strategy"] == TEMPORAL)
                           & (adversarial["variant"] == "todos")].itertuples():
        if row.dataset_id in sizes.index:
            observed = (sizes.loc[row.dataset_id, "treino"], sizes.loc[row.dataset_id, "teste"])
            if observed != (row.train_rows, row.test_rows):
                print(f"AVISO {row.dataset_id}: corte bruto {observed} != preparado "
                      f"({row.train_rows}, {row.test_rows})")

    def pick(dataset_id, strategy, variant):
        return adversarial[(adversarial.dataset_id == dataset_id) & (adversarial.test_strategy == strategy)
                           & (adversarial.variant == variant)].iloc[0]

    # Linha opcional da ULB por recorte (comando ciclo-ulb), logo abaixo da ULB.
    cycle_path = out / "ciclo_diario_ulb.csv"
    cycle = pd.read_csv(cycle_path) if cycle_path.exists() else None
    cycle_gains = pd.read_csv(out / "ciclo_diario_ulb_ganho.csv") if cycle is not None else None

    lines = []
    for dataset_id in args.datasets:
        cells = [estilo.dataset_label(dataset_id)]
        for strategy, variant in ((TEMPORAL, "todos"), (TEMPORAL, "sem-calendario"),
                                  (STRATIFIED, "todos"), (STRATIFIED, "sem-calendario")):
            r = pick(dataset_id, strategy, variant)
            cells.append(f"{estilo.number(r.auc_mean)}")
        r = pick(dataset_id, TEMPORAL, "todos")
        cells.append(f"{estilo.number(100 * r.train_fraud_rate, 2)}\\% / {estilo.number(100 * r.test_fraud_rate, 2)}\\%")
        top = gains[(gains.dataset_id == dataset_id) & (gains.test_strategy == TEMPORAL)
                    & (gains.variant == "sem-calendario")].nlargest(3, "gain")
        if pick(dataset_id, TEMPORAL, "sem-calendario").auc_mean < MIN_INFORMATIVE_AUC:
            # Sem separação, o ganho só reparte o ruído entre os atributos.
            cells.append("---")
            lines.append(" & ".join(cells) + r" \\")
            continue
        cells.append(_gain_cell(top))
        lines.append(" & ".join(cells) + r" \\")
        if dataset_id == "ulb" and cycle is not None:
            lines.append(_ulb_cycle_row(cycle, cycle_gains, estilo))
    table = (
        "\\begin{table}[H]\n    \\centering\n    \\footnotesize\n    \\setlength{\\tabcolsep}{3pt}\n"
        "    \\caption[Validação adversarial entre treino e teste]{Validação adversarial: \\acs{aucroc} "
        "(média de cinco dobras) de um XGBoost, com os parâmetros da grade, treinado para distinguir as "
        "transações do treino das do teste, no \\textit{holdout} temporal e no estratificado, com todos os "
        "atributos e sem os atributos de calendário (hora, dia da semana, fim de semana, mês, dia do mês e "
        "idade do titular); taxa de fraude do treino e do teste no temporal; e os três atributos de maior "
        "ganho no temporal sem calendário, com a fração do ganho total (omitidos quando a \\acs{aucroc} fica abaixo de 0,6). "
        + (ULB_CYCLE_CAPTION if cycle is not None else "")
        + "Diagnóstico feito depois dos resultados.}\n"
        "    \\label{tab:deslocamento-adversarial}\n"
        "    \\begin{tabularx}{\\textwidth}{lccccc>{\\raggedright\\arraybackslash}X}\n        \\toprule\n"
        "        & \\multicolumn{2}{c}{{\\bf Temporal}} & \\multicolumn{2}{c}{{\\bf Estratificado}} & & \\\\\n"
        "        \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
        "        {\\bf Base} & todos & s/ cal. & todos & s/ cal. & {\\bf Fraude tr./teste} & "
        "{\\bf Maior ganho (temporal, s/ cal.)} \\\\\n        \\midrule\n        "
        + "\n        ".join(lines)
        + "\n        \\bottomrule\n    \\end{tabularx}\n\\end{table}\n"
    )
    tables_directory = Path(args.tables_directory)
    tables_directory.mkdir(parents=True, exist_ok=True)
    (tables_directory / "tab_deslocamento.tex").write_text(table, encoding="utf-8")

    # Figura: taxa de fraude por período, com a parte de cada período.
    estilo.configure()
    # Mesma população da Sparkov e da IEEE-CIS-completa, respectivamente.
    panels = [d for d in args.datasets if d not in ("sparkov-sem-titular", "ieee-cis-sem-identidade")]
    fig, axes = plt.subplots(len(panels), 1, figsize=(estilo.PAGE_WIDTH_IN, 2.1 * len(panels)))
    colors = dict(zip(SPLIT_LABELS.values(), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for axis, dataset_id in zip(np.atleast_1d(axes), panels):
        part = rates[rates.dataset_id == dataset_id]
        periods = list(dict.fromkeys(sorted(part["period"], key=lambda p: (len(p), p))))
        totals = part.groupby("period")[["rows", "frauds"]].sum().reindex(periods)
        position = {p: i for i, p in enumerate(periods)}
        bottom = np.zeros(len(periods))
        for split in SPLIT_LABELS.values():
            s = part[part.split == split].set_index("period").reindex(periods)
            share = (s["rows"].fillna(0) / totals["rows"]).to_numpy()
            rate = 100 * totals["frauds"] / totals["rows"]
            axis.bar(range(len(periods)), share * rate.to_numpy(), bottom=bottom,
                     color=colors[split], label=split, width=0.85)
            bottom += share * rate.to_numpy()
        axis.set_ylim(0, 1.1 * np.nanmax(bottom))  # folga para a barra mais alta não encostar na moldura
        axis.set_title(estilo.dataset_label(dataset_id))
        axis.set_ylabel("fraude (%)")
        axis.set_xlabel(PERIOD_LABELS.get(dataset_id, "bloco de 30 dias"))
        step = 3 if dataset_id.startswith("sparkov") else 1
        ticks = periods[::step]
        names = [f"{p[5:7]}/{p[2:4]}" for p in ticks] if dataset_id.startswith("sparkov") else ticks
        axis.set_xticks([position[p] for p in ticks], names, rotation=0)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    estilo.legend_below(fig, handles, labels, ncol=3)
    estilo.save(fig, Path(args.figures_directory) / "deslocamento_taxa_fraude.png")
    print(table)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnóstico do deslocamento temporal (Q1).")
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("adversarial")
    a.add_argument("--prepared-data-directory", required=True)
    a.add_argument("--prepared-dataset-manifest", required=True)
    a.add_argument("--n-jobs", type=int, default=1)
    t = sub.add_parser("taxa")
    t.add_argument("--raw-directory", required=True)
    s = sub.add_parser("saidas")
    s.add_argument("--tables-directory", required=True)
    s.add_argument("--figures-directory", required=True)
    q = sub.add_parser("q1-titular")
    q.add_argument("--results-directory", required=True)
    c = sub.add_parser("ciclo-ulb")
    c.add_argument("--raw-directory", required=True)
    c.add_argument("--n-jobs", type=int, default=1)
    for p in (a, t, s, q, c):
        p.add_argument("--output-directory", required=True)
        p.add_argument("--datasets", nargs="+", default=list(DATASETS))
    args = parser.parse_args()
    {"adversarial": run_adversarial, "taxa": run_rates, "saidas": run_outputs,
     "q1-titular": run_q1_holder, "ciclo-ulb": run_ulb_daily_cycle}[args.command](args)


if __name__ == "__main__":
    main()
