"""Estima o tempo de execucao da grade, por algoritmo e por base.

A estimativa e construida a partir de medicoes reais, e nao de palpite, mas
continua sendo uma estimativa: ela extrapola de poucos pontos medidos para as
bases e protocolos que ainda nao rodaram. As premissas estao todas explicitas
abaixo, para que qualquer uma possa ser contestada e trocada.

Base de medicao
---------------
1. Medicoes proprias, feitas neste pipeline, na ULB com o protocolo
   ``stratified-holdout`` sem reamostragem (198.608 linhas de treino, 42.559
   de teste, 30 atributos). E o cenario mais caro dos algoritmos classicos,
   porque o treino inteiro entra no ajuste. Valores em ``MEASURED_ULB``.
2. Medianas das execucoes historicas do pipeline anterior, por algoritmo e
   base, extraidas de ``resultados_originais_*.tar.gz``. Cobrem CatBoost e
   XGBoost, que nao estao instalados neste ambiente. Valores em
   ``HISTORICAL_MEDIAN``.
3. Medicoes de GPU do FT-Transformer registradas na execucao anterior
   (RTX 4050), em ``TRANSFORMER_SECONDS_PER_FOLD``.

Premissas de extrapolacao
-------------------------
* Regressoes linear e logistica e Naive Bayes custam O(n x d): o tempo cresce
  **linearmente** com o numero de linhas.
* KNN custa O(n_treino x n_teste x d). Com indexacao em arvore o crescimento
  observado fica entre linear e quadratico; adota-se **n^1.5**, e o efeito de
  escolher 1,0 ou 2,0 esta no intervalo de incerteza reportado.
* CatBoost e XGBoost usam as medianas historicas diretamente, escaladas por
  linhas quando a base nao tem historico (as duas variantes da IEEE-CIS que
  nunca foram executadas).
* A PCA agora roda na Sparkov e reduz a dimensionalidade antes do ajuste, e
  os identificadores sairam dos atributos em todas as bases. Os dois efeitos
  **reduzem** o custo por dobra, e nenhum deles entra na conta: a estimativa
  e, nesse sentido, conservadora.
* O custo de preparacao (carga, particao, imputacao, codificacao,
  padronizacao, PCA, gravacao) e estimado a parte, por configuracao.

Uso:
    python3 estimate_execution_time.py
    python3 estimate_execution_time.py --format markdown
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import experiment_grid
from experiment_datasets import DATASET_DEFINITIONS


# Linhas de cada base apos a remocao de registros exatamente duplicados.
DATASET_ROW_COUNT = {
    "ulb": 283_726,
    "home-credit-completa": 1_430_155,
    "home-credit-somente-aplicacao": 307_511,
    "sparkov": 1_852_394,
    "ieee-cis-completa": 144_233,
    "ieee-cis-sem-identidade": 144_233,
    "ieee-cis-transacional": 590_540,
}
REFERENCE_DATASET = "ulb"

# Medido neste pipeline, ULB, holdout estratificado, sem reamostragem.
# Inclui a partida do Python, a copia local dos arquivos e a gravacao.
MEASURED_ULB_SECONDS = {
    "logistic-regression": 3.8,
    "naive-bayes": 2.1,
    "knn": 39.2,
}

# Medianas das execucoes historicas (pipeline anterior), em segundos por
# execucao. Usadas para CatBoost e XGBoost.
HISTORICAL_MEDIAN_SECONDS = {
    "catboost": {
        "ulb": 24.8,
        "home-credit-completa": 3284.3,
        # Sem a juncao a base tem 307.511 linhas em vez de 1,41 milhao; o
        # custo escala com as linhas.
        "home-credit-somente-aplicacao": 3284.3 * 307_511 / 1_413_701,
        "sparkov": 171.8,
        "ieee-cis-completa": 430.9,
    },
    "xgboost": {
        "ulb": 28.0,
        "home-credit-completa": 3351.1,
        "home-credit-somente-aplicacao": 3351.1 * 307_511 / 1_413_701,
        "sparkov": 198.6,
        "ieee-cis-completa": 451.2,
    },
}
# Ancoras medidas no Acer em 31/10/2026, com o codigo atual: uma dobra sem
# reamostragem da ieee-cis-transacional (temporal-holdout, card1 categorico),
# em segundos de relogio do runner (staging, ajuste, pontuacao e gravacao).
# O XGBoost levou 85 s (ajuste de 65 s) e o CatBoost 1.755 s (ajuste de
# 1.737 s), contra 4.618 s e 4.410 s pelas medianas historicas: o pipeline
# anterior era muito mais lento no XGBoost. As medianas continuam dando a
# proporcao entre as bases; o nivel de cada algoritmo vem da ancora.
BOOSTING_ANCHOR_DATASET = "ieee-cis-transacional"
MEASURED_BOOSTING_ANCHOR_SECONDS = {
    "xgboost": 85.4,
    "catboost": 1755.5,
}

# As medianas historicas vem de uma grade em que metade das configuracoes
# reamostrava o treino para 1:1, reduzindo-o drasticamente. A grade nova
# acrescenta o braco "sem reamostragem", que treina com a base inteira. Este
# fator corrige a media das duas situacoes.
NO_RESAMPLING_COST_FACTOR = 2.5

# FT-Transformer, segundos por dobra, medidos em GPU (RTX 4050) na execucao
# anterior. O custo e dominado pelo numero de transacoes e pela largura da
# base, nao pela cabeca de classificacao: mlp e lstm acrescentam uma camada
# sobre o mesmo codificador.
TRANSFORMER_SECONDS_PER_FOLD = {
    "ulb": 10.0,
    "home-credit-completa": 180.0,
    "home-credit-somente-aplicacao": 40.0,
    "sparkov": 950.0,
    "ieee-cis-completa": 75.0,
    "ieee-cis-sem-identidade": 70.0,
    "ieee-cis-transacional": 304.0,
}

# TabBERT, medido no Acer (RTX 4050, bfloat16, lote 256) em 31/10/2026, com o
# codigo atual: uma epoca supervisionada na Sparkov (1.067.850 linhas de
# treino no temporal-holdout) levou 96 s de ajuste (fit_time, que inclui a
# montagem dos tensores), e a pontuacao de validacao e teste, 24 s. Antes do
# ajuste, a leitura, o staging e o indice de janelas levaram ~440 s, com a GPU
# parada: essa parte e fixa por execucao, e nao por epoca. A medicao anterior
# (4.065 s por epoca na home-credit-completa) vinha de uma versao mais antiga
# do executor e superestimava o custo em ~55 vezes. Tudo escala com o numero
# de linhas, porque o numero de janelas causais e proporcional a ele.
TABBERT_SECONDS_PER_EPOCH_REFERENCE = 96.0
TABBERT_RUN_OVERHEAD_SECONDS_REFERENCE = 440.0 + 24.0
TABBERT_REFERENCE_DATASET = "sparkov"
# Epocas de cada regime de pre-treino mascarado.
TABBERT_EPOCHS_BY_REGIME = {
    "supervised-only": 10,
    "mlm-before-finetune": 13,
    "warmup-mlm-finetune": 13,
}

# Preparacao: segundos por configuracao, dominados pela leitura dos CSV
# brutos e pela PCA. Estimados a partir das preparacoes feitas aqui.
PREPARATION_SECONDS = {
    "ulb": 25.0,
    "home-credit-completa": 600.0,
    "home-credit-somente-aplicacao": 150.0,
    "sparkov": 420.0,
    "ieee-cis-completa": 120.0,
    "ieee-cis-sem-identidade": 110.0,
    "ieee-cis-transacional": 300.0,
}
# A preparacao do TabBERT custa mais que a das demais: alem de tokenizar todo
# campo, ela monta e grava o indice de janelas causais por titular.
SEQUENCE_PREPARATION_FACTOR = 2.0

# Bases de ablacao (item G14) tem as mesmas linhas da base de origem, entao o
# custo estimado e o dela. Os atributos removidos reduzem a largura, mas a
# estimativa fica do lado conservador.
for _definition in DATASET_DEFINITIONS.values():
    if _definition.base_dataset_id is None:
        continue
    for _table in (
        DATASET_ROW_COUNT,
        *HISTORICAL_MEDIAN_SECONDS.values(),
        TRANSFORMER_SECONDS_PER_FOLD,
        PREPARATION_SECONDS,
    ):
        if _definition.base_dataset_id in _table:
            _table.setdefault(_definition.dataset_id, _table[_definition.base_dataset_id])

# Deteccao de anomalia (item G3): sem medicao no pipeline anterior. Premissa
# explicita, usada so para dividir a grade entre maquinas (item G18): o
# Isolation Forest ajusta 100 arvores sobre subamostras de 256 e pontua em
# O(n), como a regressao logistica na ULB; o DevNet tem custo de treino fixo
# (50 epocas x 20 lotes de 512) mais a pontuacao linear. Custo por dobra e por
# braco; o DevNet roda 3 fracoes x 5 sorteios.
ANOMALY_ULB_SECONDS = {
    "isolation-forest": 4.0,
    "devnet": 12.0,
}

LINEAR_ALGORITHMS = ("logistic-regression", "naive-bayes")
# Expoente do crescimento do KNN com o numero de linhas.
KNN_SCALING_EXPONENT = 1.5
# Banda de incerteza aplicada ao total.
UNCERTAINTY_LOW = 0.6
UNCERTAINTY_HIGH = 1.8


def _historical_boosting_seconds(algorithm: str, database_type: str) -> float:
    """Custo pelas medianas historicas: so a proporcao entre bases e usada."""
    medians = HISTORICAL_MEDIAN_SECONDS[algorithm]
    if database_type in medians:
        return medians[database_type] * NO_RESAMPLING_COST_FACTOR
    # Variantes da IEEE-CIS sem historico: escala a partir da ieee-cis-completa
    # pelo numero de linhas.
    ratio = DATASET_ROW_COUNT[database_type] / DATASET_ROW_COUNT["ieee-cis-completa"]
    return medians["ieee-cis-completa"] * ratio * NO_RESAMPLING_COST_FACTOR


def seconds_per_fold(algorithm: str, database_type: str) -> float:
    """Custo estimado de uma dobra de um algoritmo em uma base."""
    row_ratio = DATASET_ROW_COUNT[database_type] / DATASET_ROW_COUNT[REFERENCE_DATASET]

    base_algorithm = algorithm.split("[", 1)[0].strip()
    if base_algorithm in LINEAR_ALGORITHMS:
        return MEASURED_ULB_SECONDS[base_algorithm] * row_ratio

    if base_algorithm == "knn":
        return MEASURED_ULB_SECONDS["knn"] * (row_ratio**KNN_SCALING_EXPONENT)

    if base_algorithm in ANOMALY_ULB_SECONDS:
        return ANOMALY_ULB_SECONDS[base_algorithm] * row_ratio

    base_algorithm = algorithm.split("[", 1)[0].strip()
    if base_algorithm in HISTORICAL_MEDIAN_SECONDS:
        anchor = MEASURED_BOOSTING_ANCHOR_SECONDS[base_algorithm] / _historical_boosting_seconds(
            base_algorithm, BOOSTING_ANCHOR_DATASET
        )
        return _historical_boosting_seconds(base_algorithm, database_type) * anchor

    if algorithm.startswith("ft-transformer"):
        return TRANSFORMER_SECONDS_PER_FOLD[database_type]

    if algorithm.startswith("tabbert"):
        regime = "mlm-before-finetune"
        if "[" in algorithm:
            regime = algorithm.split("[", 1)[1].rstrip("]").strip()
        epochs = TABBERT_EPOCHS_BY_REGIME.get(regime, 13)
        tabbert_ratio = (
            DATASET_ROW_COUNT[database_type]
            / DATASET_ROW_COUNT[TABBERT_REFERENCE_DATASET]
        )
        return (
            TABBERT_SECONDS_PER_EPOCH_REFERENCE * epochs + TABBERT_RUN_OVERHEAD_SECONDS_REFERENCE
        ) * tabbert_ratio

    raise ValueError(f"Algoritmo sem modelo de custo: {algorithm}")


def folds_for_entries(entries) -> dict[str, int]:
    """Dobras de treinamento por base em um conjunto de configuracoes."""
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        counts[entry.database_type] += (
            5 if entry.test_strategy == "stratified-k-fold" else 1
        )
    return dict(counts)


def folds_per_dataset(block: str) -> dict[str, int]:
    """Dobras de treinamento por base em um bloco da grade."""
    return folds_for_entries(experiment_grid.GRID_BUILDERS[block]())


def configurations_per_dataset(block: str) -> dict[str, int]:
    entries = experiment_grid.GRID_BUILDERS[block]()
    counts: dict[str, int] = defaultdict(int)
    for entry in entries:
        counts[entry.database_type] += 1
    return dict(counts)


def _format_duration(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 90:
        return f"{minutes:.0f}min"
    hours = minutes / 60
    if hours < 48:
        return f"{hours:.1f}h"
    return f"{hours / 24:.1f}d"


def build_estimate() -> tuple[list[dict], dict[str, float]]:
    """Devolve as linhas detalhadas e o total por bloco, em segundos."""
    rows: list[dict] = []
    block_totals: dict[str, float] = {}

    for block in experiment_grid.SUPPORTED_BLOCKS:
        entries = experiment_grid.GRID_BUILDERS[block]()
        block_total = 0.0
        # Cada algoritmo cobre so as configuracoes em que roda: os bracos
        # ponderados, so as sem reamostragem.
        for algorithm, run_entries in experiment_grid.run_entries_for(block, entries):
            for database_type, fold_count in sorted(folds_for_entries(run_entries).items()):
                total = seconds_per_fold(algorithm, database_type) * fold_count
                rows.append(
                    {
                        "bloco": block,
                        "algoritmo": algorithm,
                        "base": database_type,
                        "dobras": fold_count,
                        "s_por_dobra": seconds_per_fold(algorithm, database_type),
                        "total_s": total,
                    }
                )
                block_total += total
        block_totals[block] = block_total

    # Preparacao: uma vez por configuracao, independente do algoritmo.
    preparation_total = 0.0
    for block in experiment_grid.SUPPORTED_BLOCKS:
        factor = (
            SEQUENCE_PREPARATION_FACTOR
            if block == experiment_grid.BLOCK_TABBERT
            else 1.0
        )
        for database_type, count in configurations_per_dataset(block).items():
            total = PREPARATION_SECONDS[database_type] * count * factor
            rows.append(
                {
                    "bloco": f"preparacao-{block}",
                    "algoritmo": "preparacao",
                    "base": database_type,
                    "dobras": count,
                    "s_por_dobra": PREPARATION_SECONDS[database_type] * factor,
                    "total_s": total,
                }
            )
            preparation_total += total
    block_totals["preparacao"] = preparation_total

    return rows, block_totals


def plan_makespan_lines(plan: dict) -> list[str]:
    """Tempo estimado de cada processo de um plano e o makespan (o mais longo).

    Os tempos ja vem escalados pela calibracao de cada maquina e incluem a
    leitura pela rede; o makespan e a duracao prevista da reexecucao inteira.
    """
    lines = ["Plano por processo (grid_plan.json):"]
    makespan = 0.0
    for host, machine in sorted(plan["machines"].items()):
        lanes = machine.get("lanes") or {f"{host}": {"estimated_seconds": machine["estimated_seconds"], "pairs": machine["pairs"]}}
        for lane, entry in sorted(lanes.items()):
            seconds = float(entry["estimated_seconds"])
            makespan = max(makespan, seconds)
            lines.append(
                f"  {lane:22s} pares={len(entry['pairs']):3d}  "
                f"{_format_duration(seconds):>8s}  (faixa {_format_duration(seconds * UNCERTAINTY_LOW)} "
                f"a {_format_duration(seconds * UNCERTAINTY_HIGH)})"
            )
    lines.append(
        f"Makespan do plano: {_format_duration(makespan)} "
        f"(faixa {_format_duration(makespan * UNCERTAINTY_LOW)} a {_format_duration(makespan * UNCERTAINTY_HIGH)})"
    )
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", default="text", choices=("text", "markdown"))
    parser.add_argument(
        "--plan",
        default=None,
        help="Diretorio de um plano (grid_plan.py allocate): mostra o tempo de cada processo e o makespan.",
    )
    arguments = parser.parse_args()
    if arguments.plan:
        plan = json.loads((Path(arguments.plan) / "grid_plan.json").read_text(encoding="utf-8"))
        print("\n".join(plan_makespan_lines(plan)))
        return

    rows, block_totals = build_estimate()

    by_algorithm: dict[str, float] = defaultdict(float)
    by_algorithm_folds: dict[str, int] = defaultdict(int)
    for row in rows:
        by_algorithm[row["algoritmo"]] += row["total_s"]
        by_algorithm_folds[row["algoritmo"]] += row["dobras"]

    separator = "|" if arguments.format == "markdown" else " "
    print("Estimativa de tempo de execucao da grade revisada")
    print("=" * 64)
    print()
    print("Por algoritmo (soma de todas as bases e protocolos):")
    if arguments.format == "markdown":
        print("| Algoritmo | Dobras | Tempo estimado | Faixa |")
        print("|---|---:|---:|---|")
    for algorithm, total in sorted(by_algorithm.items(), key=lambda item: -item[1]):
        folds = by_algorithm_folds[algorithm]
        low = _format_duration(total * UNCERTAINTY_LOW)
        high = _format_duration(total * UNCERTAINTY_HIGH)
        if arguments.format == "markdown":
            print(
                f"| {algorithm} | {folds} | {_format_duration(total)} | {low} a {high} |"
            )
        else:
            print(
                f"  {algorithm:22s} dobras={folds:5d}  "
                f"{_format_duration(total):>8s}  (faixa {low} a {high})"
            )

    print()
    print("Por bloco:")
    for block, total in block_totals.items():
        print(f"  {block:24s} {_format_duration(total):>8s}")

    grand_total = sum(block_totals.values())
    print()
    print(
        f"TOTAL sequencial: {_format_duration(grand_total)} "
        f"(faixa {_format_duration(grand_total * UNCERTAINTY_LOW)} a "
        f"{_format_duration(grand_total * UNCERTAINTY_HIGH)})"
    )
    print()
    print("Detalhe por base (as cinco combinacoes mais caras):")
    for row in sorted(rows, key=lambda item: -item["total_s"])[:5]:
        print(
            f"  {row['algoritmo']:22s} {row['base']:24s} "
            f"{row['dobras']:4d} dobras x {row['s_por_dobra']:8.1f}s = "
            f"{_format_duration(row['total_s'])}"
        )


if __name__ == "__main__":
    main()
