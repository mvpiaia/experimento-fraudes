"""Gera a grade experimental de cada bloco, ja sem configuracoes duplicadas.

A grade antiga era montada em lacos aninhados de bash, o que tornava invisivel
um problema serio: varios eixos nao tinham efeito em varias bases, e as
configuracoes resultantes eram copias identicas umas das outras. A ULB nao tem
atributo categorico nem valor faltante, entao 2 codificacoes x 4 imputacoes
geravam 8 copias identicas de cada experimento, inflando a contagem de "pares"
e estreitando artificialmente os intervalos bootstrap calculados entre
configuracoes (item G9b da revisao).

Aqui a grade e gerada em um unico lugar, consultando o catalogo de bases: um
eixo so varia onde o fenomeno existe. O mesmo modulo serve de fonte para a
contagem de execucoes e para a estimativa de tempo do relatorio.

Uso:
    python3 experiment_grid.py --block classic
    python3 experiment_grid.py --block boosting --format count
    python3 experiment_grid.py --block ft-transformer --format summary
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, asdict

from experiment_datasets import (
    BLOCK_ANOMALY,
    BLOCK_BOOSTING,
    BLOCK_CLASSIC,
    BLOCK_FT_TRANSFORMER,
    BLOCK_TABBERT,
    DATASET_DEFINITIONS,
    STRATIFIED_HOLDOUT,
    TEMPORAL_HOLDOUT,
    DatasetDefinition,
    normalize_dataset_id,
    sequence_compatible_protocols,
    tabbert_protocol,
)
from lr_nb_knn_specific import FACTORIAL_CATEGORICAL_ENCODINGS


SUPPORTED_BLOCKS = (
    BLOCK_CLASSIC,
    BLOCK_BOOSTING,
    BLOCK_ANOMALY,
    BLOCK_FT_TRANSFORMER,
    BLOCK_TABBERT,
)

# Semente unica por padrao. A dispersao entre particoes e estimada pelo
# protocolo stratified-k-fold, que gera cinco dobras da mesma configuracao.
# Para repetir a grade com varias sementes, como pede o item G13 da revisao,
# basta exportar SEED_LIST="42 2026 7".
DEFAULT_SEEDS = (42,)

# Estrategias de balanceamento. A opcao "none" e a linha de base sem
# reamostragem, que nao existia na grade anterior e sem a qual nao se pode
# medir o efeito de balancear (item G10a). "random-undersampling" entra nos
# dois blocos com a MESMA razao 1:1, dando as duas familias uma estrategia
# comum e tornando a comparacao entre elas interpretavel (itens G10b e G10f).
NO_RESAMPLING = ("none", "none")
RANDOM_UNDERSAMPLING = (
    "random-undersampling",
    "sampling_strategy=auto:replacement=false",
)
RUS_SMOTE = (
    "RUS+SMOTE",
    "rus_sampling_strategy=0.5:smote_sampling_strategy=1.0:"
    "smote_k_neighbors=5:rus_replacement=false",
)
TOMEK_LINKS = ("tomek-links", "sampling_strategy=auto:n_jobs=-1")

CLASSIC_BALANCING = (NO_RESAMPLING, RANDOM_UNDERSAMPLING, RUS_SMOTE, TOMEK_LINKS)
BOOSTING_BALANCING = (NO_RESAMPLING, RANDOM_UNDERSAMPLING)
TRANSFORMER_BALANCING = (NO_RESAMPLING, RANDOM_UNDERSAMPLING)

# Desenho fatorial 2x2 da codificacao supervisionada (item C3.6 da revisao).
# Antes a grade tinha so "ordered-target" e "woe", que diferiam ao mesmo tempo na
# estatistica e no mecanismo de controle de vazamento; nenhuma diferenca medida
# podia ser atribuida a um dos dois. As quatro celulas separam os fatores.
CLASSIC_ENCODINGS = FACTORIAL_CATEGORICAL_ENCODINGS
# A remocao de registros incompletos agora vale so para o treinamento; o
# teste mantem a populacao completa (item G16).
CLASSIC_FILL_NA = ("mean", "median", "mode", "drop-incomplete-rows")
# No modo nativo "native" deixa o proprio CatBoost/XGBoost tratar a ausencia,
# sem remover linhas. Remocao, tratamento nativo e ausencia de faltantes
# ("none", so nas bases sem faltantes) tem nomes distintos (item G16).
BOOSTING_FILL_NA = ("native", "mean", "median", "mode")

TRANSFORMER_NORMALIZATIONS = ("quantile", "standard")
TRANSFORMER_CARDINALITY = (("minimum-frequency", 10), ("hashing", 1024))
# A cabeca "linear" (LayerNorm + Linear) saiu do processo; ficam apenas as
# duas variantes de classificador efetivamente comparadas.
TRANSFORMER_HEADS = ("mlp", "lstm")
# Os Transformers rodam apenas os dois protocolos de holdout. O k-fold
# multiplicaria por cinco um custo de treinamento que ja e o dominante da
# tese; os holdouts bastam para parear com os demais algoritmos.
TRANSFORMER_PROTOCOLS = ("stratified-holdout", "temporal-holdout")

# O TabBERT nao usa os protocolos das outras familias: ele precisa de janelas
# causais por titular, entao a particao e um corte cronologico DENTRO do
# historico de cada titular. O nome do protocolo registra isso no manifesto.
SEQUENCE_PARTITION_STRATEGY = "sequence-chronological"
# Discretizacao numerica em faixas: o TabBERT tokeniza todo campo.
TABBERT_NORMALIZATIONS = ("quantile-bins",)
TABBERT_NUMERIC_BINS = 10
# Janela de 15 registros: 10 de historico e 5 alvos, com passo 5.
TABBERT_WINDOW_LENGTH = 15
TABBERT_HISTORY_LENGTH = 10
TABBERT_WINDOW_STRIDE = 5
# As duas cabecas que restaram, como no FT-Transformer.
TABBERT_HEADS = ("mlp", "lstm")
# Os tres regimes de treino (item C3.19), todos com o mesmo total de epocas
# supervisionadas: `supervised-only` (S, sem pre-treino) mede a contribuicao do
# pre-treino; `mlm-before-finetune` (PM->S) o acrescenta antes do ajuste; e
# `warmup-mlm-finetune` (S->PM->S) divide o ajuste em aquecimento (um terco) e
# ajuste final (o restante), com o pre-treino entre eles. O antigo
# `mlm-after-warmup` somava o aquecimento a um ajuste completo e confundia a
# ordem das etapas com o orcamento de treino; e recusado.
# Diferente da cabeca, o regime NAO muda a preparacao: ele e um parametro de
# execucao, e por isso aparece na coluna `regime` do resultado, nao no
# configuration_md5.
TABBERT_REGIMES = ("supervised-only", "mlm-before-finetune", "warmup-mlm-finetune")
# A regressao linear saiu da grade: sua saida nao e uma probabilidade
# calibrada, pode cair fora de [0, 1] e o limiar de 0,5 nao tem interpretacao
# sobre ela. Mantida, so acrescentava uma linha de base sem significado
# decisorio.
CLASSIC_ALGORITHMS = (
    "logistic-regression",
    "naive-bayes",
    "knn",
)
BOOSTING_ALGORITHMS = ("catboost", "xgboost")
# Bloco de anomalia: o Isolation Forest roda uma vez por configuracao; o
# DevNet, uma vez por fracao de fraudes rotuladas e por semente de sorteio dos
# rotulos (item G3). As fracoes e as sementes de sorteio vivem aqui para que o
# executor e a contagem de execucoes previstas (item G17) usem os mesmos
# valores; LABEL_FRACTIONS e LABEL_DRAW_SEEDS os substituem pelo ambiente.
ANOMALY_UNSUPERVISED_ALGORITHMS = ("isolation-forest",)
ANOMALY_LABEL_BUDGET_ALGORITHMS = ("devnet",)
DEFAULT_LABEL_FRACTIONS = (0.01, 0.05, 0.10)
DEFAULT_LABEL_DRAW_SEEDS = (0, 1, 2, 3, 4)
TABBERT_ALGORITHMS = tuple(f"tabbert-{head}" for head in TABBERT_HEADS)

# Ponderacao de classe na perda, nos algoritmos que a aceitam: regressao
# logistica, CatBoost e XGBoost (itens G10 e C3.14 da revisao). As duas
# intensidades ponderadas sao bracos do eixo de balanceamento, ao lado de "sem
# reamostragem", RUS, RUS+SMOTE e Tomek Links, e por isso so rodam sobre as
# configuracoes preparadas SEM reamostragem:
#
#   none           sem ponderacao; e o unico nivel dos bracos reamostrados;
#   sqrt-balanced  braco de ponderacao moderada, com sqrt(R);
#   balanced       braco de ponderacao completa, com R.
#
# R = negativos/positivos vem do treino de cada dobra, que nesses bracos tem a
# prevalencia natural. Num treino reamostrado para 1:1, R ~ 1 e os pesos so
# repetiriam o braco none. Naive Bayes e KNN ficam de fora: o primeiro estima
# densidades condicionais e o segundo decide por votacao entre vizinhos, e
# nenhum dos dois tem nocao de peso por classe.
CLASS_WEIGHT_INTENSITIES = ("none", "sqrt-balanced", "balanced")
CLASS_WEIGHTED_CLASSIC_ALGORITHMS = frozenset({"logistic-regression"})
DEFAULT_CLASS_WEIGHT_INTENSITY = "none"
# Limiar de reserva. O limiar de cada dobra e escolhido na validacao,
# maximizando F2 (item G11-i; ver decision_threshold.py); 0,5 so vale quando a
# validacao nao tem as duas classes.
DEFAULT_THRESHOLD = 0.5


@dataclass(frozen=True)
class GridEntry:
    """Uma configuracao de preparacao, sem o algoritmo que a consumira."""

    dataset_id: str
    dataset_name: str
    source_files: str
    prediction_column: str
    auxiliary_data_is_test_data: str
    categorical_handling_mode: str
    categorical_encoding_strategy: str
    fill_na_strategy: str
    undersampling_technique: str
    undersampling_parameter_values: str
    test_strategy: str
    random_seed: int
    numeric_normalization: str = "none"
    high_cardinality_strategy: str = "none"
    high_cardinality_parameter: str = "0"

    @property
    def database_type(self) -> str:
        """Compatibilidade temporaria com consumidores da grade anterior."""
        return self.dataset_id

    def to_tsv(self) -> str:
        return "\t".join(str(value) for value in asdict(self).values())


def _configured_seeds() -> tuple[int, ...]:
    raw = os.environ.get("SEED_LIST", "").strip()
    if not raw:
        return DEFAULT_SEEDS
    return tuple(int(value) for value in raw.split())


def configured_label_fractions() -> tuple[float, ...]:
    raw = os.environ.get("LABEL_FRACTIONS", "").strip()
    if not raw:
        return DEFAULT_LABEL_FRACTIONS
    return tuple(float(value) for value in raw.split())


def configured_label_draw_seeds() -> tuple[int, ...]:
    raw = os.environ.get("LABEL_DRAW_SEEDS", "").strip()
    if not raw:
        return DEFAULT_LABEL_DRAW_SEEDS
    return tuple(int(value) for value in raw.split())


def anomaly_arms() -> list[tuple[str, float | None, int | None]]:
    """Bracos do bloco de anomalia: (algoritmo, fracao rotulada, semente de sorteio)."""
    arms: list[tuple[str, float | None, int | None]] = [
        (algorithm, None, None) for algorithm in ANOMALY_UNSUPERVISED_ALGORITHMS
    ]
    for algorithm in ANOMALY_LABEL_BUDGET_ALGORITHMS:
        for fraction in configured_label_fractions():
            for draw_seed in configured_label_draw_seeds():
                arms.append((algorithm, fraction, draw_seed))
    return arms


def _encodings_for(definition: DatasetDefinition) -> tuple[str, ...]:
    return CLASSIC_ENCODINGS if definition.has_categorical_attributes else ("none",)


def _classic_fill_na_for(definition: DatasetDefinition) -> tuple[str, ...]:
    return CLASSIC_FILL_NA if definition.has_missing_values else ("none",)


def _boosting_fill_na_for(definition: DatasetDefinition) -> tuple[str, ...]:
    return BOOSTING_FILL_NA if definition.has_missing_values else ("none",)


def _dataset_filter() -> str:
    return os.environ.get("DATASET_FILTER", "").strip()


# Filtros opcionais por eixo, no mesmo molde do DATASET_FILTER. Servem para
# montar cortes reduzidos da grade, como o estudo de variancia por semente
# (item G13): sem as variaveis, a grade e a de sempre.
PROTOCOL_FILTER_VARIABLE = "PROTOCOL_FILTER"
UNDERSAMPLING_FILTER_VARIABLE = "UNDERSAMPLING_FILTER"
FILL_NA_FILTER_VARIABLE = "FILL_NA_FILTER"


def _environment_filter(variable: str) -> str:
    return os.environ.get(variable, "").strip()


def apply_axis_filters(entries: list[GridEntry]) -> list[GridEntry]:
    """Restringe a grade aos valores pedidos de protocolo, reamostragem e imputacao.

    Protocolo e reamostragem sao filtros estritos. A imputacao so filtra onde
    ela e de fato um eixo: uma base sem valores ausentes tem apenas a entrada
    `none`, que e mantida, e os Transformers, cuja imputacao e do modelo
    (`model-managed`), tambem ficam como estao.
    """
    protocol = _environment_filter(PROTOCOL_FILTER_VARIABLE)
    undersampling = _environment_filter(UNDERSAMPLING_FILTER_VARIABLE)
    fill_na = _environment_filter(FILL_NA_FILTER_VARIABLE)
    selected = [
        entry
        for entry in entries
        if (not protocol or entry.test_strategy == protocol)
        and (not undersampling or entry.undersampling_technique == undersampling)
    ]
    if not fill_na:
        return selected
    fill_values_by_dataset: dict[str, set[str]] = {}
    for entry in selected:
        fill_values_by_dataset.setdefault(entry.database_type, set()).add(
            entry.fill_na_strategy
        )
    return [
        entry
        for entry in selected
        if entry.fill_na_strategy == fill_na
        or len(fill_values_by_dataset[entry.database_type]) == 1
    ]


def _selected_definitions(block: str) -> list[DatasetDefinition]:
    """Bases que entram no bloco, depois do filtro opcional por base.

    Cada base declara os blocos em que roda (``DatasetDefinition.blocks``): a
    ablacao da IEEE-CIS, por exemplo, so existe no boosting.
    """
    wanted = _dataset_filter()
    if wanted:
        wanted = normalize_dataset_id(wanted)
    return [
        definition
        for definition in DATASET_DEFINITIONS.values()
        if block in definition.blocks
        and (not wanted or definition.dataset_id == wanted)
    ]


def build_classic_grid() -> list[GridEntry]:
    entries: list[GridEntry] = []
    for definition in _selected_definitions(BLOCK_CLASSIC):
        for encoding in _encodings_for(definition):
            for fill_na in _classic_fill_na_for(definition):
                for technique, parameters in CLASSIC_BALANCING:
                    for protocol in definition.protocols:
                        for seed in _configured_seeds():
                            entries.append(
                                GridEntry(
                                    dataset_id=definition.dataset_id,
                                    dataset_name=definition.dataset_name,
                                    source_files=definition.source_files,
                                    prediction_column=definition.prediction_column,
                                    auxiliary_data_is_test_data=str(
                                        definition.auxiliary_data_is_test_data
                                    ).lower(),
                                    categorical_handling_mode="encoded",
                                    categorical_encoding_strategy=encoding,
                                    fill_na_strategy=fill_na,
                                    undersampling_technique=technique,
                                    undersampling_parameter_values=parameters,
                                    test_strategy=protocol,
                                    random_seed=seed,
                                )
                            )
    return apply_axis_filters(entries)


def build_boosting_grid() -> list[GridEntry]:
    entries: list[GridEntry] = []
    for definition in _selected_definitions(BLOCK_BOOSTING):
        for fill_na in _boosting_fill_na_for(definition):
            for technique, parameters in BOOSTING_BALANCING:
                for protocol in definition.protocols:
                    for seed in _configured_seeds():
                        entries.append(
                            GridEntry(
                                dataset_id=definition.dataset_id,
                                dataset_name=definition.dataset_name,
                                source_files=definition.source_files,
                                prediction_column=definition.prediction_column,
                                auxiliary_data_is_test_data=str(
                                    definition.auxiliary_data_is_test_data
                                ).lower(),
                                categorical_handling_mode="native",
                                categorical_encoding_strategy="none",
                                fill_na_strategy=fill_na,
                                undersampling_technique=technique,
                                undersampling_parameter_values=parameters,
                                test_strategy=protocol,
                                random_seed=seed,
                            )
                        )
    return apply_axis_filters(entries)


def build_anomaly_grid() -> list[GridEntry]:
    """Configuracoes nativas sem reamostragem das bases do bloco de anomalia.

    O executor de anomalia consome o manifesto nativo do boosting e pula as
    configuracoes reamostradas: Isolation Forest e DevNet aprendem com a
    prevalencia natural.
    """
    anomaly_datasets = {
        definition.dataset_id for definition in _selected_definitions(BLOCK_ANOMALY)
    }
    return [
        entry
        for entry in build_boosting_grid()
        if entry.undersampling_technique == "none" and entry.dataset_id in anomaly_datasets
    ]


def build_ft_transformer_grid() -> list[GridEntry]:
    entries: list[GridEntry] = []
    for definition in _selected_definitions(BLOCK_FT_TRANSFORMER):
        protocols = tuple(
            protocol
            for protocol in TRANSFORMER_PROTOCOLS
            if protocol in definition.protocols
        )
        for normalization in TRANSFORMER_NORMALIZATIONS:
            for strategy, parameter in TRANSFORMER_CARDINALITY:
                for technique, parameters in TRANSFORMER_BALANCING:
                    for protocol in protocols:
                        for seed in _configured_seeds():
                            entries.append(
                                GridEntry(
                                    dataset_id=definition.dataset_id,
                                    dataset_name=definition.dataset_name,
                                    source_files=definition.source_files,
                                    prediction_column=definition.prediction_column,
                                    auxiliary_data_is_test_data=str(
                                        definition.auxiliary_data_is_test_data
                                    ).lower(),
                                    categorical_handling_mode="transformer-indexed",
                                    categorical_encoding_strategy="none",
                                    fill_na_strategy="model-managed",
                                    undersampling_technique=technique,
                                    undersampling_parameter_values=parameters,
                                    test_strategy=protocol,
                                    random_seed=seed,
                                    numeric_normalization=normalization,
                                    high_cardinality_strategy=strategy,
                                    high_cardinality_parameter=str(parameter),
                                )
                            )
    return apply_axis_filters(entries)


def build_tabbert_grid() -> list[GridEntry]:
    """Grade do TabBERT, restrita as bases que tem titular e ordem.

    A ULB fica de fora: suas linhas sao transacoes anonimas, sem portador,
    entao nao ha sequencia a modelar. ``home-credit-completa-adjusted``, presente na
    grade anterior, tambem saiu: acrescentava uma linha sintetica por cliente,
    o que e manipulacao da base.

    Nao ha eixo de reamostragem: RUS e SMOTE quebrariam as janelas
    cronologicas, retirando ou inventando posicoes no meio do historico de um
    titular.

    O protocolo e o MESMO dos outros blocos, para que o TabBERT seja pareado
    com eles (item G16): ``temporal-holdout`` onde a base tem ordem temporal
    (Sparkov e IEEE-CIS), e ``stratified-holdout`` agrupado pelo titular onde
    nao tem (Home Credit, agrupada por cliente). Nos dois, as janelas usam como
    historico os registros anteriores do titular nos splits anteriores. O
    ``sequence-chronological`` antigo, corte proprio dentro de cada titular,
    saiu da grade por nao coincidir com particao alguma dos demais blocos.
    """
    entries: list[GridEntry] = []
    for definition in _selected_definitions(BLOCK_TABBERT):
        protocol = tabbert_protocol(definition)
        if protocol is None:
            continue
        for strategy, parameter in TRANSFORMER_CARDINALITY:
            for normalization in TABBERT_NORMALIZATIONS:
                for seed in _configured_seeds():
                    entries.append(
                        GridEntry(
                            dataset_id=definition.dataset_id,
                            dataset_name=definition.dataset_name,
                            source_files=definition.source_files,
                            prediction_column=definition.prediction_column,
                            auxiliary_data_is_test_data=str(
                                definition.auxiliary_data_is_test_data
                            ).lower(),
                            categorical_handling_mode="sequence-tokenized",
                            categorical_encoding_strategy="none",
                            fill_na_strategy="model-managed",
                            undersampling_technique="none",
                            undersampling_parameter_values="none",
                            test_strategy=protocol,
                            random_seed=seed,
                            numeric_normalization=normalization,
                            high_cardinality_strategy=strategy,
                            high_cardinality_parameter=str(parameter),
                        )
                    )
    return apply_axis_filters(entries)


GRID_BUILDERS = {
    BLOCK_CLASSIC: build_classic_grid,
    BLOCK_BOOSTING: build_boosting_grid,
    BLOCK_ANOMALY: build_anomaly_grid,
    BLOCK_FT_TRANSFORMER: build_ft_transformer_grid,
    BLOCK_TABBERT: build_tabbert_grid,
}


def _weighting_entries(intensity: str, entries: list[GridEntry]) -> list[GridEntry]:
    """Configuracoes em que uma intensidade roda: as ponderadas, so sem reamostragem."""
    if intensity == DEFAULT_CLASS_WEIGHT_INTENSITY:
        return entries
    return [entry for entry in entries if entry.undersampling_technique == "none"]


def run_entries_for(block: str, entries: list[GridEntry]) -> list[tuple[str, list[GridEntry]]]:
    """Devolve (algoritmo, configuracoes em que ele roda) de um bloco.

    Uma "execucao" e um par configuracao/algoritmo. Os bracos ponderados
    (sqrt-balanced, balanced) so cobrem as configuracoes sem reamostragem.
    """
    if block == BLOCK_CLASSIC:
        runs = []
        for algorithm in CLASSIC_ALGORITHMS:
            if algorithm not in CLASS_WEIGHTED_CLASSIC_ALGORITHMS:
                runs.append((algorithm, entries))
                continue
            for intensity in CLASS_WEIGHT_INTENSITIES:
                runs.append(
                    (f"{algorithm} [{intensity}]", _weighting_entries(intensity, entries))
                )
        return runs
    if block == BLOCK_BOOSTING:
        return [
            (f"{algorithm} [{intensity}]", _weighting_entries(intensity, entries))
            for algorithm in BOOSTING_ALGORITHMS
            for intensity in CLASS_WEIGHT_INTENSITIES
        ]
    if block == BLOCK_ANOMALY:
        runs = []
        for algorithm, fraction, draw_seed in anomaly_arms():
            label = algorithm if fraction is None else f"{algorithm} [{fraction:g}, sorteio {draw_seed}]"
            runs.append((label, entries))
        return runs
    if block == BLOCK_TABBERT:
        # Cada cabeca roda os tres regimes, entao sao seis execucoes por
        # configuracao preparada.
        return [
            (f"tabbert-{head} [{regime}]", entries)
            for head in TABBERT_HEADS
            for regime in TABBERT_REGIMES
        ]
    return [(f"ft-transformer-{head}", entries) for head in TRANSFORMER_HEADS]


def algorithm_runs_for(block: str, entries: list[GridEntry]) -> list[tuple[str, float, int]]:
    """Devolve (algoritmo, limiar de reserva, numero de execucoes) de um bloco.

    Cada execucao grava uma linha por dobra, entao o numero de linhas de
    resultado e maior quando o protocolo e o k-fold.
    """
    return [
        (algorithm, DEFAULT_THRESHOLD, len(run_entries))
        for algorithm, run_entries in run_entries_for(block, entries)
    ]


def result_rows_for(entries: list[GridEntry]) -> int:
    """Linhas de resultado por algoritmo: uma por dobra de cada configuracao."""
    return sum(5 if entry.test_strategy == "stratified-k-fold" else 1 for entry in entries)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", required=True, choices=SUPPORTED_BLOCKS)
    parser.add_argument(
        "--format",
        default="tsv",
        choices=(
            "tsv",
            "count",
            "summary",
            "label-fractions",
            "label-draw-seeds",
            "tabbert-regimes",
        ),
    )
    arguments = parser.parse_args()
    # Valores que o executor do TabBERT le daqui, para nao divergir da grade.
    if arguments.format == "tabbert-regimes":
        print(" ".join(TABBERT_REGIMES))
        return
    # Valores que o executor de anomalia le daqui, para nao divergir da contagem.
    if arguments.format == "label-fractions":
        print(" ".join(f"{fraction:g}" for fraction in configured_label_fractions()))
        return
    if arguments.format == "label-draw-seeds":
        print(" ".join(str(seed) for seed in configured_label_draw_seeds()))
        return
    entries = GRID_BUILDERS[arguments.block]()

    if arguments.format == "count":
        print(len(entries))
        return
    if arguments.format == "summary":
        print(f"bloco: {arguments.block}")
        print(f"configuracoes preparadas: {len(entries)}")
        per_dataset: dict[str, int] = {}
        for entry in entries:
            per_dataset[entry.database_type] = per_dataset.get(entry.database_type, 0) + 1
        for database_type, count in per_dataset.items():
            print(f"  {database_type}: {count}")
        for algorithm, _threshold, run_count in algorithm_runs_for(arguments.block, entries):
            print(f"  execucoes {algorithm}: {run_count}")
        print(f"linhas de resultado por algoritmo: {result_rows_for(entries)}")
        return

    for entry in entries:
        print(entry.to_tsv())


if __name__ == "__main__":
    main()
