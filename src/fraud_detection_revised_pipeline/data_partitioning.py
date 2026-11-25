"""Protocolos de avaliacao: particionamento de treino, validacao e teste.

A grade anterior tratava a politica de particionamento e o tamanho da reserva
como se fossem tecnicas de preparacao dos dados. Nao sao: definem o tamanho do
treino e a composicao do teste, algo que um modelo em operacao nao escolhe
(item G9g da revisao). Aqui eles voltam a ser o que sao, o protocolo de
avaliacao, e ficam reduzidos a tres opcoes explicitas:

``stratified-holdout``
    Particao unica 70/15/15, estratificada pela classe.
``stratified-k-fold``
    Cinco dobras estratificadas; cada dobra produz treino e validacao.
``temporal-holdout``
    Corte cronologico 70/15/15. Na Sparkov respeita a separacao oficial entre
    ``fraudTrain.csv`` e ``fraudTest.csv``. E o unico protocolo que estima o
    desempenho esperado em operacao (item G8).

Dois eixos da grade antiga sumiram:

* ``test_data_size``. A reserva nao tinha efeito algum em
  ``stratified-k-fold`` nem em ``stratified-train-validation-test``, porque
  reserva e remanescente eram reunidos antes da particao, e na Sparkov nao
  tinha efeito em nenhuma politica. As diferencas medidas entre reservas eram,
  portanto, ruido de particao (itens G9c e C5.28).
* ``stratified-train-test-split``. Descartava de 20% a 40% dos dados sem
  justificativa, confundindo o efeito da politica com o do tamanho do
  treinamento, e na Sparkov descartava o ``fraudTest.csv`` inteiro: 30,0% das
  transacoes e 22,2% das fraudes (item G9d).

Todos os protocolos respeitam a coluna de agrupamento da base quando ela
existe. O agrupamento usa ``GroupShuffleSplit`` e ``StratifiedGroupKFold`` do
scikit-learn, que nao exigem rotulo constante por grupo: isso cobre tanto a
Home Credit, em que o ``TARGET`` pertence ao cliente e se repete em todas as
suas linhas, quanto a Sparkov e a IEEE-CIS, em que um mesmo titular tem
transacoes legitimas e fraudulentas.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)

from experiment_datasets import (
    STRATIFIED_HOLDOUT,
    STRATIFIED_K_FOLD,
    TEMPORAL_HOLDOUT,
    DatasetDefinition,
    columns_to_drop_after_partitioning,
)
from transformer_sequence_windows import history_strata_by_split


TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15
DEFAULT_FOLD_COUNT = 5
# Marca as linhas vindas do arquivo auxiliar oficial (Sparkov). Nunca chega
# aos atributos: e removida junto com os identificadores.
AUXILIARY_SOURCE_COLUMN = "__is_auxiliary_source"
# Splits que recebem o estrato de historico: os avaliados (item C3.19).
HISTORY_STRATA_SPLITS = ("validation_data", "test_data")
# Impressao digital de cada linha bruta, carregada ate o fim do corte e
# removida antes de qualquer arquivo ser gravado.
ROW_FINGERPRINT_COLUMN = "__row_fingerprint"
SPLIT_NAMES = ("train_data", "validation_data", "test_data")


@dataclass(frozen=True)
class PartitionPlan:
    """Base concatenada e posicoes decididas para cada dobra."""

    complete_data: pd.DataFrame
    fold_positions: tuple[dict[str, np.ndarray | None], ...]
    definition: DatasetDefinition
    test_strategy: str
    retained_columns: tuple[str, ...] = ()
    history_window: tuple[int, int] | None = None


def row_fingerprints(dataframe: pd.DataFrame) -> np.ndarray:
    """Hash do conteudo bruto de cada linha, sem os marcadores internos.

    As colunas entram em ordem alfabetica, para que o hash nao dependa da
    ordem em que a carga as devolve.
    """
    columns = sorted(
        column
        for column in dataframe.columns
        if column not in {AUXILIARY_SOURCE_COLUMN, ROW_FINGERPRINT_COLUMN}
    )
    return pd.util.hash_pandas_object(dataframe[columns], index=False).to_numpy()


def partition_md5(fingerprints) -> str:
    """Identidade de um split: MD5 do CONJUNTO de linhas brutas que ele contem.

    Depende so de quais linhas caem no split, nao da ordem nem de qualquer
    transformacao posterior. Dois blocos com o mesmo partition_md5 avaliaram
    exatamente as mesmas linhas, mesmo que seus arquivos preparados difiram
    (item G16 da revisao).
    """
    ordered = np.sort(np.asarray(fingerprints, dtype=np.uint64))
    return hashlib.md5(ordered.tobytes(), usedforsecurity=False).hexdigest()


def unit_partition_md5(units) -> str:
    """Identidade de um split pelas UNIDADES de avaliacao que ele contem.

    Duas bases com linhas diferentes, mas os mesmos clientes em cada split (o
    par da Home Credit, item G16), tem o mesmo unit_partition_md5 mesmo com
    partition_md5 diferentes.
    """
    identifiers = sorted({str(unit) for unit in units})
    return hashlib.md5(
        "\n".join(identifiers).encode("utf-8"), usedforsecurity=False
    ).hexdigest()


def fold_partition_id(partition_md5_by_split: dict) -> str:
    """Identidade da dobra inteira: combina o partition_md5 de cada split.

    E a chave de pareamento entre blocos. Uma dobra sem algum split (o k-fold
    nao tem teste) entra com o campo vazio.
    """
    identity = "|".join(
        f"{split_name}={partition_md5_by_split.get(split_name, '')}"
        for split_name in SPLIT_NAMES
    )
    return hashlib.md5(identity.encode("utf-8"), usedforsecurity=False).hexdigest()


def _drop_auxiliary_columns(
    dataframe: pd.DataFrame | None,
    definition: DatasetDefinition,
    retained_columns: tuple[str, ...] = (),
) -> pd.DataFrame | None:
    """Remove identificadores, agrupamento, ordenacao e marcadores internos."""
    if dataframe is None:
        return None
    removable = set(columns_to_drop_after_partitioning(definition))
    removable.difference_update(retained_columns)
    removable.update({AUXILIARY_SOURCE_COLUMN, ROW_FINGERPRINT_COLUMN})
    present = [column for column in dataframe.columns if column in removable]
    return dataframe.drop(columns=present).reset_index(drop=True)


def _build_fold_positions(
    train_positions,
    validation_positions,
    test_positions,
) -> dict:
    """Posicoes de uma dobra, ainda sem materializar dataframes."""
    return {
        "train_data": np.asarray(train_positions, dtype=int),
        "validation_data": (
            None
            if validation_positions is None
            else np.asarray(validation_positions, dtype=int)
        ),
        "test_data": (
            None if test_positions is None else np.asarray(test_positions, dtype=int)
        ),
    }


def _finalize_fold(
    fold: dict,
    definition: DatasetDefinition,
    retained_columns: tuple[str, ...],
    fold_index: int = 0,
    history_window: tuple[int, int] | None = None,
) -> dict:
    """Calcula o partition_md5 de cada split e remove as colunas auxiliares.

    Nas bases com unidade de avaliacao, guarda antes da remocao o
    identificador da unidade de cada linha, na mesma ordem. Ele sai dos
    atributos, mas e o que permite agregar os escores por cliente depois.

    Com ``history_window`` (tamanho do historico, passo), guarda tambem o
    estrato de historico de cada linha de validacao e teste, calculado como no
    TabBERT, antes de titular e ordem sairem (item C3.19).
    """
    finalized = {"partition_md5": {}, "row_fingerprints": {}}
    if history_window is not None:
        history_length, window_stride = history_window
        strata = history_strata_by_split(
            fold, definition.dataset_id, fold_index, history_length, window_stride
        )
        finalized["history_strata"] = {
            split_name: values
            for split_name, values in strata.items()
            if split_name in HISTORY_STRATA_SPLITS
        }
    unit_column = definition.evaluation_unit_column
    if unit_column:
        finalized["evaluation_units"] = {}
        finalized["unit_partition_md5"] = {}
    for split_name in SPLIT_NAMES:
        split_data = fold.get(split_name)
        if split_data is not None and ROW_FINGERPRINT_COLUMN in split_data.columns:
            finalized["partition_md5"][split_name] = partition_md5(
                split_data[ROW_FINGERPRINT_COLUMN]
            )
            # Na ordem das linhas: a preparacao seleciona as que sobrevivem a
            # remocao de incompletos e calcula o train_population_md5 (G16).
            finalized["row_fingerprints"][split_name] = split_data[
                ROW_FINGERPRINT_COLUMN
            ].to_numpy()
        if unit_column and split_data is not None:
            finalized["evaluation_units"][split_name] = (
                split_data[unit_column].reset_index(drop=True)
            )
            finalized["unit_partition_md5"][split_name] = unit_partition_md5(
                split_data[unit_column]
            )
        finalized[split_name] = _drop_auxiliary_columns(
            split_data, definition, retained_columns
        )
    return finalized


def _can_stratify(labels: pd.Series, holdout_fraction: float) -> bool:
    """Estratificar exige pelo menos duas classes e dois exemplos em cada."""
    class_counts = labels.value_counts()
    if len(class_counts) < 2 or class_counts.min() < 2:
        return False
    holdout_count = int(round(len(labels) * holdout_fraction))
    return holdout_count >= len(class_counts) and (
        len(labels) - holdout_count >= len(class_counts)
    )


def _constant_group_labels(labels: pd.Series, groups: pd.Series) -> pd.Series | None:
    """Rotulo de cada grupo, quando ele e constante dentro do grupo; senao None."""
    labels_by_group = labels.groupby(groups.to_numpy(), sort=True)
    if (labels_by_group.nunique() > 1).any():
        return None
    return labels_by_group.first()


def _split_positions(
    data: pd.DataFrame,
    label_column: str,
    group_column: str | None,
    holdout_fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Devolve as posicoes de (restante, reserva) para um corte unico.

    Com ``group_column``, nenhum grupo aparece dos dois lados. Sem ele, o
    corte e estratificado pela classe sempre que isso for possivel.
    """
    positions = np.arange(len(data))
    labels = data[label_column]

    if group_column is not None:
        groups = data[group_column]
        group_labels = _constant_group_labels(labels, groups)
        if group_labels is not None and _can_stratify(group_labels, holdout_fraction):
            # Rotulo do grupo (cliente da Home Credit): estratifica a tabela
            # de grupos e expande o resultado para todas as linhas de cada um.
            # Assim a prevalencia por cliente se preserva em treino,
            # validacao e teste, e nenhum cliente fica dos dois lados.
            remaining_groups, holdout_groups = train_test_split(
                group_labels.index.to_numpy(),
                test_size=holdout_fraction,
                random_state=random_seed,
                shuffle=True,
                stratify=group_labels.to_numpy(),
            )
            in_holdout = groups.isin(holdout_groups).to_numpy()
            return positions[~in_holdout], positions[in_holdout]
        # Grupo com as duas classes (titular na Sparkov ou na IEEE-CIS): nao
        # ha rotulo de grupo a estratificar.
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=holdout_fraction, random_state=random_seed
        )
        remaining_positions, holdout_positions = next(
            splitter.split(positions, labels, groups=groups)
        )
        return remaining_positions, holdout_positions

    stratify_values = labels if _can_stratify(labels, holdout_fraction) else None
    remaining_positions, holdout_positions = train_test_split(
        positions,
        test_size=holdout_fraction,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_values,
    )
    return remaining_positions, holdout_positions


def _stratified_holdout(
    data: pd.DataFrame,
    definition: DatasetDefinition,
    random_seed: int,
) -> list[dict]:
    """Holdout 70/15/15 em dois cortes sucessivos."""
    label_column = definition.prediction_column
    group_column = definition.group_column

    remaining_positions, test_positions = _split_positions(
        data, label_column, group_column, TEST_FRACTION, random_seed
    )
    remaining_data = data.iloc[remaining_positions]
    # A validacao e 15% do total, isto e, 15/85 do que sobrou do primeiro corte.
    validation_fraction_of_remaining = VALIDATION_FRACTION / (1 - TEST_FRACTION)
    train_positions, validation_positions = _split_positions(
        remaining_data,
        label_column,
        group_column,
        validation_fraction_of_remaining,
        random_seed,
    )

    return [
        _build_fold_positions(
            remaining_positions[train_positions],
            remaining_positions[validation_positions],
            test_positions,
        )
    ]


def _stratified_k_fold(
    data: pd.DataFrame,
    definition: DatasetDefinition,
    random_seed: int,
    fold_count: int = DEFAULT_FOLD_COUNT,
) -> list[dict]:
    """Cinco dobras estratificadas; cada dobra devolve treino e validacao."""
    label_column = definition.prediction_column
    group_column = definition.group_column
    labels = data[label_column]

    group_labels = (
        _constant_group_labels(labels, data[group_column])
        if group_column is not None
        else None
    )
    if group_labels is not None:
        # Rotulo do grupo (cliente da Home Credit): estratifica a tabela de
        # grupos e expande para as linhas, como no holdout. O
        # StratifiedGroupKFold dependia de quantas linhas cada cliente tem, e
        # as duas Home Credit punham o mesmo cliente em dobras diferentes
        # (item G16).
        group_ids = group_labels.index.to_numpy()
        splitter = StratifiedKFold(
            n_splits=fold_count, shuffle=True, random_state=random_seed
        )
        groups = data[group_column]
        fold_iterator = (
            (
                np.flatnonzero(groups.isin(group_ids[train_groups]).to_numpy()),
                np.flatnonzero(groups.isin(group_ids[validation_groups]).to_numpy()),
            )
            for train_groups, validation_groups in splitter.split(
                group_ids, group_labels.to_numpy()
            )
        )
    elif group_column is not None:
        splitter = StratifiedGroupKFold(
            n_splits=fold_count, shuffle=True, random_state=random_seed
        )
        fold_iterator = splitter.split(data, labels, groups=data[group_column])
    else:
        splitter = StratifiedKFold(
            n_splits=fold_count, shuffle=True, random_state=random_seed
        )
        fold_iterator = splitter.split(data, labels)

    return [
        _build_fold_positions(train_positions, validation_positions, None)
        for train_positions, validation_positions in fold_iterator
    ]


def _chronological_cut(
    ordered_data: pd.DataFrame,
    definition: DatasetDefinition,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Corta uma tabela ja ordenada em treino, validacao e teste."""
    row_count = len(ordered_data)
    train_end = int(round(row_count * train_fraction))
    validation_end = train_end + int(round(row_count * validation_fraction))
    return (
        ordered_data.iloc[:train_end],
        ordered_data.iloc[train_end:validation_end],
        ordered_data.iloc[validation_end:],
    )


def _temporal_holdout(
    data: pd.DataFrame,
    definition: DatasetDefinition,
) -> list[dict]:
    """Corte cronologico: treino no passado, teste no futuro.

    Quando a base tem um arquivo de teste oficial posterior (Sparkov), ele e o
    conjunto de teste e o corte entre treino e validacao acontece dentro do
    arquivo principal. Nas demais, o corte 70/15/15 e feito sobre a ordem
    temporal completa.
    """
    order_column = definition.temporal_order_column
    if order_column is None:
        raise ValueError(
            f"{definition.database_type} nao tem coluna de ordenacao temporal; "
            f"o protocolo {TEMPORAL_HOLDOUT} nao se aplica"
        )

    ordered_data = data.sort_values(order_column, kind="mergesort")
    ordered_positions = ordered_data.index.to_numpy(dtype=int)

    if AUXILIARY_SOURCE_COLUMN in ordered_data.columns and bool(
        ordered_data[AUXILIARY_SOURCE_COLUMN].any()
    ):
        auxiliary_mask = ordered_data[AUXILIARY_SOURCE_COLUMN].astype(bool).to_numpy()
        main_positions = ordered_positions[~auxiliary_mask]
        test_positions = ordered_positions[auxiliary_mask]
        # O arquivo principal cobre o periodo anterior; dele saem treino e
        # validacao, tambem em ordem cronologica.
        main_row_count = len(main_positions)
        train_end = int(round(main_row_count * (TRAIN_FRACTION / (1 - TEST_FRACTION))))
        return [
            _build_fold_positions(
                main_positions[:train_end],
                main_positions[train_end:],
                test_positions,
            )
        ]

    row_count = len(ordered_positions)
    train_end = int(round(row_count * TRAIN_FRACTION))
    validation_end = train_end + int(round(row_count * VALIDATION_FRACTION))
    return [
        _build_fold_positions(
            ordered_positions[:train_end],
            ordered_positions[train_end:validation_end],
            ordered_positions[validation_end:],
        )
    ]


def materialize_fold(plan: PartitionPlan, fold_position: int) -> dict:
    """Materializa e finaliza somente uma dobra de um plano.

    ``fold_position`` e zero-based; o ``fold_index`` persistido continua
    one-based nos consumidores do pipeline.
    """
    positions = plan.fold_positions[fold_position]
    fold = {
        split_name: (
            None
            if positions.get(split_name) is None
            else plan.complete_data.iloc[positions[split_name]]
        )
        for split_name in SPLIT_NAMES
    }
    return _finalize_fold(
        fold,
        plan.definition,
        plan.retained_columns,
        fold_position,
        plan.history_window,
    )


def plan_partitions(
    database_train: pd.DataFrame,
    database_auxiliary: pd.DataFrame | None,
    definition: DatasetDefinition,
    test_strategy: str,
    random_seed: int,
    retained_columns: tuple[str, ...] = (),
    history_window: tuple[int, int] | None = None,
) -> PartitionPlan:
    """Decide o protocolo e devolve somente as posicoes de cada split.

    O resultado segue o contrato historico (``train_test_strategy`` mais
    ``test_info``), para que o restante do pipeline nao precise mudar. Cada
    dobra traz tambem ``partition_md5``, a identidade de cada split calculada
    sobre as linhas brutas.

    ``retained_columns`` mantem colunas que normalmente saem depois do corte;
    o TabBERT as usa para montar as janelas por titular e em ordem, e o
    esquema de atributos as exclui do modelo.

    ``history_window`` (tamanho do historico, passo) grava em cada dobra o
    estrato de historico das linhas de validacao e teste (``history_strata``).
    """
    if test_strategy not in definition.protocols:
        supported = ", ".join(definition.protocols)
        raise ValueError(
            f"{definition.database_type} nao suporta o protocolo "
            f"'{test_strategy}'. Suportados: {supported}"
        )
    if database_train is None or database_train.empty:
        raise ValueError("database_train must be a non-empty dataframe")
    label_column = definition.prediction_column
    if label_column not in database_train.columns:
        raise ValueError(f"{label_column} does not exist in database_train")

    complete_data = database_train.copy()
    complete_data[AUXILIARY_SOURCE_COLUMN] = False
    if database_auxiliary is not None and not database_auxiliary.empty:
        auxiliary_data = database_auxiliary.copy()
        auxiliary_data[AUXILIARY_SOURCE_COLUMN] = True
        complete_data = pd.concat(
            [complete_data, auxiliary_data], ignore_index=True, sort=False
        )
    complete_data = complete_data.reset_index(drop=True)
    # Linhas sem rotulo nao podem ser imputadas nem avaliadas.
    complete_data = complete_data.dropna(subset=[label_column]).reset_index(drop=True)
    complete_data[ROW_FINGERPRINT_COLUMN] = row_fingerprints(complete_data)

    if test_strategy == STRATIFIED_HOLDOUT:
        folds = _stratified_holdout(complete_data, definition, random_seed)
    elif test_strategy == STRATIFIED_K_FOLD:
        folds = _stratified_k_fold(complete_data, definition, random_seed)
    elif test_strategy == TEMPORAL_HOLDOUT:
        folds = _temporal_holdout(complete_data, definition)
    else:
        raise ValueError(f"Unknown train-test strategy: {test_strategy}")

    return PartitionPlan(
        complete_data=complete_data,
        fold_positions=tuple(folds),
        definition=definition,
        test_strategy=test_strategy,
        retained_columns=retained_columns,
        history_window=history_window,
    )


def build_partitions(
    database_train: pd.DataFrame,
    database_auxiliary: pd.DataFrame | None,
    definition: DatasetDefinition,
    test_strategy: str,
    random_seed: int,
    retained_columns: tuple[str, ...] = (),
    history_window: tuple[int, int] | None = None,
) -> dict:
    """Mantem o contrato historico, materializando todas as dobras."""
    plan = plan_partitions(
        database_train,
        database_auxiliary,
        definition,
        test_strategy,
        random_seed,
        retained_columns,
        history_window,
    )
    return {
        "train_test_strategy": test_strategy,
        "test_info": [
            materialize_fold(plan, fold_position)
            for fold_position in range(len(plan.fold_positions))
        ],
    }
