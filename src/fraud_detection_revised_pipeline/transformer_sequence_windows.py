"""Causal, target-complete sequence windows for TabBERT."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

import experiment_datasets


# As chaves vem do catalogo de bases, para que titular e ordem sejam os
# mesmos usados pelo agrupamento das particoes. Na IEEE-CIS a chave deixou de
# ser `card1`: segundo a descricao publicada pelos organizadores, card1-card6
# sao informacoes do cartao (tipo, categoria, banco emissor, pais), e nao um
# identificador de portador, de modo que muitos titulares compartilham o mesmo
# valor e as sequencias misturavam clientes diferentes (item G8-iii da
# revisao). No lugar entra o titular aproximado, combinando card1, addr1 e o
# dia da primeira transacao derivado de D1.
SEQUENCE_KEYS = {
    definition.dataset_id: definition.sequence_keys
    for definition in experiment_datasets.DATASET_DEFINITIONS.values()
    if definition.sequence_keys is not None
}

# home-credit-completa's "titular" is a loan client, not a cardholder: the client's
# current-application attributes and TARGET are the same on every one of
# their previous_application rows (the sequence "steps" are past loan
# applications, not independent transactions). Splitting each client's rows
# chronologically into train/validation/test, as done for genuine transaction
# streams below, would still let that duplicated, label-carrying information
# cross the split. These datasets instead assign each whole client to one
# split (see _split_chronologically_group_exclusive); sparkov/22 and their
# variants keep the original within-holder chronological split, which is the
# intended protocol for an actual transaction sequence.
GROUP_EXCLUSIVE_SEQUENCE_DATASETS = frozenset({"home-credit-completa"})


def _chronological_split_counts(
    record_count: int, validation_ratio: float, test_ratio: float
) -> tuple[int, int, int]:
    """Split one holder chronologically, always keeping a test record.

    Holders with a single record contribute only to the test set, and holders
    with two records skip validation: the evaluated set must never shrink
    because a holder has a short history.
    """
    if record_count <= 0:
        return 0, 0, 0
    if record_count == 1:
        return 0, 0, 1
    test_count = max(1, round(record_count * test_ratio))
    if record_count - test_count < 2:
        return record_count - test_count, 0, test_count
    validation_count = max(1, round(record_count * validation_ratio))
    return record_count - test_count - validation_count, validation_count, test_count


def split_chronologically_by_identity(
    dataframe: pd.DataFrame,
    dataset: str,
    validation_ratio: float = 0.2,
    test_ratio: float = 0.2,
    classification_column_name: str | None = None,
    random_seed: int = 42,
) -> dict:
    """Build the legacy ``sequence-chronological`` TabBERT partition.

    Oldest records of each holder train, newest test. The grid now uses the
    shared protocols (data_partitioning.build_partitions) so that TabBERT is
    paired with the other blocks; this split is kept to reproduce the old
    configurations.

    For GROUP_EXCLUSIVE_SEQUENCE_DATASETS, delegates to
    _split_chronologically_group_exclusive instead: there, the holder itself
    (not just their oldest/newest records) is assigned to a single split, at
    random under ``random_seed``. The within-holder chronological cut used for
    the other datasets is deterministic and does not depend on the seed.
    """
    dataset = experiment_datasets.normalize_dataset_id(dataset)
    if dataset not in SEQUENCE_KEYS:
        raise ValueError(f"{dataset} does not provide a supported sequence identity")
    group_column, order_column = SEQUENCE_KEYS[dataset]
    missing = {group_column, order_column}.difference(dataframe.columns)
    if missing:
        raise ValueError(f"chronological split missing columns: {sorted(missing)}")

    if dataset in GROUP_EXCLUSIVE_SEQUENCE_DATASETS:
        return _split_chronologically_group_exclusive(
            dataframe,
            group_column,
            order_column,
            classification_column_name,
            validation_ratio,
            test_ratio,
            random_seed,
        )

    parts: dict[str, list[pd.DataFrame]] = {"train": [], "validation": [], "test": []}
    for _, group in dataframe.groupby(group_column, dropna=False, sort=False):
        ordered = group.sort_values(order_column, kind="mergesort")
        train_count, validation_count, test_count = _chronological_split_counts(
            len(ordered), validation_ratio, test_ratio
        )
        parts["train"].append(ordered.iloc[:train_count])
        parts["validation"].append(
            ordered.iloc[train_count : train_count + validation_count]
        )
        parts["test"].append(ordered.iloc[len(ordered) - test_count :] if test_count else ordered.iloc[0:0])

    def concatenate(split_name: str) -> pd.DataFrame | None:
        frames = [frame for frame in parts[split_name] if not frame.empty]
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    return {
        "train_test_strategy": "model-transform-brt",
        "test_info": [
            {
                "train_data": concatenate("train"),
                "validation_data": concatenate("validation"),
                "test_data": concatenate("test"),
            }
        ],
    }


def _split_chronologically_group_exclusive(
    dataframe: pd.DataFrame,
    group_column: str,
    order_column: str,
    classification_column_name: str | None,
    validation_ratio: float,
    test_ratio: float,
    random_seed: int = 42,
) -> dict:
    """Assign each whole holder to a single split, stratified by their label.

    The assignment is random under ``random_seed``, the configuration's seed:
    with a fixed value, extra seeds in SEED_LIST would repeat the same clients
    in each split while recording distinct configurations.

    Unlike split_chronologically_by_identity's per-holder chronological
    slicing, every row of a given holder ends up in the same split here; the
    chronological order is preserved only to keep each holder's own rows
    ordered within whichever split they land in, for the windowing step that
    follows. Requires classification_column_name to be constant within each
    group (true for home-credit-completa, whose TARGET belongs to the client), since
    stratifying the split by a label that varies within a group would be
    ill-defined.
    """
    if not classification_column_name or classification_column_name not in dataframe.columns:
        raise ValueError(
            "classification_column_name is required to split "
            "GROUP_EXCLUSIVE_SEQUENCE_DATASETS and must be a column of dataframe"
        )

    ordered = dataframe.sort_values([group_column, order_column], kind="mergesort")
    labels_per_group = ordered.groupby(group_column, dropna=False)[classification_column_name]
    inconsistent_groups = labels_per_group.nunique(dropna=False)
    inconsistent_groups = inconsistent_groups[inconsistent_groups > 1]
    if not inconsistent_groups.empty:
        raise ValueError(
            "group-exclusive chronological split requires a constant label per "
            f"holder; found {len(inconsistent_groups)} holder(s) with more than "
            "one class value"
        )
    group_table = labels_per_group.first().reset_index()

    stratify_column = group_table[classification_column_name]
    can_stratify = stratify_column.value_counts().min() >= 2 if len(group_table) >= 4 else False
    remaining_groups, test_groups = train_test_split(
        group_table,
        test_size=test_ratio,
        random_state=random_seed,
        shuffle=True,
        stratify=stratify_column if can_stratify else None,
    )
    adjusted_validation_ratio = (
        validation_ratio / (1 - test_ratio) if test_ratio < 1 else 0.0
    )
    train_groups, validation_groups = (remaining_groups, remaining_groups.iloc[0:0])
    if adjusted_validation_ratio > 0 and len(remaining_groups) >= 2:
        remaining_stratify = remaining_groups[classification_column_name]
        can_stratify_remaining = (
            remaining_stratify.value_counts().min() >= 2 if len(remaining_groups) >= 4 else False
        )
        train_groups, validation_groups = train_test_split(
            remaining_groups,
            test_size=adjusted_validation_ratio,
            random_state=random_seed,
            shuffle=True,
            stratify=remaining_stratify if can_stratify_remaining else None,
        )

    def subset_for(groups: pd.DataFrame) -> pd.DataFrame | None:
        if groups.empty:
            return None
        keys = groups[[group_column]].drop_duplicates()
        subset = ordered.merge(keys, on=group_column, how="inner")
        return subset.reset_index(drop=True) if not subset.empty else None

    return {
        "train_test_strategy": "model-transform-brt-grouped",
        "test_info": [
            {
                "train_data": subset_for(train_groups),
                "validation_data": subset_for(validation_groups),
                "test_data": subset_for(test_groups),
            }
        ],
    }


SEQUENCE_SPLIT_NAMES = ("train_data", "validation_data", "test_data")


def identify_fold_rows(fold: dict, fold_index: int) -> dict:
    """Copy of a fold with stable split-local row ids and the source split.

    The row id also breaks ties in the holder ordering, so every caller that
    must reproduce the TabBERT windows uses this same function.
    """
    prepared_fold = {**fold}
    for split_name in SEQUENCE_SPLIT_NAMES:
        data = fold.get(split_name)
        if data is None:
            continue
        data = data.copy()
        data["row_id"] = [f"{fold_index}:{split_name}:{index}" for index in range(len(data))]
        data["source_split"] = split_name.removesuffix("_data")
        prepared_fold[split_name] = data
    return prepared_fold


def assign_row_identity(split_information: dict) -> dict:
    """Add stable fold/split-local row ids while retaining split separation."""
    return {
        **split_information,
        "test_info": [
            identify_fold_rows(fold, fold_index)
            for fold_index, fold in enumerate(split_information["test_info"])
        ],
    }


def history_strata_by_split(
    fold: dict,
    dataset: str,
    fold_index: int = 0,
    history_length: int = 10,
    window_stride: int = 5,
) -> dict[str, pd.Series]:
    """``base_history_length`` of every row, per split, in the split's row order.

    It is the value the TabBERT window index records for the row as a target,
    computed by the same ``build_sequence_window_index`` on the same holder
    ordering, so models that never see the history can be cut by the same
    strata (item C3.19 of the review). Only the holder and order columns are
    read.
    """
    if dataset not in SEQUENCE_KEYS:
        raise ValueError(f"{dataset} does not provide a supported sequence identity")
    group_column, order_column = SEQUENCE_KEYS[dataset]
    keys_only = {
        split_name: fold[split_name][[group_column, order_column]]
        for split_name in SEQUENCE_SPLIT_NAMES
        if fold.get(split_name) is not None
    }
    identified = identify_fold_rows(keys_only, fold_index)
    windows = build_sequence_window_index(
        identified, dataset, history_length, window_stride, window_stride
    )
    targets = windows[windows["is_target"]].set_index("row_id")["base_history_length"]
    strata = {}
    for split_name in keys_only:
        row_ids = identified[split_name]["row_id"]
        values = targets.reindex(row_ids)
        if values.isna().any():
            raise ValueError(f"{split_name} has rows that are not targets of any window")
        strata[split_name] = pd.Series(values.astype(int).to_numpy(), name="base_history_length")
    return strata


def build_sequence_window_index(
    fold: dict,
    dataset: str,
    history_length: int = 10,
    target_length: int = 5,
    window_stride: int = 5,
) -> pd.DataFrame:
    if dataset not in SEQUENCE_KEYS:
        raise ValueError(f"{dataset} does not provide a supported sequence identity")
    if window_stride != target_length:
        raise ValueError("window_stride must equal target_length to evaluate each row once")
    group_column, order_column = SEQUENCE_KEYS[dataset]
    parts = []
    for split_name in ("train_data", "validation_data", "test_data"):
        data = fold.get(split_name)
        if data is not None and not data.empty:
            required = {group_column, order_column, "row_id", "source_split"}
            missing = required.difference(data.columns)
            if missing:
                raise ValueError(f"sequence data missing columns: {sorted(missing)}")
            parts.append(data[[*required]])
    if not parts:
        return pd.DataFrame()

    all_rows = pd.concat(parts, ignore_index=True)
    rows = []
    sequence_index = 0
    split_rank = {"train": 0, "validation": 1, "test": 2}
    for holder, group in all_rows.groupby(group_column, dropna=False, sort=False):
        ordered = group.sort_values([order_column, "row_id"], kind="mergesort").reset_index(drop=True)
        for target_start in range(0, len(ordered), window_stride):
            targets = ordered.iloc[target_start : target_start + target_length]
            history = ordered.iloc[max(0, target_start - history_length) : target_start]
            pad_count = history_length - len(history)
            for position in range(pad_count):
                rows.append(
                    {
                        "holder_id": str(holder), "sequence_index": sequence_index,
                        "position_index": position, "row_id": "[PAD]",
                        "source_split": "pad", "is_target": False,
                        "base_history_length": len(history), "causal_context_length": 0,
                    }
                )
            for offset, record in enumerate(pd.concat([history, targets]).itertuples(index=False)):
                is_target = offset >= len(history)
                target_offset = offset - len(history) if is_target else -1
                target_rank = split_rank.get(record.source_split, 99)
                if is_target:
                    allowed_history = sum(
                        split_rank.get(value, 99) <= target_rank
                        for value in history["source_split"]
                    )
                    allowed_prior_targets = sum(
                        split_rank.get(value, 99) <= target_rank
                        for value in targets.iloc[:target_offset]["source_split"]
                    )
                else:
                    allowed_history = len(history)
                    allowed_prior_targets = 0
                rows.append(
                    {
                        "holder_id": str(holder), "sequence_index": sequence_index,
                        "position_index": pad_count + offset,
                        "row_id": record.row_id, "source_split": record.source_split,
                        "is_target": is_target,
                        "target_split": record.source_split if is_target else "",
                        "base_history_length": allowed_history,
                        "causal_context_length": allowed_history + allowed_prior_targets,
                        "split_rank": split_rank.get(record.source_split, 99),
                    }
                )
            sequence_index += 1
    return pd.DataFrame(rows).drop(columns=["split_rank"], errors="ignore")
