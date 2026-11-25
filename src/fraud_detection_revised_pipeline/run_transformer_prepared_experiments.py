"""Train and evaluate FT-Transformer or TabBERT from prepared manifests."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
import json
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import execution_provenance
from lr_nb_knn_specific import (
    EVALUATION_UNIT_CLIENT,
    EVALUATION_UNIT_ROW,
    PREPARED_STORAGE_FAILURE_EXIT_CODE,
    PreparedDatasetStorageError,
    aggregate_split_scores_by_unit,
    record_algorithm_failure_to_csv,
    record_algorithm_result_to_csv,
    stage_prepared_dataset_files,
)
from decision_threshold import (
    DEFAULT_THRESHOLD_STRATEGY,
    REQUESTABLE_THRESHOLD_STRATEGIES,
    SELECTED_THRESHOLD_STRATEGIES,
    THRESHOLD_STRATEGY_FIXED,
    FoldThreshold,
    select_fold_thresholds,
)
from history_strata import HISTORY_STRATUM_NAME, binary_metrics, stratum_metrics
from transformer_models import (
    TABBERT_TRAINING_REGIMES,
    FTTransformer,
    TabBERT,
    tabbert_supervised_epochs,
    tabbert_training_stages,
)
from model_metadata import count_trainable_parameters, describe_torch_constructor
from transformer_vocabulary import FIRST_VALUE_INDEX, MASK_INDEX
from experiment_datasets import get_dataset_definition
from execution_failures import check_fold_partitions
from prepared_dataset_storage import (
    EVALUATION_UNIT_COLUMN,
    HISTORY_STRATA_COLUMN,
    evaluation_unit_path,
    history_strata_path,
    normalize_manifest_dataset_identity,
)


# Cada combinacao de base + cabecalho de classificacao e um algoritmo proprio,
# no mesmo espirito de catboost/xgboost serem algoritmos distintos: cada uma
# grava em seu proprio resultados_<algorithm>.csv, com esse identificador
# completo na coluna `algorithm`.
# A cabeca "linear" (LayerNorm + Linear) saiu do processo: ficam apenas as
# duas variantes de classificador efetivamente comparadas na tese. Cada uma e
# tratada como um algoritmo proprio, com seu resultados_<modelo>.csv e sua
# linha na coluna "algorithm"; "regime" continua vindo do modelo base, nunca
# do sufixo da cabeca.
MODEL_CONFIGURATIONS: dict[str, tuple[str, str]] = {
    "ft-transformer-mlp": ("ft-transformer", "mlp"),
    "ft-transformer-lstm": ("ft-transformer", "lstm"),
    "tabbert-mlp": ("tabbert", "mlp"),
    "tabbert-lstm": ("tabbert", "lstm"),
}


def resolve_model_configuration(model_name: str) -> tuple[str, str]:
    """(base_model, classifier_head) para um identificador de --model."""
    try:
        return MODEL_CONFIGURATIONS[model_name]
    except KeyError:
        raise ValueError(f"Unknown model: {model_name}") from None


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_CONFIGURATIONS), required=True)
    parser.add_argument("--configuration-md5", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--prepared-data-directory", required=True)
    parser.add_argument(
        "--local-staging-directory",
        default=str(Path(__file__).resolve().parent / "prepared_datasets"),
        help=(
            "Local directory used while a configuration's prepared files are "
            "staged from --prepared-data-directory for processing."
        ),
    )
    parser.add_argument(
        "--tabbert-regime",
        default="mlm-before-finetune",
        choices=TABBERT_TRAINING_REGIMES,
        help="Regime de treino do TabBERT (item C3.19); o antigo mlm-after-warmup e recusado.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--mlm-epochs", type=int, default=3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Lote fixo; sem valor, ele e derivado do numero de campos da base.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["auto", "mixed", "float32"], default="auto")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,  # Acompanha RANDOM_STATE do pipeline classico (lr_nb_knn_specific.py).
        help="Semente de random, numpy, torch e CUDA; propagada aos processos auxiliares.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help=(
            "Exige algoritmos deterministas do PyTorch. Desligado por padrao: "
            "parte das operacoes de atencao perde desempenho ou nao possui "
            "implementacao determinista, o que faria a execucao falhar."
        ),
    )
    parser.add_argument(
        "--result-file",
        help="Caminho de resultados_<model>.csv; sem valor, usa o nome canonico ao lado do script.",
    )
    parser.add_argument(
        "--details-file",
        help=(
            "Caminho do .jsonl companheiro com transaction_predictions/history_strata; "
            "sem valor, usa resultados_<model>_details.jsonl ao lado do script."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        choices=list(REQUESTABLE_THRESHOLD_STRATEGIES),
        default=DEFAULT_THRESHOLD_STRATEGY,
        help=(
            "validation-f2 (padrao): limiar que maximiza F2 na validacao da dobra, "
            "aplicado ao teste. fixed: 0,5."
        ),
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help=(
            "Recomputa e regrava cada fold mesmo que ja exista uma linha de sucesso "
            "para configuration_md5/fold_index/algorithm/regime."
        ),
    )
    return parser.parse_args()


def seed_everything(seed: int, deterministic: bool) -> None:
    """Fixa todos os pontos de sorteio conhecidos da execucao."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int, base_seed: int) -> None:
    """Deriva a semente de cada processo auxiliar da semente base, nao do estado global."""
    worker_seed = (base_seed + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Lote de referencia medido em GPU: 256 amostras para uma base de 31 campos.
REFERENCE_BATCH_SIZE = 256
REFERENCE_FIELD_COUNT = 31
MINIMUM_BATCH_SIZE = 8
MAXIMUM_WORKERS = 4


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def derive_batch_size(field_count: int) -> int:
    """Manter o custo por passo comparavel entre bases de larguras diferentes."""
    if field_count <= 0:
        return REFERENCE_BATCH_SIZE
    derived = round(REFERENCE_BATCH_SIZE * REFERENCE_FIELD_COUNT / field_count)
    return max(MINIMUM_BATCH_SIZE, min(REFERENCE_BATCH_SIZE, derived))


def resolve_workers(requested: int | None, device: torch.device) -> int:
    if requested is not None:
        return max(0, requested)
    if device.type != "cuda":
        return 0
    return min(MAXIMUM_WORKERS, max(1, (os.cpu_count() or 2) // 2))


@dataclass(frozen=True)
class ExecutionSettings:
    """Condicoes de execucao resolvidas uma vez e registradas no resultado."""

    device: torch.device
    batch_size: int
    num_workers: int
    autocast_dtype: torch.dtype | None
    uses_gradient_scaler: bool
    device_name: str
    seed: int
    deterministic: bool

    @property
    def precision(self) -> str:
        if self.autocast_dtype is None:
            return "float32"
        return str(self.autocast_dtype).removeprefix("torch.")

    def loader(self, dataset, batch_size: int, shuffle: bool = False) -> DataLoader:
        generator = None
        if shuffle:
            # Gerador proprio: o embaralhamento nao depende do estado global do RNG.
            generator = torch.Generator()
            generator.manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            num_workers=self.num_workers,
            worker_init_fn=(
                partial(worker_init_fn, base_seed=self.seed)
                if self.num_workers > 0
                else None
            ),
            pin_memory=self.device.type == "cuda",
            persistent_workers=self.num_workers > 0,
        )

    def as_metadata(self) -> dict:
        # Prefixo obrigatorio: `precision` colidiria com a metrica de precisao.
        return {
            "execution_device": self.device.type,
            "execution_device_name": self.device_name,
            "execution_precision": self.precision,
            "execution_batch_size": self.batch_size,
            "execution_num_workers": self.num_workers,
            "execution_seed": self.seed,
            "execution_deterministic": self.deterministic,
        }


def build_execution_settings(args, device: torch.device, field_count: int) -> ExecutionSettings:
    uses_mixed = args.precision in {"auto", "mixed"} and device.type == "cuda"
    autocast_dtype = None
    uses_scaler = False
    if uses_mixed:
        if torch.cuda.is_bf16_supported():
            autocast_dtype = torch.bfloat16
        else:
            autocast_dtype = torch.float16
            uses_scaler = True
    # TF32 acelera o que permanecer em float32 sem custo relevante de precisao.
    torch.set_float32_matmul_precision(
        "highest" if args.precision == "float32" else "high"
    )
    return ExecutionSettings(
        device=device,
        batch_size=args.batch_size or derive_batch_size(field_count),
        num_workers=resolve_workers(args.num_workers, device),
        autocast_dtype=autocast_dtype,
        uses_gradient_scaler=uses_scaler,
        device_name=(
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        seed=args.seed,
        deterministic=args.deterministic,
    )


def choose_fold_threshold(
    scored_splits: dict[str, tuple[np.ndarray, np.ndarray]], threshold_strategy: str
) -> FoldThreshold:
    """Limiar da dobra a partir das previsoes de validacao (item G11-i).

    Os protocolos dos Transformers (holdouts e corte cronologico do TabBERT)
    sempre tem teste, entao o limiar vem da validacao da propria dobra.
    """
    validation = scored_splits.get("validation")
    validation_set = (
        None
        if validation is None
        else (np.asarray(validation[0]).astype(int), np.asarray(validation[1]))
    )
    return select_fold_thresholds(
        [validation_set], ["test" in scored_splits], threshold_strategy
    )[0]


# As metricas sao as mesmas dos estratos de historico (history_strata.py).
metrics = binary_metrics


def load_fold_rows(rows: pd.DataFrame, root: Path, fold_index: int) -> dict[str, pd.DataFrame]:
    fold_rows = rows[rows["fold_index"].astype(int) == fold_index]
    return {
        row["split"]: pd.read_csv(root / f"{row['dataset_md5']}.csv")
        for _, row in fold_rows.iterrows()
    }


def load_fold_units(rows: pd.DataFrame, root: Path, fold_index: int) -> dict[str, pd.Series]:
    """Cliente de cada linha de validacao/teste, lido do arquivo lateral (item G7)."""
    fold_rows = rows[rows["fold_index"].astype(int) == fold_index]
    units = {}
    for _, row in fold_rows.iterrows():
        path = evaluation_unit_path(root, row["dataset_md5"])
        if path.is_file():
            units[row["split"]] = pd.read_csv(path)[EVALUATION_UNIT_COLUMN]
    return units


def load_fold_history_strata(rows: pd.DataFrame, root: Path, fold_index: int) -> dict[str, np.ndarray]:
    """Estrato de historico de cada linha de validacao/teste (item C3.19), quando gravado."""
    fold_rows = rows[rows["fold_index"].astype(int) == fold_index]
    strata = {}
    for _, row in fold_rows.iterrows():
        path = history_strata_path(root, row["dataset_md5"])
        if path.is_file():
            strata[row["split"]] = pd.read_csv(path)[HISTORY_STRATA_COLUMN].to_numpy()
    return strata


def unit_level_scored_splits(
    scored_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    units_by_split: dict[str, np.ndarray],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """(alvos, escores) por cliente, com a media dos escores de cada um."""
    aggregated = {}
    for split_name, (targets, probabilities) in scored_splits.items():
        units = units_by_split.get(split_name)
        if units is None:
            continue
        if len(units) != len(targets):
            raise ValueError(
                f"{len(units)} evaluation units for {len(targets)} {split_name} predictions"
            )
        by_unit = aggregate_split_scores_by_unit(
            {"target": targets, "positive_scores": probabilities}, units
        )
        aggregated[split_name] = (
            by_unit["target"].to_numpy(), np.asarray(by_unit["positive_scores"])
        )
    return aggregated


def unit_level_results(
    scored_splits: dict[str, tuple[np.ndarray, np.ndarray]],
    units_by_split: dict[str, np.ndarray],
    threshold_strategy: str,
    model_metadata: dict,
) -> list[dict]:
    """Resultados por cliente, com limiar proprio escolhido na validacao por cliente."""
    client_splits = unit_level_scored_splits(scored_splits, units_by_split)
    if not client_splits:
        return []
    fold_threshold = choose_fold_threshold(client_splits, threshold_strategy)
    results = []
    for evaluation_split, (targets, probabilities) in client_splits.items():
        result = metrics(targets, probabilities, fold_threshold.threshold)
        result.update(
            evaluation_split=evaluation_split,
            evaluation_unit=EVALUATION_UNIT_CLIENT,
            evaluated_units=len(targets),
            classification_threshold=fold_threshold.threshold,
            threshold_strategy=fold_threshold.strategy,
            # As predicoes por transacao ja vao no registro por linha.
            transaction_predictions=[],
            **model_metadata,
        )
        results.append(result)
    return results


def autocast_context(settings: ExecutionSettings):
    return torch.autocast(
        device_type=settings.device.type,
        dtype=settings.autocast_dtype,
        enabled=settings.autocast_dtype is not None,
    )


def gradient_scaler(settings: ExecutionSettings) -> torch.amp.GradScaler:
    return torch.amp.GradScaler(
        settings.device.type, enabled=settings.uses_gradient_scaler
    )


def optimize(loss, optimizer, scaler) -> None:
    """Um passo de otimizacao, com ou sem escalonamento de gradiente."""
    if scaler.is_enabled():
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        return
    loss.backward()
    optimizer.step()


def synchronized_clock(settings) -> float:
    """Relogio monotono depois de esperar a GPU terminar o que foi enfileirado.

    Sem a sincronizacao, as chamadas CUDA sao assincronas e o tempo medido
    seria o de enfileirar o trabalho, e nao o de executa-lo (item G18).
    """
    if str(getattr(settings, "device", "cpu")).startswith("cuda"):
        torch.cuda.synchronize()
    return time.perf_counter()


def fold_timing(fit_time: float, scoring_time: float, pretraining_time: float | None = None) -> dict:
    return {
        "fit_time": round(fit_time, 6),
        "scoring_time": round(scoring_time, 6),
        "pretraining_time": "" if pretraining_time is None else round(pretraining_time, 6),
    }


def train_binary_model(model, loader, settings, epochs, learning_rate, forward):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    scaler = gradient_scaler(settings)
    model.train()
    for _ in range(epochs):
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            moved = tuple(value.to(settings.device, non_blocking=True) for value in batch)
            with autocast_context(settings):
                logits, targets = forward(model, moved)
                loss = loss_function(logits, targets.float())
            optimize(loss, optimizer, scaler)


def transformer_model_metadata(
    model_class,
    model,
    args,
    settings: ExecutionSettings,
    stage_epochs: dict[str, int],
    batch_size: int,
    **constructor_arguments,
) -> tuple[dict, dict]:
    """Hiperparametros e resumo do treino de um Transformer ajustado.

    As cardinalidades e contagens de atributos vem da base, e nao de uma
    escolha de modelagem: ficam no resumo do treino, e nos hiperparametros
    entram so os argumentos que o executor fixa.
    """
    data_derived = {
        name: constructor_arguments.pop(name)
        for name in (
            "numerical_feature_count",
            "categorical_cardinalities",
            "field_cardinalities",
        )
        if name in constructor_arguments
    }
    hyperparameters = {
        "architecture": describe_torch_constructor(model_class, **constructor_arguments),
        "training": {
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
            "batch_size": batch_size,
            "precision": settings.precision,
        },
    }
    if model_class is TabBERT:
        hyperparameters["training"].update(
            regime=args.tabbert_regime,
            mlm_epochs=args.mlm_epochs,
            mlm_mask_probability=0.15,
        )
    training_summary = {
        "trainable_parameters": count_trainable_parameters(model),
        "epochs_by_stage": stage_epochs,
        **data_derived,
    }
    return hyperparameters, training_summary


# Splits que os Transformers avaliam e gravam. A validacao entrou para que a
# selecao da melhor configuracao tenha onde acontecer sem olhar o teste (item
# G12 da revisao). O treino fica de fora aqui, e so aqui: nos blocos classico e
# de boosting suas metricas ja vinham calculadas e sao gravadas de graca, mas
# num Transformer exigiriam uma passada de inferencia inteira sobre o maior dos
# splits para produzir uma medida de ajuste, que nao entra em selecao nem em
# relato.
EVALUATED_SPLITS = ("validation", "test")


def run_ft_fold(
    data: dict[str, pd.DataFrame],
    metadata: dict,
    args,
    settings,
    units_by_split: dict[str, pd.Series] | None = None,
    history_by_split: dict[str, np.ndarray] | None = None,
) -> list[dict]:
    target = metadata["target_column"]
    categorical = metadata["categorical_columns"]
    numerical = metadata["numerical_columns"] + [
        f"{column}__is_missing" for column in metadata["numerical_columns"]
    ]
    cardinalities = [metadata["vocabularies"][column]["size"] for column in categorical]
    _base_model, classifier_head = resolve_model_configuration(getattr(args, "model", "ft-transformer-mlp"))
    model = FTTransformer(
        len(numerical), cardinalities, classifier_head=classifier_head
    ).to(settings.device)

    def tensors(frame):
        return TensorDataset(
            torch.tensor(frame[numerical].to_numpy(dtype=np.float32)),
            torch.tensor(frame[categorical].to_numpy(dtype=np.int64)),
            torch.tensor(frame[target].to_numpy(dtype=np.float32)),
        )

    fit_started_at = synchronized_clock(settings)
    loader = settings.loader(tensors(data["train"]), settings.batch_size, shuffle=True)
    train_binary_model(
        model, loader, settings, args.epochs, args.learning_rate,
        lambda current, batch: (current(batch[0], batch[1]), batch[2]),
    )
    fit_time = synchronized_clock(settings) - fit_started_at
    hyperparameters, training_summary = transformer_model_metadata(
        FTTransformer,
        model,
        args,
        settings,
        {"supervised": args.epochs},
        settings.batch_size,
        numerical_feature_count=len(numerical),
        categorical_cardinalities=cardinalities,
        classifier_head=classifier_head,
    )
    # Avalia a validacao TAMBEM, nao apenas o teste. Antes o split de avaliacao
    # era "test" quando existia, e as metricas de validacao nunca chegavam ao
    # CSV -- sem elas a selecao da melhor configuracao recaia sobre a propria
    # particao de teste em que o resultado e relatado (item G12 da revisao).
    model.eval()
    scoring_started_at = synchronized_clock(settings)
    scored_splits = {}
    for evaluation_split in EVALUATED_SPLITS:
        evaluation = data.get(evaluation_split)
        if evaluation is None or len(evaluation) == 0:
            continue
        eval_loader = settings.loader(tensors(evaluation), settings.batch_size)
        probabilities, targets = [], []
        with torch.no_grad(), autocast_context(settings):
            for numeric, categorical_values, labels in eval_loader:
                logits = model(
                    numeric.to(settings.device, non_blocking=True),
                    categorical_values.to(settings.device, non_blocking=True),
                )
                probabilities.extend(torch.sigmoid(logits).float().cpu().numpy())
                targets.extend(labels.numpy())
        scored_splits[evaluation_split] = (np.asarray(targets), np.asarray(probabilities))
    timing = fold_timing(fit_time, synchronized_clock(settings) - scoring_started_at)

    # O limiar so e conhecido depois de pontuar a validacao.
    fold_threshold = choose_fold_threshold(scored_splits, args.threshold_strategy)
    results = []
    for evaluation_split, (targets, probabilities) in scored_splits.items():
        result = metrics(targets, probabilities, fold_threshold.threshold)
        result["evaluation_split"] = evaluation_split
        result["classification_threshold"] = fold_threshold.threshold
        result["threshold_strategy"] = fold_threshold.strategy
        result["model_hyperparameters"] = hyperparameters
        result["training_summary"] = training_summary
        strata = (history_by_split or {}).get(evaluation_split)
        result["transaction_predictions"] = [
            {"prediction_index": index, "target": float(target), "probability": float(probability)}
            for index, (target, probability) in enumerate(zip(targets, probabilities))
        ]
        if strata is not None:
            # Mesmo corte do TabBERT, sobre as mesmas transacoes (item C3.19).
            for prediction, stratum in zip(result["transaction_predictions"], strata):
                prediction[HISTORY_STRATUM_NAME] = int(stratum)
            result["history_strata"] = {
                HISTORY_STRATUM_NAME: stratum_metrics(
                    targets, probabilities, strata, fold_threshold.threshold
                )
            }
        result.update(timing)
        results.append(result)
    # A ordem das predicoes e a das linhas do split (loader sem embaralhar),
    # a mesma do arquivo lateral.
    results.extend(
        unit_level_results(
            scored_splits,
            {name: units.to_numpy() for name, units in (units_by_split or {}).items()},
            args.threshold_strategy,
            {"model_hyperparameters": hyperparameters, "training_summary": training_summary, **timing},
        )
    )
    return results


WINDOW_LENGTH = 15
SPLIT_RANK = {"train": 0, "validation": 1, "test": 2}


def resolve_token_dtype(max_cardinality: int) -> np.dtype:
    """Menor tipo inteiro que acomoda os indices de vocabulario da configuracao."""
    for candidate in (np.int16, np.int32, np.int64):
        if max_cardinality <= np.iinfo(candidate).max:
            return candidate
    return np.int64


def _empty_tabbert_dataset(field_count: int, token_dtype: np.dtype) -> TensorDataset:
    return TensorDataset(
        torch.from_numpy(np.zeros((0, WINDOW_LENGTH, field_count), dtype=token_dtype)),
        torch.zeros((0, WINDOW_LENGTH), dtype=torch.float32),
        torch.zeros((0, WINDOW_LENGTH), dtype=torch.bool),
        torch.zeros((0, WINDOW_LENGTH, 2), dtype=torch.int64),
    )


def _tabbert_window_layout(windows: pd.DataFrame, target_split: str):
    """(janelas, posicoes reindexadas) com alvo no split pedido, ou None."""
    matching_sequences = windows.loc[
        windows["is_target"] & (windows["target_split"] == target_split), "sequence_index"
    ].unique()
    if len(matching_sequences) == 0:
        return None
    sequence_ids = np.sort(matching_sequences)
    full_index = pd.MultiIndex.from_product(
        [sequence_ids, range(WINDOW_LENGTH)], names=["sequence_index", "position_index"]
    )
    selected = windows[windows["sequence_index"].isin(sequence_ids)]
    # Janelas curtas nao preenchem as WINDOW_LENGTH posicoes; o reindex
    # completa as faltantes com NaN, tratadas como preenchimento.
    return sequence_ids, selected.set_index(["sequence_index", "position_index"]).reindex(full_index)


def _tabbert_records(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    return pd.concat([frame.set_index("row_id") for frame in data.values() if frame is not None])


def _tabbert_target_mask(reindexed: pd.DataFrame, target_split: str) -> np.ndarray:
    is_target = reindexed["is_target"].fillna(False).to_numpy(dtype=bool)
    return is_target & (reindexed["target_split"].to_numpy() == target_split)


def tabbert_target_units(
    data: dict[str, pd.DataFrame],
    windows: pd.DataFrame,
    target_split: str,
    unit_column: str,
) -> np.ndarray:
    """Unidade (cliente) de cada posicao-alvo, na ordem em que o modelo as pontua.

    A avaliacao percorre as janelas em ordem e pega as posicoes-alvo de cada
    uma; aqui a mesma ordem e reconstruida a partir do mesmo arranjo, lendo a
    chave de sequencia que o TabBERT ja mantem nos arquivos.
    """
    layout = _tabbert_window_layout(windows, target_split)
    if layout is None:
        return np.empty(0, dtype=object)
    _sequence_ids, reindexed = layout
    looked_up = _tabbert_records(data).reindex(reindexed["row_id"].to_numpy())
    return looked_up[unit_column].to_numpy()[_tabbert_target_mask(reindexed, target_split)]


def build_tabbert_tensors(
    data: dict[str, pd.DataFrame],
    windows: pd.DataFrame,
    metadata: dict,
    target_split: str,
    token_dtype: np.dtype = np.int64,
) -> TensorDataset:
    """Monta os tensores de janela por operacoes sobre arrays.

    Cada janela do indice tem ate WINDOW_LENGTH posicoes; janelas mais curtas
    ocorrem quando restam menos de 5 registros ao final do titular. As posicoes
    ausentes de uma janela curta sao tratadas como as de preenchimento: token
    zero, fora da mascara de alvo, exatamente como a montagem anterior deixava
    ao nunca escrever naquela posicao do array pre-zerado.
    """
    fields = metadata["categorical_columns"] + metadata["numerical_columns"]
    target_column = metadata["target_column"]
    requested_rank = SPLIT_RANK[target_split]

    layout = _tabbert_window_layout(windows, target_split)
    if layout is None:
        return _empty_tabbert_dataset(len(fields), token_dtype)
    sequence_ids, reindexed = layout

    source_rank = reindexed["source_split"].map(SPLIT_RANK).fillna(99).to_numpy()
    allowed = source_rank <= requested_rank

    looked_up = _tabbert_records(data).reindex(reindexed["row_id"].to_numpy())

    window_count = len(sequence_ids)
    field_values = looked_up[fields].to_numpy(dtype=np.float64)
    field_values = np.nan_to_num(field_values, nan=0.0)
    field_values = np.where(allowed[:, None], field_values, 0.0)
    tokens = field_values.reshape(window_count, WINDOW_LENGTH, len(fields)).astype(token_dtype)

    target_mask_flat = _tabbert_target_mask(reindexed, target_split)

    target_labels = looked_up[target_column].to_numpy(dtype=np.float64)
    target_values = np.where(target_mask_flat, np.nan_to_num(target_labels, nan=0.0), 0.0)
    target_values = target_values.reshape(window_count, WINDOW_LENGTH).astype(np.float32)
    target_mask = target_mask_flat.reshape(window_count, WINDOW_LENGTH)

    base_history = reindexed["base_history_length"].fillna(0).to_numpy(dtype=np.float64)
    causal_context = reindexed["causal_context_length"].fillna(0).to_numpy(dtype=np.float64)
    history = np.stack(
        [
            np.where(target_mask_flat, base_history, 0.0),
            np.where(target_mask_flat, causal_context, 0.0),
        ],
        axis=-1,
    ).reshape(window_count, WINDOW_LENGTH, 2).astype(np.int64)

    return TensorDataset(
        torch.from_numpy(tokens), torch.from_numpy(target_values),
        torch.from_numpy(target_mask), torch.from_numpy(history),
    )


def sequence_batch_size(settings: ExecutionSettings) -> int:
    """Cada janela carrega 15 registros, entao o lote de sequencia e menor."""
    return max(1, settings.batch_size // 8)


def train_tabbert_supervised(model, dataset, args, settings, epochs):
    loader = settings.loader(dataset, sequence_batch_size(settings), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    loss_function = nn.BCEWithLogitsLoss()
    scaler = gradient_scaler(settings)
    model.train()
    for _ in range(epochs):
        for tokens, labels, target_mask, _history in loader:
            # Os tensores de janela usam o menor tipo que cabe no vocabulario;
            # a camada de embedding exige indices int64.
            tokens = tokens.to(settings.device, dtype=torch.int64, non_blocking=True)
            labels = labels.to(settings.device, non_blocking=True)
            target_mask = target_mask.to(settings.device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            padding = tokens.eq(0).all(dim=-1)
            with autocast_context(settings):
                logits = model(tokens, padding)
                loss = loss_function(logits[target_mask], labels[target_mask])
            optimize(loss, optimizer, scaler)


def train_tabbert_mlm(model, dataset, args, settings):
    loader = settings.loader(dataset, sequence_batch_size(settings), shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scaler = gradient_scaler(settings)
    model.train()
    for _ in range(args.mlm_epochs):
        for tokens, _labels, _target_mask, _history in loader:
            tokens = tokens.to(settings.device, dtype=torch.int64, non_blocking=True)
            padding = tokens.eq(0).all(dim=-1)
            selected = (torch.rand_like(tokens.float()) < 0.15) & ~padding.unsqueeze(-1)
            if not selected.any():
                continue
            masked = tokens.clone(); masked[selected] = MASK_INDEX
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(settings):
                logits = model.masked_field_logits(masked, padding)
                losses = [
                    nn.functional.cross_entropy(field_logits[selected[:, :, index]], tokens[:, :, index][selected[:, :, index]])
                    for index, field_logits in enumerate(logits)
                    if selected[:, :, index].any()
                ]
                loss = torch.stack(losses).mean()
            optimize(loss, optimizer, scaler)


def train_tabbert_regime(model, train_data, args, settings) -> tuple[dict[str, int], float | None]:
    """Executa as etapas do regime e devolve as epocas por etapa e o tempo do MLM.

    Todos os regimes tem o mesmo total de epocas supervisionadas
    (``args.epochs``); o PM->S acrescenta o MLM antes do ajuste, e o S->PM->S
    divide o ajuste em aquecimento e ajuste final, com o MLM entre eles
    (item C3.19).
    """
    stage_epochs: dict[str, int] = {}
    pretraining_time = None
    stages = tabbert_training_stages(args.tabbert_regime)
    supervised_epochs = tabbert_supervised_epochs(stages, args.epochs)
    for stage in stages:
        if stage == "mlm":
            mlm_started_at = synchronized_clock(settings)
            train_tabbert_mlm(model, train_data, args, settings)
            pretraining_time = synchronized_clock(settings) - mlm_started_at
            stage_epochs[stage] = args.mlm_epochs
        else:
            train_tabbert_supervised(model, train_data, args, settings, supervised_epochs[stage])
            stage_epochs[stage] = supervised_epochs[stage]
    return stage_epochs, pretraining_time


def run_tabbert_fold(data, windows, metadata, args, settings, unit_column=None):
    categorical = metadata["categorical_columns"]
    numerical = metadata["numerical_columns"]
    cardinalities = [metadata["vocabularies"][column]["size"] for column in categorical]
    cardinalities += [len(metadata["numeric"]["bin_edges"][column]) - 1 + FIRST_VALUE_INDEX for column in numerical]
    _base_model, classifier_head = resolve_model_configuration(getattr(args, "model", "tabbert-mlp"))
    model = TabBERT(cardinalities, classifier_head=classifier_head).to(settings.device)
    token_dtype = resolve_token_dtype(max(cardinalities, default=1))
    fit_started_at = synchronized_clock(settings)
    train_data = build_tabbert_tensors(data, windows, metadata, "train", token_dtype)
    stage_epochs, pretraining_time = train_tabbert_regime(model, train_data, args, settings)
    # O ajuste inclui o MLM; o MLM sozinho fica tambem em pretraining_time.
    fit_time = synchronized_clock(settings) - fit_started_at
    hyperparameters, training_summary = transformer_model_metadata(
        TabBERT,
        model,
        args,
        settings,
        stage_epochs,
        sequence_batch_size(settings),
        field_cardinalities=cardinalities,
        classifier_head=classifier_head,
    )
    # Como no FT-Transformer, a validacao passa a ser avaliada e gravada, para
    # que a selecao da melhor configuracao nao recaia sobre a particao de teste
    # (item G12 da revisao).
    model.eval()
    scoring_started_at = synchronized_clock(settings)
    scored_splits = {}
    history_by_split = {}
    for evaluation_split in EVALUATED_SPLITS:
        if evaluation_split not in data:
            continue
        evaluation = build_tabbert_tensors(
            data, windows, metadata, evaluation_split, token_dtype
        )
        if len(evaluation) == 0:
            continue
        loader = settings.loader(evaluation, sequence_batch_size(settings))
        probabilities, targets, history_values = [], [], []
        with torch.no_grad(), autocast_context(settings):
            for tokens, labels, target_mask, history in loader:
                tokens = tokens.to(settings.device, dtype=torch.int64, non_blocking=True)
                target_mask = target_mask.to(settings.device, non_blocking=True)
                logits = model(tokens, tokens.eq(0).all(dim=-1))
                probabilities.extend(torch.sigmoid(logits[target_mask]).float().cpu().numpy())
                targets.extend(labels.to(settings.device)[target_mask].cpu().numpy())
                history_values.extend(history.to(settings.device)[target_mask].cpu().numpy().tolist())
        if not targets:
            continue
        scored_splits[evaluation_split] = (np.asarray(targets), np.asarray(probabilities))
        history_by_split[evaluation_split] = history_values
    timing = fold_timing(
        fit_time, synchronized_clock(settings) - scoring_started_at, pretraining_time
    )

    fold_threshold = choose_fold_threshold(scored_splits, args.threshold_strategy)
    threshold = fold_threshold.threshold
    results = []
    for evaluation_split, (targets, probabilities) in scored_splits.items():
        history_values = history_by_split[evaluation_split]
        result = metrics(targets, probabilities, threshold)
        result["evaluation_split"] = evaluation_split
        result["classification_threshold"] = threshold
        result["threshold_strategy"] = fold_threshold.strategy
        result["model_hyperparameters"] = hyperparameters
        result["training_summary"] = training_summary
        result["transaction_predictions"] = [
            {
                "target": float(target), "probability": float(probability),
                "base_history_length": int(history[0]),
                "causal_context_length": int(history[1]),
            }
            for target, probability, history in zip(targets, probabilities, history_values)
        ]
        # Transacoes, fraudes e prevalencia em cada estrato (item C3.19).
        history_array = np.asarray(history_values)
        result["history_strata"] = {
            name: stratum_metrics(targets, probabilities, history_array[:, index], threshold)
            for index, name in enumerate((HISTORY_STRATUM_NAME, "causal_context_length"))
        }
        result.update(timing)
        results.append(result)
    if unit_column:
        results.extend(
            unit_level_results(
                scored_splits,
                {
                    split_name: tabbert_target_units(data, windows, split_name, unit_column)
                    for split_name in scored_splits
                },
                args.threshold_strategy,
                {"model_hyperparameters": hyperparameters, "training_summary": training_summary, **timing},
            )
        )
    return results


def load_completed_folds(
    results_path: Path,
    configuration_md5: str,
    algorithm: str,
    regime: str,
    threshold_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
) -> set[int]:
    """Folds ja bem-sucedidos para esta configuracao/algoritmo/regime no CSV canonico.

    A estrategia de limiar entra na chave: linhas sem ``threshold_strategy`` sao
    anteriores a escolha na validacao e contam como ``fixed``.
    """
    if not results_path.is_file():
        return set()
    try:
        results = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return set()
    if results.empty:
        return set()
    matching = results[
        (results["configuration_md5"] == configuration_md5)
        & (results["algorithm"] == algorithm)
        & (results["regime"] == regime)
        & (results["execution_status"] == "success")
    ]
    recorded_strategies = (
        matching["threshold_strategy"]
        if "threshold_strategy" in matching.columns
        else pd.Series("", index=matching.index)
    ).replace("", THRESHOLD_STRATEGY_FIXED)
    if threshold_strategy == THRESHOLD_STRATEGY_FIXED:
        matching = matching[recorded_strategies == THRESHOLD_STRATEGY_FIXED]
    else:
        matching = matching[recorded_strategies.isin(SELECTED_THRESHOLD_STRATEGIES)]
    return {int(value) for value in matching["fold_index"]}


def append_details_record(details_path: Path, record: dict) -> None:
    details_directory = details_path.parent
    if str(details_directory):
        details_directory.mkdir(parents=True, exist_ok=True)
    with details_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=True) + "\n")


def stage_transformer_artifacts(
    rows: pd.DataFrame, staged_directory: Path, prepared_directory: Path
) -> None:
    """Copy each fold's vocabulary/window index file into the staged directory.

    `stage_prepared_dataset_files` only stages the `dataset_md5.csv` splits;
    the Transformer manifest also references a vocabulary file per fold and,
    for TabBERT, a window index file, both outside that function's scope.
    The manifest records the path where the configuration was prepared (the
    worker scratch, in distributed preparation), so the file is looked up by
    name in the prepared directory first, like the validator does.
    """
    for column in ("vocabulary_path", "window_index_path"):
        raw_values = rows.get(column)
        if raw_values is None:
            continue
        for raw_path in sorted({str(value) for value in raw_values.dropna() if str(value).strip()}):
            source_path = prepared_directory / Path(raw_path).name
            if not source_path.is_file():
                source_path = Path(raw_path)
            if not source_path.is_file():
                raise PreparedDatasetStorageError(
                    f"Prepared dataset file was not found: {source_path}"
                )
            try:
                shutil.copy2(source_path, staged_directory / source_path.name)
            except OSError as error:
                raise PreparedDatasetStorageError(
                    f"Could not copy prepared dataset file: {source_path}"
                ) from error


def run_configuration(args: argparse.Namespace) -> None:
    base_model, _classifier_head = resolve_model_configuration(args.model)
    root = Path(args.prepared_data_directory)
    manifest = normalize_manifest_dataset_identity(
        pd.read_csv(args.manifest, sep="\t", dtype=str)
    )
    rows = manifest[manifest["configuration_md5"].str.lower() == args.configuration_md5.lower()]
    if rows.empty:
        raise FileNotFoundError("configuration not found in manifest")
    device = resolve_device(args.device)
    seed_everything(args.seed, args.deterministic)

    result_path = Path(args.result_file or f"resultados_{args.model}.csv")
    details_path = Path(args.details_file or f"resultados_{args.model}_details.jsonl")
    regime = args.tabbert_regime if base_model == "tabbert" else ""
    completed = (
        set()
        if args.force_reprocess
        else load_completed_folds(
            result_path, args.configuration_md5, args.model, regime, args.threshold_strategy
        )
    )

    failed = 0
    local_staging_root = Path(args.local_staging_directory)
    # Todas as bases de uma configuracao sao buscadas do armazenamento
    # preparado (tipicamente o HD externo) uma unica vez aqui; os folds
    # processados a seguir leem apenas do diretorio local ja copiado.
    with stage_prepared_dataset_files(
        root, args.manifest, args.configuration_md5, local_staging_root
    ) as staged_directory:
        stage_transformer_artifacts(rows, staged_directory, root)
        for fold_index in sorted(rows["fold_index"].astype(int).unique()):
            if int(fold_index) in completed:
                continue
            fold_rows = rows[rows["fold_index"].astype(int) == fold_index]
            manifest_row = fold_rows.iloc[0]
            # MD5 do conteudo de cada arquivo desta dobra, para que a linha de
            # resultado aponte os bytes exatos de treino, validacao e teste.
            fold_split_md5 = {
                f"{row.split}_data": str(getattr(row, "content_md5", "") or "")
                for row in fold_rows.itertuples(index=False)
            }
            # Identidade de cada split pelas linhas brutas (vazia em manifestos
            # anteriores a coluna partition_md5).
            fold_partition_md5 = {
                f"{row.split}_data": str(getattr(row, "partition_md5", "") or "")
                for row in fold_rows.itertuples(index=False)
                if str(getattr(row, "partition_md5", "") or "") not in ("", "nan")
            }
            fold_unit_partition_md5 = {
                f"{row.split}_data": str(getattr(row, "unit_partition_md5", "") or "")
                for row in fold_rows.itertuples(index=False)
                if str(getattr(row, "unit_partition_md5", "") or "") not in ("", "nan")
            }
            fold_train_population_md5 = str(
                getattr(manifest_row, "train_population_md5", "") or ""
            ).replace("nan", "")
            fold_train_unit_population_md5 = str(
                getattr(manifest_row, "train_unit_population_md5", "") or ""
            ).replace("nan", "")
            settings = None
            try:
                data = load_fold_rows(rows, staged_directory, fold_index)
                vocabulary_path = staged_directory / Path(manifest_row["vocabulary_path"]).name
                metadata = json.loads(vocabulary_path.read_text(encoding="utf-8"))
                check_fold_partitions(data, metadata["target_column"], fold_index)
                field_count = len(metadata["categorical_columns"]) + len(metadata["numerical_columns"])
                if base_model == "ft-transformer":
                    # O modo indexado acrescenta um indicador de ausencia por atributo numerico.
                    field_count += len(metadata["numerical_columns"])
                settings = build_execution_settings(args, device, field_count)
                print(
                    f"execucao: dispositivo={settings.device_name} precisao={settings.precision} "
                    f"lote={settings.batch_size} workers={settings.num_workers} campos={field_count} "
                    f"semente={settings.seed} determinista={settings.deterministic}"
                )
                if base_model == "ft-transformer":
                    fold_results = run_ft_fold(
                        data,
                        metadata,
                        args,
                        settings,
                        load_fold_units(rows, staged_directory, fold_index),
                        load_fold_history_strata(rows, staged_directory, fold_index),
                    )
                else:
                    window_path = staged_directory / Path(manifest_row["window_index_path"]).name
                    windows = pd.read_csv(window_path)
                    fold_results = run_tabbert_fold(
                        data,
                        windows,
                        metadata,
                        args,
                        settings,
                        unit_column=get_dataset_definition(
                            manifest_row["database_type"]
                        ).evaluation_unit_column,
                    )
                if not fold_results:
                    raise ValueError("nenhum split foi avaliado nesta dobra")
                for fold_result in fold_results:
                    fold_result.update(settings.as_metadata())
            except PreparedDatasetStorageError:
                raise
            except Exception as error:
                print(
                    f"Erro na execucao: configuration_md5={args.configuration_md5} "
                    f"fold={fold_index} model={args.model}: {error}"
                )
                record_algorithm_failure_to_csv(
                    result_path,
                    "",
                    args.model,
                    manifest_row.get("categorical_encoding_strategy", ""),
                    manifest_row.get("fill_na_strategy", ""),
                    manifest_row.get("test_strategy", ""),
                    undersampling_technique=manifest_row.get("undersampling_technique"),
                    random_seed=manifest_row.get("random_seed", ""),
                    configuration_md5=args.configuration_md5,
                    database_type=manifest_row.get("database_type"),
                    regime=regime,
                    fold_index=int(fold_index),
                    threshold_strategy=args.threshold_strategy,
                    execution_metadata=settings.as_metadata() if settings is not None else None,
                    error=error,
                )
                failed += 1
                continue

            # Um `split_results` por unidade de avaliacao, com TODOS os splits
            # avaliados: o gravador emite uma linha por split, de modo que a
            # validacao chegue ao CSV ao lado do teste. Nas bases com unidade
            # (Home Credit), as linhas por cliente formam uma segunda entrada,
            # com limiar proprio.
            split_results_by_unit = {}
            sizes_by_unit = {}
            threshold_by_unit = {}
            details_by_split = {}
            row_sizes = {
                "train_data": len(data.get("train", [])),
                "validation_data": len(data.get("validation", [])),
                "test_data": len(data.get("test", [])),
            }
            for fold_result in fold_results:
                split_name = fold_result.pop("evaluation_split")
                unit = fold_result.pop("evaluation_unit", EVALUATION_UNIT_ROW)
                threshold_by_unit[unit] = {
                    "classification_threshold": fold_result.pop("classification_threshold"),
                    "threshold_strategy": fold_result.pop("threshold_strategy"),
                    "model_hyperparameters": fold_result.pop("model_hyperparameters"),
                    "training_summary": fold_result.pop("training_summary"),
                    # Tempos da dobra (item G18), iguais em todos os splits.
                    "fit_time": fold_result.pop("fit_time", ""),
                    "scoring_time": fold_result.pop("scoring_time", ""),
                    "pretraining_time": fold_result.pop("pretraining_time", ""),
                }
                transaction_predictions = fold_result.pop("transaction_predictions")
                history_strata = fold_result.pop("history_strata", None)
                sizes = sizes_by_unit.setdefault(unit, dict(row_sizes))
                if unit == EVALUATION_UNIT_ROW:
                    details_by_split[split_name] = (transaction_predictions, history_strata)
                else:
                    sizes[f"{split_name}_data"] = fold_result.pop("evaluated_units")
                split_results_by_unit.setdefault(unit, {})[f"{split_name}_data"] = {
                    "metrics": fold_result
                }

            execution_result = {
                "algorithm": args.model,
                "undersampling_technique": manifest_row.get("undersampling_technique"),
                "train_test_strategy": manifest_row.get("test_strategy"),
                "results": [
                    {
                        "test_info_index": int(fold_index),
                        "fold_index": int(fold_index),
                        "evaluation_unit": unit,
                        "split_md5": fold_split_md5,
                        "partition_md5": fold_partition_md5,
                        "train_population_md5": fold_train_population_md5,
                        "unit_partition_md5": fold_unit_partition_md5,
                        "train_unit_population_md5": fold_train_unit_population_md5,
                        "split_sizes": sizes_by_unit[unit],
                        "split_results": split_results,
                        **threshold_by_unit[unit],
                    }
                    for unit, split_results in split_results_by_unit.items()
                ],
            }
            record_algorithm_result_to_csv(
                result_path,
                "",
                execution_result,
                manifest_row.get("categorical_encoding_strategy", ""),
                manifest_row.get("fill_na_strategy", ""),
                None,
                configuration_md5=args.configuration_md5,
                database_type=manifest_row.get("database_type"),
                regime=regime,
                random_seed=manifest_row.get("random_seed", ""),
            )

            # Um registro de detalhes por split, com a mesma chave composta da
            # linha do CSV (configuracao, dobra, algoritmo, regime, split).
            for split_name, (
                transaction_predictions,
                history_strata,
            ) in details_by_split.items():
                details_record = {
                    "configuration_md5": args.configuration_md5,
                    "dataset_id": manifest_row.get("dataset_id"),
                    "dataset_name": manifest_row.get("dataset_name"),
                    "test_strategy": manifest_row.get("test_strategy"),
                    "random_seed": manifest_row.get("random_seed"),
                    "fold_index": int(fold_index),
                    "algorithm": args.model,
                    "regime": regime,
                    "threshold_strategy": args.threshold_strategy,
                    "split": split_name,
                    "transaction_predictions": transaction_predictions,
                }
                if history_strata is not None:
                    details_record["history_strata"] = history_strata
                append_details_record(details_path, details_record)

    print(f"Failed executions: {failed}")


def main() -> None:
    execution_provenance.guard_environment_or_exit()
    args = parse_arguments()
    try:
        run_configuration(args)
    except PreparedDatasetStorageError as error:
        print(f"Prepared storage failure: {error}")
        raise SystemExit(PREPARED_STORAGE_FAILURE_EXIT_CODE) from error


if __name__ == "__main__":
    main()
