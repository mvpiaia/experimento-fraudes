"""Motivo das falhas de execucao (item G17 da revisao).

Toda linha ``execution_status=failure`` grava uma categoria fechada e a
mensagem da excecao, para que a tabela de execucoes previstas x realizadas do
apendice diga por que cada execucao falhou. As categorias:

- ``memoria``: falta de memoria em CPU ou GPU;
- ``convergencia``: o ajuste nao convergiu ou produziu valores nao finitos;
- ``particao-vazia``: treino, validacao ou teste sem linhas, ou treino com uma
  so classe (por exemplo, depois da remocao de incompletos);
- ``armazenamento``: os arquivos preparados ficaram inacessiveis;
- ``outra``: o resto; a mensagem comeca com o tipo da excecao.

Um processo morto pelo kernel (OOM killer) nao chega a um ``except``: esse caso
fica no registro do executor (``run_storage_common.sh``).
"""

from __future__ import annotations

import pandas as pd

FAILURE_MEMORY = "memoria"
FAILURE_CONVERGENCE = "convergencia"
FAILURE_EMPTY_PARTITION = "particao-vazia"
FAILURE_STORAGE = "armazenamento"
FAILURE_OTHER = "outra"
FAILURE_CATEGORIES = (
    FAILURE_MEMORY,
    FAILURE_CONVERGENCE,
    FAILURE_EMPTY_PARTITION,
    FAILURE_STORAGE,
    FAILURE_OTHER,
)
FAILURE_MESSAGE_MAX_LENGTH = 300

# Bibliotecas nativas (XGBoost, CatBoost, PyTorch) relatam falta de memoria em
# excecoes genericas; o texto e o unico sinal.
_MEMORY_MARKERS = ("out of memory", "bad_alloc", "cannot allocate memory", "outofmemory")
_CONVERGENCE_MARKERS = (
    "did not converge",
    "failed to converge",
    "input contains nan",
    "input contains infinity",
    "singular matrix",
)


class EmptyPartitionError(ValueError):
    """Um split da dobra nao tem linhas, ou o treino tem uma so classe."""


def _type_names(error: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(error).__mro__}


def classify_failure(error: BaseException) -> str:
    """Categoria fechada da falha; ver o docstring do modulo."""
    names = _type_names(error)
    text = str(error).lower()
    if isinstance(error, EmptyPartitionError):
        return FAILURE_EMPTY_PARTITION
    if "PreparedDatasetStorageError" in names:
        return FAILURE_STORAGE
    if isinstance(error, MemoryError) or "OutOfMemoryError" in names or any(
        marker in text for marker in _MEMORY_MARKERS
    ):
        return FAILURE_MEMORY
    if names & {"ConvergenceWarning", "LinAlgError", "FloatingPointError"} or any(
        marker in text for marker in _CONVERGENCE_MARKERS
    ):
        return FAILURE_CONVERGENCE
    return FAILURE_OTHER


def describe_failure(error: BaseException) -> str:
    """Primeira linha da mensagem, truncada; em ``outra``, prefixada pelo tipo."""
    lines = str(error).strip().splitlines()
    message = lines[0] if lines else ""
    if classify_failure(error) == FAILURE_OTHER:
        message = f"{type(error).__name__}: {message}" if message else type(error).__name__
    return message[:FAILURE_MESSAGE_MAX_LENGTH]


def failure_columns(error: BaseException | None) -> dict[str, str]:
    """Colunas ``failure_category``/``failure_message`` de uma linha de falha."""
    if error is None:
        return {"failure_category": FAILURE_OTHER, "failure_message": ""}
    return {
        "failure_category": classify_failure(error),
        "failure_message": describe_failure(error),
    }


def check_fold_partitions(
    splits: dict[str, pd.DataFrame | None],
    target_column: str,
    fold_index: int | str = "",
) -> None:
    """Interrompe antes do ajuste se um split estiver vazio ou o treino tiver uma classe.

    ``splits`` mapeia o nome do split (``train_data`` ou ``train``, etc.) para o
    quadro lido. Sem esta verificacao, um treino de uma classe falha com
    mensagens diferentes em cada biblioteca e nao pode ser agregado.
    """
    for name, frame in splits.items():
        if frame is None:
            continue
        label = name.removesuffix("_data")
        if frame.empty:
            raise EmptyPartitionError(f"dobra {fold_index}: split {label} sem linhas")
        if label == "train" and target_column in frame.columns:
            classes = frame[target_column].dropna().nunique()
            if classes < 2:
                raise EmptyPartitionError(
                    f"dobra {fold_index}: split train com classe unica "
                    f"({classes} classe(s) em {target_column})"
                )


def check_loaded_folds(test_data_information: dict, target_column: str) -> None:
    """Aplica ``check_fold_partitions`` a cada dobra de ``load_prepared_data_splits``."""
    for fold in test_data_information.get("test_info", []):
        splits = {
            name: fold.get(name)
            for name in ("train_data", "validation_data", "test_data")
        }
        check_fold_partitions(splits, target_column, fold.get("fold_index", ""))
