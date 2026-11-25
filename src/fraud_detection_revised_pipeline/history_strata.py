"""Metricas por estrato de historico (item C3.19 da revisao).

O TabBERT avalia cada transacao com o historico que o titular tem ate ela, e
os resultados sao cortados pelo tamanho desse historico (``base_history_length``).
O revisor apontou que esse corte mistura efeitos: as primeiras transacoes de um
titular sao tambem as mais antigas, e precisao e F1 dependem da prevalencia de
cada estrato. Por isso o mesmo corte e aplicado aos modelos que nao usam
historico (FT-Transformer, CatBoost e XGBoost), sobre as mesmas transacoes, e
cada estrato registra o numero de transacoes, de fraudes e a prevalencia.

Este modulo nao depende do PyTorch: e usado pelos executores de boosting e de
Transformers e pela verificacao dos resultados.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

HISTORY_STRATUM_NAME = "base_history_length"
STRATUM_COUNT_KEYS = ("transactions", "positives", "prevalence")


def binary_metrics(
    targets: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Metricas da classe positiva num limiar; AUCs ausentes sem as duas classes."""
    targets = np.asarray(targets).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    result = {
        "recall": float(recall_score(targets, predictions, zero_division=0)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "f_beta_2": float(fbeta_score(targets, predictions, beta=2, zero_division=0)),
        "accuracy": float(accuracy_score(targets, predictions)),
        "false_positive_rate": float(
            ((predictions == 1) & (targets == 0)).sum() / max((targets == 0).sum(), 1)
        ),
    }
    has_both_classes = len(np.unique(targets)) > 1
    result["roc_auc"] = (
        float(roc_auc_score(targets, probabilities)) if has_both_classes else float("nan")
    )
    # AUC-PR: mais informativa que a ROC sob forte desbalanceamento, mas
    # dependente da prevalencia, entao compara algoritmos dentro de uma base.
    result["auc_pr"] = (
        float(average_precision_score(targets, probabilities))
        if has_both_classes
        else float("nan")
    )
    return result


def stratum_metrics(
    targets, probabilities, strata, threshold: float
) -> dict[str, dict[str, float]]:
    """Metricas de cada estrato, com transacoes, fraudes e prevalencia.

    Um estrato sem fraude (ou sem transacao legitima) tem as AUCs ausentes,
    e nao e falha: as demais metricas continuam definidas.
    """
    targets = np.asarray(targets).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    strata = np.asarray(strata).astype(int)
    if not len(targets) == len(probabilities) == len(strata):
        raise ValueError(
            f"strata do not align with the scored rows: {len(strata)} strata, "
            f"{len(targets)} targets, {len(probabilities)} scores"
        )
    result = {}
    for stratum in sorted(set(strata.tolist())):
        selected = strata == stratum
        stratum_targets = targets[selected]
        positives = int(stratum_targets.sum())
        result[str(stratum)] = {
            **binary_metrics(stratum_targets, probabilities[selected], threshold),
            "transactions": int(selected.sum()),
            "positives": positives,
            "prevalence": positives / int(selected.sum()),
        }
    return result


def append_history_strata_record(path: str | Path, record: dict) -> None:
    """Acrescenta uma linha ao arquivo companheiro ``*_history_strata.jsonl``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def history_strata_file(results_directory: str | Path, algorithm: str) -> Path:
    """Arquivo companheiro de estratos do boosting, ao lado do CSV de resultados."""
    return Path(results_directory) / f"resultados_{algorithm}_history_strata.jsonl"


# Chave que identifica a mesma avaliacao entre algoritmos: as mesmas
# transacoes, que precisam ter as mesmas contagens em cada estrato.
# Os registros (uma linha por configuracao, dobra, algoritmo, braco e split)
# trazem ``history_strata`` como {nome do estrato: {estrato: metricas}}; todos
# os modelos gravam ``base_history_length``, e o TabBERT tambem
# ``causal_context_length``.
CONSISTENCY_KEY_COLUMNS = ("dataset_id", "test_strategy", "random_seed", "fold_index", "split")


def strata_count_rows(records: list[dict]) -> pd.DataFrame:
    """Uma linha por registro e estrato, com as contagens (para a checagem R6)."""
    rows = []
    for record in records:
        strata = (record.get("history_strata") or {}).get(HISTORY_STRATUM_NAME) or {}
        for stratum, values in strata.items():
            rows.append(
                {
                    **{column: record.get(column) for column in CONSISTENCY_KEY_COLUMNS},
                    "algorithm": record.get("algorithm"),
                    "stratum": int(stratum),
                    "transactions": int(values["transactions"]),
                    "positives": int(values["positives"]),
                }
            )
    return pd.DataFrame(rows)


def inconsistent_strata(records: list[dict]) -> pd.DataFrame:
    """Estratos cujas contagens divergem entre algoritmos na mesma avaliacao."""
    counts = strata_count_rows(records)
    if counts.empty:
        return counts
    group_columns = [*CONSISTENCY_KEY_COLUMNS, "stratum"]
    counts[list(CONSISTENCY_KEY_COLUMNS)] = counts[list(CONSISTENCY_KEY_COLUMNS)].astype(str)
    spread = counts.groupby(group_columns).agg(
        transactions=("transactions", "nunique"),
        positives=("positives", "nunique"),
        algorithms=("algorithm", lambda values: ", ".join(sorted(set(values)))),
    )
    divergent = spread[(spread["transactions"] > 1) | (spread["positives"] > 1)]
    return divergent.reset_index()


def read_history_strata_records(paths) -> list[dict]:
    """Registros com estratos de varios ``.jsonl`` (companheiros do boosting e
    detalhes dos Transformers), sem as predicoes por transacao."""
    records = []
    for path in paths:
        with Path(path).open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("history_strata"):
                    continue
                record.pop("transaction_predictions", None)
                records.append(record)
    return records


def history_strata_paths(results_directory: str | Path) -> list[Path]:
    """Arquivos que podem trazer estratos ao lado dos CSVs de resultado."""
    directory = Path(results_directory)
    return sorted(
        {
            *directory.glob("resultados_*_history_strata.jsonl"),
            *directory.glob("resultados_*_details.jsonl"),
        }
    )
