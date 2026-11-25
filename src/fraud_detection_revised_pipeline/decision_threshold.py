"""Escolha do limiar de decisao na validacao (item G11-i da revisao).

Depois de reamostrar o treino para 1:1, ou de ponderar as classes na perda, as
probabilidades previstas deixam de corresponder a prevalencia real (0,17% a
8,7%), e o limiar de 0,5 vira uma escolha arbitraria que produz pontos de
operacao extremos. Aqui o limiar passa a ser o que maximiza F2 sobre as
previsoes de validacao, e e esse limiar que se aplica ao teste.

Onde o limiar e escolhido depende do protocolo:

``validation-f2``
    A dobra tem teste (``stratified-holdout``, ``temporal-holdout`` e a
    particao cronologica do TabBERT). O limiar vem da validacao da propria
    dobra e e aplicado ao teste.
``cross-fold-validation-f2``
    A dobra nao tem teste (``stratified-k-fold``): a validacao E o conjunto
    avaliado. Escolher o limiar nela inflaria as metricas, entao o limiar de
    cada dobra vem das previsoes de validacao das OUTRAS dobras, reunidas.
``fixed``
    Limiar informado explicitamente, ou o padrao de 0,5 quando a validacao nao
    tem as duas classes e o F2 nao se define.

F2 pesa a revocacao quatro vezes mais que a precisao, o que corresponde ao
custo assimetrico da fraude: deixar passar uma fraude custa mais que revisar
uma transacao legitima.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import precision_recall_curve


THRESHOLD_STRATEGY_FIXED = "fixed"
THRESHOLD_STRATEGY_VALIDATION_F2 = "validation-f2"
THRESHOLD_STRATEGY_CROSS_FOLD_F2 = "cross-fold-validation-f2"
# Estrategias que se pedem na linha de comando. A variante entre dobras nao se
# pede: ela e consequencia de o protocolo nao ter teste.
REQUESTABLE_THRESHOLD_STRATEGIES = (
    THRESHOLD_STRATEGY_VALIDATION_F2,
    THRESHOLD_STRATEGY_FIXED,
)
DEFAULT_THRESHOLD_STRATEGY = THRESHOLD_STRATEGY_VALIDATION_F2
SELECTED_THRESHOLD_STRATEGIES = (
    THRESHOLD_STRATEGY_VALIDATION_F2,
    THRESHOLD_STRATEGY_CROSS_FOLD_F2,
)
F_BETA = 2.0


@dataclass(frozen=True)
class FoldThreshold:
    threshold: float
    strategy: str


def select_f2_threshold(binary_targets, positive_scores) -> float | None:
    """Limiar que maximiza F2; None quando falta uma das classes.

    Prediz positivo quando ``score >= limiar``, a mesma convencao de todos os
    blocos. Em empate fica o menor limiar, que favorece a revocacao.
    """
    targets = np.asarray(binary_targets, dtype=int).reshape(-1)
    scores = np.asarray(positive_scores, dtype=float).reshape(-1)
    positives = int(targets.sum())
    if positives == 0 or positives == len(targets):
        return None
    precision, recall, thresholds = precision_recall_curve(targets, scores)
    # A ultima posicao da curva (precisao 1, revocacao 0) nao tem limiar.
    precision, recall = precision[:-1], recall[:-1]
    beta_squared = F_BETA**2
    denominator = beta_squared * precision + recall
    f_beta = np.divide(
        (1 + beta_squared) * precision * recall,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(f_beta))])


def select_fold_thresholds(
    validation_sets: list[tuple[np.ndarray, np.ndarray] | None],
    fold_has_test: list[bool],
    requested_strategy: str = DEFAULT_THRESHOLD_STRATEGY,
    fixed_threshold: float = 0.5,
) -> list[FoldThreshold]:
    """Limiar de cada dobra a partir de (alvos binarios, scores) da validacao."""
    fixed = FoldThreshold(float(fixed_threshold), THRESHOLD_STRATEGY_FIXED)
    if requested_strategy == THRESHOLD_STRATEGY_FIXED:
        return [fixed] * len(validation_sets)
    if requested_strategy != THRESHOLD_STRATEGY_VALIDATION_F2:
        raise ValueError(
            f"threshold_strategy must be one of {REQUESTABLE_THRESHOLD_STRATEGIES}, "
            f"got {requested_strategy!r}"
        )

    fold_thresholds = []
    for fold_position, own_validation in enumerate(validation_sets):
        if fold_has_test[fold_position] or len(validation_sets) == 1:
            source = [own_validation]
            strategy = THRESHOLD_STRATEGY_VALIDATION_F2
        else:
            source = [
                validation
                for other_position, validation in enumerate(validation_sets)
                if other_position != fold_position
            ]
            strategy = THRESHOLD_STRATEGY_CROSS_FOLD_F2
        source = [validation for validation in source if validation is not None]
        threshold = None
        if source:
            threshold = select_f2_threshold(
                np.concatenate([targets for targets, _scores in source]),
                np.concatenate([scores for _targets, scores in source]),
            )
        fold_thresholds.append(
            fixed if threshold is None else FoldThreshold(threshold, strategy)
        )
    return fold_thresholds
