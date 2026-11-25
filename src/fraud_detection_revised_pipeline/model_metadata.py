"""Hiperparametros efetivos e resumo do treino de cada modelo ajustado.

A linha de resultado registrava a configuracao de preparacao, mas nao o modelo:
valores escolhidos pela propria biblioteca, como o learning_rate que o CatBoost
calcula a partir do tamanho da base, ficavam invisiveis, e dois resultados
diferentes nao podiam ser explicados depois. Aqui cada modelo e descrito pelo
que ele expoe, e nao pelo nome do algoritmo, de modo que um estimador novo com
a interface do scikit-learn ja e coberto sem mudar este modulo.
"""

from __future__ import annotations

import inspect
import json
import math
from typing import Any

import numpy as np


# Atributos que o ajuste produz e que explicam o treino sem depender do
# algoritmo: iteracoes ate convergir, arvores efetivas, atributos vistos,
# tamanho da subamostra do Isolation Forest.
TRAINING_SUMMARY_ATTRIBUTES = (
    "n_iter_",
    "n_features_in_",
    "max_samples_",
    "tree_count_",
)


def _xgboost_effective_parameters(model) -> dict[str, Any]:
    """Configuracao efetiva do booster, achatada em um unico dicionario.

    O get_params() do XGBoost devolve None para todo valor deixado no padrao;
    o valor efetivo so aparece na configuracao do booster ja treinado.
    """
    configuration = json.loads(model.get_booster().save_config())
    learner = configuration.get("learner", {})
    effective: dict[str, Any] = {}
    for section in ("generic_param", "learner_train_param"):
        effective.update(learner.get(section, {}))
    for name, values in learner.get("gradient_booster", {}).items():
        if name.endswith("_param") and isinstance(values, dict):
            effective.update(values)
    return effective


def describe_model_hyperparameters(model) -> dict[str, Any]:
    """Hiperparametros efetivos de um modelo ajustado.

    A ordem dos testes importa: o CatBoost tambem tem get_params(), mas so o
    get_all_params() traz os padroes que ele resolveu sozinho.
    """
    if model is None:
        return {}
    if hasattr(model, "get_all_params"):
        try:
            return dict(model.get_all_params())
        except Exception:
            # Antes do ajuste o CatBoost nao tem os parametros resolvidos.
            pass
    if hasattr(model, "get_params"):
        parameters = dict(model.get_params(deep=False))
        if hasattr(model, "get_booster"):
            try:
                effective = _xgboost_effective_parameters(model)
            except Exception:
                effective = {}
            for name, value in parameters.items():
                if value is None and name in effective:
                    parameters[name] = effective[name]
        return parameters
    return {}


def summarize_training(model) -> dict[str, Any]:
    """O que o ajuste produziu, quando o modelo expoe; vazio caso contrario."""
    if model is None:
        return {}
    summary: dict[str, Any] = {}
    for attribute in TRAINING_SUMMARY_ATTRIBUTES:
        try:
            value = getattr(model, attribute)
        except Exception:
            continue
        if value is not None and not callable(value):
            summary[attribute] = value
    if hasattr(model, "estimators_"):
        summary["estimator_count"] = len(model.estimators_)
    if hasattr(model, "get_best_iteration"):
        best_iteration = model.get_best_iteration()
        if best_iteration is not None:
            summary["best_iteration"] = best_iteration
    if hasattr(model, "get_booster"):
        try:
            summary["boosted_rounds"] = model.get_booster().num_boosted_rounds()
        except Exception:
            pass
    return summary


def describe_torch_constructor(module_class, **explicit_arguments) -> dict[str, Any]:
    """Argumentos do construtor de um modulo PyTorch, padroes incluidos.

    Os modulos nao guardam os argumentos com que foram criados; a assinatura
    do construtor da os padroes, e os argumentos explicitos os substituem.
    """
    signature = inspect.signature(module_class.__init__)
    arguments = {
        name: parameter.default
        for name, parameter in signature.parameters.items()
        if name != "self" and parameter.default is not inspect.Parameter.empty
    }
    arguments.update(explicit_arguments)
    return arguments


def count_trainable_parameters(module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def _json_compatible(value):
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_compatible(value.tolist())
    if isinstance(value, np.generic):
        return _json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        # NaN e infinito nao sao JSON valido; o missing=nan do XGBoost cai aqui.
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def serialize_metadata(values: dict[str, Any] | None) -> str:
    """JSON deterministico: mesmas entradas, mesma celula de CSV."""
    return json.dumps(
        _json_compatible(values or {}),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
