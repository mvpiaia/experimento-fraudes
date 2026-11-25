'''
Tratamento de dados das diversas bases consideradas para a realização de 3 algoritmos propostos:
- regressao logística
- naive bayes
- knn

criação: 2026.05.24

HISTÓRICO DE MODIFICAÇÕES
2026.05.24 - criação do código                          - piaia

'''

import ast
import base64
import hashlib
import inspect
import json
from contextlib import contextmanager
from pathlib import Path
import shutil
import tempfile

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    fbeta_score,
    f1_score,
    multilabel_confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler, TargetEncoder
from scipy import stats
import statsmodels.api as sm
import os
import argparse
import time

import data_partitioning
import execution_provenance
import experiment_datasets
import missing_screening
import preparation_report
from model_metadata import (
    describe_model_hyperparameters,
    serialize_metadata,
    summarize_training,
)
from decision_threshold import (
    DEFAULT_THRESHOLD_STRATEGY,
    REQUESTABLE_THRESHOLD_STRATEGIES,
    SELECTED_THRESHOLD_STRATEGIES,
    THRESHOLD_STRATEGY_FIXED,
    select_fold_thresholds,
)
from execution_failures import check_loaded_folds, failure_columns
from prepared_dataset_storage import (
    CATEGORICAL_HANDLING_MODE_NATIVE,
    CATEGORICAL_HANDLING_MODE_TRANSFORMER,
    DEFAULT_DATA_DIRECTORY,
    DEFAULT_PREPARED_DATA_DIRECTORY,
    DEFAULT_RANDOM_SEED,
    EVALUATION_UNIT_SPLITS,
    FIRST_FOLD_INDEX,
    SPLIT_FILE_LABELS,
    build_prepared_dataset_md5,
    column_registry_path,
    compute_file_content_md5,
    evaluation_unit_path,
    history_strata_path,
    is_compressed_split,
    load_prepared_data_splits,
    materialize_prepared_split,
    normalize_manifest_dataset_identity,
    prepared_split_path,
    normalize_categorical_encoding_strategy,
    normalize_categorical_handling_mode,
)

# Limiar de decisao do modo fixo e de reserva. Por padrao o limiar e escolhido
# na validacao, maximizando F2, e aplicado ao teste (item G11-i da revisao):
# depois de reamostrar ou ponderar, as probabilidades deixam de corresponder a
# prevalencia real (0,17% a 8,7%) e 0,5 vira uma escolha arbitraria. Ver
# decision_threshold.py. 0,5 so vale no modo fixed ou quando a validacao nao
# tem as duas classes.
DEFAULT_CLASSIFICATION_THRESHOLD = 0.5

# Ponderacao de classe na perda (itens G10 e C3.14 da revisao). Sao bracos do
# eixo de balanceamento, ao lado de "sem reamostragem", RUS, RUS+SMOTE e Tomek
# Links, e nao um eixo cruzado com eles:
#
#   none           sem ponderacao (o unico nivel usado nos bracos reamostrados);
#   sqrt-balanced  braco de ponderacao moderada, com sqrt(R);
#   balanced       braco de ponderacao completa, com R.
#
# Os dois bracos ponderados so rodam sobre configuracoes SEM reamostragem. Num
# treino ja reamostrado para 1:1, R ~ 1 e os pesos repetiriam o braco none; com
# Tomek Links a razao mal muda e a ponderacao corrigiria o desbalanceamento duas
# vezes. A razao R = negativos/positivos e calculada EXCLUSIVAMENTE sobre o
# treino de cada dobra, que nesses bracos tem a prevalencia natural; nunca
# sobre o teste nem sobre a base inteira. Os mesmos niveis valem para
# regressao logistica, XGBoost e CatBoost.
#
# A raiz quadrada aumenta a importancia das fraudes sem fazer cada fraude
# compensar integralmente todas as observacoes legitimas excedentes, o que e
# util quando o balanceamento completo eleva demais os falsos positivos. A
# literatura fundamenta o aprendizado sensivel a custos e a ponderacao por
# frequencia (King e Zeng, 2001; Elkan, 2001; He e Garcia, 2009; Cui et al.,
# 2019), mas nao demonstra que R ou sqrt(R) sejam universalmente otimos: por
# isso os tres valores entram como grade, e nao como escolha.
CLASS_WEIGHT_NONE = "none"
CLASS_WEIGHT_SQRT_BALANCED = "sqrt-balanced"
CLASS_WEIGHT_BALANCED = "balanced"
CLASS_WEIGHT_INTENSITIES = (
    CLASS_WEIGHT_NONE,
    CLASS_WEIGHT_SQRT_BALANCED,
    CLASS_WEIGHT_BALANCED,
)
DEFAULT_CLASS_WEIGHT_INTENSITY = CLASS_WEIGHT_NONE


def is_resampled(undersampling_technique):
    """True quando a configuracao reamostra o treino."""
    return str(undersampling_technique or "none").strip().lower() not in ("", "none")


def validate_class_weight_arm(class_weight_intensity, undersampling_technique):
    """Rejeita ponderacao combinada com reamostragem: ela e um braco proprio."""
    intensity = str(class_weight_intensity or CLASS_WEIGHT_NONE).strip().lower()
    if intensity != CLASS_WEIGHT_NONE and is_resampled(undersampling_technique):
        raise ValueError(
            f"class_weight_intensity={intensity} is a balancing arm of its own and "
            "only runs on configurations without resampling, got "
            f"undersampling_technique={undersampling_technique}"
        )


def compute_class_imbalance_ratio(train_target, class_values):
    """R = negativos/positivos, medido so no treinamento da dobra."""
    values = pd.Series(train_target).dropna()
    if len(class_values) != 2:
        return None
    negative_value, positive_value = class_values[0], class_values[-1]
    positive_count = int((values == positive_value).sum())
    negative_count = int((values == negative_value).sum())
    if positive_count == 0:
        return None
    return negative_count / positive_count


def resolve_logistic_class_weight(intensity, imbalance_ratio, class_values):
    """Traduz a intensidade no parametro class_weight do scikit-learn.

    Na intensidade completa usa-se a string ``balanced``, e nao o dicionario
    equivalente: as duas produzem a mesma razao relativa entre as classes, mas
    ``balanced`` normaliza os pesos, e a escala global da funcao de perda
    interage com a regularizacao da regressao logistica.
    """
    normalized = str(intensity or CLASS_WEIGHT_NONE).strip().lower()
    if normalized not in CLASS_WEIGHT_INTENSITIES:
        raise ValueError(
            f"class_weight_intensity must be one of {CLASS_WEIGHT_INTENSITIES}, "
            f"got {intensity!r}"
        )
    if normalized == CLASS_WEIGHT_NONE or imbalance_ratio is None:
        return None
    if normalized == CLASS_WEIGHT_BALANCED:
        return "balanced"
    negative_value, positive_value = class_values[0], class_values[-1]
    return {negative_value: 1.0, positive_value: float(imbalance_ratio) ** 0.5}

RELATIVE_DATA_PATH = str(DEFAULT_DATA_DIRECTORY)
CURRENT_PATH = os.getcwd()

_GPU_DEPENDENCIES = None
PREPARED_STORAGE_FAILURE_EXIT_CODE = 75


class PreparedDatasetStorageError(OSError):
    """Signal that prepared files could not be read from external storage."""


def _categorical_columns(dataframe):
    return dataframe.select_dtypes(include=['object', 'category', 'string']).columns


def _prepare_categorical_data(dataframe):
    prepared_dataframe = dataframe.copy()
    categorical_columns = _categorical_columns(prepared_dataframe)
    for column in categorical_columns:
        prepared_dataframe[column] = (
            prepared_dataframe[column]
            .astype('string')
            .fillna('__missing__')
        )
    return prepared_dataframe


def _default_fill_value(series):
    if pd.api.types.is_numeric_dtype(series):
        return 0
    return '__missing__'


def _load_gpu_dependencies():
    global _GPU_DEPENDENCIES

    if _GPU_DEPENDENCIES is not None:
        return _GPU_DEPENDENCIES

    try:
        import cupy as cp
        from cuml.linear_model import LinearRegression as CumulLinearRegression
    except ImportError as error:
        _GPU_DEPENDENCIES = {
            "available": False,
            "error": error,
        }
        return _GPU_DEPENDENCIES

    _GPU_DEPENDENCIES = {
        "available": True,
        "cp": cp,
        "linear_regression_class": CumulLinearRegression,
    }
    return _GPU_DEPENDENCIES


def _resolve_execution_device(execution_device):
    normalized_execution_device = str(execution_device or "auto").strip().lower()
    if normalized_execution_device not in {"auto", "cpu", "gpu"}:
        raise ValueError("execution_device must be one of: auto, cpu, gpu")

    if normalized_execution_device == "cpu":
        return {"device": "cpu"}

    gpu_dependencies = _load_gpu_dependencies()
    if gpu_dependencies["available"]:
        return {
            "device": "gpu",
            "cp": gpu_dependencies["cp"],
            "linear_regression_class": gpu_dependencies["linear_regression_class"],
        }

    if normalized_execution_device == "gpu":
        raise RuntimeError(
            "GPU execution was requested, but CuPy/cuML are not available in the environment"
        ) from gpu_dependencies["error"]

    return {"device": "cpu"}


def _to_numpy_array(values):
    if isinstance(values, np.ndarray):
        return values
    if hasattr(values, "to_numpy"):
        return values.to_numpy()
    if hasattr(values, "get"):
        return values.get()
    return np.asarray(values)


def _to_gpu_array(data, cp):
    if isinstance(data, pd.DataFrame):
        return cp.asarray(data.to_numpy(dtype=float))
    if isinstance(data, pd.Series):
        return cp.asarray(data.to_numpy(dtype=float))
    return cp.asarray(np.asarray(data, dtype=float))


def _fit_linear_regression_model(train_features, train_target, execution_backend):
    if execution_backend["device"] == "gpu":
        model = execution_backend["linear_regression_class"]()
        model.fit(
            _to_gpu_array(train_features, execution_backend["cp"]),
            _to_gpu_array(train_target, execution_backend["cp"]),
        )
        return model

    model = LinearRegression()
    model.fit(train_features, train_target)
    return model


def _predict_linear_regression_values(model, features, execution_backend):
    if execution_backend["device"] == "gpu":
        predicted_values = model.predict(
            _to_gpu_array(features, execution_backend["cp"])
        )
        return _to_numpy_array(predicted_values).ravel()

    return np.asarray(model.predict(features)).ravel()


def _model_intercept_to_scalar(model):
    intercept_values = _to_numpy_array(model.intercept_).ravel()
    if intercept_values.size == 0:
        return 0.0
    return float(intercept_values[0])


def _model_coefficients_to_series(model, feature_columns):
    coefficients = _to_numpy_array(model.coef_).ravel()
    return pd.Series(coefficients, index=feature_columns)


def _resolve_fill_value(series, fillna_strategy):
    if fillna_strategy == 'mean':
        fill_value = series.mean() if pd.api.types.is_numeric_dtype(series) else np.nan
    elif fillna_strategy == 'median':
        fill_value = series.median() if pd.api.types.is_numeric_dtype(series) else np.nan
    elif fillna_strategy == 'mode':
        mode_values = series.mode(dropna=True)
        fill_value = mode_values.iloc[0] if not mode_values.empty else np.nan
    else:
        fill_value = np.nan

    if pd.isna(fill_value):
        return _default_fill_value(series)
    return fill_value


# definir a estrategia de codificação dos dados categóricos
def categorical_encoding_strategy_operation(merged_data, categorical_encoding_strategy, target_column):
    normalized_strategy = _normalize_categorical_encoding_strategy(categorical_encoding_strategy)
    if normalized_strategy == 'deferred':
        # Preserve raw missing values until each training split has fitted its
        # imputation values.
        return merged_data.copy()

    prepared_data = _prepare_categorical_data(merged_data)
    match normalized_strategy:
        case 'onehot':
            merged_data_encoded = pd.get_dummies(prepared_data, dtype='int')
        case 'label':
            from sklearn.preprocessing import LabelEncoder
            label_encoders = {}
            merged_data_encoded = prepared_data.copy()
            for column in _categorical_columns(prepared_data):
                label_encoders[column] = LabelEncoder()
                merged_data_encoded[column] = label_encoders[column].fit_transform(
                    prepared_data[column]
                )
        case 'ordinal':
            from sklearn.preprocessing import OrdinalEncoder
            ordinal_encoder = OrdinalEncoder(
                handle_unknown='use_encoded_value',
                unknown_value=-1,
            )
            merged_data_encoded = prepared_data.copy()
            for column in _categorical_columns(prepared_data):
                merged_data_encoded[column] = ordinal_encoder.fit_transform(
                    prepared_data[[column]]
                ).ravel()
        case 'target':


            if target_column not in prepared_data.columns:
                raise KeyError(f"Target column '{target_column}' was not found")

            merged_data_encoded = prepared_data.copy()
            categorical_feature_columns = _categorical_columns(
                prepared_data.drop(columns=[target_column])
            )
            if not categorical_feature_columns.empty:
                target_encoder = TargetEncoder(target_type='binary')
                merged_data_encoded[categorical_feature_columns] = (
                    target_encoder.fit_transform(
                        prepared_data[categorical_feature_columns],
                        prepared_data[target_column],
                    )
                )
        case _ if normalized_strategy in SUPERVISED_CATEGORICAL_ENCODINGS:
            if target_column not in prepared_data.columns:
                raise KeyError(f"Target column '{target_column}' was not found")
            categorical_feature_columns = list(
                _categorical_columns(prepared_data.drop(columns=[target_column]))
            )
            merged_data_encoded = prepared_data.copy()
            if categorical_feature_columns:
                encoder = _build_supervised_categorical_encoder(
                    normalized_strategy, categorical_feature_columns
                )
                merged_data_encoded[categorical_feature_columns] = encoder.fit_transform(
                    prepared_data[categorical_feature_columns], prepared_data[target_column]
                )
                merged_data_encoded = _finite_encoded_dataframe(merged_data_encoded)
        case None:
            merged_data_encoded = prepared_data.drop(columns=_categorical_columns(prepared_data)).copy()
        case _:
            raise ValueError(f"Unsupported categorical encoding strategy '{categorical_encoding_strategy}'")
    return merged_data_encoded

# estratégia de preenchimento dos dados faltantes
def fillna_strategy_operation(merged_data, fillna_strategy):
    merged_data_filled = merged_data.copy()
    merged_data_filled = merged_data_filled.replace([np.inf, -np.inf], np.nan)
    if fillna_strategy == 'deferred':
        return merged_data_filled

    cols_with_nan = merged_data_filled.columns[merged_data_filled.isna().any()].tolist()
    # for col in cols_with_nan:
    #     print(f"Column '{col}' has {merged_data_filled[col].isna().sum()} missing values.")

    if fillna_strategy is None:
        return merged_data_filled.dropna()

    for col in cols_with_nan:
        fill_value = _resolve_fill_value(merged_data_filled[col], fillna_strategy)
        merged_data_filled[col] = merged_data_filled[col].fillna(fill_value)

    remaining_nan_columns = merged_data_filled.columns[merged_data_filled.isna().any()].tolist()
    for col in remaining_nan_columns:
        merged_data_filled[col] = merged_data_filled[col].fillna(_default_fill_value(merged_data_filled[col]))

    return merged_data_filled


# Remocao de registros incompletos. Na versao anterior esta estrategia era
# representada por fill_na_strategy=None e removia linhas de TODAS as
# particoes, inclusive do teste. Isso mudava a populacao avaliada de uma
# estrategia para outra e invalidava a comparacao entre elas; na Home Credit e
# na IEEE-CIS, onde ha colunas com mais de 90% de faltantes, eliminaria quase
# todas as linhas (itens G15c e G16 da revisao). Agora a remocao vale apenas
# para o treinamento: validacao e teste mantem a populacao completa e recebem
# a estatistica calculada no treino. Antes da remocao, as colunas com faltantes
# acima do limiar escolhido no treino saem de todos os splits
# (missing_screening.py), senao o treino ficaria quase vazio.
DROP_INCOMPLETE_ROWS_STRATEGY = missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY
# Estatistica usada para completar validacao e teste quando o treino foi
# reduzido por remocao de registros.
DROP_STRATEGY_HELDOUT_FILL = "median"


def _fit_imputation_values(train_data, fillna_strategy, target_column):
    """Fit feature imputation values using one training split only."""
    if fillna_strategy is None:
        return None

    effective_strategy = (
        DROP_STRATEGY_HELDOUT_FILL
        if fillna_strategy == DROP_INCOMPLETE_ROWS_STRATEGY
        else fillna_strategy
    )
    feature_data = train_data.drop(columns=[target_column], errors='ignore')
    return {
        column: _resolve_fill_value(feature_data[column], effective_strategy)
        for column in feature_data.columns
    }


def _apply_imputation_values(dataframe, fill_values, target_column):
    """Apply training-derived feature values without modifying target labels."""
    if dataframe is None:
        return None

    imputed_data = dataframe.copy().replace([np.inf, -np.inf], np.nan)
    if target_column in imputed_data.columns:
        imputed_data = imputed_data.dropna(subset=[target_column])

    if fill_values is None:
        return imputed_data.dropna()

    applicable_values = {
        column: value
        for column, value in fill_values.items()
        if column in imputed_data.columns
    }
    imputed_data = imputed_data.fillna(applicable_values)

    feature_columns = [
        column for column in imputed_data.columns if column != target_column
    ]
    for column in feature_columns:
        if imputed_data[column].isna().any():
            imputed_data[column] = imputed_data[column].fillna(
                _default_fill_value(imputed_data[column])
            )
    return imputed_data


def _impute_data_splits(test_data_information, fillna_strategy, target_column):
    """Fit imputation on train and apply it to every held-out dataframe."""
    imputed_information = {**test_data_information, 'test_info': []}
    for split in test_data_information['test_info']:
        screened_out_columns = []
        if fillna_strategy == DROP_INCOMPLETE_ROWS_STRATEGY and 'missing_screening' not in split:
            split = missing_screening.decide_fold_removal(split, target_column)
        imputed_split = {**split}
        raw_train_data = split['train_data'].replace([np.inf, -np.inf], np.nan)
        if fillna_strategy == DROP_INCOMPLETE_ROWS_STRATEGY:
            # So o treino perde linhas; a estatistica para completar validacao
            # e teste e calculada depois da remocao, sobre o treino efetivo.
            # As linhas vem da decisao da dobra, que numa variante de ablacao
            # foi tomada com as colunas da base de origem (item G16).
            screened_out_columns = [
                column
                for column in split['missing_screening'].screened_out_columns
                if column in raw_train_data.columns
            ]
            raw_train_data = raw_train_data.iloc[
                split['retained_train_positions']
            ].drop(columns=screened_out_columns)
            if raw_train_data.isna().any().any():
                raise ValueError(
                    "drop-incomplete-rows left missing values in the retained "
                    "training rows; the removal decision does not match the data"
                )
        elif target_column in raw_train_data.columns:
            raw_train_data = raw_train_data.dropna(subset=[target_column])
        fill_values = _fit_imputation_values(
            raw_train_data,
            fillna_strategy,
            target_column,
        )
        imputed_split['train_data'] = _apply_imputation_values(
            raw_train_data, fill_values, target_column
        )
        for split_name in ('validation_data', 'test_data'):
            held_out = split.get(split_name)
            if held_out is not None and screened_out_columns:
                held_out = held_out.drop(columns=screened_out_columns)
            imputed_split[split_name] = _apply_imputation_values(
                held_out, fill_values, target_column
            )
        imputed_information['test_info'].append(imputed_split)
    return imputed_information


RANDOM_STATE = 42

# A codificacao supervisionada e um desenho fatorial 2x2 (item C3.6 da revisao).
# A revisao apontou que comparar "ordered target" com "weight of evidence"
# misturava dois fatores: a ESTATISTICA que resume a categoria e o MECANISMO que
# impede a linha de ver o proprio rotulo. Os dois eixos agora sao independentes,
# e o nome da estrategia declara a celula: "<estatistica>:<mecanismo>".
ENCODING_FACTOR_SEPARATOR = ':'

ENCODING_STATISTIC_SMOOTHED_MEAN = 'smoothed-mean'
ENCODING_STATISTIC_LOG_ODDS = 'log-odds'
ENCODING_STATISTICS = (
    ENCODING_STATISTIC_SMOOTHED_MEAN,
    ENCODING_STATISTIC_LOG_ODDS,
)

ENCODING_CONTROL_K_FOLD = 'k-fold'
ENCODING_CONTROL_PERMUTATION_PREFIX = 'permutation-prefix'
ENCODING_LEAKAGE_CONTROLS = (
    ENCODING_CONTROL_K_FOLD,
    ENCODING_CONTROL_PERMUTATION_PREFIX,
)

# As quatro celulas do fatorial. Variar uma coordenada por vez isola o efeito.
FACTORIAL_CATEGORICAL_ENCODINGS = tuple(
    f'{statistic}{ENCODING_FACTOR_SEPARATOR}{control}'
    for statistic in ENCODING_STATISTICS
    for control in ENCODING_LEAKAGE_CONTROLS
)

# Numero de blocos do mecanismo de prefixo. O esquema ordenado original encodifica
# cada linha com as linhas que a precedem numa permutacao, o que exigiria um
# reajuste por linha. Com blocos expansivos o custo cai para um reajuste por
# bloco, mantendo a propriedade essencial: nenhuma linha participa da estatistica
# que a codifica. Dez blocos deixam 10% das linhas (o primeiro bloco, que nao tem
# prefixo algum) com o valor a priori, contra 20% se fossem cinco.
PERMUTATION_PREFIX_BLOCK_COUNT = 10
CROSS_FITTING_FOLD_COUNT = 5

# Estatisticas que nao fazem parte do fatorial, mantidas por compatibilidade com
# resultados anteriores. Todas usam cross-fitting como mecanismo.
LEGACY_SUPERVISED_ENCODINGS = {
    'target': ENCODING_CONTROL_K_FOLD,
    'james-stein': ENCODING_CONTROL_K_FOLD,
    # A ordered target encoding como publicada: prefixo POR LINHA, calculado
    # internamente pelo CatBoostEncoder. Fica fora da grade porque seu mecanismo
    # nao tem equivalente para a log-razao, o que era exatamente o confundimento
    # relatado. Serve como braco de referencia para medir o quanto a aproximacao
    # por blocos se afasta da formulacao original.
    'ordered-target': ENCODING_CONTROL_PERMUTATION_PREFIX,
}

SUPERVISED_CATEGORICAL_ENCODINGS = {
    *FACTORIAL_CATEGORICAL_ENCODINGS,
    *LEGACY_SUPERVISED_ENCODINGS,
}


def _normalize_categorical_encoding_strategy(strategy):
    if strategy is None:
        return None
    # O mapa de apelidos vive em prepared_dataset_storage porque a identidade
    # da configuracao tambem precisa dele: duas grafias da mesma celula do
    # fatorial nao podem gerar dois configuration_md5.
    return normalize_categorical_encoding_strategy(strategy)


def parse_encoding_factors(strategy):
    """Decompoe a estrategia nas duas coordenadas do fatorial.

    Devolve `(estatistica, mecanismo)`. Para as estrategias herdadas, que nao
    pertencem ao fatorial, a estatistica e o proprio nome da estrategia: elas
    nao sao comparaveis no eixo de estatistica porque nao tem par no outro
    mecanismo.
    """
    normalized_strategy = _normalize_categorical_encoding_strategy(strategy)
    if normalized_strategy is None:
        return (None, None)
    if normalized_strategy in FACTORIAL_CATEGORICAL_ENCODINGS:
        statistic, control = normalized_strategy.split(ENCODING_FACTOR_SEPARATOR)
        return (statistic, control)
    if normalized_strategy in LEGACY_SUPERVISED_ENCODINGS:
        return (normalized_strategy, LEGACY_SUPERVISED_ENCODINGS[normalized_strategy])
    return (normalized_strategy, None)


def _encoding_statistic(strategy):
    return parse_encoding_factors(strategy)[0]


def _encoding_leakage_control(strategy):
    return parse_encoding_factors(strategy)[1]


def _build_supervised_categorical_encoder(strategy, categorical_columns):
    """Constroi o objeto que calcula a ESTATISTICA, sem nenhum mecanismo embutido.

    O mecanismo de controle de vazamento e aplicado por fora, em
    `_encode_training_rows_with_leakage_control`, para que seja literalmente o
    mesmo codigo nas duas estatisticas.
    """
    statistic = _encoding_statistic(strategy)
    if statistic == 'target':
        return TargetEncoder(target_type='binary', random_state=RANDOM_STATE)

    try:
        from category_encoders import CatBoostEncoder, JamesSteinEncoder, WOEEncoder
    except ImportError as error:
        raise ImportError(
            "The 'category-encoders' package is required for James-Stein, "
            "smoothed-mean and log-odds categorical encoding"
        ) from error

    if statistic == 'james-stein':
        return JamesSteinEncoder(
            cols=list(categorical_columns),
            handle_missing='value',
            handle_unknown='value',
        )
    if statistic in (ENCODING_STATISTIC_SMOOTHED_MEAN, 'ordered-target'):
        # Media do alvo por categoria com suavizacao aditiva: (soma + a*prior) /
        # (contagem + a). Sob o mecanismo de prefixo chamado linha a linha, este
        # mesmo objeto reproduz a ordered target encoding.
        return CatBoostEncoder(
            cols=list(categorical_columns),
            a=1.0,
            sigma=None,
            handle_missing='value',
            handle_unknown='value',
            random_state=RANDOM_STATE,
        )
    if statistic == ENCODING_STATISTIC_LOG_ODDS:
        # Additive smoothing keeps category and class probabilities away from zero.
        return WOEEncoder(
            cols=list(categorical_columns),
            regularization=1.0,
            handle_missing='value',
            handle_unknown='value',
            randomized=False,
        )
    raise ValueError(f"Unsupported supervised encoding strategy '{strategy}'")


def _encoding_prior_value(strategy, target):
    """Valor atribuido as linhas que o mecanismo deixa sem prefixo algum.

    Nao e um preenchimento arbitrario: e o valor que cada estatistica assume na
    ausencia de qualquer observacao. Para a media suavizada, a prevalencia do
    treino; para a log-razao, zero, que significa "sem evidencia".
    """
    statistic = _encoding_statistic(strategy)
    if statistic == ENCODING_STATISTIC_LOG_ODDS:
        return 0.0
    return float(pd.to_numeric(target, errors='coerce').mean())


def _finite_encoded_dataframe(dataframe):
    return dataframe.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _frame_md5(dataframe):
    row_hashes = pd.util.hash_pandas_object(dataframe, index=False).to_numpy()
    return hashlib.md5(row_hashes.tobytes(), usedforsecurity=False).hexdigest()


def _count_non_finite(dataframe):
    return int((~np.isfinite(dataframe.to_numpy(dtype=float))).sum())


def _encode_training_rows_with_leakage_control(
    train_data, target, categorical_columns, strategy
):
    """Codifica as linhas de treino sem que nenhuma veja o proprio rotulo.

    Esta funcao e o ponto do desenho fatorial: a estatistica entra pelo objeto
    que `_build_supervised_categorical_encoder` devolve, e o mecanismo entra pelo
    particionamento das linhas feito aqui. Trocar um fator nao mexe no outro, que
    e o que torna a diferenca medida atribuivel.
    """
    encoded_train = train_data.copy()
    # Allocate numeric columns before assigning fold results (pandas string
    # extension columns reject floating-point encoder output).
    encoded_train[categorical_columns] = np.nan
    control = _encoding_leakage_control(strategy)

    if strategy == 'ordered-target':
        # Braco de referencia, fora do fatorial: o prefixo POR LINHA calculado
        # internamente pelo CatBoostEncoder, isto e, a ordered target encoding
        # como publicada. Mantido para medir a distancia entre ela e a
        # aproximacao por blocos, nao para comparar com a log-razao.
        random_generator = np.random.default_rng(RANDOM_STATE)
        permuted_positions = random_generator.permutation(len(train_data))
        ordered_encoder = _build_supervised_categorical_encoder(
            strategy, categorical_columns
        )
        ordered_values = ordered_encoder.fit_transform(
            train_data.iloc[permuted_positions][categorical_columns],
            target.iloc[permuted_positions],
        )
        encoded_train.loc[
            encoded_train.index[permuted_positions], categorical_columns
        ] = np.asarray(ordered_values)
        return encoded_train

    if control == ENCODING_CONTROL_PERMUTATION_PREFIX:
        # Prefixo por blocos expansivos sobre uma unica permutacao deterministica.
        # O bloco k e codificado por um ajuste feito apenas nos blocos anteriores;
        # o bloco 0, que nao tem prefixo, recebe o valor a priori da estatistica.
        random_generator = np.random.default_rng(RANDOM_STATE)
        permuted_positions = random_generator.permutation(len(train_data))
        blocks = np.array_split(
            permuted_positions,
            min(PERMUTATION_PREFIX_BLOCK_COUNT, len(train_data)),
        )
        prior_value = _encoding_prior_value(strategy, target)
        consumed_positions = []
        for block_positions in blocks:
            if len(block_positions) == 0:
                continue
            block_index = encoded_train.index[block_positions]
            prefix_positions = (
                np.concatenate(consumed_positions) if consumed_positions else None
            )
            consumed_positions.append(block_positions)
            # Um prefixo sem as duas classes nao define nenhuma das estatisticas,
            # entao vale o a priori, igual ao bloco inicial.
            if (
                prefix_positions is None
                or target.iloc[prefix_positions].nunique(dropna=True) < 2
            ):
                encoded_train.loc[block_index, categorical_columns] = prior_value
                continue
            prefix_encoder = _build_supervised_categorical_encoder(
                strategy, categorical_columns
            )
            prefix_encoder.fit(
                train_data.iloc[prefix_positions][categorical_columns],
                target.iloc[prefix_positions],
            )
            encoded_train.loc[block_index, categorical_columns] = np.asarray(
                prefix_encoder.transform(
                    train_data.iloc[block_positions][categorical_columns]
                )
            )
        return encoded_train

    minimum_class_count = int(target.value_counts().min())
    number_of_splits = min(
        CROSS_FITTING_FOLD_COUNT, minimum_class_count, len(train_data)
    )
    if number_of_splits < 2:
        encoded_train[categorical_columns] = _encoding_prior_value(strategy, target)
        return encoded_train

    splitter = StratifiedKFold(
        n_splits=number_of_splits, shuffle=True, random_state=RANDOM_STATE
    )
    for fit_indexes, transform_indexes in splitter.split(train_data, target):
        fold_encoder = _build_supervised_categorical_encoder(
            strategy, categorical_columns
        )
        fold_encoder.fit(
            train_data.iloc[fit_indexes][categorical_columns],
            target.iloc[fit_indexes],
        )
        encoded_train.loc[
            encoded_train.index[transform_indexes], categorical_columns
        ] = np.asarray(
            fold_encoder.transform(
                train_data.iloc[transform_indexes][categorical_columns]
            )
        )
    return encoded_train


def _encode_supervised_data_splits(test_data_information, strategy, target_column):
    """Cross-fit training categories and transform held-out data without labels."""
    normalized_strategy = _normalize_categorical_encoding_strategy(strategy)
    if normalized_strategy not in SUPERVISED_CATEGORICAL_ENCODINGS:
        return test_data_information

    encoded_information = {**test_data_information, 'test_info': []}
    for split in test_data_information['test_info']:
        train_data = _prepare_categorical_data(split['train_data'])
        categorical_columns = list(
            _categorical_columns(train_data.drop(columns=[target_column]))
        )
        if not categorical_columns:
            encoded_information['test_info'].append(split)
            continue

        target = train_data[target_column]
        if (
            _encoding_statistic(normalized_strategy) == ENCODING_STATISTIC_LOG_ODDS
            and target.nunique(dropna=True) != 2
        ):
            raise ValueError('Log-odds (WoE) encoding requires a binary target')

        encoded_train = _encode_training_rows_with_leakage_control(
            train_data, target, categorical_columns, normalized_strategy
        )

        final_encoder = _build_supervised_categorical_encoder(
            normalized_strategy, categorical_columns
        )
        final_encoder.fit(train_data[categorical_columns], target)

        # O codificador pode gerar valores nao finitos; eles viram 0, mas a
        # contagem vai para o relatorio de preparacao (item G15).
        non_finite_replaced = {'train': _count_non_finite(encoded_train[categorical_columns])}
        encoded_split = {**split, 'train_data': _finite_encoded_dataframe(encoded_train)}
        for split_name in ('validation_data', 'test_data'):
            held_out_data = split.get(split_name)
            if held_out_data is None or held_out_data.empty:
                encoded_split[split_name] = held_out_data
                continue
            prepared_held_out = _prepare_categorical_data(held_out_data)
            prepared_held_out[categorical_columns] = np.asarray(final_encoder.transform(
                prepared_held_out[categorical_columns]
            ))
            non_finite_replaced[split_name.removesuffix('_data')] = _count_non_finite(
                prepared_held_out[categorical_columns]
            )
            encoded_split[split_name] = _finite_encoded_dataframe(prepared_held_out)
        encoded_split['encoding_non_finite_replaced'] = non_finite_replaced
        # Hash do bloco categorico logo apos a codificacao, antes da
        # padronizacao e da PCA: e o que permite verificar que codificacoes
        # diferentes produzem atributos diferentes (verificacao P12).
        encoded_split['encoded_categorical_md5'] = {
            split_name.removesuffix('_data'): _frame_md5(encoded_split[split_name][categorical_columns])
            for split_name in ('train_data', 'validation_data', 'test_data')
            if encoded_split.get(split_name) is not None and not encoded_split[split_name].empty
        }
        encoded_information['test_info'].append(encoded_split)
    return encoded_information


def _parse_strategy(strategy):
    if strategy is None:
        raise ValueError("test_strategy must be informed")

    strategy_parts = str(strategy).split(":")
    strategy_name = strategy_parts[0].strip()
    strategy_values = [
        float(value) if "." in value else int(value)
        for value in strategy_parts[1:]
        if str(value).strip() != ""
    ]
    return strategy_name, strategy_values


def _parse_resampling_parameter(value: str):
    """Convert a resampling parameter to a Python value when possible."""
    normalized_value = value.strip()
    if not normalized_value:
        raise ValueError("Resampling parameters cannot be empty")

    normalized_literals = {
        "true": True,
        "false": False,
        "none": None,
        "null": None,
    }
    if normalized_value.lower() in normalized_literals:
        return normalized_literals[normalized_value.lower()]

    try:
        return ast.literal_eval(normalized_value)
    except (SyntaxError, ValueError):
        return normalized_value


def _build_resampling_parameters(
    parameter_string: str | None,
    positional_parameter_names: tuple[str, ...],
):
    if parameter_string is None or not str(parameter_string).strip():
        return {}

    parameter_parts = str(parameter_string).split(":")
    uses_named_parameters = ["=" in part for part in parameter_parts]
    if any(uses_named_parameters) and not all(uses_named_parameters):
        raise ValueError("Use either positional or named resampling parameters, not both")

    if all(uses_named_parameters):
        parameters = {}
        for parameter_part in parameter_parts:
            parameter_name, parameter_value = parameter_part.split("=", maxsplit=1)
            parameter_name = parameter_name.strip()
            if not parameter_name:
                raise ValueError("A resampling parameter name cannot be empty")
            if parameter_name in parameters:
                raise ValueError(f"Duplicate resampling parameter: {parameter_name}")
            parameters[parameter_name] = _parse_resampling_parameter(parameter_value)
        return parameters

    if len(parameter_parts) > len(positional_parameter_names):
        raise ValueError(
            f"Expected at most {len(positional_parameter_names)} parameters, "
            f"but received {len(parameter_parts)}"
        )

    return {
        parameter_name: _parse_resampling_parameter(parameter_value)
        for parameter_name, parameter_value in zip(
            positional_parameter_names,
            parameter_parts,
        )
    }


def _resample_with_rus_smote(
    dataframe: pd.DataFrame,
    target_column: str,
    parameter_string: str | None,
) -> pd.DataFrame:
    """Apply RandomUnderSampler followed by SMOTE using deterministic seeds."""
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError as error:
        raise ImportError(
            "The 'imbalanced-learn' package is required for RUS+SMOTE"
        ) from error

    parameter_names = (
        "rus_sampling_strategy",
        "smote_sampling_strategy",
        "smote_k_neighbors",
        "rus_replacement",
    )
    parameters = _build_resampling_parameters(parameter_string, parameter_names)
    unknown_parameters = sorted(set(parameters).difference(parameter_names))
    if unknown_parameters:
        raise ValueError(
            f"Unsupported RUS+SMOTE parameters: {', '.join(unknown_parameters)}"
        )

    rus_sampling_strategy = parameters.get("rus_sampling_strategy", 0.5)
    smote_sampling_strategy = parameters.get("smote_sampling_strategy", 1.0)
    smote_k_neighbors = int(parameters.get("smote_k_neighbors", 5))
    rus_replacement = bool(parameters.get("rus_replacement", False))
    if smote_k_neighbors < 1:
        raise ValueError("smote_k_neighbors must be at least 1")

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    under_sampler = RandomUnderSampler(
        sampling_strategy=rus_sampling_strategy,
        replacement=rus_replacement,
        random_state=RANDOM_STATE,
    )
    under_sampled_features, under_sampled_target = under_sampler.fit_resample(
        features, target
    )

    minority_count = int(pd.Series(under_sampled_target).value_counts().min())
    if minority_count < 2:
        raise ValueError(
            "RUS+SMOTE requires at least two minority-class rows after "
            "Random Under-Sampling"
        )
    effective_neighbor_count = min(smote_k_neighbors, minority_count - 1)
    over_sampler = SMOTE(
        sampling_strategy=smote_sampling_strategy,
        k_neighbors=effective_neighbor_count,
        random_state=RANDOM_STATE,
    )
    resampled_features, resampled_target = over_sampler.fit_resample(
        under_sampled_features,
        under_sampled_target,
    )

    if not isinstance(resampled_features, pd.DataFrame):
        resampled_features = pd.DataFrame(
            resampled_features,
            columns=features.columns,
        )
    if not isinstance(resampled_target, pd.Series):
        resampled_target = pd.Series(resampled_target, name=target_column)
    result = resampled_features.reset_index(drop=True)
    result[target_column] = resampled_target.reset_index(drop=True)
    return result


def resample_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    resampling_technique: str,
    parameter_string: str | None = None,
) -> pd.DataFrame:
    """Apply an imbalanced-learn sampling technique to a dataframe.

    Positional parameters must be separated by ``:``. Named parameters in the
    ``name=value:name=value`` format are also accepted. Named parameters are
    recommended because each imbalanced-learn technique accepts different
    options. The returned dataframe keeps the input column order, target
    included, so the resampled training split has the same schema as
    validation and test. Stochastic techniques always use ``RANDOM_STATE`` (42)
    so experiments remain reproducible.
    """
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("dataframe must be a non-empty pandas DataFrame")
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found")

    normalized_technique = str(resampling_technique).strip().lower().replace("_", "-")
    if normalized_technique in {"rus+smote", "rus-smote"}:
        return _resample_with_rus_smote(
            dataframe,
            target_column,
            parameter_string,
        )[list(dataframe.columns)]

    teus_technique_names = {
        "teus",
        "hybrid-evolutionary-under-sampling",
        "tomek-evolutionary-under-sampling",
    }
    if normalized_technique in teus_technique_names:
        from teus_under_sampling import TEUSUnderSampler

        sampler_class = TEUSUnderSampler
        positional_parameter_names = (
            "alpha",
            "beta",
            "gamma",
            "quantiles",
            "population_size",
            "generations",
            "crossover_probability",
            "mutation_rate",
            "random_state",
        )
    else:
        try:
            from imblearn.under_sampling import (
                ClusterCentroids,
                EditedNearestNeighbours,
                NearMiss,
                RandomUnderSampler,
                TomekLinks,
            )
        except ImportError as error:
            raise ImportError(
                "The 'imbalanced-learn' package is required for data resampling"
            ) from error

        technique_definitions = {
            "random-undersampling": (
                RandomUnderSampler,
                ("sampling_strategy", "random_state", "replacement"),
            ),
            "random-under-sampler": (
                RandomUnderSampler,
                ("sampling_strategy", "random_state", "replacement"),
            ),
            "tomek-links": (TomekLinks, ("sampling_strategy", "n_jobs")),
            "edited-nearest-neighbours": (
                EditedNearestNeighbours,
                ("sampling_strategy", "n_neighbors", "kind_sel", "n_jobs"),
            ),
            "edited-nearest-neighbors": (
                EditedNearestNeighbours,
                ("sampling_strategy", "n_neighbors", "kind_sel", "n_jobs"),
            ),
            "near-miss": (
                NearMiss,
                (
                    "sampling_strategy",
                    "version",
                    "n_neighbors",
                    "n_neighbors_ver3",
                    "n_jobs",
                ),
            ),
            "cluster-centroids": (
                ClusterCentroids,
                ("sampling_strategy", "random_state", "voting"),
            ),
        }
        if normalized_technique not in technique_definitions:
            supported_techniques = sorted(
                {*technique_definitions, *teus_technique_names, "rus+smote"}
            )
            raise ValueError(
                f"Unsupported resampling technique '{resampling_technique}'. "
                f"Supported techniques: {', '.join(supported_techniques)}"
            )

        sampler_class, positional_parameter_names = technique_definitions[
            normalized_technique
        ]
    sampler_parameters = _build_resampling_parameters(
        parameter_string,
        positional_parameter_names,
    )
    if "random_state" in positional_parameter_names:
        sampler_parameters["random_state"] = RANDOM_STATE
    try:
        sampler = sampler_class(**sampler_parameters)
    except TypeError as error:
        raise ValueError(
            f"Invalid parameters for resampling technique '{resampling_technique}': {error}"
        ) from error

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    resampled_features, resampled_target = sampler.fit_resample(features, target)

    if not isinstance(resampled_features, pd.DataFrame):
        resampled_features = pd.DataFrame(resampled_features, columns=features.columns)
    if not isinstance(resampled_target, pd.Series):
        resampled_target = pd.Series(resampled_target, name=target_column)

    result = resampled_features.reset_index(drop=True)
    result[target_column] = resampled_target.reset_index(drop=True)
    # Na ordem da entrada: no bloco nativo ha atributos derivados depois do
    # alvo, e validacao e teste mantem essa ordem.
    return result[list(dataframe.columns)]


def _drop_columns(data, removal_column_list):
    if data is None:
        return None

    columns_to_remove = [column for column in removal_column_list if column in data.columns]
    return data.drop(columns=columns_to_remove).reset_index(drop=True)


def _build_result(train_data, validation_data=None, test_data=None, removal_column_list=None):
    removal_column_list = removal_column_list or []
    return {
        "train_data": _drop_columns(train_data, removal_column_list),
        "validation_data": _drop_columns(validation_data, removal_column_list),
        "test_data": _drop_columns(test_data, removal_column_list),
    }


def _normalize_target_column_case(dataframe, target_column):
    if target_column in dataframe.columns:
        return dataframe

    matching_columns = [
        column
        for column in dataframe.columns
        if str(column).lower() == target_column.lower()
    ]
    if len(matching_columns) != 1:
        return dataframe

    return dataframe.rename(columns={matching_columns[0]: target_column})


def train_test_data_split(
    database_train,
    database_auxiliary,
    database_type,
    test_strategy,
    random_seed=RANDOM_STATE,
):
    """Create train, validation and test splits under one evaluation protocol.

    Toda a logica de particao vive em ``data_partitioning``; aqui fica apenas
    a ponte, para que o restante do pipeline continue chamando um unico ponto.
    O protocolo e descrito so por ``test_strategy``: o antigo eixo
    ``test_data_size`` foi eliminado porque nao alterava os dados em duas das
    quatro politicas e, nas outras, confundia o efeito da politica com o do
    tamanho do treinamento (itens G9c e G9d da revisao).

    O respeito ao agrupamento (cliente na Home Credit, cartao na Sparkov,
    titular aproximado na IEEE-CIS) e a remocao dos identificadores sao
    decididos pelo catalogo de bases, nao pelo chamador.
    """
    definition = experiment_datasets.get_dataset_definition(database_type)
    return data_partitioning.build_partitions(
        database_train,
        database_auxiliary,
        definition,
        test_strategy,
        random_seed,
    )


def _fit_standardization_values(train_data, target_column):
    """Media e desvio padrao dos atributos numericos, so no treinamento.

    A versao anterior so padronizava as configuracoes que recebiam PCA, o que
    deixava KNN, SMOTE, Tomek Links, regressao logistica e Naive Bayes sem
    escalonamento na ULB e na Sparkov. Como todos eles dependem de distancia
    ou de escala, e como na ULB ``Amount`` chega a 25.691 contra componentes
    principais de variancia unitaria, esses atributos dominavam as distancias
    euclidianas (item G14a). Agora a padronizacao vale para toda configuracao.
    """
    feature_data = train_data.drop(columns=[target_column], errors="ignore")
    numeric_features = feature_data.select_dtypes(include=[np.number])
    if numeric_features.empty:
        return None
    mean_values = numeric_features.mean()
    standard_deviations = numeric_features.std(ddof=0)
    # Atributo constante no treino nao carrega informacao; dividir por zero
    # geraria infinito, entao o desvio vira 1 e a coluna fica centrada em 0.
    standard_deviations = standard_deviations.replace(0, 1.0).fillna(1.0)
    return mean_values.fillna(0.0), standard_deviations


def _apply_standardization_values(dataframe, standardization_values, target_column):
    if dataframe is None or dataframe.empty or standardization_values is None:
        return dataframe
    mean_values, standard_deviations = standardization_values
    standardized = dataframe.copy()
    for column in mean_values.index:
        if column == target_column or column not in standardized.columns:
            continue
        if not pd.api.types.is_numeric_dtype(standardized[column]):
            continue
        standardized[column] = (
            standardized[column] - mean_values[column]
        ) / standard_deviations[column]
    return standardized


def standardize_data_splits(split_information, target_column):
    """Ajusta a padronizacao no treino de cada dobra e aplica nas demais."""
    standardized_information = {**split_information, "test_info": []}
    for fold in split_information["test_info"]:
        standardization_values = _fit_standardization_values(
            fold["train_data"], target_column
        )
        standardized_information["test_info"].append(
            {
                **fold,
                "train_data": _apply_standardization_values(
                    fold["train_data"], standardization_values, target_column
                ),
                "validation_data": _apply_standardization_values(
                    fold.get("validation_data"), standardization_values, target_column
                ),
                "test_data": _apply_standardization_values(
                    fold.get("test_data"), standardization_values, target_column
                ),
            }
        )
    return standardized_information


def require_numeric_complete_features(features: pd.DataFrame, context: str) -> None:
    """Recusa atributos nao numericos ou incompletos (item G15 da revisao).

    A versao anterior descartava as colunas nao numericas e completava os
    faltantes com a mediana em silencio, dentro da PCA e dos executores. Foi
    esse comportamento tolerante que escondeu o erro do G15: as categoricas
    nunca chegavam ao modelo e os eixos de imputacao e codificacao nao tinham
    efeito. Agora uma entrada assim e erro de preparacao, nao algo a corrigir.
    """
    non_numeric_columns = [
        column
        for column in features.columns
        if not pd.api.types.is_numeric_dtype(features[column])
    ]
    if non_numeric_columns:
        raise ValueError(
            f"{context} requires numeric features; non-numeric columns: "
            f"{non_numeric_columns[:10]}"
        )
    non_finite = ~np.isfinite(features.to_numpy(dtype=float))
    if non_finite.any():
        incomplete_columns = list(features.columns[non_finite.any(axis=0)])
        raise ValueError(
            f"{context} requires complete features; columns with missing or "
            f"non-finite values: {incomplete_columns[:10]}"
        )


def _linear_regression_features_and_target(
    data,
    prediction_column_name,
    feature_column_list=None,
    trained_feature_columns=None,
):
    if data is None or data.empty:
        return None, None
    if prediction_column_name not in data.columns:
        raise ValueError(f"{prediction_column_name} does not exist in data")

    feature_column_list = feature_column_list or [
        column for column in data.columns if column != prediction_column_name
    ]
    missing_feature_columns = [
        column for column in feature_column_list if column not in data.columns
    ]
    if missing_feature_columns:
        raise ValueError(f"Feature columns not found in data: {missing_feature_columns}")

    data_without_empty_target = data.dropna(subset=[prediction_column_name]).copy()
    if data_without_empty_target.empty:
        return None, None

    features = data_without_empty_target[feature_column_list]
    target = data_without_empty_target[prediction_column_name]

    if trained_feature_columns is not None:
        absent_columns = [
            column for column in trained_feature_columns if column not in features.columns
        ]
        if absent_columns:
            raise ValueError(
                f"Held-out split lacks training columns: {absent_columns[:10]}"
            )
        features = features[list(trained_feature_columns)]

    require_numeric_complete_features(features, "Classic executor")
    return features.astype(float), target


def _linear_regression_class_prediction(predicted_values, class_values, classification_threshold):
    class_values = np.array(sorted(class_values))
    if len(class_values) == 2:
        negative_class = class_values[0]
        positive_class = class_values[1]
        return np.where(predicted_values >= classification_threshold, positive_class, negative_class)

    distances = np.abs(predicted_values.reshape(-1, 1) - class_values.reshape(1, -1))
    return class_values[np.argmin(distances, axis=1)]


def _roc_auc(real_values, prediction_scores, class_values):
    """Calculate ROC-AUC from continuous scores, returning NaN when undefined."""
    if prediction_scores is None:
        return np.nan

    class_values = sorted(class_values)
    scores = np.asarray(prediction_scores)
    try:
        if len(class_values) == 2:
            if scores.ndim == 2:
                scores = scores[:, 1]
            binary_target = (np.asarray(real_values) == class_values[-1]).astype(int)
            return roc_auc_score(binary_target, scores)

        return roc_auc_score(
            real_values,
            scores,
            labels=class_values,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        # ROC-AUC is undefined when a split does not contain all required classes.
        return np.nan


def _average_precision(real_values, prediction_scores, class_values):
    """AUC-PR (precisao media), devolvendo NaN quando indefinida.

    Sob forte desbalanceamento a curva precisao-revocacao e mais informativa
    que a ROC: a ROC usa a taxa de falsos positivos, cujo denominador e o
    numero de negativos, enorme nestas bases, de modo que um aumento grande em
    termos absolutos de falsos positivos mal desloca a curva. A AUC-PR mede
    exatamente o que custa em operacao -- quantos alarmes sao falsos --, e por
    isso o item G11-iii da revisao pede as duas (Saito e Rehmsmeier, 2015;
    Davis e Goadrich, 2006).

    Diferentemente da AUC-ROC, a AUC-PR DEPENDE DA PREVALENCIA: sua linha de
    base e a propria prevalencia da classe positiva, e nao 0,5. Ela compara
    algoritmos dentro de uma mesma base, nunca entre bases com prevalencias
    diferentes.
    """
    if prediction_scores is None:
        return np.nan

    class_values = sorted(class_values)
    scores = np.asarray(prediction_scores)
    try:
        if len(class_values) != 2:
            # Fora do caso binario a precisao media exige uma convencao de
            # media que nao tem leitura operacional aqui.
            return np.nan
        if scores.ndim == 2:
            scores = scores[:, 1]
        binary_target = (np.asarray(real_values) == class_values[-1]).astype(int)
        if binary_target.sum() == 0:
            # Sem exemplos positivos a precisao media e indefinida. O
            # scikit-learn devolve 0,0 com um aviso, o que se confundiria com
            # "desempenho pessimo" numa tabela de resultados; NaN diz o que de
            # fato aconteceu, e e o mesmo que a AUC-ROC faz aqui.
            return np.nan
        return average_precision_score(binary_target, scores)
    except ValueError:
        return np.nan


def _false_positive_rate(real_values, predicted_classes, class_values):
    """Falsos positivos sobre negativos reais.

    Binario: taxa da classe positiva (mesma convencao de pos_label das demais
    metricas). Multiclasse: micro-media um-contra-o-resto, somando FP e TN de
    cada classe antes de dividir, no mesmo espirito de average="micro".
    """
    labels = sorted(class_values)
    matrices = multilabel_confusion_matrix(real_values, predicted_classes, labels=labels)
    if len(labels) == 2:
        positive_index = labels.index(labels[-1])
        true_negative, false_positive, _false_negative, _true_positive = matrices[
            positive_index
        ].ravel()
    else:
        true_negative = sum(matrix[0, 0] for matrix in matrices)
        false_positive = sum(matrix[0, 1] for matrix in matrices)
    denominator = false_positive + true_negative
    return float(false_positive) / denominator if denominator else 0.0


def _linear_regression_metrics(
    real_values,
    predicted_classes,
    class_values,
    prediction_scores=None,
):
    average_strategy = "binary" if len(class_values) == 2 else "micro"
    positive_label = sorted(class_values)[-1] if len(class_values) == 2 else 1

    return {
        "f1": f1_score(
            real_values,
            predicted_classes,
            average=average_strategy,
            pos_label=positive_label,
            zero_division=0,
        ),
        "f_beta_2": fbeta_score(
            real_values,
            predicted_classes,
            beta=2.0,
            average=average_strategy,
            pos_label=positive_label,
            zero_division=0,
        ),
        "accuracy": accuracy_score(real_values, predicted_classes),
        "precision": precision_score(
            real_values,
            predicted_classes,
            average=average_strategy,
            pos_label=positive_label,
            zero_division=0,
        ),
        "recall": recall_score(
            real_values,
            predicted_classes,
            average=average_strategy,
            pos_label=positive_label,
            zero_division=0,
        ),
        "roc_auc": _roc_auc(real_values, prediction_scores, class_values),
        "auc_pr": _average_precision(real_values, prediction_scores, class_values),
        "false_positive_rate": _false_positive_rate(
            real_values, predicted_classes, class_values
        ),
    }


def _linear_regression_predict(
    model,
    feature_scaler,
    data,
    prediction_column_name,
    feature_columns,
    split_name,
    class_values,
    classification_threshold,
    execution_backend,
):
    features, target = _linear_regression_features_and_target(
        data,
        prediction_column_name,
        trained_feature_columns=feature_columns,
    )
    if features is None:
        return None

    predicted_values = _predict_linear_regression_values(
        model,
        feature_scaler.transform(features),
        execution_backend,
    )
    predicted_classes = _linear_regression_class_prediction(
        predicted_values,
        class_values,
        classification_threshold,
    )
    return {
        "split_name": split_name,
        "metrics": _linear_regression_metrics(
            target, predicted_classes, class_values, predicted_values
        ),
        "predictions": pd.DataFrame(
            {
                "real_value": target.to_numpy(),
                "predicted_value": predicted_classes,
                "linear_regression_value": predicted_values,
            },
            index=target.index,
        ),
    }


def execute_linear_regression(
    test_data_information,
    prediction_column_name,
    feature_column_list=None,
    classification_threshold=0.5,
    execution_device="auto",
    undersampling_technique=None,
    undersampling_parameter_values=None,
):
    """
    Execute one linear regression model for each object inside test_data_information["test_info"].

    Each test_info item must contain train_data and can optionally contain validation_data
    and test_data. Under-sampling, when configured, is applied only to train_data. The
    parameter values may use the colon-separated positional or named format accepted by
    resample_dataframe. For binary classification targets, the linear-regression value is
    converted to a class using classification_threshold.
    """
    if not test_data_information or "test_info" not in test_data_information:
        raise ValueError("test_data_information must contain a test_info list")
    if undersampling_parameter_values and not undersampling_technique:
        raise ValueError(
            "undersampling_technique is required when undersampling_parameter_values "
            "is informed"
        )

    execution_backend = _resolve_execution_device(execution_device)
    results = []
    for test_index, test_info in enumerate(test_data_information["test_info"]):
        train_data = test_info.get("train_data")
        if train_data is None or train_data.empty:
            raise ValueError(f"test_info[{test_index}] must contain non-empty train_data")
        if undersampling_technique:
            train_data = resample_dataframe(
                train_data,
                prediction_column_name,
                undersampling_technique,
                undersampling_parameter_values,
            )

        train_features, train_target = _linear_regression_features_and_target(
            train_data,
            prediction_column_name,
            feature_column_list=feature_column_list,
        )
        if train_features is None:
            raise ValueError(f"test_info[{test_index}] does not have train rows with target values")
        class_values = sorted(train_target.dropna().unique())
        if len(class_values) < 2:
            raise ValueError(f"test_info[{test_index}] must contain at least two target classes")

        fit_started_at = time.perf_counter()
        feature_scaler = StandardScaler()
        scaled_train_features = feature_scaler.fit_transform(train_features)
        model = _fit_linear_regression_model(
            scaled_train_features,
            train_target,
            execution_backend,
        )
        fit_time = time.perf_counter() - fit_started_at

        feature_columns = list(train_features.columns)
        scoring_started_at = time.perf_counter()
        split_results = {
            "train_data": _linear_regression_predict(
                model,
                feature_scaler,
                train_data,
                prediction_column_name,
                feature_columns,
                "train_data",
                class_values,
                classification_threshold,
                execution_backend,
            )
        }
        for split_name in ["validation_data", "test_data"]:
            split_results[split_name] = _linear_regression_predict(
                model,
                feature_scaler,
                test_info.get(split_name),
                prediction_column_name,
                feature_columns,
                split_name,
                class_values,
                classification_threshold,
                execution_backend,
            )

        scoring_time = time.perf_counter() - scoring_started_at

        results.append(
            {
                "test_info_index": test_index,
                # Tempos da dobra (item G18): ajuste, com a padronizacao, e
                # pontuacao dos tres conjuntos.
                "fit_time": round(fit_time, 6),
                "scoring_time": round(scoring_time, 6),
                # Indice 1-based da dobra e MD5 dos arquivos lidos, ambos
                # propagados do manifesto ate a linha de resultado.
                "fold_index": test_info.get("fold_index", test_index + FIRST_FOLD_INDEX),
                "split_md5": test_info.get("split_md5", {}),
                "partition_md5": test_info.get("partition_md5", {}),
                "train_population_md5": test_info.get("train_population_md5", ""),
                "unit_partition_md5": test_info.get("unit_partition_md5", {}),
                "train_unit_population_md5": test_info.get("train_unit_population_md5", ""),
                "evaluation_units": test_info.get("evaluation_units", {}),
                "model": model,
                "feature_scaler": feature_scaler,
                "feature_columns": feature_columns,
                "class_values": class_values,
                "classification_threshold": classification_threshold,
                "execution_device": execution_backend["device"],
                "intercept": _model_intercept_to_scalar(model),
                "coefficients": _model_coefficients_to_series(model, feature_columns),
                "split_sizes": {
                    "train_data": _dataframe_size(train_data),
                    "validation_data": _dataframe_size(test_info.get("validation_data")),
                    "test_data": _dataframe_size(test_info.get("test_data")),
                },
                "split_results": split_results,
            }
        )

    return {
        "algorithm": "linear-regression",
        "prediction_column_name": prediction_column_name,
        "train_test_strategy": test_data_information.get("train_test_strategy"),
        "execution_device": execution_backend["device"],
        "undersampling_technique": undersampling_technique,
        "undersampling_parameter_values": undersampling_parameter_values,
        "results": results,
    }


def _score_split(
    target,
    class_probabilities,
    model_classes,
    class_values,
    split_name,
    probability_column,
):
    """Scores de um split, ainda sem limiar de decisao.

    O limiar so e conhecido depois que a validacao de todas as dobras foi
    pontuada (item G11-i), entao a conversao em classes e as metricas ficam
    para _split_result_from_scores.
    """
    model_classes = np.asarray(model_classes)
    if len(class_values) == 2:
        positive_class_index = list(model_classes).index(sorted(class_values)[-1])
        positive_scores = class_probabilities[:, positive_class_index]
        metric_scores = positive_scores
        multiclass_predictions = None
    else:
        multiclass_predictions = model_classes[np.argmax(class_probabilities, axis=1)]
        positive_scores = np.max(class_probabilities, axis=1)
        metric_scores = class_probabilities
    return {
        "split_name": split_name,
        "target": target,
        "positive_scores": positive_scores,
        "metric_scores": metric_scores,
        "multiclass_predictions": multiclass_predictions,
        "probability_column": probability_column,
    }


def _split_result_from_scores(split_scores, class_values, classification_threshold):
    if split_scores is None:
        return None
    target = split_scores["target"]
    if split_scores["multiclass_predictions"] is not None:
        predicted_classes = split_scores["multiclass_predictions"]
    else:
        negative_class, positive_class = sorted(class_values)[0], sorted(class_values)[-1]
        predicted_classes = np.where(
            split_scores["positive_scores"] >= classification_threshold,
            positive_class,
            negative_class,
        )
    return {
        "split_name": split_scores["split_name"],
        "metrics": _linear_regression_metrics(
            target, predicted_classes, class_values, split_scores["metric_scores"]
        ),
        "predictions": pd.DataFrame(
            {
                "real_value": target.to_numpy(),
                "predicted_value": predicted_classes,
                split_scores["probability_column"]: split_scores["positive_scores"],
            },
            index=target.index,
        ),
    }


def apply_decision_thresholds(
    fold_results,
    threshold_strategy=DEFAULT_THRESHOLD_STRATEGY,
    classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
):
    """Escolhe o limiar de cada dobra e converte os scores em resultados.

    Recebe as dobras com ``split_scores`` e devolve-as com ``split_results``,
    ``classification_threshold`` e ``threshold_strategy``. As regras de onde o
    limiar e escolhido estao em decision_threshold.py.
    """
    validation_sets = []
    fold_has_test = []
    for fold_result in fold_results:
        split_scores = fold_result["split_scores"]
        class_values = fold_result["class_values"]
        validation = split_scores.get("validation_data")
        if validation is None or len(class_values) != 2:
            validation_sets.append(None)
        else:
            binary_target = (
                np.asarray(validation["target"]) == sorted(class_values)[-1]
            ).astype(int)
            validation_sets.append((binary_target, validation["positive_scores"]))
        fold_has_test.append(split_scores.get("test_data") is not None)

    fold_thresholds = select_fold_thresholds(
        validation_sets,
        fold_has_test,
        threshold_strategy,
        classification_threshold,
    )
    for fold_result, fold_threshold in zip(fold_results, fold_thresholds):
        split_scores = fold_result.pop("split_scores")
        fold_result["classification_threshold"] = fold_threshold.threshold
        fold_result["threshold_strategy"] = fold_threshold.strategy
        fold_result["split_results"] = {
            split_name: _split_result_from_scores(
                scores, fold_result["class_values"], fold_threshold.threshold
            )
            for split_name, scores in split_scores.items()
        }
    return fold_results


# Unidade em que as metricas de uma linha de resultado foram calculadas.
EVALUATION_UNIT_ROW = "row"
EVALUATION_UNIT_CLIENT = "client"


def aggregate_split_scores_by_unit(split_scores, units):
    """Escores de um split agregados por unidade (cliente), pela media.

    Na Home Credit com juncao o TARGET e do cliente e se repete nos seus
    pedidos anteriores; medir por linha pesaria cada cliente pelo numero de
    pedidos (item G7). Aqui cada cliente vira um ponto, com o escore medio das
    suas linhas e o seu rotulo unico.
    """
    frame = pd.DataFrame(
        {
            "unit": np.asarray(units),
            "target": np.asarray(split_scores["target"]),
            "score": np.asarray(split_scores["positive_scores"], dtype=float),
        }
    )
    if (frame.groupby("unit")["target"].nunique() > 1).any():
        raise ValueError("an evaluation unit has rows with different labels")
    grouped = frame.groupby("unit", sort=True).agg(
        target=("target", "first"), score=("score", "mean")
    )
    scores = grouped["score"].to_numpy()
    return {
        **split_scores,
        "target": pd.Series(grouped["target"].to_numpy()),
        "positive_scores": scores,
        "metric_scores": scores,
        "multiclass_predictions": None,
    }


def build_unit_level_folds(fold_results):
    """Copias das dobras pontuadas com os escores agregados por cliente.

    So os splits com unidade registrada (validacao e teste) entram; o treino
    pode ter sido reamostrado e nao e avaliado por cliente. Devolve lista
    vazia nas bases sem unidade de avaliacao.
    """
    unit_folds = []
    for fold_result in fold_results:
        units_by_split = fold_result.get("evaluation_units") or {}
        split_scores = {
            split_name: aggregate_split_scores_by_unit(scores, units_by_split[split_name])
            for split_name, scores in fold_result["split_scores"].items()
            if scores is not None and split_name in units_by_split
        }
        if not split_scores:
            continue
        unit_fold = {
            key: value
            for key, value in fold_result.items()
            if key not in ("split_scores", "evaluation_units")
        }
        unit_fold["split_scores"] = split_scores
        unit_fold["evaluation_unit"] = EVALUATION_UNIT_CLIENT
        # Tamanho em clientes nos splits avaliados por cliente.
        unit_fold["split_sizes"] = {
            **fold_result.get("split_sizes", {}),
            **{
                split_name: len(scores["target"])
                for split_name, scores in split_scores.items()
            },
        }
        unit_folds.append(unit_fold)
    return unit_folds


def apply_decision_thresholds_by_unit(
    fold_results,
    threshold_strategy=DEFAULT_THRESHOLD_STRATEGY,
    classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
):
    """Limiar e metricas por linha e, onde ha unidade, tambem por cliente.

    As dobras por cliente escolhem o proprio limiar, na validacao por cliente,
    com a mesma regra das dobras por linha. Devolve as dobras por linha
    seguidas das dobras por cliente.
    """
    unit_folds = build_unit_level_folds(fold_results)
    apply_decision_thresholds(fold_results, threshold_strategy, classification_threshold)
    if unit_folds:
        apply_decision_thresholds(unit_folds, threshold_strategy, classification_threshold)
    return fold_results + unit_folds


def _logistic_regression_predict(
    model,
    feature_scaler,
    data,
    prediction_column_name,
    feature_columns,
    split_name,
    class_values,
):
    features, target = _linear_regression_features_and_target(
        data,
        prediction_column_name,
        trained_feature_columns=feature_columns,
    )
    if features is None:
        return None

    return _score_split(
        target,
        model.predict_proba(feature_scaler.transform(features)),
        model.classes_,
        class_values,
        split_name,
        "logistic_regression_probability",
    )


def execute_logistic_regression(
    test_data_information,
    prediction_column_name,
    feature_column_list=None,
    classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    threshold_strategy=DEFAULT_THRESHOLD_STRATEGY,
    execution_device="auto",
    undersampling_technique=None,
    undersampling_parameter_values=None,
    class_weight_intensity=DEFAULT_CLASS_WEIGHT_INTENSITY,
):
    """Train and evaluate logistic regression on the configured data splits."""
    if not test_data_information or "test_info" not in test_data_information:
        raise ValueError("test_data_information must contain a test_info list")
    if undersampling_parameter_values and not undersampling_technique:
        raise ValueError(
            "undersampling_technique is required when undersampling_parameter_values "
            "is informed"
        )
    if execution_device == "gpu":
        raise ValueError("logistic regression currently supports CPU execution only")

    results = []
    for test_index, test_info in enumerate(test_data_information["test_info"]):
        train_data = test_info.get("train_data")
        if train_data is None or train_data.empty:
            raise ValueError(f"test_info[{test_index}] must contain non-empty train_data")
        if undersampling_technique:
            train_data = resample_dataframe(
                train_data,
                prediction_column_name,
                undersampling_technique,
                undersampling_parameter_values,
            )

        train_features, train_target = _linear_regression_features_and_target(
            train_data,
            prediction_column_name,
            feature_column_list=feature_column_list,
        )
        if train_features is None:
            raise ValueError(
                f"test_info[{test_index}] does not have train rows with target values"
            )
        class_values = sorted(train_target.dropna().unique())
        if len(class_values) != 2:
            raise ValueError(
                f"test_info[{test_index}] must contain exactly two target classes"
            )

        fit_started_at = time.perf_counter()
        feature_scaler = StandardScaler()
        scaled_train_features = feature_scaler.fit_transform(train_features)
        # R vem do treino desta dobra (prevalencia natural nos bracos
        # ponderados): nunca do teste, nunca da base inteira.
        imbalance_ratio = compute_class_imbalance_ratio(train_target, class_values)
        class_weight = resolve_logistic_class_weight(
            class_weight_intensity, imbalance_ratio, class_values
        )
        model = LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight=class_weight,
        )
        model.fit(scaled_train_features, train_target)
        fit_time = time.perf_counter() - fit_started_at

        feature_columns = list(train_features.columns)
        scoring_started_at = time.perf_counter()
        split_scores = {}
        for split_name in ["train_data", "validation_data", "test_data"]:
            split_data = (
                train_data if split_name == "train_data" else test_info.get(split_name)
            )
            split_scores[split_name] = _logistic_regression_predict(
                model,
                feature_scaler,
                split_data,
                prediction_column_name,
                feature_columns,
                split_name,
                class_values,
            )

        scoring_time = time.perf_counter() - scoring_started_at

        results.append(
            {
                "test_info_index": test_index,
                # Tempos da dobra (item G18): ajuste, com a padronizacao, e
                # pontuacao dos tres conjuntos.
                "fit_time": round(fit_time, 6),
                "scoring_time": round(scoring_time, 6),
                # Indice 1-based da dobra e MD5 dos arquivos lidos, ambos
                # propagados do manifesto ate a linha de resultado.
                "fold_index": test_info.get("fold_index", test_index + FIRST_FOLD_INDEX),
                "split_md5": test_info.get("split_md5", {}),
                "partition_md5": test_info.get("partition_md5", {}),
                "train_population_md5": test_info.get("train_population_md5", ""),
                "unit_partition_md5": test_info.get("unit_partition_md5", {}),
                "train_unit_population_md5": test_info.get("train_unit_population_md5", ""),
                "evaluation_units": test_info.get("evaluation_units", {}),
                "model": model,
                "feature_scaler": feature_scaler,
                "feature_columns": feature_columns,
                "class_values": class_values,
                "class_weight_intensity": class_weight_intensity,
                "class_weight_ratio": imbalance_ratio,
                "execution_device": "cpu",
                "intercept": float(model.intercept_[0]),
                "coefficients": pd.Series(model.coef_[0], index=feature_columns),
                "split_sizes": {
                    "train_data": _dataframe_size(train_data),
                    "validation_data": _dataframe_size(
                        test_info.get("validation_data")
                    ),
                    "test_data": _dataframe_size(test_info.get("test_data")),
                },
                "split_scores": split_scores,
            }
        )

    results = apply_decision_thresholds_by_unit(
        results, threshold_strategy, classification_threshold
    )

    return {
        "algorithm": "logistic-regression",
        "prediction_column_name": prediction_column_name,
        "train_test_strategy": test_data_information.get("train_test_strategy"),
        "execution_device": "cpu",
        "undersampling_technique": undersampling_technique,
        "undersampling_parameter_values": undersampling_parameter_values,
        "results": results,
    }


def _naive_bayes_predict(
    model,
    data,
    prediction_column_name,
    feature_columns,
    split_name,
    class_values,
):
    features, target = _linear_regression_features_and_target(
        data,
        prediction_column_name,
        trained_feature_columns=feature_columns,
    )
    if features is None:
        return None

    return _score_split(
        target,
        model.predict_proba(features),
        model.classes_,
        class_values,
        split_name,
        "naive_bayes_probability",
    )


def execute_naive_bayes(
    test_data_information,
    prediction_column_name,
    feature_column_list=None,
    classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    threshold_strategy=DEFAULT_THRESHOLD_STRATEGY,
    execution_device="auto",
    undersampling_technique=None,
    undersampling_parameter_values=None,
):
    """Train and evaluate Gaussian Naive Bayes on the configured data splits."""
    if not test_data_information or "test_info" not in test_data_information:
        raise ValueError("test_data_information must contain a test_info list")
    if undersampling_parameter_values and not undersampling_technique:
        raise ValueError(
            "undersampling_technique is required when undersampling_parameter_values "
            "is informed"
        )
    if execution_device == "gpu":
        raise ValueError("naive Bayes currently supports CPU execution only")

    results = []
    for test_index, test_info in enumerate(test_data_information["test_info"]):
        train_data = test_info.get("train_data")
        if train_data is None or train_data.empty:
            raise ValueError(f"test_info[{test_index}] must contain non-empty train_data")
        if undersampling_technique:
            train_data = resample_dataframe(
                train_data,
                prediction_column_name,
                undersampling_technique,
                undersampling_parameter_values,
            )

        train_features, train_target = _linear_regression_features_and_target(
            train_data,
            prediction_column_name,
            feature_column_list=feature_column_list,
        )
        if train_features is None:
            raise ValueError(
                f"test_info[{test_index}] does not have train rows with target values"
            )
        class_values = sorted(train_target.dropna().unique())
        if len(class_values) < 2:
            raise ValueError(
                f"test_info[{test_index}] must contain at least two target classes"
            )

        fit_started_at = time.perf_counter()
        model = GaussianNB()
        model.fit(train_features, train_target)
        fit_time = time.perf_counter() - fit_started_at

        feature_columns = list(train_features.columns)
        scoring_started_at = time.perf_counter()
        split_scores = {}
        for split_name in ["train_data", "validation_data", "test_data"]:
            split_data = (
                train_data if split_name == "train_data" else test_info.get(split_name)
            )
            split_scores[split_name] = _naive_bayes_predict(
                model,
                split_data,
                prediction_column_name,
                feature_columns,
                split_name,
                class_values,
            )

        scoring_time = time.perf_counter() - scoring_started_at

        results.append(
            {
                "test_info_index": test_index,
                # Tempos da dobra (item G18): ajuste, com a padronizacao, e
                # pontuacao dos tres conjuntos.
                "fit_time": round(fit_time, 6),
                "scoring_time": round(scoring_time, 6),
                # Indice 1-based da dobra e MD5 dos arquivos lidos, ambos
                # propagados do manifesto ate a linha de resultado.
                "fold_index": test_info.get("fold_index", test_index + FIRST_FOLD_INDEX),
                "split_md5": test_info.get("split_md5", {}),
                "partition_md5": test_info.get("partition_md5", {}),
                "train_population_md5": test_info.get("train_population_md5", ""),
                "unit_partition_md5": test_info.get("unit_partition_md5", {}),
                "train_unit_population_md5": test_info.get("train_unit_population_md5", ""),
                "evaluation_units": test_info.get("evaluation_units", {}),
                "model": model,
                "feature_columns": feature_columns,
                "class_values": class_values,
                "execution_device": "cpu",
                "split_sizes": {
                    "train_data": _dataframe_size(train_data),
                    "validation_data": _dataframe_size(test_info.get("validation_data")),
                    "test_data": _dataframe_size(test_info.get("test_data")),
                },
                "split_scores": split_scores,
            }
        )

    results = apply_decision_thresholds_by_unit(
        results, threshold_strategy, classification_threshold
    )

    return {
        "algorithm": "naive-bayes",
        "prediction_column_name": prediction_column_name,
        "train_test_strategy": test_data_information.get("train_test_strategy"),
        "execution_device": "cpu",
        "undersampling_technique": undersampling_technique,
        "undersampling_parameter_values": undersampling_parameter_values,
        "results": results,
    }


def _knn_predict(
    model,
    feature_scaler,
    data,
    prediction_column_name,
    feature_columns,
    split_name,
    class_values,
):
    features, target = _linear_regression_features_and_target(
        data,
        prediction_column_name,
        trained_feature_columns=feature_columns,
    )
    if features is None:
        return None

    return _score_split(
        target,
        model.predict_proba(feature_scaler.transform(features)),
        model.classes_,
        class_values,
        split_name,
        "knn_probability",
    )


def execute_knn(
    test_data_information,
    prediction_column_name,
    feature_column_list=None,
    classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    threshold_strategy=DEFAULT_THRESHOLD_STRATEGY,
    execution_device="auto",
    undersampling_technique=None,
    undersampling_parameter_values=None,
):
    """Train and evaluate a K-nearest neighbors classifier on the data splits."""
    if not test_data_information or "test_info" not in test_data_information:
        raise ValueError("test_data_information must contain a test_info list")
    if undersampling_parameter_values and not undersampling_technique:
        raise ValueError(
            "undersampling_technique is required when undersampling_parameter_values "
            "is informed"
        )
    if execution_device == "gpu":
        raise ValueError("KNN currently supports CPU execution only")

    results = []
    for test_index, test_info in enumerate(test_data_information["test_info"]):
        train_data = test_info.get("train_data")
        if train_data is None or train_data.empty:
            raise ValueError(f"test_info[{test_index}] must contain non-empty train_data")
        if undersampling_technique:
            train_data = resample_dataframe(
                train_data,
                prediction_column_name,
                undersampling_technique,
                undersampling_parameter_values,
            )

        train_features, train_target = _linear_regression_features_and_target(
            train_data,
            prediction_column_name,
            feature_column_list=feature_column_list,
        )
        if train_features is None:
            raise ValueError(
                f"test_info[{test_index}] does not have train rows with target values"
            )
        class_values = sorted(train_target.dropna().unique())
        if len(class_values) < 2:
            raise ValueError(
                f"test_info[{test_index}] must contain at least two target classes"
            )

        fit_started_at = time.perf_counter()
        feature_scaler = StandardScaler()
        scaled_train_features = feature_scaler.fit_transform(train_features)
        neighbor_count = min(5, len(train_features))
        model = KNeighborsClassifier(n_neighbors=neighbor_count)
        model.fit(scaled_train_features, train_target)
        fit_time = time.perf_counter() - fit_started_at

        feature_columns = list(train_features.columns)
        scoring_started_at = time.perf_counter()
        split_scores = {}
        for split_name in ["train_data", "validation_data", "test_data"]:
            split_data = (
                train_data if split_name == "train_data" else test_info.get(split_name)
            )
            split_scores[split_name] = _knn_predict(
                model,
                feature_scaler,
                split_data,
                prediction_column_name,
                feature_columns,
                split_name,
                class_values,
            )

        scoring_time = time.perf_counter() - scoring_started_at

        results.append(
            {
                "test_info_index": test_index,
                # Tempos da dobra (item G18): ajuste, com a padronizacao, e
                # pontuacao dos tres conjuntos.
                "fit_time": round(fit_time, 6),
                "scoring_time": round(scoring_time, 6),
                # Indice 1-based da dobra e MD5 dos arquivos lidos, ambos
                # propagados do manifesto ate a linha de resultado.
                "fold_index": test_info.get("fold_index", test_index + FIRST_FOLD_INDEX),
                "split_md5": test_info.get("split_md5", {}),
                "partition_md5": test_info.get("partition_md5", {}),
                "train_population_md5": test_info.get("train_population_md5", ""),
                "unit_partition_md5": test_info.get("unit_partition_md5", {}),
                "train_unit_population_md5": test_info.get("train_unit_population_md5", ""),
                "evaluation_units": test_info.get("evaluation_units", {}),
                "model": model,
                "feature_scaler": feature_scaler,
                "feature_columns": feature_columns,
                "class_values": class_values,
                "neighbor_count": neighbor_count,
                "execution_device": "cpu",
                "split_sizes": {
                    "train_data": _dataframe_size(train_data),
                    "validation_data": _dataframe_size(test_info.get("validation_data")),
                    "test_data": _dataframe_size(test_info.get("test_data")),
                },
                "split_scores": split_scores,
            }
        )

    results = apply_decision_thresholds_by_unit(
        results, threshold_strategy, classification_threshold
    )

    return {
        "algorithm": "knn",
        "prediction_column_name": prediction_column_name,
        "train_test_strategy": test_data_information.get("train_test_strategy"),
        "execution_device": "cpu",
        "undersampling_technique": undersampling_technique,
        "undersampling_parameter_values": undersampling_parameter_values,
        "results": results,
    }


def execute_prediction_algorithm(algorithm, test_data_information, **parameters):
    """Dispatch execution to a supported prediction algorithm."""
    normalized_algorithm = str(algorithm).strip().lower().replace("_", "-")
    algorithm_executors = {
        "linear-regression": execute_linear_regression,
        "logistic-regression": execute_logistic_regression,
        "naive-bayes": execute_naive_bayes,
        "knn": execute_knn,
    }
    if normalized_algorithm not in algorithm_executors:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Supported algorithms: "
            f"{', '.join(sorted(algorithm_executors))}"
        )
    executor = algorithm_executors[normalized_algorithm]
    # Naive Bayes e KNN nao tem nocao de peso por classe: o primeiro estima
    # densidades condicionais, o segundo decide por votacao entre vizinhos.
    # Passar a intensidade a eles seria um erro de tipo, entao o parametro e
    # entregue apenas a quem sabe usa-lo.
    accepted = inspect.signature(executor).parameters
    applicable = {
        name: value for name, value in parameters.items() if name in accepted
    }
    return executor(test_data_information, **applicable)


def _dataframe_size(data):
    if data is None:
        return 0
    return len(data)


_HELD_OUT_SPLIT_LABELS = frozenset(
    SPLIT_FILE_LABELS[split_name] for split_name in EVALUATION_UNIT_SPLITS
)


def _requires_evaluation_units(manifest_rows, dataset_md5):
    """True quando o split e de validacao/teste de uma base com unidade de avaliacao."""
    if not {"database_type", "split"}.issubset(manifest_rows.columns):
        return False
    rows = manifest_rows[manifest_rows["dataset_md5"].str.strip().str.lower() == dataset_md5]
    for row in rows.itertuples(index=False):
        if row.split not in _HELD_OUT_SPLIT_LABELS:
            continue
        try:
            definition = experiment_datasets.get_dataset_definition(row.database_type)
        except ValueError:
            continue
        if definition.evaluation_unit_column:
            return True
    return False


def _requires_history_strata(manifest_rows, dataset_md5):
    """True quando o split avaliado deve trazer o estrato de historico (item C3.19).

    Vale para os modos nativo e do FT-Transformer, nas bases e no protocolo
    do TabBERT (experiment_datasets.records_history_strata).
    """
    required_columns = {"database_type", "split", "test_strategy", "categorical_handling_mode"}
    if not required_columns.issubset(manifest_rows.columns):
        return False
    rows = manifest_rows[manifest_rows["dataset_md5"].str.strip().str.lower() == dataset_md5]
    for row in rows.itertuples(index=False):
        if row.split not in _HELD_OUT_SPLIT_LABELS:
            continue
        if normalize_categorical_handling_mode(row.categorical_handling_mode) not in {
            CATEGORICAL_HANDLING_MODE_NATIVE,
            CATEGORICAL_HANDLING_MODE_TRANSFORMER,
        }:
            continue
        try:
            definition = experiment_datasets.get_dataset_definition(row.database_type)
        except ValueError:
            continue
        test_strategy = str(row.test_strategy).split(":", maxsplit=1)[0]
        if experiment_datasets.records_history_strata(definition, test_strategy):
            return True
    return False


def _records_column_registry(manifest_rows, dataset_md5):
    """True quando o manifesto registra o hash de colunas desse arquivo (item G14).

    Manifestos anteriores nao tem a coluna; nesse caso o JSON lateral nao e
    exigido.
    """
    if "feature_schema_md5" not in manifest_rows.columns:
        return False
    rows = manifest_rows[manifest_rows["dataset_md5"].str.strip().str.lower() == dataset_md5]
    recorded = rows["feature_schema_md5"].fillna("").astype(str).str.strip()
    return bool((recorded != "").any())


def _preparation_report_files(manifest_rows, source, configuration_md5):
    """Relatorios de preparacao prometidos pelo manifesto (item G15).

    Ausente ou com MD5 diferente do gravado, o relatorio nao audita mais o
    arquivo lido: e falha de armazenamento, como os demais laterais.
    Manifestos sem a coluna nao exigem relatorio.
    """
    if "preparation_report_md5" not in manifest_rows.columns:
        return []
    report_paths = []
    recorded = manifest_rows[["fold_index", "preparation_report_md5"]].fillna("")
    for fold_index, report_md5 in recorded.drop_duplicates().itertuples(index=False):
        report_md5 = str(report_md5).strip().lower()
        if not report_md5:
            continue
        report_path = preparation_report.preparation_report_path(
            source, configuration_md5, int(fold_index)
        )
        if not report_path.is_file():
            raise PreparedDatasetStorageError(
                f"Preparation report file was not found: {report_path}"
            )
        if compute_file_content_md5(report_path) != report_md5:
            raise PreparedDatasetStorageError(
                f"Preparation report does not match the manifest MD5: {report_path}"
            )
        report_paths.append(report_path)
    return report_paths


@contextmanager
def stage_prepared_dataset_files(
    source_directory,
    manifest_path,
    configuration_md5,
    staging_root,
):
    """Copy all split files for one configuration to an isolated local directory."""
    normalized_configuration_md5 = str(configuration_md5).strip().lower()
    source = Path(source_directory)
    manifest = normalize_manifest_dataset_identity(
        pd.read_csv(manifest_path, sep="\t", dtype=str)
    )
    required_columns = {"configuration_md5", "dataset_md5"}
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise ValueError(
            "Prepared dataset manifest is missing columns: "
            f"{', '.join(missing_columns)}"
        )

    matching_rows = manifest[
        manifest["configuration_md5"].str.lower()
        == normalized_configuration_md5
    ]
    if matching_rows.empty:
        raise FileNotFoundError(
            "No manifest rows matched configuration MD5 "
            f"'{normalized_configuration_md5}'"
        )

    source_files = []
    expected_split_md5: dict[Path, str] = {}
    for dataset_md5_value in matching_rows["dataset_md5"].drop_duplicates():
        dataset_md5 = str(dataset_md5_value).strip().lower()
        if len(dataset_md5) != 32 or any(
            character not in "0123456789abcdef" for character in dataset_md5
        ):
            raise ValueError(f"Invalid dataset_md5 in manifest: {dataset_md5_value}")
        source_path = prepared_split_path(source, dataset_md5)
        if source_path is None:
            raise PreparedDatasetStorageError(
                "Prepared dataset file was not found: "
                f"{source / f'{dataset_md5}.csv'} (nor .csv.zst)"
            )
        if "content_md5" in matching_rows:
            recorded = matching_rows.loc[
                matching_rows["dataset_md5"].str.strip().str.lower() == dataset_md5, "content_md5"
            ].dropna()
            if not recorded.empty and str(recorded.iloc[0]).strip():
                expected_split_md5[source_path] = str(recorded.iloc[0]).strip().lower()
        source_files.append(source_path)
        units_path = evaluation_unit_path(source, dataset_md5)
        if units_path.is_file():
            source_files.append(units_path)
        elif _requires_evaluation_units(matching_rows, dataset_md5):
            # Sem o arquivo lateral a base seria avaliada so por linha, em
            # silencio. Como os demais arquivos ausentes, vira falha de
            # armazenamento: a configuracao e repetida, nunca descartada.
            raise PreparedDatasetStorageError(
                f"Evaluation unit file was not found: {units_path}"
            )
        strata_path = history_strata_path(source, dataset_md5)
        if strata_path.is_file():
            source_files.append(strata_path)
        elif _requires_history_strata(matching_rows, dataset_md5):
            # Sem o estrato o modelo perderia, em silencio, o corte por
            # historico (item C3.19). Falha de armazenamento, como os demais.
            raise PreparedDatasetStorageError(
                f"History strata file was not found: {strata_path}"
            )
        registry_path = column_registry_path(source, dataset_md5)
        if registry_path.is_file():
            source_files.append(registry_path)
        elif _records_column_registry(matching_rows, dataset_md5):
            # O manifesto promete o registro de colunas; sem o JSON o arquivo
            # nao pode ser auditado. Como os demais laterais, e falha de
            # armazenamento: a configuracao e repetida, nunca descartada.
            raise PreparedDatasetStorageError(
                f"Column registry file was not found: {registry_path}"
            )
    source_files.extend(
        _preparation_report_files(matching_rows, source, normalized_configuration_md5)
    )

    local_staging_root = Path(staging_root)
    local_staging_root.mkdir(parents=True, exist_ok=True)
    resolved_source = source.resolve()
    resolved_staging_root = local_staging_root.resolve()
    if (
        resolved_staging_root == resolved_source
        or resolved_source in resolved_staging_root.parents
    ):
        raise ValueError(
            "The local staging directory must be outside the source directory"
        )
    staged_directory = Path(
        tempfile.mkdtemp(
            prefix=f"{normalized_configuration_md5}_",
            dir=resolved_staging_root,
        )
    ).resolve()
    try:
        for source_path in source_files:
            try:
                if is_compressed_split(source_path):
                    # Os executores leem sempre o CSV local; a copia e
                    # descomprimida e conferida pelo content_md5 na escrita.
                    materialize_prepared_split(
                        source_path,
                        staged_directory / source_path.name.removesuffix(".zst"),
                        expected_split_md5.get(source_path),
                    )
                else:
                    shutil.copy2(source_path, staged_directory / source_path.name)
            except OSError as error:
                raise PreparedDatasetStorageError(
                    f"Could not copy prepared dataset file: {source_path}"
                ) from error
        yield staged_directory
    finally:
        if staged_directory.parent != resolved_staging_root:
            raise RuntimeError(
                f"Refusing to clean an unexpected directory: {staged_directory}"
            )
        if staged_directory == resolved_source or resolved_source in staged_directory.parents:
            raise RuntimeError(
                f"Refusing to clean the prepared dataset source: {resolved_source}"
            )
        shutil.rmtree(staged_directory)


# Esquema canonico compartilhado por classicos, boosting nativo e Transformers.
# `regime` e `fold_index`/`split` permitem parear a mesma unidade de avaliacao
# entre algoritmos; `regime` fica sempre vazio aqui, pois nenhum destes
# algoritmos tem variantes de treinamento como os dois regimes do TabBERT.
RESULT_CSV_COLUMNS = [
    "configuration_md5",
    "dataset_id",
    "dataset_name",
    "database_type",
    "dataset_filename",
    "algorithm",
    "regime",
    "fold_index",
    "split",
    # Unidade das metricas: row (cada registro) ou client (escores agregados
    # por cliente, so nas bases com unidade de avaliacao; item G7).
    "evaluation_unit",
    "execution_status",
    # Motivo da falha (item G17): categoria fechada e mensagem da excecao;
    # vazios nas linhas de sucesso. Ver execution_failures.py.
    "failure_category",
    "failure_message",
    "undersampling_technique",
    "random_seed",
    "classification_threshold",
    # Como o limiar foi obtido: validation-f2 (validacao da propria dobra,
    # aplicado ao teste), cross-fold-validation-f2 (validacao das outras
    # dobras, no k-fold) ou fixed. Ver decision_threshold.py (item G11-i).
    "threshold_strategy",
    # Intensidade da ponderacao de classe e a razao R = negativos/positivos
    # efetivamente medida no treino daquela dobra. Guardar R, e nao so o
    # rotulo da intensidade, permite reconstruir o peso aplicado.
    "class_weight_intensity",
    "class_weight_ratio",
    # MD5 do conteudo de cada arquivo efetivamente lido. Permite auditar, a
    # partir da linha de resultado, exatamente quais bytes de treino,
    # validacao e teste produziram aquela metrica.
    "train_dataset_md5",
    "validation_dataset_md5",
    "test_dataset_md5",
    # Identidade da dobra calculada sobre as linhas brutas de treino, validacao
    # e teste, antes de qualquer transformacao. E a chave de pareamento entre
    # blocos: mesmo partition_id, mesmas linhas avaliadas (item G16).
    "partition_id",
    # Linhas brutas do treino efetivo, depois da remocao de incompletos e
    # antes da reamostragem: base e variante de ablacao com o mesmo valor
    # treinaram com as mesmas linhas (item G16).
    "train_population_id",
    # Unidades de avaliacao (clientes) de cada split: pareia bases com linhas
    # diferentes e os mesmos clientes, como as duas Home Credit; vazio nas
    # bases sem unidade de avaliacao.
    "unit_partition_id",
    # Clientes do treino efetivo, nas bases com unidade de avaliacao: no braco
    # de remocao, as duas Home Credit com o mesmo valor treinaram com os
    # mesmos clientes (item G16).
    "train_unit_population_id",
    "train_df_size",
    "validation_df_size",
    "test_df_size",
    "categorical_encoding_strategy",
    "fill_na_strategy",
    "train_validation_test_strategy",
    "categorical_apply_time",
    "fillna_apply_time",
    "dataset_setup_time",
    "algorithm_time",
    "elapsed_time",
    "f1",
    "f_beta_2",
    "accuracy",
    "precision",
    "recall",
    "roc_auc",
    # AUC-PR (precisao media). Sob forte desbalanceamento ela e mais
    # informativa que a AUC-ROC, mas depende da prevalencia: compara
    # algoritmos DENTRO de uma base, nunca entre bases (item G11-iii).
    "auc_pr",
    "false_positive_rate",
    # Metadados de execucao dos Transformers; vazios para os demais algoritmos,
    # que rodam em CPU de forma unica e nao tem lote/precisao/semente a registrar.
    "execution_device",
    "execution_device_name",
    "execution_precision",
    "execution_batch_size",
    "execution_num_workers",
    "execution_seed",
    "execution_deterministic",
    # Fraudes rotuladas que o algoritmo recebeu no treino (item G3 da
    # revisao): -1 quando o eixo nao se aplica, isto e, em todo algoritmo
    # supervisionado com rotulos completos; 0 no Isolation Forest; k no DevNet.
    "label_budget",
    # Fracao das fraudes do treino que ficou rotulada, o parametro do braco do
    # DevNet (0,01, 0,05 e 0,10, como sugere a revisao): label_budget e o
    # numero efetivo da dobra, ceil(fracao x fraudes do treino). -1 quando o
    # eixo nao se aplica; 0 no Isolation Forest.
    "label_fraction",
    # JSON com os hiperparametros efetivos do modelo da dobra, inclusive os
    # padroes resolvidos pela biblioteca, e JSON com o que o treino produziu.
    # Ver model_metadata.py.
    "model_hyperparameters",
    "training_summary",
    # Tempos por dobra (item G18): fit_time e o ajuste do modelo (no TabBERT,
    # MLM e ajuste fino juntos; o MLM tambem em pretraining_time) e
    # scoring_time a pontuacao dos tres conjuntos. algorithm_time continua
    # sendo o total da execucao de todas as dobras.
    "fit_time",
    "scoring_time",
    "pretraining_time",
    # Proveniencia (item G18): commit, ambiente travado e maquina de cada
    # linha. A reexecucao roda em tres maquinas; estas colunas dizem de onde
    # veio cada resultado e nunca entram em chave de retomada ou pareamento.
    # Ver execution_provenance.py.
    *execution_provenance.PROVENANCE_COLUMNS,
]

# Valor de label_budget quando o algoritmo recebe todos os rotulos do treino.
LABEL_BUDGET_NOT_APPLICABLE = -1
LABEL_FRACTION_NOT_APPLICABLE = -1


def _recorded_numeric_values(recorded_results, column, missing_value):
    """Valores numericos de uma coluna, com o valor implicito das linhas antigas.

    A leitura e numerica, e nao textual, porque um CSV regravado pelo pandas
    pode trazer "-1.0" onde foi escrito "-1".
    """
    if column not in recorded_results.columns:
        return pd.Series(missing_value, index=recorded_results.index, dtype=float)
    return pd.to_numeric(recorded_results[column], errors="coerce").fillna(missing_value)


def is_experiment_already_recorded(
    results_directory,
    configuration_md5,
    algorithm,
    classification_threshold=None,
    class_weight_intensity=None,
    threshold_strategy=None,
    label_budget=None,
    execution_seed=None,
    label_fraction=None,
):
    """Return whether an algorithm/configuration pair completed successfully.

    A estrategia de limiar faz parte da chave: uma execucao antiga com limiar
    fixo nao pode impedir a execucao com limiar escolhido na validacao. Linhas
    sem a coluna ``threshold_strategy`` sao anteriores a ela e contam como
    ``fixed``. O valor do limiar so entra na chave no modo ``fixed``, porque no
    modo escolhido ele e resultado da execucao, nao parametro.

    ``label_budget``, quando informado, tambem entra na chave: orcamentos
    diferentes da mesma configuracao sao execucoes diferentes. Linhas sem a
    coluna sao anteriores a ela e contam como ``-1``. ``execution_seed``,
    quando informado, distingue os sorteios de rotulos do DevNet.
    ``label_fraction``, quando informado, identifica o braco do DevNet: o
    numero de fraudes rotuladas so e conhecido depois de carregar a dobra.
    """
    normalized_configuration_md5 = str(configuration_md5).strip().lower()
    normalized_algorithm = str(algorithm).strip().lower()
    normalized_threshold = (
        None if classification_threshold is None else f"{float(classification_threshold):g}"
    )
    normalized_intensity = (
        None
        if class_weight_intensity is None
        else str(class_weight_intensity).strip().lower()
    )
    normalized_threshold_strategy = (
        None
        if threshold_strategy is None
        else str(threshold_strategy).strip().lower()
    )

    results_directory_path = Path(results_directory)
    # ``resultas_`` is a historical typo that is part of the persisted-results
    # compatibility contract.  New files keep using ``resultados_``.
    candidate_result_files = sorted(
        {
            *results_directory_path.glob("resultados_*.csv"),
            *results_directory_path.glob("resultas_*.csv"),
        }
    )
    for results_path in candidate_result_files:
        try:
            recorded_results = pd.read_csv(
                results_path,
                dtype=str,
                keep_default_na=False,
            )
        except (pd.errors.EmptyDataError, ValueError):
            # Empty or malformed result files cannot identify an experiment.
            continue

        required_identity_columns = {"configuration_md5", "algorithm"}
        if not required_identity_columns.issubset(recorded_results.columns):
            continue

        recorded_results, statuses_were_added = _backfill_legacy_execution_statuses(
            recorded_results
        )
        if statuses_were_added:
            recorded_results.to_csv(results_path, index=False)

        configuration_matches = (
            recorded_results["configuration_md5"].str.strip().str.lower()
            == normalized_configuration_md5
        )
        algorithm_matches = (
            recorded_results["algorithm"].str.strip().str.lower()
            == normalized_algorithm
        )
        identity_matches = configuration_matches & algorithm_matches
        if normalized_threshold_strategy is not None:
            recorded_strategies = (
                recorded_results["threshold_strategy"]
                if "threshold_strategy" in recorded_results.columns
                else pd.Series("", index=recorded_results.index)
            ).fillna("").astype(str).str.strip().str.lower().replace("", THRESHOLD_STRATEGY_FIXED)
            if normalized_threshold_strategy == THRESHOLD_STRATEGY_FIXED:
                identity_matches = identity_matches & (
                    recorded_strategies == THRESHOLD_STRATEGY_FIXED
                )
            else:
                identity_matches = identity_matches & recorded_strategies.isin(
                    SELECTED_THRESHOLD_STRATEGIES
                )
        if (
            normalized_threshold is not None
            and "classification_threshold" in recorded_results.columns
        ):
            recorded_thresholds = pd.to_numeric(
                recorded_results["classification_threshold"], errors="coerce"
            ).map(lambda value: "" if pd.isna(value) else f"{value:g}")
            identity_matches = identity_matches & (
                recorded_thresholds == normalized_threshold
            )

        if (
            normalized_intensity is not None
            and "class_weight_intensity" in recorded_results.columns
        ):
            identity_matches = identity_matches & (
                recorded_results["class_weight_intensity"]
                .fillna("")
                .astype(str)
                .str.strip()
                .str.lower()
                == normalized_intensity
            )

        if label_budget is not None:
            identity_matches = identity_matches & (
                _recorded_numeric_values(
                    recorded_results, "label_budget", LABEL_BUDGET_NOT_APPLICABLE
                )
                == int(label_budget)
            )

        if label_fraction is not None:
            identity_matches = identity_matches & np.isclose(
                _recorded_numeric_values(
                    recorded_results, "label_fraction", LABEL_FRACTION_NOT_APPLICABLE
                ),
                float(label_fraction),
            )

        if execution_seed not in (None, ""):
            recorded_seeds = (
                pd.to_numeric(recorded_results["execution_seed"], errors="coerce")
                if "execution_seed" in recorded_results.columns
                else pd.Series(np.nan, index=recorded_results.index)
            )
            identity_matches = identity_matches & (recorded_seeds == int(execution_seed))

        matching_results = recorded_results[identity_matches]
        if matching_results.empty:
            continue

        successful_results = (
            matching_results["execution_status"].str.strip().str.lower()
            == "success"
        )
        if successful_results.any():
            return True

    return False


def _backfill_legacy_execution_statuses(results_dataframe):
    """Infer and persist statuses where legacy metrics provide a clear outcome."""
    updated_results = results_dataframe.copy()
    status_column_was_added = "execution_status" not in updated_results.columns
    if status_column_was_added:
        algorithm_column_index = updated_results.columns.get_loc("algorithm")
        updated_results.insert(algorithm_column_index + 1, "execution_status", "")

    normalized_statuses = (
        updated_results["execution_status"].fillna("").astype(str).str.strip().str.lower()
    )
    legacy_rows = normalized_statuses == ""
    metric_columns = [
        column
        for column in (
            "f1", "f_beta_2", "accuracy", "precision", "recall", "roc_auc", "auc_pr"
        )
        if column in updated_results.columns
    ]
    if not legacy_rows.any() or not metric_columns:
        return updated_results, status_column_was_added

    numeric_metrics = updated_results.loc[legacy_rows, metric_columns].apply(
        pd.to_numeric,
        errors="coerce",
    )
    inferred_successes = (
        numeric_metrics.notna() & numeric_metrics.ne(-1)
    ).any(axis=1)
    inferred_failures = numeric_metrics.notna().all(axis=1) & numeric_metrics.eq(-1).all(
        axis=1
    )

    updated_results.loc[inferred_successes.index[inferred_successes], "execution_status"] = (
        "success"
    )
    updated_results.loc[inferred_failures.index[inferred_failures], "execution_status"] = (
        "failure"
    )
    results_were_updated = bool(
        status_column_was_added
        or inferred_successes.any()
        or inferred_failures.any()
    )
    return updated_results, results_were_updated


def _ensure_results_csv_schema(output_csv_filename):
    if not os.path.exists(output_csv_filename):
        return

    try:
        current_results_dataframe = pd.read_csv(output_csv_filename)
    except pd.errors.EmptyDataError:
        return

    current_results_dataframe, statuses_were_added = (
        _backfill_legacy_execution_statuses(current_results_dataframe)
    )
    if (
        list(current_results_dataframe.columns) == RESULT_CSV_COLUMNS
        and not statuses_were_added
    ):
        return

    updated_results_dataframe = current_results_dataframe.reindex(
        columns=RESULT_CSV_COLUMNS
    )
    updated_results_dataframe.to_csv(output_csv_filename, index=False)


# Ordem em que os splits de uma dobra sao gravados. E a ordem de leitura
# natural de uma tabela de resultados, nao uma prioridade: todos os splits
# disponiveis viram linha.
RECORDED_SPLIT_ORDER = ("train_data", "validation_data", "test_data")


def _iterate_execution_metrics(execution_result):
    """Itera (nome_do_split, metricas) por TODOS os splits disponiveis.

    A versao anterior escolhia um unico split, com prioridade
    test > validation > train. Como os tres protocolos sempre produzem teste,
    nenhuma linha `split=validation` chegava ao CSV: as metricas de validacao
    eram calculadas e descartadas na escrita, e a fase de analise nao tinha em
    que selecionar a melhor configuracao -- a selecao acabava recaindo sobre a
    metrica de teste, que e exatamente a queixa do item G12 da revisao.

    Gravar uma linha por split nao custa calculo algum nos blocos classico e de
    boosting, onde as metricas dos tres splits ja estavam em `split_results`.
    """
    split_results = execution_result.get("split_results", {})
    for split_name in RECORDED_SPLIT_ORDER:
        split_result = split_results.get(split_name)
        if split_result is None:
            continue
        metrics = split_result.get("metrics")
        if metrics is None:
            continue
        yield split_name.removesuffix("_data"), metrics


def _select_execution_metrics(execution_result):
    """Devolve (nome_do_split, metricas) do split reportado por convencao.

    Mantida para quem precisa de um unico split representativo. A prioridade
    test > validation > train e a mesma de antes, e e tambem o padrao de leitura
    do dashboard (`DEFAULT_EVALUATION_SPLIT`). A GRAVACAO, porem, nao usa mais
    esta funcao: ela percorre todos os splits.
    """
    split_results = execution_result.get("split_results", {})
    for split_name in ("test_data", "validation_data", "train_data"):
        split_result = split_results.get(split_name)
        if split_result is not None:
            return split_name.removesuffix("_data"), split_result.get("metrics")
    return None, None


def _fold_model_metadata(execution_result):
    """Orcamento de rotulos e metadados do modelo de uma dobra.

    Executores que nao guardam o modelo na dobra (boosting, Transformers,
    anomalia) entregam os metadados prontos; os classicos guardam o modelo
    ajustado em ``model``, e os metadados sao extraidos dele aqui.
    """
    model = execution_result.get("model")
    hyperparameters = execution_result.get("model_hyperparameters")
    if hyperparameters is None:
        hyperparameters = describe_model_hyperparameters(model)
    training_summary = execution_result.get("training_summary")
    if training_summary is None:
        training_summary = summarize_training(model)
    return {
        "label_budget": execution_result.get(
            "label_budget", LABEL_BUDGET_NOT_APPLICABLE
        ),
        "label_fraction": execution_result.get(
            "label_fraction", LABEL_FRACTION_NOT_APPLICABLE
        ),
        "model_hyperparameters": serialize_metadata(hyperparameters),
        "training_summary": serialize_metadata(training_summary),
    }


# Escores do teste por linha (item G20): companheiro ``.jsonl`` do CSV de
# resultados, com um registro por (dobra, unidade de avaliacao). Permitem o
# bootstrap pareado da diferenca de F2 entre duas execucoes avaliadas nas
# mesmas linhas (analise_hipoteses.py). Escores em float32 little-endian e
# rotulos binarios em bits (np.packbits), ambos em base64, para que cada
# registro caiba numa linha e os arquivos de maquinas diferentes possam ser
# concatenados como os demais companheiros.
TEST_SCORES_FILE_SUFFIX = "_test_scores.jsonl"
TEST_SCORES_RECORD_TYPE = "test_scores"
TEST_SCORES_ENCODING = "scores=float32-le-base64;targets=packbits-base64"
TEST_SCORES_SPLIT = "test_data"


def test_scores_path(output_csv_filename):
    """``resultados_<algoritmo>.csv`` -> ``resultados_<algoritmo>_test_scores.jsonl``."""
    base = str(output_csv_filename)
    if base.endswith(".csv"):
        base = base[: -len(".csv")]
    return base + TEST_SCORES_FILE_SUFFIX


def encode_test_scores(predictions, class_values):
    """Escores e rotulos de um split ja pontuado, codificados para o ``.jsonl``.

    Devolve None fora do caso binario ou sem a coluna de escore.
    """
    if predictions is None or len(class_values) != 2:
        return None
    score_columns = [
        column
        for column in predictions.columns
        if column not in ("real_value", "predicted_value")
    ]
    if len(score_columns) != 1:
        return None
    positive_class = sorted(class_values)[-1]
    targets = (predictions["real_value"].to_numpy() == positive_class).astype(np.uint8)
    scores = predictions[score_columns[0]].to_numpy(dtype="<f4")
    return {
        "rows": int(len(targets)),
        "positives": int(targets.sum()),
        "encoding": TEST_SCORES_ENCODING,
        "scores": base64.b64encode(scores.tobytes()).decode("ascii"),
        "targets": base64.b64encode(np.packbits(targets).tobytes()).decode("ascii"),
    }


def decode_test_scores(record):
    """Inverso de encode_test_scores: (rotulos 0/1, escores float32)."""
    rows = int(record["rows"])
    scores = np.frombuffer(base64.b64decode(record["scores"]), dtype="<f4")
    targets = np.unpackbits(
        np.frombuffer(base64.b64decode(record["targets"]), dtype=np.uint8)
    )[:rows]
    return targets.astype(np.uint8), scores[:rows].astype(np.float64)


def append_test_score_records(path, records):
    if not records:
        return
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "a", encoding="utf-8") as stream:
        for record in records:
            clean = {key: _json_scalar(value) for key, value in record.items()}
            stream.write(json.dumps(clean, allow_nan=False) + "\n")


def _json_scalar(value):
    """Escalares numpy viram nativos; NaN e infinito viram null."""
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def record_algorithm_result_to_csv(
    output_csv_filename,
    dataset_filename,
    algorithm_execution_result,
    categorical_encoding_strategy,
    fill_na_strategy,
    elapsed_time,
    categorical_apply_time=None,
    fillna_apply_time=None,
    dataset_setup_time=None,
    algorithm_time=None,
    configuration_md5=None,
    database_type=None,
    dataset_id=None,
    dataset_name=None,
    regime="",
    random_seed=DEFAULT_RANDOM_SEED,
    classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    class_weight_intensity=DEFAULT_CLASS_WEIGHT_INTENSITY,
    threshold_strategy=THRESHOLD_STRATEGY_FIXED,
    record_test_scores=True,
):
    """
    Append algorithm execution metrics to a CSV file.

    Com ``record_test_scores``, os escores do teste de cada dobra vao para o
    companheiro ``test_scores_path(output_csv_filename)``. Execucoes cujos
    resultados nao trazem as predicoes (os Transformers, que gravam as suas no
    ``_details.jsonl``) nao geram registro.

    O limiar e a estrategia gravados sao os de cada dobra, quando a execucao
    os escolheu; os argumentos so valem para quem nao escolhe limiar.

    Uma linha por (dobra, split): para cada dobra em
    algorithm_execution_result["results"], todos os splits avaliados viram
    linha, distinguidos pela coluna `split`. A selecao da melhor configuracao,
    na fase de analise, deve usar as linhas `split=validation` e reservar
    `split=test` para o relato final (item G12 da revisao).
    """
    if not algorithm_execution_result or "results" not in algorithm_execution_result:
        raise ValueError("algorithm_execution_result must contain a results list")

    dataset_identity = experiment_datasets.resolve_dataset_identity(
        dataset_id=dataset_id,
        database_type=database_type,
        dataset_name=dataset_name,
    )

    rows = []
    score_records = []
    for fold_position, execution_result in enumerate(algorithm_execution_result["results"]):
        # Dobras sao numeradas a partir de 1; politicas sem dobras gravam a
        # particao unica como dobra 1, nunca como vazio.
        fold_index = execution_result.get(
            "fold_index", fold_position + FIRST_FOLD_INDEX
        )
        if record_test_scores:
            test_split = (execution_result.get("split_results") or {}).get(TEST_SCORES_SPLIT)
            encoded = encode_test_scores(
                (test_split or {}).get("predictions"),
                execution_result.get("class_values", ()),
            )
            if encoded is not None:
                score_records.append(
                    {
                        "record_type": TEST_SCORES_RECORD_TYPE,
                        "configuration_md5": configuration_md5,
                        **dataset_identity,
                        "algorithm": algorithm_execution_result.get("algorithm"),
                        "regime": regime,
                        "fold_index": int(fold_index),
                        "split": "test",
                        "evaluation_unit": execution_result.get(
                            "evaluation_unit", EVALUATION_UNIT_ROW
                        ),
                        "train_validation_test_strategy": algorithm_execution_result.get(
                            "train_test_strategy"
                        ),
                        "undersampling_technique": algorithm_execution_result.get(
                            "undersampling_technique"
                        ),
                        "class_weight_intensity": execution_result.get(
                            "class_weight_intensity", class_weight_intensity
                        ),
                        "threshold_strategy": execution_result.get(
                            "threshold_strategy", threshold_strategy
                        ),
                        "classification_threshold": execution_result.get(
                            "classification_threshold", classification_threshold
                        ),
                        "categorical_encoding_strategy": categorical_encoding_strategy,
                        "fill_na_strategy": fill_na_strategy,
                        "random_seed": random_seed,
                        "test_partition_md5": (
                            execution_result.get("partition_md5") or {}
                        ).get(TEST_SCORES_SPLIT, ""),
                        "test_dataset_md5": (
                            execution_result.get("split_md5") or {}
                        ).get(TEST_SCORES_SPLIT, ""),
                        **encoded,
                    }
                )
        split_sizes = execution_result.get("split_sizes", {})
        split_md5 = execution_result.get("split_md5", {})
        partition_md5_by_split = execution_result.get("partition_md5", {})
        partition_id = (
            data_partitioning.fold_partition_id(partition_md5_by_split)
            if partition_md5_by_split
            else ""
        )
        unit_partition_md5_by_split = execution_result.get("unit_partition_md5", {})
        unit_partition_id = (
            data_partitioning.fold_partition_id(unit_partition_md5_by_split)
            if unit_partition_md5_by_split
            else ""
        )
        fold_metadata = _fold_model_metadata(execution_result)
        for split_name, metrics in _iterate_execution_metrics(execution_result):
            rows.append(
                {
                    "configuration_md5": configuration_md5,
                    **dataset_identity,
                    "dataset_filename": dataset_filename,
                    "algorithm": algorithm_execution_result.get("algorithm"),
                    "regime": regime,
                    "fold_index": fold_index,
                    "split": split_name,
                    "evaluation_unit": execution_result.get(
                        "evaluation_unit", EVALUATION_UNIT_ROW
                    ),
                    "execution_status": "success",
                    "undersampling_technique": algorithm_execution_result.get(
                        "undersampling_technique"
                    ),
                    "random_seed": random_seed,
                    "classification_threshold": execution_result.get(
                        "classification_threshold", classification_threshold
                    ),
                    "threshold_strategy": execution_result.get(
                        "threshold_strategy", threshold_strategy
                    ),
                    "class_weight_intensity": execution_result.get(
                        "class_weight_intensity", class_weight_intensity
                    ),
                    "class_weight_ratio": execution_result.get("class_weight_ratio", ""),
                    "train_dataset_md5": split_md5.get("train_data", ""),
                    "validation_dataset_md5": split_md5.get("validation_data", ""),
                    "test_dataset_md5": split_md5.get("test_data", ""),
                    "partition_id": partition_id,
                    "train_population_id": execution_result.get("train_population_md5", ""),
                    "unit_partition_id": unit_partition_id,
                    "train_unit_population_id": execution_result.get(
                        "train_unit_population_md5", ""
                    ),
                    "train_df_size": split_sizes.get("train_data", 0),
                    "validation_df_size": split_sizes.get("validation_data", 0),
                    "test_df_size": split_sizes.get("test_data", 0),
                    "categorical_encoding_strategy": categorical_encoding_strategy,
                    "fill_na_strategy": fill_na_strategy,
                    "train_validation_test_strategy": algorithm_execution_result.get("train_test_strategy"),
                    "categorical_apply_time": categorical_apply_time,
                    "fillna_apply_time": fillna_apply_time,
                    "dataset_setup_time": dataset_setup_time,
                    "algorithm_time": algorithm_time,
                    "elapsed_time": elapsed_time,
                    "f1": metrics.get("f1"),
                    "f_beta_2": metrics.get("f_beta_2"),
                    "accuracy": metrics.get("accuracy"),
                    "precision": metrics.get("precision"),
                    "recall": metrics.get("recall"),
                    "roc_auc": metrics.get("roc_auc"),
                    "auc_pr": metrics.get("auc_pr"),
                    "false_positive_rate": metrics.get("false_positive_rate"),
                    "execution_device": metrics.get("execution_device"),
                    "execution_device_name": metrics.get("execution_device_name"),
                    "execution_precision": metrics.get("execution_precision"),
                    "execution_batch_size": metrics.get("execution_batch_size"),
                    "execution_num_workers": metrics.get("execution_num_workers"),
                    # Os Transformers gravam a semente nas metricas; o DevNet,
                    # cuja semente e a do sorteio dos rotulos, na dobra.
                    "execution_seed": metrics.get(
                        "execution_seed", execution_result.get("execution_seed")
                    ),
                    "execution_deterministic": metrics.get("execution_deterministic"),
                    **fold_metadata,
                    "fit_time": execution_result.get("fit_time", ""),
                    "scoring_time": execution_result.get("scoring_time", ""),
                    "pretraining_time": execution_result.get("pretraining_time", ""),
                    **execution_provenance.collect_provenance(),
                }
            )

    if not rows:
        raise ValueError("No metrics were found to record")

    output_directory = os.path.dirname(output_csv_filename)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    _ensure_results_csv_schema(output_csv_filename)
    results_dataframe = pd.DataFrame(rows, columns=RESULT_CSV_COLUMNS)
    results_dataframe.to_csv(
        output_csv_filename,
        mode="a",
        header=not os.path.exists(output_csv_filename),
        index=False,
    )
    append_test_score_records(test_scores_path(output_csv_filename), score_records)

    return results_dataframe


def record_algorithm_failure_to_csv(
    output_csv_filename,
    dataset_filename,
    algorithm,
    categorical_encoding_strategy,
    fill_na_strategy,
    train_validation_test_strategy,
    undersampling_technique=None,
    random_seed=DEFAULT_RANDOM_SEED,
    classification_threshold=DEFAULT_CLASSIFICATION_THRESHOLD,
    class_weight_intensity=DEFAULT_CLASS_WEIGHT_INTENSITY,
    threshold_strategy=DEFAULT_THRESHOLD_STRATEGY,
    class_weight_ratio="",
    train_dataset_md5="",
    validation_dataset_md5="",
    test_dataset_md5="",
    train_df_size=0,
    validation_df_size=0,
    test_df_size=0,
    elapsed_time=None,
    categorical_apply_time=None,
    fillna_apply_time=None,
    dataset_setup_time=None,
    algorithm_time=None,
    configuration_md5=None,
    database_type=None,
    dataset_id=None,
    dataset_name=None,
    regime="",
    fold_index="",
    split="",
    execution_metadata=None,
    label_budget=LABEL_BUDGET_NOT_APPLICABLE,
    label_fraction=LABEL_FRACTION_NOT_APPLICABLE,
    error=None,
    fit_time="",
    scoring_time="",
):
    # Metadados de execucao (dispositivo, precisao, lote, ...) podem ja ter
    # sido resolvidos antes da falha; sem eles, as colunas ficam vazias.
    execution_metadata = execution_metadata or {}
    dataset_identity = experiment_datasets.resolve_dataset_identity(
        dataset_id=dataset_id,
        database_type=database_type,
        dataset_name=dataset_name,
    )
    row = {
        "configuration_md5": configuration_md5,
        **dataset_identity,
        "dataset_filename": dataset_filename,
        "algorithm": algorithm,
        "regime": regime,
        # A execucao falha antes de identificar folds individualmente, entao
        # fold_index/split ficam vazios salvo quando o chamador os conhece.
        "fold_index": fold_index,
        "split": split,
        "evaluation_unit": EVALUATION_UNIT_ROW,
        "execution_status": "failure",
        **failure_columns(error),
        "undersampling_technique": undersampling_technique,
        "random_seed": random_seed,
        "classification_threshold": classification_threshold,
        "threshold_strategy": threshold_strategy,
        "class_weight_intensity": class_weight_intensity,
        "class_weight_ratio": class_weight_ratio,
        "train_dataset_md5": train_dataset_md5,
        "validation_dataset_md5": validation_dataset_md5,
        "test_dataset_md5": test_dataset_md5,
        "partition_id": "",
        "train_population_id": "",
        "unit_partition_id": "",
        "train_unit_population_id": "",
        "train_df_size": train_df_size,
        "validation_df_size": validation_df_size,
        "test_df_size": test_df_size,
        "categorical_encoding_strategy": categorical_encoding_strategy,
        "fill_na_strategy": fill_na_strategy,
        "train_validation_test_strategy": train_validation_test_strategy,
        "categorical_apply_time": categorical_apply_time,
        "fillna_apply_time": fillna_apply_time,
        "dataset_setup_time": dataset_setup_time,
        "algorithm_time": algorithm_time,
        "elapsed_time": elapsed_time,
        "f1": -1,
        "f_beta_2": -1,
        "accuracy": -1,
        "precision": -1,
        "recall": -1,
        "roc_auc": -1,
        "auc_pr": -1,
        "false_positive_rate": -1,
        "execution_device": execution_metadata.get("execution_device", ""),
        "execution_device_name": execution_metadata.get("execution_device_name", ""),
        "execution_precision": execution_metadata.get("execution_precision", ""),
        "execution_batch_size": execution_metadata.get("execution_batch_size", ""),
        "execution_num_workers": execution_metadata.get("execution_num_workers", ""),
        "execution_seed": execution_metadata.get("execution_seed", ""),
        "execution_deterministic": execution_metadata.get("execution_deterministic", ""),
        # A falha pode ocorrer antes do ajuste: sem modelo, nada a descrever.
        "label_budget": label_budget,
        "label_fraction": label_fraction,
        "model_hyperparameters": serialize_metadata({}),
        "training_summary": serialize_metadata({}),
        # Tempos medidos ate a falha, quando o chamador os conhece.
        "fit_time": fit_time,
        "scoring_time": scoring_time,
        "pretraining_time": "",
        **execution_provenance.collect_provenance(),
    }

    output_directory = os.path.dirname(output_csv_filename)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    _ensure_results_csv_schema(output_csv_filename)
    results_dataframe = pd.DataFrame([row], columns=RESULT_CSV_COLUMNS)
    results_dataframe.to_csv(
        output_csv_filename,
        mode="a",
        header=not os.path.exists(output_csv_filename),
        index=False,
    )

    return results_dataframe


def _parse_nullable_value(value):
    if value is None:
        return None

    normalized_value = str(value).strip()
    if normalized_value.lower() in ["", "none", "null"]:
        return None
    return normalized_value


def _parse_bool(value):
    if isinstance(value, bool):
        return value

    normalized_value = str(value).strip().lower()
    if normalized_value in ["true", "1", "yes", "y", "sim", "s"]:
        return True
    if normalized_value in ["false", "0", "no", "n", "nao", "não"]:
        return False

    raise argparse.ArgumentTypeError(
        "auxiliary_data_is_test_data must be true or false"
    )


def parse_script_arguments():
    parser = argparse.ArgumentParser(
        description="Run a regression experiment configuration."
    )
    parser.add_argument(
        "--algorithm",
        default="linear-regression",
        choices=["linear-regression", "logistic-regression", "naive-bayes", "knn"],
        help="Prediction algorithm. Defaults to linear-regression.",
    )
    parser.add_argument(
        "--dataset-id", "--database-type", "--database_type",
        dest="database_type", required=True,
    )
    parser.add_argument("--dataset-filename", "--dataset_filename", required=True)
    parser.add_argument("--prediction-column-name", "--prediction_column_name", required=True)
    parser.add_argument(
        "--auxiliary-data-is-test-data",
        "--auxiliary_data_is_test_data",
        default=True,
        type=_parse_bool,
        help="Retained for compatibility; dataset selection occurs during preparation.",
    )
    parser.add_argument(
        "--categorical-encoding-strategy",
        "--categorical_encoding_strategy",
        "--categorical-encoding-estrategy",
        "--categorical_encoding_estrategy",
        dest="categorical_encoding_strategy",
        default=None,
        type=_parse_nullable_value,
    )
    parser.add_argument(
        "--fill-na-strategy",
        "--fill_na_strategy",
        "--fillna-strategy",
        "--fillna_strategy",
        "--fill-ba-strategy",
        "--fill_ba_strategy",
        dest="fill_na_strategy",
        default=None,
        type=_parse_nullable_value,
    )
    parser.add_argument(
        "--categorical-handling-mode",
        "--categorical_handling_mode",
        default=None,
        type=_parse_nullable_value,
        help=(
            "Categorical handling mode used during preparation. Omit for legacy "
            "manifests that do not include this column."
        ),
    )
    parser.add_argument("--test-strategy", "--test_strategy", required=True)
    parser.add_argument(
        "--execution-device",
        "--execution_device",
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help=(
            "Choose where the model runs: auto, cpu or gpu. Logistic regression "
            "naive Bayes and KNN currently support auto/cpu."
        ),
    )
    parser.add_argument(
        "--random-seed",
        "--random_seed",
        default=DEFAULT_RANDOM_SEED,
        type=int,
        help=(
            "Seed used when preparing this configuration. Part of the "
            "configuration identity, so repetitions with different seeds are "
            "distinct, comparable configurations."
        ),
    )
    parser.add_argument(
        "--classification-threshold",
        "--classification_threshold",
        default=DEFAULT_CLASSIFICATION_THRESHOLD,
        type=float,
        help=(
            "Decision threshold used when --threshold-strategy is fixed, and the "
            "fallback when the validation split lacks one of the classes."
        ),
    )
    parser.add_argument(
        "--threshold-strategy",
        "--threshold_strategy",
        default=DEFAULT_THRESHOLD_STRATEGY,
        choices=list(REQUESTABLE_THRESHOLD_STRATEGIES),
        help=(
            "validation-f2 (default) picks the threshold that maximises F2 on the "
            "validation predictions and applies it to the test split; in "
            "stratified-k-fold each fold uses the other folds' validation. fixed "
            "uses --classification-threshold. Recorded in the result row and part "
            "of the already-recorded key."
        ),
    )
    parser.add_argument(
        "--class-weight-intensity",
        "--class_weight_intensity",
        default=DEFAULT_CLASS_WEIGHT_INTENSITY,
        choices=list(CLASS_WEIGHT_INTENSITIES),
        help=(
            "Class weighting intensity applied during fitting: none, sqrt-balanced "
            "or balanced. sqrt-balanced and balanced are balancing arms of their "
            "own and only run on configurations without resampling, where R = "
            "negatives/positives is the natural ratio of this fold's training "
            "split. Execution parameter: it does not change configuration_md5, but "
            "it is recorded and is part of the already-recorded key. Only logistic "
            "regression uses it among the classic algorithms."
        ),
    )
    parser.add_argument(
        "--undersampling-technique",
        "--undersampling_technique",
        default=None,
        type=_parse_nullable_value,
        help=(
            "Under-sampling technique applied only to training data. Examples: "
            "random-undersampling, tomek-links, edited-nearest-neighbours, "
            "near-miss, cluster-centroids, RUS+SMOTE or teus."
        ),
    )
    parser.add_argument(
        "--undersampling-parameter-values",
        "--undersampling_parameter_values",
        default=None,
        type=_parse_nullable_value,
        help=(
            "Colon-separated under-sampling parameters, for example "
            "'sampling_strategy=0.5:random_state=42'. RUS+SMOTE accepts "
            "rus_sampling_strategy, smote_sampling_strategy, "
            "smote_k_neighbors and rus_replacement."
        ),
    )
    parser.add_argument(
        "--prepared-data-directory",
        "--prepared_data_directory",
        default=str(DEFAULT_PREPARED_DATA_DIRECTORY),
        help="Directory containing CSVs created by prepare_lr_nb_knn_datasets.py.",
    )
    parser.add_argument(
        "--results-directory",
        "--results_directory",
        default=str(Path(__file__).resolve().parent),
        help="Directory where resultados_<algorithm>.csv is appended.",
    )
    parser.add_argument(
        "--local-staging-directory",
        "--local_staging_directory",
        default=str(Path(__file__).resolve().parent / "prepared_datasets"),
        help="Local directory used while reading one prepared configuration.",
    )
    parser.add_argument(
        "--prepared-dataset-md5",
        "--prepared_dataset_md5",
        default=None,
        help="MD5 identity emitted by prepare_lr_nb_knn_datasets.py.",
    )
    parser.add_argument(
        "--prepared-dataset-manifest",
        "--prepared_dataset_manifest",
        default=None,
        help=(
            "TSV mapping configuration parameters and fold/split MD5 files. "
            "Defaults to prepared_dataset_configurations.tsv in the data directory."
        ),
    )
    parser.add_argument(
        "--force-reprocess",
        "--force_reprocess",
        action="store_true",
        help=(
            "Recompute and append a new result row even if a successful one is "
            "already recorded for this configuration_md5/algorithm."
        ),
    )
    return parser.parse_args()

# Aplica PCA no dataframe informado como parâmetro, removendo, obviamente a coluna que representa o objetivo da classificação
def _apply_pca(input_training_data, input_validation_data, input_test_data, target_column):
    def _prepare_pca_features(dataframe, expected_columns=None):
        if dataframe is None:
            return None, None
        if dataframe.empty:
            return pd.DataFrame(index=dataframe.index), None

        target = dataframe[target_column].copy() if target_column in dataframe.columns else None
        features = dataframe.loc[:, dataframe.columns != target_column]
        if expected_columns is not None:
            absent_columns = [column for column in expected_columns if column not in features.columns]
            if absent_columns:
                raise ValueError(f"PCA held-out split lacks training columns: {absent_columns[:10]}")
            features = features[list(expected_columns)]
        if features.empty:
            raise ValueError("PCA requires at least one numeric feature column")
        # Sem select_dtypes nem fillna: entrada invalida e erro (item G15).
        require_numeric_complete_features(features, "PCA")
        return features.astype(float), target

    def _build_pca_dataframe(transformed_data, index, target):
        pca_columns = [f'PC{i+1}' for i in range(transformed_data.shape[1])]
        output_dataframe = pd.DataFrame(transformed_data, columns=pca_columns, index=index)
        if target is not None:
            output_dataframe[target_column] = target.to_numpy()
        return output_dataframe

    # 1. SEPARAÇÃO: Remove a classe (e mantém o index para segurança)
    X_train_features, y_train_target = _prepare_pca_features(input_training_data)
    trained_feature_columns = X_train_features.columns

    # 2. NORMALIZAÇÃO: Obrigatória antes do PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_features)

    # 3. PCA: Roda apenas nas variáveis preditoras
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.95)

    X_train_pca = pca.fit_transform(X_train_scaled)

    # 4. RECONSTRUÇÃO: Cria o DataFrame do PCA mantendo o index original
    input_train_data_pca_df = _build_pca_dataframe(
        X_train_pca,
        input_training_data.index,
        y_train_target,
    )

    input_validation_data_pca_df = pd.DataFrame()
    if input_validation_data is not None and not input_validation_data.empty:
        X_validation_features, y_validation_target = _prepare_pca_features(
            input_validation_data,
            expected_columns=trained_feature_columns,
        )

        X_validation_scaled = scaler.transform(X_validation_features)
        X_validation_pca = pca.transform(X_validation_scaled)

        input_validation_data_pca_df = _build_pca_dataframe(
            X_validation_pca,
            input_validation_data.index,
            y_validation_target,
        )

    input_test_data_pca_df = pd.DataFrame()
    if input_test_data is not None and not input_test_data.empty:
        X_test_features, y_test_target = _prepare_pca_features(
            input_test_data,
            expected_columns=trained_feature_columns,
        )

        X_test_scaled = scaler.transform(X_test_features)
        X_test_pca = pca.transform(X_test_scaled)

        input_test_data_pca_df = _build_pca_dataframe(
            X_test_pca,
            input_test_data.index,
            y_test_target,
        )

    # Resumo gravado no relatorio de preparacao (item G15: componentes retidos).
    pca_summary = {
        "pca_input_columns": int(X_train_features.shape[1]),
        "pca_components": int(pca.n_components_),
        "pca_explained_variance": float(pca.explained_variance_ratio_.sum()),
    }
    return (
        input_train_data_pca_df.copy(),
        input_validation_data_pca_df.copy(),
        input_test_data_pca_df.copy(),
        pca_summary,
    )

IEEE_CIS_DATABASE_TYPES = experiment_datasets.IEEE_CIS_DATABASE_TYPES
SUPPORTED_DATABASE_TYPES = tuple(sorted(experiment_datasets.DATASET_DEFINITIONS))


def load_data(database_type):
    """Load one experimental dataset, already stripped of leakage shortcuts.

    A carga ficou puramente estrutural: le os arquivos, deriva os atributos
    relativos (hora do dia, dia da semana, idade, distancia) e devolve o
    quadro bruto. Codificacao categorica, imputacao, padronizacao e PCA NAO
    acontecem aqui, porque todas sao transformacoes dependentes dos dados e
    precisam ser ajustadas apenas no treinamento de cada particao. A versao
    anterior aplicava PCA e imputacao sobre a base inteira antes de
    particionar em algumas bases, o que vazava informacao do teste para o
    treino (itens G14 e G15 da revisao).

    Devolve ``(dados_principais, dados_auxiliares, definicao_da_base)``. O
    arquivo auxiliar so existe na Sparkov, onde ``fraudTest.csv`` cobre um
    periodo posterior ao de ``fraudTrain.csv``.
    """
    experiment_datasets.CURRENT_PATH = CURRENT_PATH
    experiment_datasets.RELATIVE_DATA_PATH = RELATIVE_DATA_PATH
    return experiment_datasets.load_raw_dataset(database_type)


if __name__ == "__main__":
    import execution_provenance

    execution_provenance.guard_environment_or_exit()
    args = parse_script_arguments()
    execution_start_time = time.perf_counter()
    RELATIVE_DATA_PATH = str(DEFAULT_DATA_DIRECTORY)
    CURRENT_PATH = os.getcwd()
    print(f"Current path: {CURRENT_PATH}")

    algorithm_name = args.algorithm
    result_csv_filename = os.path.join(
        args.results_directory,
        f"resultados_{algorithm_name}.csv",
    )
    dataset_configuration = {
        "database_type": args.database_type,
        "dataset_filename": args.dataset_filename,
        "prediction_column_name": args.prediction_column_name,
        "auxiliary_data_is_test_data": args.auxiliary_data_is_test_data,
    }
    categorical_encoding_strategy = args.categorical_encoding_strategy
    categorical_handling_mode = (
        normalize_categorical_handling_mode(args.categorical_handling_mode)
        if args.categorical_handling_mode is not None
        else None
    )
    fill_na_strategy = args.fill_na_strategy
    test_strategy = args.test_strategy
    execution_device = args.execution_device
    undersampling_technique = args.undersampling_technique
    undersampling_parameter_values = args.undersampling_parameter_values
    random_seed = args.random_seed
    classification_threshold = args.classification_threshold
    threshold_strategy = args.threshold_strategy
    class_weight_intensity = args.class_weight_intensity

    expected_dataset_md5 = build_prepared_dataset_md5(
        dataset_configuration["database_type"],
        categorical_encoding_strategy,
        fill_na_strategy,
        test_strategy,
        dataset_configuration["prediction_column_name"],
        undersampling_technique,
        undersampling_parameter_values,
        dataset_configuration["auxiliary_data_is_test_data"],
        categorical_handling_mode,
        random_seed,
    )
    prepared_dataset_md5 = args.prepared_dataset_md5 or expected_dataset_md5

    try:
        if len(prepared_dataset_md5) != 32 or any(
            character not in "0123456789abcdef"
            for character in prepared_dataset_md5.lower()
        ):
            raise ValueError("prepared_dataset_md5 must contain 32 hexadecimal characters")
        if prepared_dataset_md5.lower() != expected_dataset_md5:
            raise ValueError(
                "prepared_dataset_md5 does not match the informed dataset "
                "preparation parameters"
            )
        normalized_prepared_dataset_md5 = prepared_dataset_md5.lower()
        validate_class_weight_arm(class_weight_intensity, undersampling_technique)
        if not args.force_reprocess and is_experiment_already_recorded(
            os.path.dirname(result_csv_filename),
            normalized_prepared_dataset_md5,
            algorithm_name,
            (
                classification_threshold
                if threshold_strategy == THRESHOLD_STRATEGY_FIXED
                else None
            ),
            class_weight_intensity,
            threshold_strategy,
        ):
            print(
                "Skipping experiment already recorded: "
                f"configuration_md5={normalized_prepared_dataset_md5}, "
                f"algorithm={algorithm_name}, "
                f"threshold_strategy={threshold_strategy}, "
                f"class_weight={class_weight_intensity}"
            )
            raise SystemExit(0)
        prepared_dataset_manifest = args.prepared_dataset_manifest or os.path.join(
            args.prepared_data_directory,
            "prepared_dataset_configurations.tsv",
        )
        local_prepared_data_root = Path(args.local_staging_directory)
        with stage_prepared_dataset_files(
            args.prepared_data_directory,
            prepared_dataset_manifest,
            prepared_dataset_md5,
            local_prepared_data_root,
        ) as staged_data_directory:
            test_data_information = load_prepared_data_splits(
                staged_data_directory,
                prepared_dataset_manifest,
                normalized_prepared_dataset_md5,
                test_strategy,
            )
            check_loaded_folds(
                test_data_information, dataset_configuration["prediction_column_name"]
            )
            algorithm_execution_result = execute_prediction_algorithm(
                algorithm_name,
                test_data_information,
                prediction_column_name=dataset_configuration["prediction_column_name"],
                execution_device=execution_device,
                classification_threshold=classification_threshold,
                threshold_strategy=threshold_strategy,
                class_weight_intensity=class_weight_intensity,
                undersampling_technique=None,
                undersampling_parameter_values=None,
            )
            # Preserve preparation metadata in the results without sampling twice.
            algorithm_execution_result["undersampling_technique"] = (
                undersampling_technique
            )
            algorithm_execution_result["undersampling_parameter_values"] = (
                undersampling_parameter_values
            )
            record_algorithm_result_to_csv(
                result_csv_filename,
                dataset_configuration["dataset_filename"],
                algorithm_execution_result,
                categorical_encoding_strategy,
                fill_na_strategy,
                round(time.perf_counter() - execution_start_time, 6),
                configuration_md5=normalized_prepared_dataset_md5,
                database_type=dataset_configuration["database_type"],
                random_seed=random_seed,
                classification_threshold=classification_threshold,
                class_weight_intensity=class_weight_intensity,
                threshold_strategy=threshold_strategy,
            )
    except Exception as error:
        print(
            "Error running experiment: "
            f"{dataset_configuration['database_type']}, "
            f"encoding={categorical_encoding_strategy}, "
            f"fillna={fill_na_strategy}, "
            f"strategy={test_strategy}, "
            f"seed={random_seed}, "
            f"threshold_strategy={threshold_strategy}, "
            f"class_weight={class_weight_intensity}, "
            f"undersampling={undersampling_technique}, "
            f"undersampling_parameters={undersampling_parameter_values}: {error}"
        )
        record_algorithm_failure_to_csv(
            result_csv_filename,
            dataset_configuration["dataset_filename"],
            algorithm_name,
            categorical_encoding_strategy,
            fill_na_strategy,
            test_strategy,
            undersampling_technique=undersampling_technique,
            random_seed=random_seed,
            classification_threshold=classification_threshold,
            threshold_strategy=threshold_strategy,
            class_weight_intensity=class_weight_intensity,
            elapsed_time=round(time.perf_counter() - execution_start_time, 6),
            configuration_md5=(
                prepared_dataset_md5.lower()
                if prepared_dataset_md5
                else expected_dataset_md5.lower()
            ),
            database_type=dataset_configuration["database_type"],
            error=error,
        )
        if isinstance(error, PreparedDatasetStorageError):
            raise SystemExit(PREPARED_STORAGE_FAILURE_EXIT_CODE) from error
        raise
