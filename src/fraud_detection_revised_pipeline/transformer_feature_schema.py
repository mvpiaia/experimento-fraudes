"""Semantic feature typing used by the Transformer preparation pipeline."""

from __future__ import annotations

import pandas as pd

import experiment_datasets


# APARTMENTS_MODE, BASEMENTAREA_MODE, RATE_DOWN_PAYMENT e RATE_INTEREST_* sao
# continuas (estatisticas do imovel e taxas normalizadas em [0, 1]) e ficam
# numericas, como na home-credit-somente-aplicacao. A mudanca tem revisao de
# identidade propria (TRANSFORMER_SCHEMA_REVISIONS em prepared_dataset_storage).
HOME_CREDIT_CATEGORICAL = frozenset(
    {
        "SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_TYPE", "CODE_GENDER",
        "FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_TYPE_SUITE",
        "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE", "FLAG_MOBIL", "FLAG_EMP_PHONE",
        "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL",
        "OCCUPATION_TYPE", "WEEKDAY_APPR_PROCESS_START",
        "REG_REGION_NOT_LIVE_REGION", "REG_REGION_NOT_WORK_REGION",
        "LIVE_REGION_NOT_WORK_REGION", "REG_CITY_NOT_LIVE_CITY",
        "REG_CITY_NOT_WORK_CITY", "LIVE_CITY_NOT_WORK_CITY",
        "ORGANIZATION_TYPE",
        "FONDKAPREMONT_MODE", "HOUSETYPE_MODE", "WALLSMATERIAL_MODE",
        "EMERGENCYSTATE_MODE", "FLAG_LAST_APPL_PER_CONTRACT",
        "NAME_CASH_LOAN_PURPOSE",
        "NAME_CONTRACT_STATUS", "NAME_PAYMENT_TYPE", "CODE_REJECT_REASON",
        "NAME_CLIENT_TYPE", "NAME_GOODS_CATEGORY", "NAME_PORTFOLIO",
        "NAME_PRODUCT_TYPE", "CHANNEL_TYPE", "NAME_SELLER_INDUSTRY",
        "NAME_YIELD_GROUP", "PRODUCT_COMBINATION",
        "NFLAG_INSURED_ON_APPROVAL", "is_synthetic_current",
        *(f"FLAG_DOCUMENT_{index}" for index in range(2, 22)),
    }
)

# Os codigos card1-card5, addr1 e addr2 vem do catalogo
# (``categorical_code_columns``), nao desta lista: assim todos os blocos
# partem da mesma declaracao (item G14).
IEEE_CIS_TRANSACTION_CATEGORICAL = frozenset({"TransactionID"})

IEEE_CIS_IDENTITY_CATEGORICAL = frozenset(
    {
        "id_01", "id_03", "id_04", "id_05", "id_06", "id_07", "id_08",
        "id_09", "id_10", "id_11", "id_13", "id_14", "id_17", "id_18",
        "id_19", "id_20", "id_21", "id_22", "id_24", "id_25", "id_26",
        "id_32",
    }
)


DATASET_CATEGORICAL_OVERRIDES = {
    "home-credit-completa": HOME_CREDIT_CATEGORICAL,
    # cc_num e indexValue sao identificadores e ja saem dos atributos na
    # carga (item G14b da revisao), entao nao aparecem mais aqui.
    "sparkov": frozenset({"zip", "is_weekend", "category", "state", "job", "gender", "merchant", "city"}),
    "ieee-cis-completa": IEEE_CIS_TRANSACTION_CATEGORICAL | IEEE_CIS_IDENTITY_CATEGORICAL,
    "ieee-cis-transacional": IEEE_CIS_TRANSACTION_CATEGORICAL,
}

# Colunas que existem nos arquivos preparados mas NUNCA podem virar atributo
# do modelo. Elas sao necessarias rio acima -- para juntar tabelas, agrupar
# particoes e ordenar as janelas causais do TabBERT -- e por isso sobrevivem ao
# descarte no modo sequencial, onde o indice de janelas as referencia. Se
# chegassem ao modelo como campo, ele aprenderia a reconhecer o cliente ou a
# transacao em vez de generalizar a partir dos atributos: e exatamente o
# problema do item G14b da revisao, em que trans_num, unico por transacao,
# explicava o XGBoost sinalizar ~80% das transacoes da Sparkov.
#
# A lista vem do catalogo de bases, de modo que identificador, agrupamento,
# marca de tempo e chaves de sequencia tenham uma unica fonte de verdade.
def _build_identifier_columns() -> dict[str, frozenset[str]]:
    identifier_columns: dict[str, frozenset[str]] = {}
    for definition in experiment_datasets.DATASET_DEFINITIONS.values():
        columns = set(
            experiment_datasets.columns_to_drop_after_partitioning(definition)
        )
        if definition.sequence_keys is not None:
            # So o titular e sempre excluido. A coluna de ordem do TabBERT
            # pode ser um atributo legitimo: DAYS_DECISION, na Home Credit, e
            # o numero de dias ENTRE o pedido anterior e o atual, uma medida
            # relativa com significado proprio. Ja TransactionDT e unix_time
            # sao marcas absolutas e saem por estarem na lista de descarte da
            # base, nao por serem chave de sequencia.
            holder_column, _order_column = definition.sequence_keys
            columns.add(holder_column)
        identifier_columns[definition.dataset_id] = frozenset(columns)
    return identifier_columns


DATASET_IDENTIFIER_COLUMNS = _build_identifier_columns()

HIGH_CARDINALITY_COLUMNS = frozenset(
    {
        "SK_ID_CURR", "SK_ID_PREV", "cc_num", "merchant", "street",
        "trans_num", "card1", "TransactionID", "DeviceInfo", "id_02",
    }
)


def categorical_overrides(dataset_id: str) -> frozenset[str]:
    """Categoricos declarados: lista da base (ou da base de origem) mais os codigos.

    Uma variante de ablacao herda a lista da base de origem; as colunas que ela
    remove saem de ``technical_columns`` em ``resolve_feature_types``.
    """
    definition = experiment_datasets.get_dataset_definition(dataset_id)
    base = experiment_datasets.resolve_base_definition(definition)
    return DATASET_CATEGORICAL_OVERRIDES.get(
        base.dataset_id, frozenset()
    ) | frozenset(definition.categorical_code_columns)


def resolve_feature_types(
    dataset: str,
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[list[str], list[str]]:
    """Return categorical and numerical columns using dataset semantics."""
    dataset_id = experiment_datasets.normalize_dataset_id(dataset)
    technical_columns = {"row_id", "source_split"} | DATASET_IDENTIFIER_COLUMNS.get(
        dataset_id, frozenset()
    )
    features = [
        column for column in dataframe.columns
        if column != target_column and column not in technical_columns
    ]
    overrides = categorical_overrides(dataset_id)
    inferred = set(
        dataframe[features].select_dtypes(include=["object", "category", "string"])
    )
    categorical = [
        column for column in features if column in overrides or column in inferred
    ]
    numerical = [column for column in features if column not in categorical]
    return categorical, numerical
