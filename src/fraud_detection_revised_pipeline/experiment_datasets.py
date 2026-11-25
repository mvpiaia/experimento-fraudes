"""Catalogo das bases experimentais: carga, atributos descartados e politica de PCA.

Este modulo concentra tudo o que depende de *qual* base esta sendo usada, para
que o restante do pipeline fique generico. As decisoes aqui implementadas vem
do relatorio de revisao da dissertacao:

* ``eqm-ref-21`` e ``eqm-ref-21-pca`` nao existem mais. Eram republicacoes da
  base da ULB e produziam resultados identicos aos de ``eqm-ref-18``, com o
  efeito colateral de dar peso dobrado a ULB em toda media agregada (item G9a).
* ``eqm-ref-18-pca`` tambem foi removida. Os atributos ``V1``-``V28`` da ULB ja
  sao componentes principais publicados; aplicar uma segunda PCA sobre eles e,
  na pratica, uma rotacao seguida do descarte de componentes, e confundia o
  efeito da reducao com o da base (itens G9a e C5.31). Por isso a PCA agora e
  uma propriedade fixa de cada base, e nao um eixo da grade: roda em todas,
  menos na ULB.
* Colunas identificadoras saem dos atributos em todas as bases. Elas permitiam
  memorizacao e explicam comportamentos degenerados ja observados, como o
  XGBoost sinalizando a maioria das transacoes da Sparkov por causa de
  ``trans_num``, unico por transacao (item G14b/G14c).
* Marcas de tempo absolutas (``Time``, ``unix_time``, ``TransactionDT``) deixam
  de ser atributos e passam a servir apenas para ordenar e particionar; delas
  derivam atributos relativos, como hora do dia e dia da semana (item G14d).
* As sete colunas de ``previous_application`` homonimas das de
  ``application_data`` deixaram de ser descartadas como "redundantes": elas
  descrevem o pedido anterior, e nao o atual, e agora entram com prefixo
  ``PREV_`` (item C3.26).
* A IEEE-CIS ganha um identificador aproximado de titular, porque ``card1``
  sozinho e informacao do cartao, e nao do portador (item G8-iii).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd


# Blocos de execucao que consomem uma base. A grade e o executor de anomalia
# so percorrem as bases que declaram o bloco; o padrao e participar de todos.
BLOCK_CLASSIC = "classic"
BLOCK_BOOSTING = "boosting"
BLOCK_ANOMALY = "anomaly"
BLOCK_FT_TRANSFORMER = "ft-transformer"
BLOCK_TABBERT = "tabbert"
ALL_BLOCKS = (
    BLOCK_CLASSIC,
    BLOCK_BOOSTING,
    BLOCK_ANOMALY,
    BLOCK_FT_TRANSFORMER,
    BLOCK_TABBERT,
)

# Diretorio das bases brutas; preenchido pelo modulo chamador.
RELATIVE_DATA_PATH = "../datasets"
CURRENT_PATH = os.getcwd()


@dataclass(frozen=True)
class DatasetDefinition:
    """Tudo o que o pipeline precisa saber sobre uma base experimental."""

    dataset_id: str
    dataset_name: str
    legacy_aliases: tuple[str, ...]
    source_files: str
    prediction_column: str
    # Arquivo auxiliar e o conjunto de teste oficial (so a Sparkov tem um).
    auxiliary_data_is_test_data: bool = False
    # Atributos removidos antes de qualquer ajuste: identificadores, dados
    # pessoais e marcas de tempo absolutas.
    identifier_columns_to_drop: tuple[str, ...] = ()
    # Coluna usada para ordenar cronologicamente; nunca entra como atributo.
    temporal_order_column: str | None = None
    # Coluna de agrupamento que nenhuma particao pode dividir.
    group_column: str | None = None
    # PCA de 95% da variancia, ajustada so no treino de cada particao, e
    # aplicada EXCLUSIVAMENTE no bloco classico. Boosting e Transformers nunca
    # a recebem: ela eliminaria a semantica individual dos atributos, impediria
    # o tratamento nativo das variaveis categoricas e alteraria a
    # representacao por campo usada pelos Transformers. Alem disso, por ser nao
    # supervisionada, a PCA preserva variancia dos preditores, nao
    # necessariamente informacao relevante para a classificacao de fraude.
    applies_pca_in_classic_block: bool = False
    # Eixos da grade que so fazem sentido quando a base realmente tem o
    # fenomeno. A ULB nao tem atributo categorico nem valor faltante: variar
    # codificacao e imputacao nela gerava 8 copias identicas de cada
    # experimento, inflando a contagem de "pares" e estreitando
    # artificialmente os intervalos bootstrap (item G9b da revisao).
    has_categorical_attributes: bool = True
    has_missing_values: bool = True
    # (titular, ordem) usados pelo TabBERT para montar janelas causais. Sem
    # eles a base nao tem sequencia por titular e fica fora do TabBERT: e o
    # caso da ULB, cujas linhas sao transacoes anonimas sem portador.
    sequence_keys: tuple[str, str] | None = None
    protocols: tuple[str, ...] = ()
    # Coluna da unidade a que o rotulo pertence quando ela abrange varias
    # linhas. Na Home Credit com juncao o TARGET e do cliente e se repete nos
    # seus ~4,9 pedidos: medir por linha pesaria cada cliente pelo numero de
    # pedidos anteriores (item G7). Com ela, os escores sao agregados por
    # unidade antes das metricas. O identificador nunca entra nos atributos.
    evaluation_unit_column: str | None = None
    # Dominio da base. As de controle (Home Credit: inadimplencia, nao fraude)
    # ficam fora dos agregados das bases de fraude (item G7-c).
    domain: str = "fraud"
    # Valores monetarios: recebem log1p antes da padronizacao nos modelos
    # sensíveis a escala (bloco classico, Isolation Forest e DevNet). Assimetria
    # de 13 a 37 deixava uma transacao a 80-173 desvios-padrao da media, e a
    # cauda alta tem 0% de fraude acima do p99,9 na ULB e na Sparkov (item G14).
    monetary_columns: tuple[str, ...] = ()
    # Codigos categoricos guardados como numeros (IEEE-CIS: card1-card5, addr1,
    # addr2). Viram texto na carga, para que todo bloco os trate como categorias.
    categorical_code_columns: tuple[str, ...] = ()
    # Atributos derivados pela carga; documentam o registro de colunas.
    derived_columns: tuple[str, ...] = ()
    # Variante de ablacao: herda carregador, chaves e protocolos de
    # ``base_dataset_id`` e remove ``ablated_columns`` depois do corte, de modo
    # que as duas bases particionam exatamente as mesmas linhas.
    base_dataset_id: str | None = None
    ablated_columns: tuple[str, ...] = ()
    # Par de ablacao com linhas diferentes, mas as mesmas unidades de
    # avaliacao (as duas Home Credit). O pareamento e pelo
    # unit_partition_md5, nao pelo partition_md5 (item G16).
    paired_dataset_id: str | None = None
    # Blocos em que a base entra.
    blocks: tuple[str, ...] = ALL_BLOCKS
    notes: str = ""

    @property
    def database_type(self) -> str:
        """Espelho temporario do identificador canonico para APIs legadas."""
        return self.dataset_id


DOMAIN_FRAUD = "fraud"
DOMAIN_OUT_OF_DOMAIN_CONTROL = "out-of-domain-control"


STRATIFIED_HOLDOUT = "stratified-holdout"
STRATIFIED_K_FOLD = "stratified-k-fold"
TEMPORAL_HOLDOUT = "temporal-holdout"
SUPPORTED_PROTOCOLS = (STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD, TEMPORAL_HOLDOUT)

ALL_PROTOCOLS = (STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD, TEMPORAL_HOLDOUT)
SHUFFLED_PROTOCOLS = (STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD)


# Atributos que a carga deriva (nao existem nos arquivos de origem).
SPARKOV_DERIVED_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "month",
    "day_of_month",
    "age_years",
    "customer_merchant_km",
)
IEEE_CIS_DERIVED_COLUMNS = ("approximate_holder_id", "hour_of_day", "day_of_week")
# Codigos de cartao e endereco da IEEE-CIS. Os organizadores os classificam
# como categoricos; sao numeros so por acidente de armazenamento.
IEEE_CIS_CODE_COLUMNS = ("card1", "card2", "card3", "card5", "addr1", "addr2")
# Atributos que recompoem o titular na Sparkov (983 titulares, ~1 valor cada).
# O cc_num continua identificando o titular, mas so agrupa e ordena.
SPARKOV_HOLDER_ATTRIBUTES = ("zip", "lat", "long", "city", "city_pop", "job")
# Atributos de alta cardinalidade ou deriva da IEEE-CIS-completa.
IEEE_CIS_HIGH_CARDINALITY_ATTRIBUTES = ("DeviceInfo", "id_31", "id_33", "card1")
# Colunas de train_identity.csv, exceto a chave TransactionID.
IEEE_CIS_IDENTITY_ATTRIBUTES = (
    *(f"id_{index:02d}" for index in range(1, 39)),
    "DeviceType",
    "DeviceInfo",
)


DATASET_DEFINITIONS: dict[str, DatasetDefinition] = {
    "ulb": DatasetDefinition(
        dataset_id="ulb",
        dataset_name="ULB",
        legacy_aliases=("eqm-ref-18",),
        source_files="ref-18/creditcard.csv",
        prediction_column="Class",
        identifier_columns_to_drop=("Time",),
        temporal_order_column="Time",
        applies_pca_in_classic_block=False,
        has_categorical_attributes=False,
        has_missing_values=False,
        protocols=ALL_PROTOCOLS,
        monetary_columns=("Amount",),
        derived_columns=("hour_of_day",),
        notes=(
            "ULB. V1-V28 ja sao componentes principais publicados, entao nao "
            "recebe PCA adicional. Time vira hora do dia e sai dos atributos."
        ),
    ),
    "home-credit-completa": DatasetDefinition(
        dataset_id="home-credit-completa",
        dataset_name="Home-Credit-completa",
        legacy_aliases=("eqm-ref-19",),
        source_files="ref-19/application_data.csv+ref-19/previous_application.csv",
        prediction_column="TARGET",
        identifier_columns_to_drop=("SK_ID_CURR", "SK_ID_PREV"),
        temporal_order_column=None,
        group_column="SK_ID_CURR",
        applies_pca_in_classic_block=True,
        # A "sequencia" do cliente sao seus pedidos anteriores, ordenados por
        # DAYS_DECISION, e nao transacoes independentes.
        sequence_keys=("SK_ID_CURR", "DAYS_DECISION"),
        protocols=SHUFFLED_PROTOCOLS,
        evaluation_unit_column="SK_ID_CURR",
        domain=DOMAIN_OUT_OF_DOMAIN_CONTROL,
        notes=(
            "Home Credit. Controle fora do dominio de fraude: o alvo e "
            "inadimplencia e a unidade e o cliente. Juncao a esquerda com "
            "previous_application: 307.511 clientes em 1.430.155 linhas; quem "
            "nao tem pedido anterior fica com uma linha e os PREV_* ausentes. "
            "A juncao replica cada cliente em ~4,7 linhas, entao toda particao "
            "e feita por SK_ID_CURR e os identificadores saem dos atributos."
        ),
    ),
    "home-credit-somente-aplicacao": DatasetDefinition(
        dataset_id="home-credit-somente-aplicacao",
        dataset_name="Home-Credit-somente-aplicação",
        legacy_aliases=("eqm-ref-19-application",),
        source_files="ref-19/application_data.csv",
        prediction_column="TARGET",
        identifier_columns_to_drop=("SK_ID_CURR",),
        temporal_order_column=None,
        # Uma linha por cliente, mas agrupada por SK_ID_CURR mesmo assim: o
        # corte passa pela tabela de clientes, como na completa, e as duas
        # bases poem cada cliente no mesmo split (item G16).
        group_column="SK_ID_CURR",
        applies_pca_in_classic_block=True,
        protocols=SHUFFLED_PROTOCOLS,
        evaluation_unit_column="SK_ID_CURR",
        paired_dataset_id="home-credit-completa",
        domain=DOMAIN_OUT_OF_DOMAIN_CONTROL,
        notes=(
            "Home Credit sem a juncao com previous_application: 307.511 "
            "pedidos, um por cliente. Pareada com home-credit-completa, "
            "isola o efeito "
            "da juncao -- tanto a informacao que o historico acrescenta quanto "
            "a replicacao de cada cliente em ~4,9 linhas, que era a origem do "
            "vazamento apontado no item G7. Como aqui cada cliente aparece uma "
            "unica vez, a particao aleatoria ja e segura por construcao, e o "
            "contraste entre as duas mostra quanto do desempenho anterior vinha "
            "do historico e quanto vinha da replicacao."
        ),
    ),
    "sparkov": DatasetDefinition(
        dataset_id="sparkov",
        dataset_name="Sparkov",
        legacy_aliases=("eqm-ref-20",),
        source_files="ref-20/fraudTrain.csv+ref-20/fraudTest.csv",
        prediction_column="is_fraud",
        auxiliary_data_is_test_data=True,
        identifier_columns_to_drop=(
            "indexValue",
            "cc_num",
            "trans_num",
            "first",
            "last",
            "street",
            "dob",
            "unix_time",
            "trans_date_trans_time",
        ),
        temporal_order_column="trans_date_trans_time",
        group_column="cc_num",
        applies_pca_in_classic_block=True,
        has_missing_values=False,
        sequence_keys=("cc_num", "unix_time"),
        protocols=ALL_PROTOCOLS,
        monetary_columns=("amt",),
        derived_columns=SPARKOV_DERIVED_COLUMNS,
        notes=(
            "Sparkov (sintetica). Identificadores e dados pessoais saem dos "
            "atributos; ficam hora, dia da semana, idade, as coordenadas de "
            "cliente e estabelecimento e a distancia entre elas. A localizacao "
            "de uma transacao e um atributo legitimo para identificar fraude, "
            "entao as quatro coordenadas permanecem. O protocolo temporal usa "
            "os arquivos oficiais fraudTrain/fraudTest."
        ),
    ),
    "ieee-cis-completa": DatasetDefinition(
        dataset_id="ieee-cis-completa",
        dataset_name="IEEE-CIS-completa",
        legacy_aliases=("eqm-ref-22",),
        source_files="ref-22/train_transaction.csv+ref-22/train_identity.csv",
        prediction_column="isFraud",
        identifier_columns_to_drop=("TransactionID", "TransactionDT"),
        temporal_order_column="TransactionDT",
        group_column="approximate_holder_id",
        applies_pca_in_classic_block=True,
        sequence_keys=("approximate_holder_id", "TransactionDT"),
        protocols=ALL_PROTOCOLS,
        monetary_columns=("TransactionAmt",),
        categorical_code_columns=IEEE_CIS_CODE_COLUMNS,
        derived_columns=IEEE_CIS_DERIVED_COLUMNS,
        notes=(
            "IEEE-CIS, juncao interna com identidade: 144.233 transacoes, "
            "todos os atributos."
        ),
    ),
    "ieee-cis-transacional": DatasetDefinition(
        dataset_id="ieee-cis-transacional",
        dataset_name="IEEE-CIS-transacional",
        legacy_aliases=("eqm-ref-22-transaction",),
        source_files="ref-22/train_transaction.csv",
        prediction_column="isFraud",
        identifier_columns_to_drop=("TransactionID", "TransactionDT"),
        temporal_order_column="TransactionDT",
        group_column="approximate_holder_id",
        applies_pca_in_classic_block=True,
        sequence_keys=("approximate_holder_id", "TransactionDT"),
        protocols=ALL_PROTOCOLS,
        monetary_columns=("TransactionAmt",),
        categorical_code_columns=IEEE_CIS_CODE_COLUMNS,
        derived_columns=IEEE_CIS_DERIVED_COLUMNS,
        notes=(
            "IEEE-CIS, todas as 590.540 transacoes, sem ler o arquivo de "
            "identidade. Pareada com ieee-cis-sem-identidade, isola o efeito "
            "do volume com o conjunto de atributos fixo."
        ),
    ),
}

def _build_variant(base_dataset_id: str, **overrides) -> DatasetDefinition:
    """Base de ablacao: copia a base de origem e muda so o declarado.

    Sem apelido legado, salvo quando a variante substitui uma base que ja
    existia como eqm-ref-NN (a IEEE-CIS-sem-identidade).
    """
    return replace(
        DATASET_DEFINITIONS[base_dataset_id],
        base_dataset_id=base_dataset_id,
        **{"legacy_aliases": (), **overrides},
    )


DATASET_DEFINITIONS["sparkov-sem-titular"] = _build_variant(
    "sparkov",
    dataset_id="sparkov-sem-titular",
    dataset_name="Sparkov-sem-titular",
    ablated_columns=SPARKOV_HOLDER_ATTRIBUTES,
    notes=(
        "Sparkov sem os atributos que recompoem o titular: zip, lat, long, "
        "city, city_pop e job tem cerca de um valor por titular e, no "
        "temporal-holdout, permitem reconhecer o mesmo titular no treino e no "
        "teste. O cc_num continua identificando o titular, mas so para "
        "agrupar e ordenar; localizacao nao basta, porque varias pessoas "
        "dividem o mesmo local. customer_merchant_km, derivada das "
        "coordenadas antes do descarte, permanece."
    ),
)
DATASET_DEFINITIONS["ieee-cis-completa-sem-alta-cardinalidade"] = _build_variant(
    "ieee-cis-completa",
    dataset_id="ieee-cis-completa-sem-alta-cardinalidade",
    dataset_name="IEEE-CIS-completa-sem-alta-cardinalidade",
    ablated_columns=IEEE_CIS_HIGH_CARDINALITY_ATTRIBUTES,
    categorical_code_columns=tuple(
        column
        for column in IEEE_CIS_CODE_COLUMNS
        if column not in IEEE_CIS_HIGH_CARDINALITY_ATTRIBUTES
    ),
    blocks=(BLOCK_BOOSTING,),
    notes=(
        "Ablacao da IEEE-CIS-completa sem DeviceInfo, id_31, id_33 e card1 "
        "(categorias raras, deriva temporal e codigo de cartao com ate 13,5 mil "
        "niveis). Roda so no XGBoost e no CatBoost, sobre as mesmas linhas e "
        "partições da base de origem; o identificador aproximado de titular "
        "continua construido com card1 e addr1 antes da remocao."
    ),
)

DATASET_DEFINITIONS["ieee-cis-sem-identidade"] = _build_variant(
    "ieee-cis-completa",
    dataset_id="ieee-cis-sem-identidade",
    dataset_name="IEEE-CIS-sem-identidade",
    legacy_aliases=("eqm-ref-22-noidentity",),
    ablated_columns=IEEE_CIS_IDENTITY_ATTRIBUTES,
    notes=(
        "IEEE-CIS, mesmas 144.233 transacoes de ieee-cis-completa sem as colunas "
        "de identidade. E uma variante de colunas: carrega a base de origem e "
        "remove a identidade depois do corte, de modo que as particoes (e o "
        "partition_md5) sao as da base e, na remocao de incompletos, as linhas "
        "removidas do treino tambem (item G16)."
    ),
)

IEEE_CIS_DATASET_IDS = frozenset(
    {
        "ieee-cis-completa",
        "ieee-cis-sem-identidade",
        "ieee-cis-transacional",
        "ieee-cis-completa-sem-alta-cardinalidade",
    }
)
# Nome mantido para consumidores internos ainda em migracao; os valores ja sao
# os identificadores canonicos.
IEEE_CIS_DATABASE_TYPES = IEEE_CIS_DATASET_IDS

DATASET_ID_BY_ALIAS = {
    alias: definition.dataset_id
    for definition in DATASET_DEFINITIONS.values()
    for alias in definition.legacy_aliases
}
# Colunas de previous_application homonimas das de application_data. Elas
# descrevem o pedido anterior; recebem prefixo em vez de serem descartadas.
PREVIOUS_APPLICATION_OVERLAPPING_COLUMNS = (
    "NAME_CONTRACT_TYPE",
    "AMT_ANNUITY",
    "AMT_CREDIT",
    "AMT_GOODS_PRICE",
    "WEEKDAY_APPR_PROCESS_START",
    "HOUR_APPR_PROCESS_START",
    "NAME_TYPE_SUITE",
)


def datasets_supporting_sequence_models() -> list[str]:
    """Bases com titular e ordem, isto e, elegiveis ao TabBERT.

    A ULB fica de fora: suas linhas sao transacoes anonimas, sem identificador
    de portador, entao nao ha sequencia a modelar. A antiga variante ajustada,
    que a grade anterior usava, tambem saiu: ela acrescentava uma linha
    sintetica por cliente, o que e uma manipulacao da base.
    """
    return [
        definition.dataset_id
        for definition in DATASET_DEFINITIONS.values()
        if definition.sequence_keys is not None and BLOCK_TABBERT in definition.blocks
    ]


def sequence_compatible_protocols(definition: DatasetDefinition) -> tuple[str, ...]:
    """Protocolos compartilhados que preservam as janelas causais do TabBERT.

    O TabBERT monta, para cada titular, janelas com os registros anteriores ao
    alvo. Um protocolo serve se nenhum alvo depender de historico que caia num
    split posterior ao seu:

    * ``temporal-holdout``: o corte e global no tempo, entao para todo titular
      o treino vem antes da validacao, que vem antes do teste;
    * os embaralhados, quando agrupam pelo PROPRIO titular da sequencia: cada
      titular cai inteiro num unico split, com seu historico completo.

    Fora disso, o embaralhamento espalharia os registros de um mesmo titular
    entre os splits, e as janelas ficariam com buracos no meio do historico.
    """
    if definition.sequence_keys is None:
        return ()
    holder_column = definition.sequence_keys[0]
    return tuple(
        protocol
        for protocol in definition.protocols
        if protocol == TEMPORAL_HOLDOUT or definition.group_column == holder_column
    )


def tabbert_protocol(definition: DatasetDefinition) -> str | None:
    """Protocolo em que o TabBERT roda numa base, ou None se ela nao entra nele.

    O temporal quando a base tem ordem temporal; senao, o holdout embaralhado
    agrupado pelo titular. E o mesmo protocolo dos outros blocos, para que o
    TabBERT seja pareado com eles (item G16).
    """
    if BLOCK_TABBERT not in definition.blocks:
        return None
    protocols = sequence_compatible_protocols(definition)
    for protocol in (TEMPORAL_HOLDOUT, STRATIFIED_HOLDOUT):
        if protocol in protocols:
            return protocol
    return None


def records_history_strata(definition: DatasetDefinition, test_strategy: str | None) -> bool:
    """True quando uma configuracao grava o estrato de historico de cada transacao.

    So as bases do TabBERT, no protocolo dele: e ai que o corte pelo tamanho do
    historico pode ser comparado entre o TabBERT e os modelos que nao usam
    historico (item C3.19 da revisao).
    """
    return bool(test_strategy) and tabbert_protocol(definition) == test_strategy


def normalize_dataset_id(dataset_identifier: str) -> str:
    """Resolve um identificador canonico ou alias historico."""
    normalized = str(dataset_identifier).strip()
    if normalized in DATASET_DEFINITIONS:
        return normalized
    if normalized in DATASET_ID_BY_ALIAS:
        return DATASET_ID_BY_ALIAS[normalized]
    supported = ", ".join(sorted(DATASET_DEFINITIONS))
    raise ValueError(
        f"Base de dados desconhecida: {dataset_identifier}. Suportadas: {supported}"
    )


def get_dataset_definition(dataset_identifier: str) -> DatasetDefinition:
    try:
        return DATASET_DEFINITIONS[normalize_dataset_id(dataset_identifier)]
    except (TypeError, ValueError):
        raise


def resolve_dataset_identity(
    dataset_id: str | None = None,
    database_type: str | None = None,
    dataset_name: str | None = None,
) -> dict[str, str]:
    """Valida campos persistidos e devolve a identidade canonica completa."""

    def present(value: str | None) -> str | None:
        if value is None:
            return None
        normalized = str(value).strip()
        return None if normalized == "" or normalized.lower() == "nan" else normalized

    provided_identifiers = [
        value for value in (present(dataset_id), present(database_type)) if value
    ]
    if not provided_identifiers:
        raise ValueError("Dataset identity requires dataset_id or database_type")
    resolved_ids = {normalize_dataset_id(value) for value in provided_identifiers}
    if len(resolved_ids) != 1:
        raise ValueError(
            "Conflicting dataset identity fields: " + ", ".join(provided_identifiers)
        )
    canonical_id = resolved_ids.pop()
    definition = DATASET_DEFINITIONS[canonical_id]
    provided_name = present(dataset_name)
    if provided_name is not None and provided_name != definition.dataset_name:
        raise ValueError(
            f"Conflicting dataset_name for {canonical_id}: {provided_name!r}; "
            f"expected {definition.dataset_name!r}"
        )
    return {
        "dataset_id": canonical_id,
        "dataset_name": definition.dataset_name,
        "database_type": canonical_id,
    }


def _dataset_path(*path_parts: str) -> str:
    return os.path.join(CURRENT_PATH, RELATIVE_DATA_PATH, *path_parts)


def _add_haversine_distance(
    dataframe: pd.DataFrame,
    latitude_column: str,
    longitude_column: str,
    other_latitude_column: str,
    other_longitude_column: str,
    output_column: str,
) -> pd.DataFrame:
    """Distancia em km entre cliente e estabelecimento.

    Substitui quatro coordenadas brutas por um unico atributo com significado
    operacional; as coordenadas isoladas identificam o cliente.
    """
    earth_radius_km = 6371.0
    latitude = np.radians(dataframe[latitude_column].astype(float))
    other_latitude = np.radians(dataframe[other_latitude_column].astype(float))
    latitude_difference = other_latitude - latitude
    longitude_difference = np.radians(
        dataframe[other_longitude_column].astype(float)
        - dataframe[longitude_column].astype(float)
    )
    central_angle = (
        np.sin(latitude_difference / 2) ** 2
        + np.cos(latitude) * np.cos(other_latitude) * np.sin(longitude_difference / 2) ** 2
    )
    dataframe[output_column] = (
        2 * earth_radius_km * np.arcsin(np.sqrt(central_angle.clip(0, 1)))
    )
    return dataframe


def _load_ulb(definition: DatasetDefinition) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    data = pd.read_csv(_dataset_path("ref-18", "creditcard.csv"))
    # Time e o numero de segundos desde a primeira transacao da base. Como
    # atributo absoluto ele so serve para localizar a transacao no periodo
    # coletado; dele fica a hora do dia, que tem significado operacional.
    data["hour_of_day"] = (data["Time"] / 3600.0) % 24
    return data, None


def _load_home_credit(
    definition: DatasetDefinition,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    application_data = pd.read_csv(_dataset_path("ref-19", "application_data.csv"))
    previous_application_data = pd.read_csv(
        _dataset_path("ref-19", "previous_application.csv")
    )
    previous_application_data = previous_application_data.rename(
        columns={
            column: f"PREV_{column}"
            for column in PREVIOUS_APPLICATION_OVERLAPPING_COLUMNS
            if column in previous_application_data.columns
        }
    )
    # Juncao a esquerda (item G16): a interna descartava os 16.454 clientes
    # sem pedido anterior, com inadimplencia de 5,96% contra 8,19% dos demais,
    # e o par com a home-credit-somente-aplicacao comparava populacoes
    # diferentes. Cliente sem historico fica com uma linha e os PREV_*
    # ausentes; a ausencia e informativa.
    merged_data = pd.merge(
        application_data,
        previous_application_data,
        on="SK_ID_CURR",
        how="left",
    )
    return merged_data, None


def _load_home_credit_application_only(
    definition: DatasetDefinition,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Home Credit sem a juncao: uma linha por cliente.

    A versao com juncao replica cada cliente em ~4,9 linhas, com os mesmos
    atributos do pedido atual e o mesmo TARGET. Esta visao le apenas
    ``application_data.csv``, de modo que nenhum cliente se repete. Alem de
    eliminar por construcao a origem do vazamento, ela nao sofre o vies de
    selecao da juncao interna, que descartava os 16.454 clientes sem pedidos
    anteriores (307.511 menos 291.057). Desde o item G16 a completa usa juncao
    a esquerda, e as duas tem os mesmos clientes.
    """
    return pd.read_csv(_dataset_path("ref-19", "application_data.csv")), None


def _load_sparkov(
    definition: DatasetDefinition,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    def prepare(frame: pd.DataFrame) -> pd.DataFrame:
        frame = frame.rename(columns={"Unnamed: 0": "indexValue"})
        transaction_time = pd.to_datetime(frame["trans_date_trans_time"])
        frame["trans_date_trans_time"] = transaction_time
        frame["hour"] = transaction_time.dt.hour
        frame["day_of_week"] = transaction_time.dt.dayofweek
        frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
        frame["month"] = transaction_time.dt.month
        frame["day_of_month"] = transaction_time.dt.day
        # Idade no momento da transacao, em anos; substitui a data de
        # nascimento, que e um dado pessoal identificador.
        birth_date = pd.to_datetime(frame["dob"], errors="coerce")
        frame["age_years"] = (
            (transaction_time - birth_date).dt.days / 365.25
        ).astype(float)
        frame = _add_haversine_distance(
            frame, "lat", "long", "merch_lat", "merch_long", "customer_merchant_km"
        )
        return frame

    train_data = prepare(pd.read_csv(_dataset_path("ref-20", "fraudTrain.csv")))
    test_data = prepare(pd.read_csv(_dataset_path("ref-20", "fraudTest.csv")))
    return train_data, test_data


# Colunas que, combinadas, aproximam o titular na IEEE-CIS.
APPROXIMATE_HOLDER_COLUMNS = ("card1", "addr1")


def _build_approximate_holder_id(dataframe: pd.DataFrame) -> pd.Series:
    """Identificador aproximado de titular para a IEEE-CIS.

    Segundo a descricao publicada pelos organizadores, ``card1``-``card6`` sao
    informacoes do cartao (tipo, categoria, banco emissor, pais), e nao um
    identificador de portador: muitos titulares compartilham o mesmo valor, de
    modo que agrupar so por ``card1`` mistura clientes diferentes na mesma
    sequencia (item G8-iii da revisao).

    A combinacao adotada e ``card1`` + ``addr1``, a mesma usada na literatura
    de referencia da competicao. Continua sendo uma aproximacao: dois
    titulares com o mesmo tipo de cartao e a mesma regiao de endereco caem no
    mesmo grupo, e um titular que muda de endereco se divide em dois. Como o
    agrupamento e usado para *separar* particoes, o erro e conservador na
    direcao que importa: juntar demais nunca cria vazamento, apenas torna a
    separacao mais rigorosa do que o necessario.
    """
    components = [
        dataframe[column].astype("string").fillna("NA")
        if column in dataframe.columns
        else pd.Series("NA", index=dataframe.index, dtype="string")
        for column in APPROXIMATE_HOLDER_COLUMNS
    ]
    return components[0].str.cat(components[1:], sep="|")


def _convert_code_columns_to_text(
    dataframe: pd.DataFrame, code_columns: tuple[str, ...]
) -> pd.DataFrame:
    """Converte codigos numericos em categorias de texto sem parte decimal.

    ``315.0`` vira ``"315"``; o ausente continua ausente. Com o tipo de texto,
    todo bloco reconhece a coluna como categorica sem lista propria.
    """
    converted = dataframe.copy()
    for column in code_columns:
        if column not in converted.columns:
            continue
        numeric = pd.to_numeric(converted[column], errors="coerce")
        present = numeric.dropna()
        if (present % 1 != 0).any():
            raise ValueError(
                f"Codigo categorico {column} tem valores nao inteiros; "
                "nao pode ser convertido para texto sem parte decimal"
            )
        text = numeric.astype("Int64").astype("string")
        converted[column] = text.astype(object).where(text.notna(), np.nan)
    return converted


def _load_ieee_cis(
    definition: DatasetDefinition,
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Carrega uma das tres visoes rotuladas, sem tocar nos arquivos de teste.

    Os arquivos ``test_transaction.csv`` e ``test_identity.csv`` da competicao
    nao tem rotulo e nunca sao lidos por este pipeline.
    """
    # Uma variante de ablacao carrega exatamente como a base de origem.
    dataset_id = resolve_base_definition(definition).dataset_id
    transaction_data = pd.read_csv(_dataset_path("ref-22", "train_transaction.csv"))
    transaction_data = pd.concat(
        [
            transaction_data,
            _build_approximate_holder_id(transaction_data).rename(
                "approximate_holder_id"
            ),
        ],
        axis=1,
    )
    # Depois do identificador de titular, que usa card1 e addr1 como numeros
    # originais. A lista vem da base de origem para que a variante tenha
    # linhas identicas (e o mesmo partition_md5) mesmo ao tirar card1.
    transaction_data = _convert_code_columns_to_text(
        transaction_data, resolve_base_definition(definition).categorical_code_columns
    )

    if dataset_id == "ieee-cis-transacional":
        return transaction_data, None

    identity_data = pd.read_csv(_dataset_path("ref-22", "train_identity.csv"))
    matched_with_identity = transaction_data.merge(
        identity_data,
        on="TransactionID",
        how="inner",
        sort=False,
        validate="one_to_one",
    )
    return matched_with_identity, None


_DATASET_LOADERS = {
    "ulb": _load_ulb,
    "home-credit-completa": _load_home_credit,
    "home-credit-somente-aplicacao": _load_home_credit_application_only,
    "sparkov": _load_sparkov,
    "ieee-cis-completa": _load_ieee_cis,
    "ieee-cis-sem-identidade": _load_ieee_cis,
    "ieee-cis-transacional": _load_ieee_cis,
    "ieee-cis-completa-sem-alta-cardinalidade": _load_ieee_cis,
    "sparkov-sem-titular": _load_sparkov,
}


def drop_exact_duplicate_records(
    dataframe: pd.DataFrame, database_type: str
) -> pd.DataFrame:
    """Remove registros exatamente repetidos, com todas as colunas de origem.

    A base da ULB traz 1.081 linhas identicas em todas as 31 colunas,
    inclusive ``Time``. Sao registros repetidos na ingestao, e nao transacoes
    distintas: mantidos, uma particao aleatoria espalha copias do mesmo
    registro entre treino e teste, e o modelo acerta por memorizacao. O KNN e
    o caso extremo, porque o vizinho mais proximo de uma linha de teste passa
    a ser a sua propria copia no treino.

    Esta e uma limpeza de qualidade do dado, nao uma alteracao do desenho
    experimental: nenhuma transacao distinta e descartada. Linhas que ficam
    identicas apenas DEPOIS do descarte da marca de tempo sao transacoes
    diferentes que coincidem nos atributos, e permanecem na base.
    """
    duplicate_count = int(dataframe.duplicated().sum())
    if duplicate_count == 0:
        return dataframe
    print(
        f"deduplicacao={database_type}|registros_identicos_removidos={duplicate_count}"
    )
    return dataframe.drop_duplicates().reset_index(drop=True)


def _derive_temporal_features(
    dataframe: pd.DataFrame, definition: DatasetDefinition
) -> pd.DataFrame:
    """Deriva hora e dia da semana da IEEE-CIS a partir de TransactionDT."""
    if definition.dataset_id not in IEEE_CIS_DATASET_IDS:
        return dataframe
    seconds = dataframe["TransactionDT"].astype(float)
    # Uma concatenacao unica: a IEEE-CIS tem centenas de colunas e inserir uma
    # de cada vez fragmenta o quadro.
    derived = pd.DataFrame(
        {
            "hour_of_day": (seconds / 3600.0) % 24,
            "day_of_week": ((seconds / (3600.0 * 24)) % 7).astype(int),
        },
        index=dataframe.index,
    )
    return pd.concat([dataframe, derived], axis=1)


def _same_value(first: object, second: object) -> bool:
    if pd.isna(first) and pd.isna(second):
        return True
    return bool(first == second)


def _column_is_constant(series_list: list[pd.Series]) -> bool:
    """True quando a coluna tem um unico valor em todos os quadros.

    O ausente conta como valor: uma coluna so com ``T`` e ausentes NAO e
    constante, porque a ausencia carrega informacao. Uma coluna toda ausente e.
    """
    representatives = []
    for series in series_list:
        if len(series) == 0:
            continue
        if series.nunique(dropna=False) > 1:
            return False
        representatives.append(series.iloc[0])
    if not representatives:
        return False
    first = representatives[0]
    return all(_same_value(first, other) for other in representatives[1:])


def find_constant_columns(
    frames: list[pd.DataFrame],
    protected_columns: set[str],
) -> list[str]:
    """Colunas com valor unico no dado principal mais o auxiliar.

    O criterio nao usa o rotulo: uma coluna constante em todas as linhas nao
    carrega informacao de nenhum split, entao descarta-la antes do corte nao
    vaza nada. Calcular por dobra daria listas diferentes entre dobras e
    impediria documentar uma lista fixa por base.
    """
    present = [frame for frame in frames if frame is not None]
    constants = []
    for column in present[0].columns:
        if column in protected_columns:
            continue
        if _column_is_constant([frame[column] for frame in present if column in frame]):
            constants.append(column)
    return constants


def _protected_columns(definition: DatasetDefinition) -> set[str]:
    """Colunas que nunca entram no criterio de constante."""
    protected = {definition.prediction_column}
    protected.update(definition.identifier_columns_to_drop)
    for optional_column in (
        definition.group_column,
        definition.temporal_order_column,
        definition.evaluation_unit_column,
    ):
        if optional_column:
            protected.add(optional_column)
    if definition.sequence_keys:
        protected.update(definition.sequence_keys)
    return protected


def load_raw_dataset(
    dataset_identifier: str,
) -> tuple[pd.DataFrame, pd.DataFrame | None, DatasetDefinition]:
    """Carrega a base bruta, derivando atributos relativos e o agrupamento.

    As colunas identificadoras ainda NAO sao removidas aqui: a de agrupamento
    e a de ordenacao temporal sao necessarias para particionar. O descarte
    acontece depois da particao, em ``drop_identifier_columns``. As colunas
    constantes (``find_constant_columns``) saem ja na carga, e a procedencia
    das colunas fica em ``train_data.attrs["column_provenance"]`` para o
    registro de colunas dos arquivos preparados.
    """
    definition = get_dataset_definition(dataset_identifier)
    dataset_id = definition.dataset_id
    loader = _DATASET_LOADERS[dataset_id]
    train_data, auxiliary_data = loader(definition)
    train_data = drop_exact_duplicate_records(train_data, dataset_id)
    train_data = _derive_temporal_features(train_data, definition)
    if auxiliary_data is not None:
        auxiliary_data = drop_exact_duplicate_records(
            auxiliary_data, f"{dataset_id}-auxiliar"
        )
        auxiliary_data = _derive_temporal_features(auxiliary_data, definition)

    loaded_columns = list(train_data.columns)
    constant_columns = find_constant_columns(
        [train_data, auxiliary_data], _protected_columns(definition)
    )
    if constant_columns:
        print(
            f"colunas_constantes={dataset_id}|removidas={','.join(constant_columns)}"
        )
        train_data = train_data.drop(columns=constant_columns)
        if auxiliary_data is not None:
            auxiliary_data = auxiliary_data.drop(
                columns=[c for c in constant_columns if c in auxiliary_data.columns]
            )
    derived_columns = [
        column
        for column in _derived_columns_of(definition)
        if column in loaded_columns
    ]
    train_data.attrs["column_provenance"] = {
        "source_columns": [c for c in loaded_columns if c not in derived_columns],
        "derived_columns": derived_columns,
        "constant_columns": constant_columns,
    }
    return train_data, auxiliary_data, definition


def _derived_columns_of(definition: DatasetDefinition) -> tuple[str, ...]:
    return resolve_base_definition(definition).derived_columns


def columns_to_drop_after_partitioning(definition: DatasetDefinition) -> list[str]:
    """Colunas que saem dos atributos depois de cumprirem seu papel.

    Inclui os identificadores declarados, a coluna de agrupamento e a de
    ordenacao temporal: nenhuma delas pode chegar ao modelo.
    """
    columns = list(definition.identifier_columns_to_drop)
    for auxiliary_column in (definition.group_column, definition.temporal_order_column):
        if auxiliary_column and auxiliary_column not in columns:
            columns.append(auxiliary_column)
    for ablated_column in definition.ablated_columns:
        if ablated_column not in columns:
            columns.append(ablated_column)
    return columns


def identifier_columns_to_drop(definition: DatasetDefinition) -> list[str]:
    """Colunas removidas por serem identificador, agrupamento ou ordenacao."""
    return [
        column
        for column in columns_to_drop_after_partitioning(definition)
        if column not in definition.ablated_columns
    ]


def resolve_base_definition(definition: DatasetDefinition) -> DatasetDefinition:
    """Base de origem de uma variante; a propria base quando nao e variante."""
    if definition.base_dataset_id is None:
        return definition
    return DATASET_DEFINITIONS[definition.base_dataset_id]


def describe_catalog() -> pd.DataFrame:
    """Quadro-resumo das bases, usado pelos scripts de validacao e relatorio."""
    return pd.DataFrame(
        [
            {
                "dataset_id": definition.dataset_id,
                "dataset_name": definition.dataset_name,
                "database_type": definition.dataset_id,
                "legacy_aliases": ",".join(definition.legacy_aliases),
                "source_files": definition.source_files,
                "prediction_column": definition.prediction_column,
                "applies_pca_in_classic_block": definition.applies_pca_in_classic_block,
                "domain": definition.domain,
                "group_column": definition.group_column or "",
                "evaluation_unit_column": definition.evaluation_unit_column or "",
                "temporal_order_column": definition.temporal_order_column or "",
                "protocols": ",".join(definition.protocols),
                "dropped_columns": ",".join(
                    columns_to_drop_after_partitioning(definition)
                ),
            }
            for definition in DATASET_DEFINITIONS.values()
        ]
    )
