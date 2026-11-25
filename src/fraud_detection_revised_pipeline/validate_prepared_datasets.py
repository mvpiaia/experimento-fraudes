"""Valida as bases preparadas: vazamento, integridade e consistencia.

Este script existe porque o relatorio de revisao apontou varios indicios de
vazamento e de erro no pipeline que so se detectam olhando para os arquivos
efetivamente gravados, e nao para o codigo que os gera. Ele roda em dois
modos complementares.

``--mode prepared`` audita o que esta no disco, lendo o manifesto e os CSVs:

    P1  integridade  -- o MD5 do conteudo de cada arquivo bate com o do
                        manifesto, isto e, o arquivo lido na execucao e
                        exatamente o que foi preparado;
    P2  sobreposicao -- nenhuma linha aparece em mais de uma particao da
                        mesma dobra;
    P3  identificadores -- nenhuma coluna identificadora, de agrupamento ou
                        de tempo absoluto sobrou nos atributos;
    P4  prevalencia  -- validacao e teste mantem a prevalencia da base; so o
                        treino pode ter sido reamostrado;
    P5  dobras       -- os indices comecam em 1 e sao contiguos;
    P6  duplicatas   -- duas configuracoes distintas nunca produzem o mesmo
                        conjunto de arquivos (era o caso de eqm-ref-18 x
                        eqm-ref-21 e das 8 copias por base sem categoricas);
    P7  bases banidas -- eqm-ref-21 e suas variantes nao aparecem;
    P8  vocabulario -- nos modos de Transformer, nenhum identificador,
                        agrupamento ou marca de tempo entra como CAMPO do
                        modelo, mesmo quando a coluna sobrevive no arquivo;
    P9  pareamento  -- mesma base de origem, protocolo, semente, dobra e split
                       tem o mesmo partition_md5 em todos os blocos e
                       variantes; nos pares por unidade, o mesmo
                       unit_partition_md5.

``--mode partition`` refaz a particao a partir da base bruta, com a mesma
semente, e checa o que os arquivos preparados ja nao conseguem mostrar,
porque as colunas foram removidas:

    D1  grupos       -- nenhum cliente, cartao ou titular aparece em duas
                        particoes ao mesmo tempo;
    D2  tempo        -- no protocolo temporal, todo registro de treino e
                        anterior a todo registro de teste;
    D3  rotulo       -- a coluna alvo esta presente e tem as duas classes.

``--mode prepared`` tambem roda P10-P16 e ``--mode results`` roda R1-R6,
que perguntam se cada eixo da grade muda o que deveria mudar (itens G15 e
G17); ver ``preparation_sanity_checks.py``. Com ``--appendix-output DIR``, as
tabelas do apendice sao gravadas mesmo quando ha falhas.

Uso:
    python3 validate_prepared_datasets.py --mode prepared --appendix-output apendice/
    python3 validate_prepared_datasets.py --mode results --manifest m.tsv
    python3 validate_prepared_datasets.py --mode results --manifest m.tsv \
        --blocks classic boosting --execution-log registro_execucoes.csv
    python3 validate_prepared_datasets.py --mode results --results antigos/*.csv  # so R2, legado
    python3 validate_prepared_datasets.py --mode prepared
    python3 validate_prepared_datasets.py --mode partition --dataset sparkov
    python3 validate_prepared_datasets.py --mode partition  # todas as bases
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

import data_partitioning
import equivalent_configurations as equivalent
import experiment_datasets
import history_strata
import lr_nb_knn_specific as pipeline
import preparation_sanity_checks as sanity
from data_partitioning import AUXILIARY_SOURCE_COLUMN
from experiment_datasets import (
    DATASET_DEFINITIONS,
    TEMPORAL_HOLDOUT,
    columns_to_drop_after_partitioning,
    get_dataset_definition,
)
from prepared_dataset_storage import (
    DEFAULT_PREPARED_DATA_DIRECTORY,
    DEFAULT_RANDOM_SEED,
    FIRST_FOLD_INDEX,
    compute_split_content_md5,
    prepared_split_path,
    read_prepared_split_csv,
    normalize_manifest_dataset_identity,
)


BANNED_DATABASE_TYPES = frozenset(
    {"eqm-ref-21", "eqm-ref-21-pca", "eqm-ref-18-pca"}
)
# Tolerancia da comparacao de prevalencia entre validacao/teste e a base.
# Particoes finitas nao reproduzem a prevalencia exata; o que precisa ser
# detectado e um desvio grosseiro, da ordem do que a reamostragem causaria.
PREVALENCE_TOLERANCE = 0.02
# Identidade da linha de origem, usada somente durante a auditoria.
SOURCE_ROW_INDEX_COLUMN = "__source_row_index"
# Modos em que as colunas de identidade sobrevivem no arquivo preparado de
# proposito: o indice de janelas causais do TabBERT referencia o titular e a
# ordem, entao elas nao podem ser descartadas na gravacao. A garantia de que
# nao viram atributo do modelo e dada por P8, que le o vocabulario.
SEQUENCE_HANDLING_MODE = "sequence-tokenized"
TRANSFORMER_HANDLING_MODES = frozenset(
    {SEQUENCE_HANDLING_MODE, "transformer-indexed"}
)


@dataclass
class CheckOutcome:
    check_id: str
    scope: str
    passed: bool
    detail: str

    def render(self) -> str:
        status = "OK  " if self.passed else "FALHA"
        return f"[{status}] {self.check_id} {self.scope}: {self.detail}"


class ValidationReport:
    def __init__(self) -> None:
        self.outcomes: list[CheckOutcome] = []

    def add(self, check_id: str, scope: str, passed: bool, detail: str) -> None:
        outcome = CheckOutcome(check_id, scope, passed, detail)
        self.outcomes.append(outcome)
        print(outcome.render())

    @property
    def failures(self) -> list[CheckOutcome]:
        return [outcome for outcome in self.outcomes if not outcome.passed]

    def summarize(self) -> int:
        print()
        print(f"Verificacoes executadas: {len(self.outcomes)}")
        print(f"Falhas: {len(self.failures)}")
        for failure in self.failures:
            print(f"  - {failure.check_id} {failure.scope}: {failure.detail}")
        return 1 if self.failures else 0


def _row_fingerprints(dataframe: pd.DataFrame) -> set[int]:
    """Impressao digital de cada linha, para detectar sobreposicao.

    Usa o hash de linha do pandas sobre as colunas ordenadas, de modo que a
    comparacao nao dependa da ordem das colunas entre particoes.
    """
    ordered = dataframe.reindex(sorted(dataframe.columns), axis=1)
    return set(pd.util.hash_pandas_object(ordered, index=False).tolist())


def _positive_rate(dataframe: pd.DataFrame, target_column: str) -> float | None:
    if dataframe is None or dataframe.empty or target_column not in dataframe.columns:
        return None
    values = pd.to_numeric(dataframe[target_column], errors="coerce").dropna()
    if values.empty:
        return None
    return float((values > 0).mean())


# --------------------------------------------------------------------------
# Modo "prepared": auditoria dos arquivos ja gravados
# --------------------------------------------------------------------------


def resolve_prepared_artifact(prepared_directory: Path, recorded_path: str) -> Path:
    """Arquivo lateral (vocabulario, janelas) no diretorio validado.

    O manifesto grava o caminho de onde a configuracao foi preparada; na
    preparacao distribuida e o scratch do worker, que nao existe mais. Como o
    staging dos executores, procura pelo nome no diretorio validado e so usa o
    caminho gravado quando o arquivo nao esta la.
    """
    local = prepared_directory / Path(recorded_path).name
    return local if local.is_file() else Path(recorded_path)


def validate_prepared_files(
    manifest_path: Path,
    prepared_directory: Path,
    report: ValidationReport,
    configuration_limit: int,
) -> None:
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    if not {"dataset_id", "database_type"}.intersection(manifest.columns):
        raise ValueError("Manifesto sem dataset_id ou database_type legado")
    identity_values = pd.Series("", index=manifest.index, dtype=str)
    if "database_type" in manifest.columns:
        identity_values = manifest["database_type"].fillna("").astype(str)
    if "dataset_id" in manifest.columns:
        canonical_values = manifest["dataset_id"].fillna("").astype(str)
        identity_values = canonical_values.where(canonical_values.str.strip() != "", identity_values)
    banned_mask = identity_values.isin(BANNED_DATABASE_TYPES)
    banned_present = sorted(set(identity_values[banned_mask]))
    report.add(
        "P7",
        "manifesto",
        not banned_present,
        "nenhuma base banida presente"
        if not banned_present
        else f"bases banidas encontradas: {', '.join(banned_present)}",
    )
    # Bases banidas nao pertencem ao catálogo novo e já foram diagnosticadas
    # em P7. As demais linhas passam pelo mesmo resolvedor usado na execução.
    manifest = normalize_manifest_dataset_identity(manifest.loc[~banned_mask])

    configuration_md5_list = list(manifest["configuration_md5"].dropna().unique())
    if configuration_limit > 0:
        configuration_md5_list = configuration_md5_list[:configuration_limit]

    # P6 precisa olhar todas as configuracoes de uma vez.
    file_set_by_configuration: dict[str, frozenset[str]] = {}

    for configuration_md5 in configuration_md5_list:
        rows = manifest[manifest["configuration_md5"] == configuration_md5]
        database_type = rows["database_type"].iloc[0]
        target_column = rows["prediction_column_name"].iloc[0]
        handling_mode = "encoded"
        if "categorical_handling_mode" in rows.columns:
            handling_mode = (
                str(rows["categorical_handling_mode"].iloc[0] or "encoded")
                .strip()
                .lower()
            )
        definition = (
            get_dataset_definition(database_type)
            if database_type in DATASET_DEFINITIONS
            else None
        )
        scope = f"{database_type}/{configuration_md5[:8]}"

        # P1 -- integridade do conteudo
        integrity_problems = []
        content_md5_values = []
        for row in rows.itertuples(index=False):
            path = prepared_split_path(prepared_directory, row.dataset_md5)
            if path is None:
                integrity_problems.append(f"{row.dataset_md5}.csv ausente")
                continue
            actual = compute_split_content_md5(path)
            content_md5_values.append(actual)
            recorded = str(getattr(row, "content_md5", "") or "").strip().lower()
            if recorded and recorded != actual:
                integrity_problems.append(
                    f"{path.name}: manifesto {recorded}, arquivo {actual}"
                )
        report.add(
            "P1",
            scope,
            not integrity_problems,
            "MD5 de conteudo confere em todos os arquivos"
            if not integrity_problems
            else "; ".join(integrity_problems),
        )
        file_set_by_configuration[configuration_md5] = frozenset(content_md5_values)

        # P8 -- nenhum identificador entra como CAMPO do modelo.
        #
        # Nos modos de Transformer, o vocabulario de cada dobra lista
        # exatamente os campos que o modelo recebe. E a leitura mais direta
        # possivel da pergunta "o identificador virou atributo?": nao depende
        # de inferir o que o pre-processamento fez, apenas do que ele gravou.
        if handling_mode in TRANSFORMER_HANDLING_MODES and definition is not None:
            banned_as_feature = set(
                columns_to_drop_after_partitioning(definition)
            )
            if definition.sequence_keys is not None:
                banned_as_feature.add(definition.sequence_keys[0])
            vocabulary_problems: list[str] = []
            checked_vocabularies = 0
            for vocabulary_value in sorted(
                {
                    str(value)
                    for value in rows.get(
                        "vocabulary_path", pd.Series(dtype=str)
                    ).dropna()
                    if str(value).strip()
                }
            ):
                vocabulary_file = resolve_prepared_artifact(
                    prepared_directory, vocabulary_value
                )
                if not vocabulary_file.is_file():
                    vocabulary_problems.append(f"{vocabulary_file.name} ausente")
                    continue
                metadata = json.loads(vocabulary_file.read_text(encoding="utf-8"))
                checked_vocabularies += 1
                model_fields = set(metadata.get("categorical_columns", [])) | set(
                    metadata.get("numerical_columns", [])
                )
                intruders = sorted(model_fields & banned_as_feature)
                if intruders:
                    vocabulary_problems.append(
                        f"{vocabulary_file.name}: {', '.join(intruders)}"
                    )
            if checked_vocabularies or vocabulary_problems:
                report.add(
                    "P8",
                    scope,
                    not vocabulary_problems,
                    f"{checked_vocabularies} vocabulario(s) sem identificador, "
                    "agrupamento ou marca de tempo entre os campos do modelo"
                    if not vocabulary_problems
                    else "identificadores entre os campos do modelo: "
                    + "; ".join(vocabulary_problems),
                )

        # P5 -- numeracao das dobras
        fold_indexes = sorted({int(value) for value in rows["fold_index"]})
        expected = list(range(FIRST_FOLD_INDEX, FIRST_FOLD_INDEX + len(fold_indexes)))
        report.add(
            "P5",
            scope,
            fold_indexes == expected,
            f"dobras {fold_indexes} comecam em {FIRST_FOLD_INDEX} e sao contiguas"
            if fold_indexes == expected
            else f"dobras inesperadas: {fold_indexes}",
        )

        for fold_index in fold_indexes:
            fold_rows = rows[rows["fold_index"].astype(int) == fold_index]
            fold_scope = f"{scope}/dobra-{fold_index}"
            splits: dict[str, pd.DataFrame] = {}
            for row in fold_rows.itertuples(index=False):
                path = prepared_split_path(prepared_directory, row.dataset_md5)
                if path is not None:
                    splits[row.split] = read_prepared_split_csv(path)
            if not splits:
                continue

            # P2 -- sobreposicao de atributos entre particoes.
            #
            # Uma coincidencia aqui tem duas causas possiveis, e elas pedem
            # reacoes diferentes. (a) A mesma linha de origem foi parar em
            # duas particoes: seria um defeito do particionamento, e e o que
            # esta verificacao procura. (b) Transacoes DISTINTAS coincidem em
            # todos os atributos restantes, o que acontece quando a base tem
            # registros repetidos nos atributos depois do descarte da marca
            # de tempo. Nesse caso a propria particao, isoladamente, ja
            # contem repeticoes internas. Quando isso ocorre, a coincidencia
            # entre particoes e uma propriedade do dado, nao do codigo, e o
            # item e registrado como observacao.
            fingerprints = {
                name: _row_fingerprints(frame) for name, frame in splits.items()
            }
            source_has_duplicate_features = any(
                len(fingerprints[name]) < len(frame)
                for name, frame in splits.items()
            )
            overlaps = []
            split_names = sorted(fingerprints)
            for first_index, first in enumerate(split_names):
                for second in split_names[first_index + 1 :]:
                    shared = fingerprints[first] & fingerprints[second]
                    if shared:
                        overlaps.append(f"{first}x{second}={len(shared)} linhas")
            if not overlaps:
                report.add(
                    "P2",
                    fold_scope,
                    True,
                    "nenhuma linha compartilhada entre treino, validacao e teste",
                )
            elif source_has_duplicate_features:
                report.add(
                    "P2",
                    fold_scope,
                    True,
                    "observacao: "
                    + "; ".join(overlaps)
                    + " -- a base tem transacoes distintas com atributos "
                    "identicos (ha repeticao dentro de uma mesma particao), "
                    "entao a coincidencia vem do dado, nao do particionamento; "
                    "a disjuncao por linha e provada em D4",
                )
            else:
                report.add(
                    "P2",
                    fold_scope,
                    False,
                    "; ".join(overlaps)
                    + " -- sem repeticao interna, isso indica reuso da mesma "
                    "linha entre particoes",
                )

            # P3 -- identificadores removidos
            if definition is not None:
                banned_columns = set(columns_to_drop_after_partitioning(definition))
                banned_columns.add(AUXILIARY_SOURCE_COLUMN)
                leftovers = sorted(
                    {
                        column
                        for frame in splits.values()
                        for column in frame.columns
                        if column in banned_columns
                    }
                )
                if handling_mode == SEQUENCE_HANDLING_MODE and leftovers:
                    # No TabBERT essas colunas PRECISAM sobreviver: o indice de
                    # janelas causais referencia o titular e a ordem. O que nao
                    # pode acontecer e elas virarem campo do modelo, e isso e
                    # P8 que verifica, lendo o vocabulario.
                    report.add(
                        "P3",
                        fold_scope,
                        True,
                        "esperado no modo sequencial: "
                        + ", ".join(leftovers)
                        + " sobrevivem porque o indice de janelas causais as "
                        "referencia; P8 verifica que nao viram campo do modelo",
                    )
                else:
                    report.add(
                        "P3",
                        fold_scope,
                        not leftovers,
                        "nenhum identificador, agrupamento ou marca de tempo nos atributos"
                        if not leftovers
                        else f"colunas proibidas presentes: {', '.join(leftovers)}",
                    )

            # P4 -- prevalencia preservada fora do treino
            train_rate = _positive_rate(splits.get("train"), target_column)
            heldout_rates = {
                name: _positive_rate(splits.get(name), target_column)
                for name in ("validation", "test")
                if splits.get(name) is not None
            }
            comparable = [rate for rate in heldout_rates.values() if rate is not None]
            if len(comparable) >= 2:
                spread = max(comparable) - min(comparable)
                # No corte cronologico por titular a composicao das particoes
                # muda por construcao: titulares com um unico registro vao so
                # para o teste e os com dois pulam a validacao, para que o
                # conjunto avaliado nao encolha por causa de historicos
                # curtos. A prevalencia entre validacao e teste, portanto, nao
                # tem por que coincidir.
                prevalence_is_structural = handling_mode == SEQUENCE_HANDLING_MODE
                formatted_train_rate = (
                    train_rate if train_rate is None else round(train_rate, 4)
                )
                if spread <= PREVALENCE_TOLERANCE:
                    detail = (
                        f"prevalencia validacao/teste {['%.4f' % r for r in comparable]}"
                        f" (treino {formatted_train_rate})"
                    )
                elif prevalence_is_structural:
                    detail = (
                        f"esperado no corte cronologico por titular: validacao e "
                        f"teste divergem em {spread:.4f} ({heldout_rates}) porque "
                        "titulares com um unico registro vao so para o teste e os "
                        "com dois pulam a validacao, para que o conjunto avaliado "
                        "nao encolha por causa de historicos curtos"
                    )
                else:
                    detail = (
                        f"validacao e teste divergem em {spread:.4f}: {heldout_rates}"
                    )
                report.add(
                    "P4",
                    fold_scope,
                    prevalence_is_structural or spread <= PREVALENCE_TOLERANCE,
                    detail,
                )
            elif comparable:
                report.add(
                    "P4",
                    fold_scope,
                    True,
                    f"prevalencia fora do treino {comparable[0]:.4f} "
                    f"(treino {train_rate if train_rate is None else round(train_rate, 4)})",
                )

    # P6 -- configuracoes distintas nao podem produzir os mesmos arquivos
    duplicates: dict[frozenset[str], list[str]] = {}
    for configuration_md5, file_set in file_set_by_configuration.items():
        if not file_set:
            continue
        duplicates.setdefault(file_set, []).append(configuration_md5)
    # Um eixo inerte (alta cardinalidade numa base sem colunas de alta
    # cardinalidade) produz gemeas legitimas: mesmos splits e vocabularios, e
    # so uma delas e executada (equivalent_configurations).
    twins = equivalent.twin_map(equivalent.inert_high_cardinality_groups(manifest))
    duplicated_groups = [
        group
        for group in duplicates.values()
        if len(group) > 1 and len({twins.get(md5, md5) for md5 in group}) > 1
    ]
    inert_note = (
        f"; {len(twins)} gemea(s) por eixo de alta cardinalidade inerte, "
        "executadas pelo representante"
        if twins
        else ""
    )
    report.add(
        "P6",
        "manifesto",
        not duplicated_groups,
        f"{len(file_set_by_configuration)} configuracoes produzem conjuntos de arquivos distintos"
        + inert_note
        if not duplicated_groups
        else "configuracoes com arquivos identicos: "
        + "; ".join(", ".join(group[:4]) for group in duplicated_groups[:5]),
    )

    validate_partition_pairing(manifest, report)


def validate_partition_pairing(manifest: pd.DataFrame, report: ValidationReport) -> None:
    """P9 -- mesma base, protocolo, semente, dobra e split: mesmas linhas.

    Os blocos preparam em momentos diferentes e recomputam a particao. Ela so
    coincide se a carga da base, a semente, o codigo e as bibliotecas forem os
    mesmos; o partition_md5, calculado sobre as linhas brutas de cada split,
    torna isso verificavel. Uma divergencia aqui quebra o pareamento entre
    familias (item G16 da revisao).

    O grupo e a base de origem: uma variante de ablacao precisa ter a mesma
    particao da sua base. Nos pares por unidade (as duas Home Credit), que tem
    linhas diferentes, o criterio e o mesmo unit_partition_md5.
    """
    if "partition_md5" not in manifest.columns:
        report.add(
            "P9",
            "manifesto",
            False,
            "manifesto sem a coluna partition_md5: prepare novamente ou rode um "
            "prepare_*.sh para migrar o cabecalho",
        )
        return
    partition_values = manifest["partition_md5"].fillna("").str.strip().str.lower()
    identified = manifest[partition_values != ""].assign(
        partition_md5=partition_values[partition_values != ""]
    )
    missing_modes = sorted(
        manifest.loc[partition_values == "", "categorical_handling_mode"]
        .fillna("encoded")
        .unique()
    )
    if missing_modes:
        report.add(
            "P9",
            "manifesto",
            True,
            "linhas preparadas antes do partition_md5 ficam fora da verificacao: "
            + ", ".join(missing_modes),
        )
    identified = identified.assign(
        base_dataset=identified["database_type"].map(sanity.base_dataset_id)
    )
    key_columns = [
        "base_dataset",
        "test_strategy",
        "random_seed",
        "auxiliary_data_is_test_data",
        "fold_index",
        "split",
    ]
    for key, group in identified.groupby(key_columns, dropna=False):
        modes = sorted(group["categorical_handling_mode"].fillna("encoded").unique())
        distinct = group["partition_md5"].unique()
        database_type, protocol, seed, _auxiliary, fold_index, split = key
        scope = f"{database_type}/{protocol}/semente-{seed}/dobra-{fold_index}/{split}"
        datasets = ", ".join(sorted(group["database_type"].unique()))
        report.add(
            "P9",
            scope,
            len(distinct) == 1,
            f"mesma particao em {len(group['configuration_md5'].unique())} "
            f"configuracoes ({', '.join(modes)}; {datasets})"
            if len(distinct) == 1
            else f"{len(distinct)} particoes diferentes entre {', '.join(modes)} "
            f"({datasets})",
        )
    _validate_unit_pairing(manifest, report)


def _validate_unit_pairing(manifest: pd.DataFrame, report: ValidationReport) -> None:
    """P9 nos pares por unidade: os mesmos clientes em cada split."""
    paired = manifest.assign(
        unit_pair=manifest["database_type"].map(sanity.unit_pair_id)
    )
    paired = paired[paired["unit_pair"].notna()]
    if paired.empty:
        return
    if "unit_partition_md5" not in paired.columns:
        report.add(
            "P9", "manifesto", False,
            "manifesto sem a coluna unit_partition_md5: os pares por unidade "
            "nao podem ser verificados",
        )
        return
    paired = paired.assign(
        unit_partition_md5=paired["unit_partition_md5"].fillna("").astype(str)
        .str.strip().str.lower()
    )
    key_columns = ["unit_pair", "test_strategy", "random_seed", "fold_index", "split"]
    for key, group in paired.groupby(key_columns, dropna=False):
        pair, protocol, seed, fold_index, split = key
        datasets = sorted(group["database_type"].unique())
        if len(datasets) < 2:
            continue
        distinct = set(group["unit_partition_md5"])
        passed = len(distinct) == 1 and "" not in distinct
        report.add(
            "P9",
            f"{pair}/{protocol}/semente-{seed}/dobra-{fold_index}/{split}",
            passed,
            f"mesmos clientes em {', '.join(datasets)}"
            if passed
            else f"clientes diferentes entre {', '.join(datasets)}",
        )


# --------------------------------------------------------------------------
# Modo "partition": refaz a particao a partir da base bruta
# --------------------------------------------------------------------------


def validate_partitioning(
    database_type: str,
    report: ValidationReport,
    random_seed: int,
) -> None:
    definition = get_dataset_definition(database_type)
    database_train, database_auxiliary, _ = pipeline.load_data(database_type)

    for protocol in definition.protocols:
        scope = f"{database_type}/{protocol}"
        # Reconstroi a particao sem remover as colunas auxiliares, que sao
        # justamente o que precisa ser auditado aqui.
        complete_data = database_train.copy()
        complete_data[AUXILIARY_SOURCE_COLUMN] = False
        if database_auxiliary is not None and not database_auxiliary.empty:
            auxiliary = database_auxiliary.copy()
            auxiliary[AUXILIARY_SOURCE_COLUMN] = True
            complete_data = pd.concat(
                [complete_data, auxiliary], ignore_index=True, sort=False
            )
        complete_data = complete_data.reset_index(drop=True)
        complete_data = complete_data.dropna(
            subset=[definition.prediction_column]
        ).reset_index(drop=True)
        # As dobras sao reindexadas ao serem montadas, entao a identidade da
        # linha de origem viaja como coluna. Durante a auditoria nada e
        # descartado das dobras, de modo que ela chega intacta a D4.
        complete_data[SOURCE_ROW_INDEX_COLUMN] = complete_data.index

        if protocol == data_partitioning.TEMPORAL_HOLDOUT:
            folds = data_partitioning._temporal_holdout(complete_data, _AuditDefinition(definition))
        elif protocol == data_partitioning.STRATIFIED_HOLDOUT:
            folds = data_partitioning._stratified_holdout(
                complete_data, _AuditDefinition(definition), random_seed
            )
        else:
            folds = data_partitioning._stratified_k_fold(
                complete_data, _AuditDefinition(definition), random_seed
            )

        for fold_position, fold in enumerate(folds):
            fold_scope = f"{scope}/dobra-{fold_position + FIRST_FOLD_INDEX}"
            # Os partidores devolvem posicoes; a auditoria materializa a tabela
            # completa, com as colunas auxiliares que a preparacao removeria.
            splits = {
                name.removesuffix("_data"): complete_data.iloc[positions]
                for name, positions in fold.items()
                if name in data_partitioning.SPLIT_NAMES
                and positions is not None
                and len(positions) > 0
            }

            # D3 -- rotulo presente e com as duas classes no treino
            train = splits.get("train")
            classes = (
                sorted(train[definition.prediction_column].dropna().unique())
                if train is not None
                else []
            )
            report.add(
                "D3",
                fold_scope,
                len(classes) >= 2,
                f"treino tem as classes {classes} em {len(train)} linhas"
                if len(classes) >= 2
                else f"treino degenerado: classes {classes}",
            )

            # D4 -- disjuncao por linha de origem.
            #
            # Esta e a prova forte de que nenhuma linha e reutilizada: as
            # particoes sao fatias posicionais da mesma tabela, e o indice
            # original de cada uma e preservado ate aqui. Se os conjuntos de
            # indices forem disjuntos, nenhuma transacao aparece em duas
            # particoes, independentemente de atributos coincidentes.
            index_sets = {
                name: set(frame[SOURCE_ROW_INDEX_COLUMN])
                for name, frame in splits.items()
                if SOURCE_ROW_INDEX_COLUMN in frame.columns
            }
            shared_indexes = []
            names = sorted(index_sets)
            for first_index, first in enumerate(names):
                for second in names[first_index + 1 :]:
                    shared = index_sets[first] & index_sets[second]
                    if shared:
                        shared_indexes.append(f"{first}x{second}={len(shared)} linhas")
            total_rows = sum(len(values) for values in index_sets.values())
            report.add(
                "D4",
                fold_scope,
                not shared_indexes,
                f"{total_rows} linhas distribuidas em particoes disjuntas por indice "
                f"de origem ({', '.join(f'{n}={len(v)}' for n, v in index_sets.items())})"
                if not shared_indexes
                else "; ".join(shared_indexes),
            )

            # D1 -- grupos disjuntos
            if definition.group_column:
                group_sets = {
                    name: set(frame[definition.group_column].dropna().unique())
                    for name, frame in splits.items()
                    if definition.group_column in frame.columns
                }
                shared_messages = []
                names = sorted(group_sets)
                for first_index, first in enumerate(names):
                    for second in names[first_index + 1 :]:
                        shared = group_sets[first] & group_sets[second]
                        if shared:
                            shared_messages.append(
                                f"{first}x{second}={len(shared)} grupos"
                            )
                total_groups = sum(len(values) for values in group_sets.values())
                if protocol == TEMPORAL_HOLDOUT:
                    # No protocolo temporal o grupo DEVE atravessar a
                    # fronteira. O que separa as particoes e o periodo, nao o
                    # titular, e esse e justamente o cenario de operacao: um
                    # sistema real conhece o historico do cartao e precisa
                    # decidir sobre a proxima transacao dele. Exigir grupos
                    # disjuntos aqui mediria outra coisa, a generalizacao para
                    # titulares nunca vistos. A garantia que importa neste
                    # protocolo e a ordem temporal, verificada em D2.
                    report.add(
                        "D1",
                        fold_scope,
                        True,
                        "esperado no protocolo temporal: "
                        + ("; ".join(shared_messages) or "nenhum grupo compartilhado")
                        + f" -- as particoes sao separadas pelo periodo, nao por "
                        f"'{definition.group_column}'; a garantia deste protocolo e D2",
                    )
                else:
                    report.add(
                        "D1",
                        fold_scope,
                        not shared_messages,
                        f"{total_groups} grupos de '{definition.group_column}' totalmente "
                        "separados entre as particoes"
                        if not shared_messages
                        else "; ".join(shared_messages),
                    )

            # D2 -- ordem temporal respeitada
            if protocol == TEMPORAL_HOLDOUT and definition.temporal_order_column:
                order_column = definition.temporal_order_column
                boundaries = {
                    name: (frame[order_column].min(), frame[order_column].max())
                    for name, frame in splits.items()
                    if order_column in frame.columns
                }
                violations = []
                if "train" in boundaries and "test" in boundaries:
                    if boundaries["train"][1] > boundaries["test"][0]:
                        violations.append(
                            f"treino termina em {boundaries['train'][1]} depois do "
                            f"inicio do teste em {boundaries['test'][0]}"
                        )
                if "train" in boundaries and "validation" in boundaries:
                    if boundaries["train"][1] > boundaries["validation"][0]:
                        violations.append(
                            f"treino termina em {boundaries['train'][1]} depois do "
                            f"inicio da validacao em {boundaries['validation'][0]}"
                        )
                report.add(
                    "D2",
                    fold_scope,
                    not violations,
                    "treino inteiramente anterior a validacao e ao teste: "
                    + ", ".join(
                        f"{name}=[{start}, {end}]"
                        for name, (start, end) in boundaries.items()
                    )
                    if not violations
                    else "; ".join(violations),
                )


class _AuditDefinition:
    """Expoe a definicao da base sem descartar as colunas auditadas.

    ``build_partitions`` remove identificadores, agrupamento e marca de tempo
    ao montar cada dobra, que e exatamente o comportamento desejado em
    producao. Para auditar o agrupamento e a ordem temporal e preciso ver
    essas colunas, entao este adaptador zera a lista de descarte e deixa o
    restante da definicao intacto.
    """

    def __init__(self, definition) -> None:
        self._definition = definition

    def __getattr__(self, name: str):
        return getattr(self._definition, name)

    @property
    def identifier_columns_to_drop(self) -> tuple[str, ...]:
        return ()

    @property
    def group_column(self) -> str | None:
        return self._definition.group_column

    @property
    def temporal_order_column(self) -> str | None:
        return self._definition.temporal_order_column


def _patched_columns_to_drop(definition) -> list[str]:
    """Durante a auditoria nada e descartado das dobras."""
    if isinstance(definition, _AuditDefinition):
        return []
    return columns_to_drop_after_partitioning(definition)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepared", "partition", "results"), required=True)
    parser.add_argument(
        "--results",
        nargs="*",
        default=None,
        help="CSVs de resultado do modo results. Padrao: resultados_*.csv ao lado do script.",
    )
    parser.add_argument(
        "--history-strata",
        nargs="*",
        default=None,
        help=(
            "Arquivos .jsonl com estratos de historico (R6). Padrao: "
            "resultados_*_history_strata.jsonl e resultados_*_details.jsonl ao lado dos resultados."
        ),
    )
    parser.add_argument(
        "--appendix-output",
        default=None,
        help="Diretorio onde gravar as tabelas do apendice (CSV e Markdown).",
    )
    parser.add_argument(
        "--prepared-data-directory",
        default=str(DEFAULT_PREPARED_DATA_DIRECTORY),
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Padrao: prepared_dataset_configurations.tsv no diretorio preparado.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Base a auditar no modo partition. Sem valor, audita todas.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--blocks",
        nargs="*",
        default=None,
        choices=sanity.RUN_BLOCKS,
        help="Blocos da grade prevista no R4. Padrao: os blocos presentes nos resultados.",
    )
    parser.add_argument(
        "--execution-log",
        default=None,
        help=(
            "Registro dos executores (registro_execucoes.csv), que explica as "
            "execucoes ausentes no R4. Padrao: ao lado dos resultados."
        ),
    )
    parser.add_argument(
        "--configuration-limit",
        type=int,
        default=0,
        help="Audita apenas as N primeiras configuracoes do manifesto (0 = todas).",
    )
    arguments = parser.parse_args()

    pipeline.RELATIVE_DATA_PATH = "../datasets"
    pipeline.CURRENT_PATH = str(Path(__file__).resolve().parent)
    experiment_datasets.RELATIVE_DATA_PATH = pipeline.RELATIVE_DATA_PATH
    experiment_datasets.CURRENT_PATH = pipeline.CURRENT_PATH

    report = ValidationReport()
    prepared_directory = Path(arguments.prepared_data_directory)
    manifest_path = Path(
        arguments.manifest
        or prepared_directory / "prepared_dataset_configurations.tsv"
    )
    appendix_tables: dict[str, pd.DataFrame] = {}

    if arguments.mode == "prepared":
        if not manifest_path.is_file():
            print(f"Manifesto nao encontrado: {manifest_path}", file=sys.stderr)
            return 2
        validate_prepared_files(
            manifest_path,
            prepared_directory,
            report,
            arguments.configuration_limit,
        )
        manifest = _load_verified_manifest(manifest_path)
        reports = sanity.load_fold_reports(manifest, prepared_directory)
        sanity.check_preparation_reports(
            manifest, prepared_directory, reports, report, arguments.configuration_limit
        )
        sanity.check_axis_effects(manifest, reports, report)
        sanity.check_missing_screening(manifest, prepared_directory, reports, report)
        sanity.check_pca_record(manifest, reports, report)
        sanity.check_configuration_identity(manifest, report)
        sanity.check_training_population(manifest, report)
        appendix_tables.update(sanity.appendix_tables(manifest, reports))
        appendix_tables["appendix_inert_axes.csv"] = equivalent.groups_table(
            equivalent.inert_high_cardinality_groups(manifest)
        )
    elif arguments.mode == "results":
        result_paths = (
            [Path(path) for path in arguments.results]
            if arguments.results
            else sorted(Path(__file__).resolve().parent.glob("resultados_*.csv"))
        )
        results = sanity.load_results(result_paths)
        if results.empty:
            print("Nenhum resultado encontrado.", file=sys.stderr)
            return 2
        # Sem manifesto, so R2 roda: e o modo legado, para CSVs antigos.
        legacy = arguments.manifest is None and not manifest_path.is_file()
        appendix_tables["appendix_axis_effects.csv"] = sanity.check_one_axis_pairs(
            results, report, legacy
        )
        if not legacy:
            manifest = _load_verified_manifest(manifest_path)
            reports = sanity.load_fold_reports(manifest, prepared_directory)
            sanity.check_results_traceability(results, manifest, report)
            sanity.check_training_sizes(results, reports, report)
            sanity.check_result_pairing(results, report)
            strata_paths = (
                [Path(path) for path in arguments.history_strata]
                if arguments.history_strata
                else history_strata.history_strata_paths(Path(result_paths[0]).resolve().parent)
            )
            sanity.check_history_strata(
                history_strata.read_history_strata_records(strata_paths), report
            )
            log_path = Path(
                arguments.execution_log
                or Path(result_paths[0]).resolve().parent / "registro_execucoes.csv"
            )
            inert_groups = equivalent.inert_high_cardinality_groups(manifest)
            run_counts, run_failures = sanity.check_run_counts(
                results,
                report,
                execution_log=sanity.load_execution_log(log_path),
                blocks=tuple(arguments.blocks) if arguments.blocks else None,
                twins=equivalent.twin_map(inert_groups),
            )
            appendix_tables["appendix_inert_axes.csv"] = equivalent.groups_table(inert_groups)
            appendix_tables["appendix_run_counts.csv"] = run_counts
            appendix_tables["appendix_run_failures.csv"] = run_failures
            appendix_tables.update(sanity.appendix_tables(manifest, reports))
    else:
        # Durante a auditoria as dobras preservam as colunas auxiliares.
        data_partitioning.columns_to_drop_after_partitioning = _patched_columns_to_drop
        database_types = (
            [arguments.dataset] if arguments.dataset else list(DATASET_DEFINITIONS)
        )
        for database_type in database_types:
            validate_partitioning(database_type, report, arguments.random_seed)

    if arguments.appendix_output:
        written = sanity.write_appendix(Path(arguments.appendix_output), appendix_tables)
        print(f"Apendice gravado em {arguments.appendix_output}: {len(written)} arquivos")
    return report.summarize()


def _load_verified_manifest(manifest_path: Path) -> pd.DataFrame:
    """Manifesto sem bases banidas e com dataset_id canonico, como em P1-P9."""
    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    identity = manifest.get("dataset_id", manifest.get("database_type")).fillna("")
    manifest = manifest.loc[~identity.isin(BANNED_DATABASE_TYPES)]
    return normalize_manifest_dataset_identity(manifest)


if __name__ == "__main__":
    raise SystemExit(main())
