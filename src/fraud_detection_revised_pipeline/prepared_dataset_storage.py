"""Store and retrieve experiment-ready dataframe splits using descriptive names.

Diferencas em relacao a versao anterior do pipeline, todas motivadas pelo
relatorio de revisao da dissertacao:

* ``test_data_size`` deixou de existir. O tamanho da reserva era inerte em
  duas politicas e descartava de 20% a 40% dos dados em outra (item G9c/G9d),
  entao o eixo foi removido da identidade da configuracao. O protocolo de
  avaliacao e descrito apenas por ``test_strategy``.
* ``random_seed`` entrou na identidade, para que repeticoes com sementes
  diferentes sejam configuracoes distintas e comparaveis (item G13).
* ``fold_index`` e 1-based. Quando a politica nao produz dobras, o indice e
  sempre 1, nunca vazio (pedido explicito do orientando).
* Cada arquivo gravado carrega, alem do MD5 de identidade (deterministico a
  partir de configuracao/dobra/split), um MD5 de *conteudo*, que e o que
  permite auditar depois se o arquivo lido na execucao e exatamente o que foi
  preparado.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from missing_screening import DROP_INCOMPLETE_ROWS_STRATEGY, DROP_SCREENING_RULE
from experiment_datasets import (
    get_dataset_definition,
    normalize_dataset_id,
    resolve_dataset_identity,
)


DEFAULT_DATA_DIRECTORY = Path("/media/marcospiaia/dayasets")
DEFAULT_PREPARED_DATA_DIRECTORY = DEFAULT_DATA_DIRECTORY
CATEGORICAL_HANDLING_MODE_ENCODED = "encoded"
CATEGORICAL_HANDLING_MODE_NATIVE = "native"
CATEGORICAL_HANDLING_MODE_TRANSFORMER = "transformer-indexed"
CATEGORICAL_HANDLING_MODE_SEQUENCE = "sequence-tokenized"
CATEGORICAL_HANDLING_MODES = frozenset(
    {
        CATEGORICAL_HANDLING_MODE_ENCODED,
        CATEGORICAL_HANDLING_MODE_NATIVE,
        CATEGORICAL_HANDLING_MODE_TRANSFORMER,
        CATEGORICAL_HANDLING_MODE_SEQUENCE,
    }
)
SPLIT_NAMES = ("train_data", "validation_data", "test_data")
SPLIT_FILE_LABELS = {
    "train_data": "train",
    "validation_data": "validation",
    "test_data": "test",
}
# Primeira dobra gravada. As politicas sem dobras produzem exatamente uma
# particao, registrada com este mesmo indice.
FIRST_FOLD_INDEX = 1
# Arquivo lateral com o identificador da unidade de avaliacao de cada linha de
# um split (o cliente, na Home Credit com juncao). Fica fora do CSV de
# atributos para nunca chegar ao modelo, e e achado pelo nome do split, sem
# coluna nova no manifesto. Gravado so para validacao e teste: o treino pode
# perder ou ganhar linhas na reamostragem, e a avaliacao por unidade so mede
# os conjuntos reservados.
EVALUATION_UNIT_FILE_SUFFIX = ".groups.csv"
EVALUATION_UNIT_COLUMN = "evaluation_unit"
EVALUATION_UNIT_SPLITS = ("validation_data", "test_data")

# Arquivo lateral com o estrato de historico de cada linha de validacao e
# teste (``base_history_length``, como no indice de janelas do TabBERT), nas
# bases e no protocolo do TabBERT (item C3.19). Fica fora do CSV de atributos,
# como o das unidades, e e achado pelo nome do split.
HISTORY_STRATA_FILE_SUFFIX = ".history.csv"
HISTORY_STRATA_COLUMN = "base_history_length"

# Revisao do pre-processamento, parte da identidade da configuracao. Muda
# quando os arquivos preparados mudam de conteudo sem que a grade mude: ``g14``
# acrescentou log1p nos valores monetarios, codigos categoricos, descarte de
# constantes e as bases de ablacao. Sem ela, o reaproveitamento por
# configuration_md5 retomaria, em silencio, arquivos preparados antes.
# ``g15`` acrescentou a triagem de colunas do drop-incomplete-rows, o relatorio
# de preparacao por dobra e as protecoes que recusam entradas invalidas.
PREPROCESSING_REVISION = "g15"
# Revisoes do esquema de atributos dos Transformers por base
# (transformer_feature_schema). Entram so no md5 dos modos Transformer da base
# listada, como a regra da triagem entra so no drop-incomplete-rows.
#   hc-numeric-rates: APARTMENTS_MODE, BASEMENTAREA_MODE, RATE_DOWN_PAYMENT,
#   RATE_INTEREST_PRIMARY e RATE_INTEREST_PRIVILEGED deixam de ser categoricas
#   (sao taxas e estatisticas normalizadas continuas), em 08/11/2026.
TRANSFORMER_SCHEMA_REVISIONS = {"home-credit-completa": "hc-numeric-rates"}

# Arquivo lateral com as colunas de cada arquivo preparado: o que entrou, o que
# saiu (e por que), o que foi derivado e transformado. Achado pelo nome do
# arquivo; o manifesto guarda so o hash do esquema (``feature_schema_md5``).
COLUMN_REGISTRY_FILE_SUFFIX = ".columns.json"


def column_registry_path(directory: str | Path, dataset_md5: str) -> Path:
    return Path(directory) / f"{dataset_md5}{COLUMN_REGISTRY_FILE_SUFFIX}"


def compute_feature_schema_md5(columns) -> str:
    """MD5 da lista ordenada de colunas do arquivo, unida por ``|``."""
    return hashlib.md5(
        "|".join(str(column) for column in columns).encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


def evaluation_unit_path(directory: str | Path, dataset_md5: str) -> Path:
    return Path(directory) / f"{dataset_md5}{EVALUATION_UNIT_FILE_SUFFIX}"


def history_strata_path(directory: str | Path, dataset_md5: str) -> Path:
    return Path(directory) / f"{dataset_md5}{HISTORY_STRATA_FILE_SUFFIX}"
DEFAULT_RANDOM_SEED = 42


def normalize_manifest_dataset_identity(manifest: pd.DataFrame) -> pd.DataFrame:
    """Return manifest rows with canonical identity columns, without rewriting it."""
    if not {"dataset_id", "database_type"}.intersection(manifest.columns):
        raise ValueError(
            "Prepared dataset manifest is missing dataset_id/database_type"
        )
    normalized = manifest.copy()
    identities = []
    for row_index, row in normalized.iterrows():
        try:
            identities.append(
                resolve_dataset_identity(
                    dataset_id=row.get("dataset_id"),
                    database_type=row.get("database_type"),
                    dataset_name=row.get("dataset_name"),
                )
            )
        except ValueError as error:
            raise ValueError(
                f"Invalid dataset identity in manifest row {row_index + 2}: {error}"
            ) from error
    for column in ("dataset_id", "dataset_name", "database_type"):
        normalized[column] = [identity[column] for identity in identities]
    return normalized


@dataclass(frozen=True)
class PreparedDatasetFile:
    dataset_md5: str
    fold_index: int
    split_label: str
    path: Path
    content_md5: str = ""
    # Identidade do split calculada sobre as linhas brutas, antes de qualquer
    # transformacao; igual entre blocos que avaliaram as mesmas linhas.
    partition_md5: str = ""
    # Hash das colunas do arquivo; igual ao do JSON lateral de colunas.
    feature_schema_md5: str = ""
    vocabulary_md5: str | None = None
    vocabulary_path: Path | None = None
    window_index_path: Path | None = None
    # Relatorio de preparacao da dobra e triagem do drop-incomplete-rows
    # (item G15); vazios nos blocos que nao gravam relatorio.
    preparation_report_md5: str = ""
    missing_screening_threshold: str = ""
    missing_screening_rule: str = ""
    screened_out_columns: str = ""
    # Pareamento (item G16): linhas brutas do treino efetivo, depois da
    # remocao de incompletos, e conjunto de unidades de avaliacao do split.
    train_population_md5: str = ""
    unit_partition_md5: str = ""
    # Clientes do treino efetivo, nas bases com unidade de avaliacao.
    train_unit_population_md5: str = ""


def compute_file_content_md5(path: str | Path) -> str:
    """Return the MD5 of a file's bytes, read in chunks."""
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Armazenamento compacto (compact-prepared-storage): um split pode estar no
# diretorio preparado como <dataset_md5>.csv ou <dataset_md5>.csv.zst. Nos dois
# casos o content_md5 e o MD5 do CSV descomprimido, e os executores so leem o
# CSV que o staging materializa no diretorio local.
SPLIT_FILE_SUFFIX = ".csv"
COMPRESSED_SPLIT_FILE_SUFFIX = ".csv.zst"
STORAGE_METADATA_FILE_SUFFIX = ".storage.json"
ZSTD_COMPRESSION_LEVEL = 3
_CHUNK_BYTES = 1024 * 1024


class PreparedSplitDecompressorError(RuntimeError):
    """Split armazenado comprimido sem o binario ``zstd`` disponivel.

    Nao herda de OSError: nao e falha do HD, e os runners nao devem esperar
    por ele (codigo 75), e sim falhar com esta mensagem.
    """


def zstd_command() -> str:
    command = shutil.which("zstd")
    if command is None:
        raise PreparedSplitDecompressorError(
            "zstd is required to read compressed prepared splits (.csv.zst); "
            "install it with: sudo apt install zstd"
        )
    return command


def is_compressed_split(path: str | Path) -> bool:
    return str(path).endswith(COMPRESSED_SPLIT_FILE_SUFFIX)


def prepared_split_path(directory: str | Path, dataset_md5: str) -> Path | None:
    """O split de ``dataset_md5`` no diretorio: CSV, senao comprimido, senao None."""
    directory = Path(directory)
    for suffix in (SPLIT_FILE_SUFFIX, COMPRESSED_SPLIT_FILE_SUFFIX):
        path = directory / f"{dataset_md5}{suffix}"
        if path.is_file():
            return path
    return None


class _DecompressedStream:
    """stdout do ``zstd -dc``, que registra se a leitura chegou ao fim."""

    def __init__(self, process: subprocess.Popen):
        self._process = process
        self.reached_end = False

    def read(self, size: int = -1) -> bytes:
        data = self._process.stdout.read(size)
        if not data and size != 0:
            self.reached_end = True
        return data

    def readable(self) -> bool:
        return True

    def __iter__(self):
        return iter(lambda: self.read(_CHUNK_BYTES), b"")


@contextlib.contextmanager
def open_prepared_split(path: str | Path, compressed: bool | None = None):
    """Fluxo binario com o CSV do split, aceito por ``pd.read_csv``.

    Um ``.zst`` corrompido lido ate o fim levanta OSError. Uma leitura
    interrompida (``nrows``) apenas encerra o descompressor.
    """
    path = Path(path)
    if not (is_compressed_split(path) if compressed is None else compressed):
        with open(path, "rb") as handle:
            yield handle
        return
    process = subprocess.Popen(
        [zstd_command(), "-dcq", "--", str(path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stream = _DecompressedStream(process)
    try:
        yield stream
    finally:
        if not stream.reached_end:
            process.kill()
        process.stdout.close()
        error = process.stderr.read().decode("utf-8", "replace").strip()
        process.stderr.close()
        return_code = process.wait()
        if stream.reached_end and return_code != 0:
            raise OSError(f"could not decompress {path}: {error or return_code}")


def read_prepared_split_csv(path: str | Path, **read_csv_arguments) -> pd.DataFrame:
    with open_prepared_split(path) as handle:
        return pd.read_csv(handle, **read_csv_arguments)


def split_content_digest(path: str | Path, compressed: bool | None = None) -> tuple[str, int]:
    """MD5 e tamanho do CSV descomprimido (ou do proprio CSV)."""
    digest = hashlib.md5(usedforsecurity=False)
    size = 0
    with open_prepared_split(path, compressed) as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def compute_split_content_md5(path: str | Path) -> str:
    return split_content_digest(path)[0]


def materialize_prepared_split(
    path: str | Path, destination: str | Path, expected_md5: str | None = None
) -> Path:
    """Grava o CSV do split em ``destination`` e confere o MD5 na escrita.

    Divergencia remove a copia e levanta ValueError.
    """
    destination = Path(destination)
    temporary = destination.with_name(f".{destination.name}.tmp")
    digest = hashlib.md5(usedforsecurity=False)
    try:
        with open_prepared_split(path) as source, open(temporary, "wb") as target:
            for chunk in iter(lambda: source.read(_CHUNK_BYTES), b""):
                digest.update(chunk)
                target.write(chunk)
        if expected_md5 and digest.hexdigest() != expected_md5.strip().lower():
            raise ValueError(
                f"content_md5 mismatch for {Path(path).name}: "
                f"{digest.hexdigest()} != {expected_md5.strip().lower()}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def compress_prepared_split(csv_path: str | Path, expected_md5: str | None = None) -> tuple[Path, int]:
    """Cria ``<dataset_md5>.csv.zst`` ao lado do CSV, conferido antes de aparecer.

    Devolve o caminho e o tamanho do CSV. O CSV original nao e removido.
    """
    csv_path = Path(csv_path)
    target = csv_path.with_name(csv_path.name + ".zst")
    temporary = csv_path.with_name(f".{target.name}.tmp")
    try:
        subprocess.run(
            [
                zstd_command(), f"-{ZSTD_COMPRESSION_LEVEL}", "-T0", "-q", "-f",
                "-o", str(temporary), "--", str(csv_path),
            ],
            check=True,
            capture_output=True,
        )
        compressed_md5, content_bytes = split_content_digest(temporary, compressed=True)
        expected = (expected_md5 or compute_file_content_md5(csv_path)).strip().lower()
        if compressed_md5 != expected:
            raise ValueError(
                f"content_md5 mismatch for {csv_path.name}: {compressed_md5} != {expected}"
            )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target, content_bytes


def storage_metadata_path(directory: str | Path, configuration_md5: str) -> Path:
    return Path(directory) / f"{configuration_md5}{STORAGE_METADATA_FILE_SUFFIX}"


def write_storage_metadata(
    directory: str | Path,
    configuration_md5: str,
    rows,
    content_bytes: dict[str, int] | None = None,
) -> Path:
    """Grava nome, codec e tamanhos de cada split da configuracao.

    ``content_bytes`` evita descomprimir de novo o que acabou de ser medido.
    """
    directory = Path(directory)
    content_bytes = content_bytes or {}
    splits = {}
    for row in rows:
        dataset_md5 = str(row["dataset_md5"]).strip().lower()
        if dataset_md5 in splits:
            continue
        path = prepared_split_path(directory, dataset_md5)
        if path is None:
            raise FileNotFoundError(f"prepared split not found: {dataset_md5} in {directory}")
        compressed = is_compressed_split(path)
        size = content_bytes.get(dataset_md5)
        if size is None:
            size = split_content_digest(path)[1] if compressed else path.stat().st_size
        splits[dataset_md5] = {
            "stored_name": path.name,
            "codec": "zstd" if compressed else "none",
            "stored_bytes": path.stat().st_size,
            "content_bytes": size,
            "content_md5": str(row.get("content_md5", "")).strip().lower(),
        }
    path = storage_metadata_path(directory, configuration_md5)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps({"configuration_md5": configuration_md5, "splits": splits}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    return path


def read_storage_metadata(directory: str | Path, configuration_md5: str) -> dict | None:
    path = storage_metadata_path(directory, configuration_md5)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# Faixas numericas que o preparador assume quando --numeric-bins nao e passado.
# O FT-Transformer nao discretiza, mas o valor entra na sua identidade desde a
# primeira revisao; manifestos e executores dependem desse md5.
DEFAULT_NUMERIC_BINS = 10


def transformer_identity_parameters(
    categorical_handling_mode: str | None,
    numeric_normalization: str | None,
    high_cardinality_strategy: str | None,
    high_cardinality_parameter: int | None,
    numeric_bins: int | None,
    window_length: int | None,
    history_length: int | None,
    window_stride: int | None,
) -> tuple:
    """Parametros de Transformer que entram no md5, na ordem de ``build_prepared_dataset_md5``.

    Regra unica do preparador e do plano distribuido: normalizacao,
    cardinalidade e faixas em todo modo Transformer; janelas so no
    ``sequence-tokenized``. Fora dos Transformers, nenhum entra.
    """
    sequence = categorical_handling_mode == CATEGORICAL_HANDLING_MODE_SEQUENCE
    transformer = sequence or categorical_handling_mode == CATEGORICAL_HANDLING_MODE_TRANSFORMER
    if not transformer:
        return (None,) * 7
    return (
        numeric_normalization,
        high_cardinality_strategy,
        high_cardinality_parameter,
        numeric_bins,
        window_length if sequence else None,
        history_length if sequence else None,
        window_stride if sequence else None,
    )


def build_prepared_dataset_md5(
    dataset: str,
    categorical_encoding: str | None,
    fill_na_strategy: str | None,
    test_strategy: str,
    prediction_column: str,
    undersampling_technique: str | None,
    undersampling_parameters: str | None,
    auxiliary_data_is_test_data: bool,
    categorical_handling_mode: str | None = None,
    random_seed: int = DEFAULT_RANDOM_SEED,
    numeric_normalization: str | None = None,
    high_cardinality_strategy: str | None = None,
    high_cardinality_parameter: int | None = None,
    numeric_bins: int | None = None,
    window_length: int | None = None,
    history_length: int | None = None,
    window_stride: int | None = None,
    include_transformer_schema_revision: bool = True,
) -> str:
    """Return the MD5 identity of the canonical preparation parameter string.

    ``include_transformer_schema_revision=False`` reproduz a identidade de
    antes de uma revisao de esquema (``TRANSFORMER_SCHEMA_REVISIONS``); serve
    so para localizar no plano as configuracoes substituidas.
    """
    parameter_values = (
        normalize_dataset_id(dataset),
        normalize_categorical_encoding_strategy(categorical_encoding),
        fill_na_strategy,
        test_strategy,
        prediction_column,
        undersampling_technique,
        undersampling_parameters,
        bool(auxiliary_data_is_test_data),
        int(random_seed),
        PREPROCESSING_REVISION,
    )
    if fill_na_strategy == DROP_INCOMPLETE_ROWS_STRATEGY:
        # A regra da triagem muda os arquivos; mudar o piso cria configuracoes
        # novas em vez de sobrescrever as antigas (item G15c).
        parameter_values = (*parameter_values, DROP_SCREENING_RULE)
    if categorical_handling_mode is not None:
        normalized_categorical_handling_mode = normalize_categorical_handling_mode(
            categorical_handling_mode
        )
        parameter_values = (
            *parameter_values,
            normalized_categorical_handling_mode,
        )
        if normalized_categorical_handling_mode in {
            CATEGORICAL_HANDLING_MODE_TRANSFORMER,
            CATEGORICAL_HANDLING_MODE_SEQUENCE,
        }:
            parameter_values = (
                *parameter_values,
                numeric_normalization,
                high_cardinality_strategy,
                high_cardinality_parameter,
                numeric_bins,
                window_length,
                history_length,
                window_stride,
            )
            schema_revision = TRANSFORMER_SCHEMA_REVISIONS.get(normalize_dataset_id(dataset))
            if schema_revision and include_transformer_schema_revision:
                # A tipagem das colunas mudou so nesta base: a identidade
                # muda nela, e as demais configuracoes ficam como estavam.
                parameter_values = (*parameter_values, schema_revision)
    canonical_parameter_string = "|".join(
        "none" if value is None or str(value).strip() == "" else str(value)
        for value in parameter_values
    )
    return hashlib.md5(
        canonical_parameter_string.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()


# Apelidos da codificacao categorica supervisionada. Ficam aqui, e nao no
# executor, porque a identidade da configuracao PRECISA normaliza-los: duas
# grafias da mesma celula do fatorial produziriam dois configuration_md5
# distintos para dados identicos, que e a duplicata que o item G9b eliminou.
CATEGORICAL_ENCODING_ALIASES: dict[str, str] = {
    "catboost": "ordered-target",
    "catboost-encoder": "ordered-target",
    "ordered-target-encoding": "ordered-target",
    # "woe" sempre significou log-razao com cross-fitting de cinco dobras, que e
    # exatamente a celula "log-odds:k-fold" do fatorial 2x2 (item C3.6).
    "woe": "log-odds:k-fold",
    "weight-of-evidence": "log-odds:k-fold",
}


def normalize_categorical_encoding_strategy(value: str | None) -> str | None:
    """Resolve apelidos para o nome canonico da estrategia de codificacao."""
    if value is None:
        return None
    normalized_value = str(value).strip().lower().replace("_", "-")
    if normalized_value == "":
        return None
    return CATEGORICAL_ENCODING_ALIASES.get(normalized_value, normalized_value)


def normalize_categorical_handling_mode(value: str | None) -> str:
    """Normalize the categorical handling mode used by preparation manifests."""
    if value is None or str(value).strip() == "":
        return CATEGORICAL_HANDLING_MODE_ENCODED

    normalized_value = str(value).strip().lower().replace("_", "-")
    if normalized_value not in CATEGORICAL_HANDLING_MODES:
        supported_modes = ", ".join(sorted(CATEGORICAL_HANDLING_MODES))
        raise ValueError(
            "categorical_handling_mode must be one of: "
            f"{supported_modes}"
        )
    return normalized_value


def save_fold_vocabulary(
    output_directory: str | Path,
    configuration_md5: str,
    fold_index: int,
    metadata: dict,
) -> tuple[str, Path]:
    """Persist deterministic preprocessing metadata for one training fold."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
    vocabulary_md5 = hashlib.md5(
        serialized.encode("utf-8"), usedforsecurity=False
    ).hexdigest()
    path = destination / f"{configuration_md5}_fold-{fold_index}_vocabulary.json"
    path.write_text(serialized + "\n", encoding="utf-8")
    return vocabulary_md5, path


def save_window_index(
    output_directory: str | Path,
    configuration_md5: str,
    fold_index: int,
    dataframe: pd.DataFrame,
) -> Path:
    """Persist the long-form sequence index for one fold."""
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{configuration_md5}_fold-{fold_index}_windows.csv"
    dataframe.to_csv(path, index=False)
    return path


def write_column_registry(
    directory: str | Path,
    dataset_md5: str,
    columns: list[str],
    registry: dict | None,
) -> str:
    """Grava o JSON lateral de colunas de um arquivo e devolve o hash do esquema.

    ``registry`` traz o que e comum a todos os arquivos da configuracao
    (origem, removidas por motivo, derivadas, transformadas, categoricas); o
    que depende do arquivo -- as colunas e o hash -- vem do proprio quadro.
    """
    feature_schema_md5 = compute_feature_schema_md5(columns)
    payload = {
        "preprocessing_revision": PREPROCESSING_REVISION,
        "columns": columns,
        "feature_schema_md5": feature_schema_md5,
        "source_columns": [],
        "dropped_columns": {"identifier": [], "constant": [], "ablation": []},
        "derived_columns": [],
        "log_transformed_columns": [],
        "categorical_columns": [],
        **(registry or {}),
    }
    payload["columns"] = columns
    payload["feature_schema_md5"] = feature_schema_md5
    with open(column_registry_path(directory, dataset_md5), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return feature_schema_md5


def save_prepared_data_splits(
    test_data_information: dict,
    output_directory: str | Path,
    configuration_md5: str,
    first_fold_index: int = FIRST_FOLD_INDEX,
) -> list[PreparedDatasetFile]:
    """Write each non-empty fold/split dataframe and return the created paths.

    As dobras sao numeradas a partir de 1. Politicas de particao sem dobras
    gravam uma unica particao, tambem com indice 1.
    """
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    created_files: list[PreparedDatasetFile] = []

    for fold_index, split_information in enumerate(
        test_data_information.get("test_info", []), start=first_fold_index
    ):
        partition_md5_by_split = split_information.get("partition_md5", {})
        unit_partition_md5_by_split = split_information.get("unit_partition_md5", {})
        # Sem remocao de incompletos (ou sem relatorio, nos Transformers), o
        # treino efetivo e o proprio treino da particao.
        train_population_md5 = split_information.get(
            "train_population_md5"
        ) or partition_md5_by_split.get("train_data", "")
        train_unit_population_md5 = split_information.get(
            "train_unit_population_md5"
        ) or unit_partition_md5_by_split.get("train_data", "")
        for split_name in SPLIT_NAMES:
            dataframe = split_information.get(split_name)
            if dataframe is None or dataframe.empty:
                continue
            split_label = SPLIT_FILE_LABELS[split_name]
            file_identity = f"{configuration_md5}|{fold_index}|{split_label}"
            dataset_md5 = hashlib.md5(
                file_identity.encode("utf-8"),
                usedforsecurity=False,
            ).hexdigest()
            output_path = destination / f"{dataset_md5}.csv"
            dataframe.to_csv(output_path, index=False)
            feature_schema_md5 = write_column_registry(
                destination,
                dataset_md5,
                list(dataframe.columns),
                test_data_information.get("column_registry"),
            )
            units = split_information.get("evaluation_units", {}).get(split_name)
            if units is not None and split_name in EVALUATION_UNIT_SPLITS:
                if len(units) != len(dataframe):
                    raise ValueError(
                        f"Evaluation units of {split_name} do not align with its "
                        f"{len(dataframe)} rows: got {len(units)}"
                    )
                pd.DataFrame({EVALUATION_UNIT_COLUMN: list(units)}).to_csv(
                    evaluation_unit_path(destination, dataset_md5), index=False
                )
            strata = split_information.get("history_strata", {}).get(split_name)
            if strata is not None and split_name in EVALUATION_UNIT_SPLITS:
                if len(strata) != len(dataframe):
                    raise ValueError(
                        f"History strata of {split_name} do not align with its "
                        f"{len(dataframe)} rows: got {len(strata)}"
                    )
                pd.DataFrame({HISTORY_STRATA_COLUMN: list(strata)}).to_csv(
                    history_strata_path(destination, dataset_md5), index=False
                )
            created_files.append(
                PreparedDatasetFile(
                    dataset_md5=dataset_md5,
                    fold_index=fold_index,
                    split_label=split_label,
                    path=output_path,
                    content_md5=compute_file_content_md5(output_path),
                    partition_md5=partition_md5_by_split.get(split_name, ""),
                    feature_schema_md5=feature_schema_md5,
                    train_population_md5=train_population_md5,
                    unit_partition_md5=unit_partition_md5_by_split.get(split_name, ""),
                    train_unit_population_md5=train_unit_population_md5,
                )
            )

    if not created_files:
        raise ValueError("No non-empty prepared dataset splits were available to save")
    return created_files


def declared_code_columns(manifest_rows: pd.DataFrame) -> list[str]:
    """Codigos categoricos declarados no catalogo para as bases das linhas.

    Um CSV nao guarda tipo: ``"315"`` volta como numero, e o CatBoost, o
    XGBoost e os detectores de anomalia passariam a ver ``card1`` como
    quantidade. A leitura devolve essas colunas como texto (item G14).
    """
    identity_column = "dataset_id" if "dataset_id" in manifest_rows.columns else (
        "database_type"
    )
    columns: list[str] = []
    for dataset in manifest_rows[identity_column].dropna().unique():
        definition = get_dataset_definition(dataset)
        for column in definition.categorical_code_columns:
            if column not in columns:
                columns.append(column)
    return columns


def load_prepared_data_splits(
    input_directory: str | Path,
    manifest_path: str | Path,
    configuration_md5: str,
    test_strategy: str,
) -> dict:
    """Load a configuration's CSVs from manifest rows and reconstruct its folds.

    Cada dobra devolvida carrega ``fold_index`` (1-based) e ``split_md5``, um
    dicionario com o MD5 do conteudo de cada arquivo efetivamente lido. Esses
    MD5 sao gravados na linha de resultado, de modo que cada metrica aponte
    para os bytes exatos de treino, validacao e teste que a produziram.
    """
    source = Path(input_directory)
    manifest = normalize_manifest_dataset_identity(
        pd.read_csv(manifest_path, sep="\t", dtype=str)
    )
    required_columns = {
        "configuration_md5",
        "dataset_md5",
        "fold_index",
        "split",
    }
    missing_columns = sorted(required_columns.difference(manifest.columns))
    if missing_columns:
        raise ValueError(
            f"Prepared dataset manifest is missing columns: {', '.join(missing_columns)}"
        )
    if "categorical_handling_mode" not in manifest.columns:
        manifest["categorical_handling_mode"] = CATEGORICAL_HANDLING_MODE_ENCODED
    matching_rows = manifest[
        manifest["configuration_md5"].str.lower() == configuration_md5.lower()
    ]
    if matching_rows.empty:
        raise FileNotFoundError(
            f"No manifest rows matched configuration MD5 '{configuration_md5}'"
        )
    categorical_handling_modes = {
        normalize_categorical_handling_mode(value)
        for value in matching_rows["categorical_handling_mode"].dropna().unique()
    }
    if not categorical_handling_modes:
        categorical_handling_modes = {CATEGORICAL_HANDLING_MODE_ENCODED}
    if len(categorical_handling_modes) > 1:
        raise ValueError(
            "Prepared dataset manifest has multiple categorical handling modes "
            f"for configuration MD5 '{configuration_md5}'"
        )

    # So o modo nativo preserva as categorias como texto; nos demais o codigo
    # ja foi codificado (ou indexado) numericamente na preparacao.
    text_dtypes = (
        {
            column: str
            for column in declared_code_columns(matching_rows)
        }
        if next(iter(categorical_handling_modes)) == CATEGORICAL_HANDLING_MODE_NATIVE
        else None
    )

    folds: dict[int, dict] = {}
    reverse_split_labels = {
        split_label: split_name
        for split_name, split_label in SPLIT_FILE_LABELS.items()
    }
    for row in matching_rows.itertuples(index=False):
        dataset_md5 = row.dataset_md5.lower()
        if len(dataset_md5) != 32 or any(
            character not in "0123456789abcdef" for character in dataset_md5
        ):
            raise ValueError(f"Invalid dataset_md5 in manifest: {row.dataset_md5}")
        fold_index = int(row.fold_index)
        if fold_index < FIRST_FOLD_INDEX:
            raise ValueError(
                f"fold_index must start at {FIRST_FOLD_INDEX}; found {fold_index}"
            )
        split_label = row.split
        if split_label not in reverse_split_labels:
            raise ValueError(f"Unknown prepared dataset split '{split_label}'")
        path = prepared_split_path(source, dataset_md5)
        if path is None:
            raise FileNotFoundError(
                f"Prepared dataset file was not found: {source / f'{dataset_md5}.csv'}"
            )
        fold = folds.setdefault(
            fold_index,
            {
                **{split_name: None for split_name in SPLIT_NAMES},
                "fold_index": fold_index,
                "split_md5": {},
                "partition_md5": {},
                "unit_partition_md5": {},
                "train_population_md5": "",
                "train_unit_population_md5": "",
            },
        )
        split_name = reverse_split_labels[split_label]
        if fold[split_name] is not None:
            raise ValueError(
                f"Duplicate manifest row for fold {fold_index}, split {split_label}"
            )
        content_md5 = compute_split_content_md5(path)
        recorded_content_md5 = str(getattr(row, "content_md5", "") or "").strip().lower()
        if recorded_content_md5 and recorded_content_md5 != content_md5:
            # O arquivo mudou depois de preparado. Seguir adiante gravaria uma
            # metrica atribuida a dados que nao sao os daquela configuracao.
            raise ValueError(
                f"Prepared file {path.name} does not match the manifest: "
                f"expected content MD5 {recorded_content_md5}, found {content_md5}"
            )
        fold[split_name] = read_prepared_split_csv(path, dtype=text_dtypes)
        fold["split_md5"][split_name] = content_md5
        recorded_schema_md5 = str(getattr(row, "feature_schema_md5", "") or "").strip().lower()
        if recorded_schema_md5 and recorded_schema_md5 != "nan":
            observed_schema_md5 = compute_feature_schema_md5(fold[split_name].columns)
            if observed_schema_md5 != recorded_schema_md5:
                # As colunas do arquivo nao sao as registradas na preparacao.
                raise ValueError(
                    f"Prepared file {path.name} columns do not match the manifest: "
                    f"expected feature schema MD5 {recorded_schema_md5}, "
                    f"found {observed_schema_md5}"
                )
        units_path = evaluation_unit_path(source, dataset_md5)
        if units_path.is_file():
            units = pd.read_csv(units_path)[EVALUATION_UNIT_COLUMN]
            if len(units) != len(fold[split_name]):
                raise ValueError(
                    f"{units_path.name} does not align with {path.name}: "
                    f"{len(units)} units for {len(fold[split_name])} rows"
                )
            fold.setdefault("evaluation_units", {})[split_name] = units
        strata_path = history_strata_path(source, dataset_md5)
        if strata_path.is_file():
            strata = pd.read_csv(strata_path)[HISTORY_STRATA_COLUMN]
            if len(strata) != len(fold[split_name]):
                raise ValueError(
                    f"{strata_path.name} does not align with {path.name}: "
                    f"{len(strata)} strata for {len(fold[split_name])} rows"
                )
            fold.setdefault("history_strata", {})[split_name] = strata
        recorded_partition_md5 = str(getattr(row, "partition_md5", "") or "").strip().lower()
        if recorded_partition_md5 and recorded_partition_md5 != "nan":
            fold["partition_md5"][split_name] = recorded_partition_md5
        recorded_unit_md5 = str(getattr(row, "unit_partition_md5", "") or "").strip().lower()
        if recorded_unit_md5 and recorded_unit_md5 != "nan":
            fold["unit_partition_md5"][split_name] = recorded_unit_md5
        recorded_population = str(getattr(row, "train_population_md5", "") or "").strip().lower()
        if recorded_population and recorded_population != "nan":
            fold["train_population_md5"] = recorded_population
        recorded_unit_population = str(
            getattr(row, "train_unit_population_md5", "") or ""
        ).strip().lower()
        if recorded_unit_population and recorded_unit_population != "nan":
            fold["train_unit_population_md5"] = recorded_unit_population

    return {
        "train_test_strategy": str(test_strategy).split(":", maxsplit=1)[0],
        "categorical_handling_mode": next(iter(categorical_handling_modes)),
        "test_info": [folds[index] for index in sorted(folds)],
    }
