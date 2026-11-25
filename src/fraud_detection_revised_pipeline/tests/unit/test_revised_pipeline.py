"""Testes das garantias que o relatorio de revisao exigiu do pipeline.

Cada teste nomeia o item da revisao que ele protege, para que uma mudanca
futura que o quebre deixe claro qual conclusao da dissertacao fica em risco.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import data_partitioning  # noqa: E402
import decision_threshold  # noqa: E402
import experiment_datasets  # noqa: E402
import experiment_grid  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
from experiment_datasets import (  # noqa: E402
    STRATIFIED_HOLDOUT,
    STRATIFIED_K_FOLD,
    TEMPORAL_HOLDOUT,
    DatasetDefinition,
)
from prepared_dataset_storage import (  # noqa: E402
    FIRST_FOLD_INDEX,
    build_prepared_dataset_md5,
    save_prepared_data_splits,
)


# ---------------------------------------------------------------------------
# Catalogo de bases
# ---------------------------------------------------------------------------


def test_eqm_ref_21_and_duplicated_views_are_gone():
    """G9a: as republicacoes da ULB davam peso dobrado a ela nas medias."""
    for banned in ("eqm-ref-21", "eqm-ref-21-pca", "eqm-ref-18-pca"):
        assert banned not in experiment_datasets.DATASET_DEFINITIONS
        with pytest.raises(ValueError, match="desconhecida"):
            experiment_datasets.get_dataset_definition(banned)


def test_pca_runs_everywhere_except_the_ulb():
    """C5.31: V1-V28 da ULB ja sao componentes principais publicados."""
    definitions = experiment_datasets.DATASET_DEFINITIONS
    assert definitions["ulb"].applies_pca_in_classic_block is False
    for dataset_id, definition in definitions.items():
        if dataset_id != "ulb":
            assert definition.applies_pca_in_classic_block is True


def test_identifier_columns_leave_the_attributes():
    """G14b: trans_num e unico por transacao e induzia memorizacao."""
    sparkov = experiment_datasets.get_dataset_definition("eqm-ref-20")
    dropped = experiment_datasets.columns_to_drop_after_partitioning(sparkov)
    for identifier in ("cc_num", "trans_num", "first", "last", "street", "dob"):
        assert identifier in dropped
    # A marca de tempo absoluta so serve para ordenar e particionar.
    assert "trans_date_trans_time" in dropped

    ieee = experiment_datasets.get_dataset_definition("eqm-ref-22")
    ieee_dropped = experiment_datasets.columns_to_drop_after_partitioning(ieee)
    assert "TransactionID" in ieee_dropped
    assert "TransactionDT" in ieee_dropped


def test_exact_duplicate_records_are_removed_before_partitioning():
    """Registros repetidos na origem viram copias entre treino e teste."""
    frame = pd.DataFrame(
        {"feature": [1, 1, 2, 3], "Class": [0, 0, 1, 0]}
    )
    deduplicated = experiment_datasets.drop_exact_duplicate_records(frame, "teste")
    assert len(deduplicated) == 3


# ---------------------------------------------------------------------------
# Identidade da configuracao
# ---------------------------------------------------------------------------


def test_configuration_identity_has_no_test_data_size_and_has_the_seed():
    """G9c e G13: a reserva saiu da identidade; a semente entrou."""
    base_arguments = dict(
        dataset="eqm-ref-18",
        categorical_encoding=None,
        fill_na_strategy="mean",
        test_strategy=STRATIFIED_HOLDOUT,
        prediction_column="Class",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        categorical_handling_mode="encoded",
    )
    first = build_prepared_dataset_md5(**base_arguments, random_seed=42)
    second = build_prepared_dataset_md5(**base_arguments, random_seed=7)
    assert first != second, "sementes diferentes devem ser configuracoes diferentes"

    repeated = build_prepared_dataset_md5(**base_arguments, random_seed=42)
    assert first == repeated, "a identidade deve ser deterministica"


def test_protocol_changes_the_configuration_identity():
    arguments = dict(
        dataset="eqm-ref-18",
        categorical_encoding=None,
        fill_na_strategy="mean",
        prediction_column="Class",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        categorical_handling_mode="encoded",
        random_seed=42,
    )
    identities = {
        build_prepared_dataset_md5(**arguments, test_strategy=protocol)
        for protocol in (STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD, TEMPORAL_HOLDOUT)
    }
    assert len(identities) == 3


# ---------------------------------------------------------------------------
# Particionamento
# ---------------------------------------------------------------------------


def _synthetic_frame(row_count: int = 400, group_count: int = 40) -> pd.DataFrame:
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "client_id": np.repeat(np.arange(group_count), row_count // group_count),
            "event_time": np.arange(row_count),
            "feature": generator.normal(size=row_count),
            "label": generator.integers(0, 2, size=row_count),
        }
    )


def _synthetic_definition(**overrides) -> DatasetDefinition:
    arguments = dict(
        dataset_id="sintetica",
        dataset_name="Sintética",
        legacy_aliases=(),
        source_files="memoria",
        prediction_column="label",
        identifier_columns_to_drop=(),
        temporal_order_column="event_time",
        group_column="client_id",
        protocols=(STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD, TEMPORAL_HOLDOUT),
    )
    arguments.update(overrides)
    return DatasetDefinition(**arguments)


@pytest.mark.parametrize(
    "protocol", [STRATIFIED_HOLDOUT, STRATIFIED_K_FOLD, TEMPORAL_HOLDOUT]
)
def test_group_never_appears_in_two_partitions(protocol):
    """G7 e G8: um cliente nos dois lados permite memorizar o rotulo."""
    frame = _synthetic_frame()
    definition = _synthetic_definition()
    # A coluna de agrupamento e removida das dobras; para auditar e preciso
    # olhar a particao antes do descarte.
    original_drop = data_partitioning.columns_to_drop_after_partitioning
    data_partitioning.columns_to_drop_after_partitioning = lambda _definition: []
    try:
        partitions = data_partitioning.build_partitions(
            frame, None, definition, protocol, random_seed=42
        )
    finally:
        data_partitioning.columns_to_drop_after_partitioning = original_drop

    for fold in partitions["test_info"]:
        group_sets = {
            name: set(frame_part["client_id"])
            for name, frame_part in fold.items()
            if name in data_partitioning.SPLIT_NAMES
            and frame_part is not None
            and not frame_part.empty
        }
        names = sorted(group_sets)
        for first_index, first in enumerate(names):
            for second in names[first_index + 1 :]:
                assert not (group_sets[first] & group_sets[second]), (
                    f"{protocol}: grupos compartilhados entre {first} e {second}"
                )


def test_temporal_holdout_keeps_training_in_the_past():
    """G8: sortear treino e teste no mesmo periodo superestima o desempenho."""
    frame = _synthetic_frame()
    definition = _synthetic_definition(group_column=None)
    original_drop = data_partitioning.columns_to_drop_after_partitioning
    data_partitioning.columns_to_drop_after_partitioning = lambda _definition: []
    try:
        partitions = data_partitioning.build_partitions(
            frame, None, definition, TEMPORAL_HOLDOUT, random_seed=42
        )
    finally:
        data_partitioning.columns_to_drop_after_partitioning = original_drop

    fold = partitions["test_info"][0]
    assert fold["train_data"]["event_time"].max() < fold["validation_data"]["event_time"].min()
    assert fold["validation_data"]["event_time"].max() < fold["test_data"]["event_time"].min()


def test_identifier_and_grouping_columns_do_not_reach_the_model():
    frame = _synthetic_frame()
    definition = _synthetic_definition(identifier_columns_to_drop=("feature",))
    partitions = data_partitioning.build_partitions(
        frame, None, definition, STRATIFIED_HOLDOUT, random_seed=42
    )
    for fold in partitions["test_info"]:
        for split_name in data_partitioning.SPLIT_NAMES:
            frame_part = fold[split_name]
            if frame_part is None:
                continue
            assert "client_id" not in frame_part.columns
            assert "event_time" not in frame_part.columns
            assert "feature" not in frame_part.columns
            assert data_partitioning.AUXILIARY_SOURCE_COLUMN not in frame_part.columns


def test_unsupported_protocol_is_rejected_per_dataset():
    """A Home Credit nao tem ordem temporal de transacao."""
    home_credit = experiment_datasets.get_dataset_definition("eqm-ref-19")
    assert TEMPORAL_HOLDOUT not in home_credit.protocols
    with pytest.raises(ValueError, match="nao suporta o protocolo"):
        data_partitioning.build_partitions(
            pd.DataFrame({"TARGET": [0, 1]}),
            None,
            home_credit,
            TEMPORAL_HOLDOUT,
            random_seed=42,
        )


# ---------------------------------------------------------------------------
# Transformacoes ajustadas somente no treino
# ---------------------------------------------------------------------------


def test_standardization_is_fitted_on_training_only():
    """G14a: KNN, SMOTE e Naive Bayes dependem da escala dos atributos."""
    split_information = {
        "test_info": [
            {
                "train_data": pd.DataFrame(
                    {"value": [0.0, 10.0, 20.0, 30.0], "label": [0, 0, 1, 1]}
                ),
                "validation_data": None,
                # O teste tem outra escala de proposito: se a padronizacao
                # fosse ajustada nele, sua media iria a zero.
                "test_data": pd.DataFrame(
                    {"value": [100.0, 200.0], "label": [0, 1]}
                ),
            }
        ]
    }
    standardized = pipeline.standardize_data_splits(split_information, "label")
    fold = standardized["test_info"][0]

    assert fold["train_data"]["value"].mean() == pytest.approx(0.0, abs=1e-9)
    assert fold["test_data"]["value"].mean() > 1.0, (
        "a media do teste so seria zero se a padronizacao tivesse sido "
        "ajustada nele, o que seria vazamento"
    )
    assert fold["train_data"]["label"].tolist() == [0, 0, 1, 1]


def test_dropping_incomplete_rows_touches_training_only():
    """G16: remover linhas do teste mudava a populacao avaliada."""
    split_information = {
        "test_info": [
            {
                # A linha incompleta e legitima: removê-la preserva todas as
                # fraudes, entao a triagem (G15c) mantem a coluna.
                "train_data": pd.DataFrame(
                    {"value": [1.0, np.nan, 3.0, 4.0], "label": [0, 0, 1, 1]}
                ),
                "validation_data": pd.DataFrame(
                    {"value": [np.nan, 2.0], "label": [0, 1]}
                ),
                "test_data": pd.DataFrame(
                    {"value": [np.nan, 5.0, 6.0], "label": [1, 0, 1]}
                ),
            }
        ]
    }
    imputed = pipeline._impute_data_splits(
        split_information,
        pipeline.DROP_INCOMPLETE_ROWS_STRATEGY,
        "label",
    )
    fold = imputed["test_info"][0]

    assert len(fold["train_data"]) == 3, "o treino perde a linha incompleta"
    assert len(fold["validation_data"]) == 2, "a validacao mantem a populacao"
    assert len(fold["test_data"]) == 3, "o teste mantem a populacao"
    assert not fold["test_data"]["value"].isna().any()


# ---------------------------------------------------------------------------
# Armazenamento e resultados
# ---------------------------------------------------------------------------


def test_fold_index_starts_at_one(tmp_path):
    """Pedido explicito: sem dobras, o indice e 1, nunca vazio."""
    split_information = {
        "test_info": [
            {
                "train_data": pd.DataFrame({"value": [1], "label": [0]}),
                "validation_data": None,
                "test_data": pd.DataFrame({"value": [2], "label": [1]}),
            }
        ]
    }
    created = save_prepared_data_splits(split_information, tmp_path, "a" * 32)
    assert {file.fold_index for file in created} == {FIRST_FOLD_INDEX}
    assert FIRST_FOLD_INDEX == 1


def test_each_saved_file_carries_its_content_md5(tmp_path):
    """Pedido explicito: gravar o MD5 de treino, validacao e teste."""
    split_information = {
        "test_info": [
            {
                "train_data": pd.DataFrame({"value": [1, 2], "label": [0, 1]}),
                "validation_data": pd.DataFrame({"value": [3], "label": [0]}),
                "test_data": pd.DataFrame({"value": [4], "label": [1]}),
            }
        ]
    }
    created = save_prepared_data_splits(split_information, tmp_path, "b" * 32)
    assert {file.split_label for file in created} == {"train", "validation", "test"}
    for file in created:
        assert len(file.content_md5) == 32
        # O MD5 e do conteudo, nao da identidade: arquivos com conteudos
        # diferentes nao podem colidir.
        assert file.content_md5 != file.dataset_md5


def test_result_schema_records_seed_threshold_and_split_md5():
    columns = pipeline.RESULT_CSV_COLUMNS
    assert "test_data_size" not in columns
    for required in (
        "random_seed",
        "classification_threshold",
        "train_dataset_md5",
        "validation_dataset_md5",
        "test_dataset_md5",
        "fold_index",
        "split",
    ):
        assert required in columns


def test_threshold_is_part_of_the_already_recorded_key(tmp_path):
    """Sem isso, a execucao em 0,7 seria pulada pela de 0,5."""
    results = pd.DataFrame(
        [
            {
                "configuration_md5": "c" * 32,
                "algorithm": "logistic-regression",
                "execution_status": "success",
                "classification_threshold": 0.5,
            }
        ]
    )
    results.to_csv(tmp_path / "resultados_logistic-regression.csv", index=False)

    assert pipeline.is_experiment_already_recorded(
        tmp_path, "c" * 32, "logistic-regression", 0.5
    )
    assert not pipeline.is_experiment_already_recorded(
        tmp_path, "c" * 32, "logistic-regression", 0.7
    )


def test_historical_resultas_filename_is_still_discovered(tmp_path):
    pd.DataFrame(
        [
            {
                "configuration_md5": "f" * 32,
                "algorithm": "logistic-regression",
                "execution_status": "success",
            }
        ]
    ).to_csv(tmp_path / "resultas_logistic-regression.csv", index=False)

    assert pipeline.is_experiment_already_recorded(
        tmp_path, "f" * 32, "logistic-regression"
    )


# ---------------------------------------------------------------------------
# Grade experimental
# ---------------------------------------------------------------------------


def test_grid_does_not_repeat_identical_configurations():
    """G9b: na ULB, 2 codificacoes x 4 imputacoes geravam 8 copias iguais."""
    entries = experiment_grid.build_classic_grid()
    ulb_entries = [
        entry for entry in entries if entry.dataset_id == "ulb"
    ]
    assert {entry.categorical_encoding_strategy for entry in ulb_entries} == {"none"}
    assert {entry.fill_na_strategy for entry in ulb_entries} == {"none"}

    ieee_entries = [
        entry for entry in entries if entry.dataset_id == "ieee-cis-completa"
    ]
    assert {entry.categorical_encoding_strategy for entry in ieee_entries} == {
        "smoothed-mean:k-fold",
        "smoothed-mean:permutation-prefix",
        "log-odds:k-fold",
        "log-odds:permutation-prefix",
    }

    # Nenhuma configuracao da grade pode se repetir.
    signatures = [entry.to_tsv() for entry in entries]
    assert len(signatures) == len(set(signatures))


def test_grid_includes_a_no_resampling_baseline():
    """G10a: sem esse braco nao se pode medir o efeito de balancear."""
    for builder in (
        experiment_grid.build_classic_grid,
        experiment_grid.build_boosting_grid,
    ):
        techniques = {entry.undersampling_technique for entry in builder()}
        assert "none" in techniques
        assert "random-undersampling" in techniques, (
            "as duas familias precisam de uma estrategia comum (G10f)"
        )


def test_transformer_grid_drops_the_linear_head():
    runs = experiment_grid.algorithm_runs_for(
        experiment_grid.BLOCK_FT_TRANSFORMER,
        experiment_grid.build_ft_transformer_grid(),
    )
    algorithms = {algorithm for algorithm, _threshold, _count in runs}
    assert algorithms == {"ft-transformer-mlp", "ft-transformer-lstm"}


def test_tabbert_grid_only_covers_datasets_with_a_holder():
    """A ULB nao tem portador; nao ha sequencia a modelar nela."""
    entries = experiment_grid.build_tabbert_grid()
    database_types = {entry.database_type for entry in entries}
    assert "ulb" not in database_types
    assert database_types == set(
        experiment_datasets.datasets_supporting_sequence_models()
    )
    # A base sintetica da grade anterior nao volta: ela acrescentava uma linha
    # inventada por cliente.
    assert "home-credit-adjusted" not in database_types


def test_tabbert_grid_never_resamples():
    """RUS e SMOTE quebrariam as janelas cronologicas por titular."""
    for entry in experiment_grid.build_tabbert_grid():
        assert entry.undersampling_technique == "none"
        assert entry.categorical_handling_mode == "sequence-tokenized"


def test_tabbert_uses_a_protocol_shared_with_the_other_blocks():
    """G16: o TabBERT avalia as mesmas particoes dos outros blocos."""
    shared = {
        (entry.database_type, entry.test_strategy, entry.random_seed)
        for builder in (
            experiment_grid.build_classic_grid,
            experiment_grid.build_boosting_grid,
            experiment_grid.build_ft_transformer_grid,
        )
        for entry in builder()
    }
    for entry in experiment_grid.build_tabbert_grid():
        assert (entry.database_type, entry.test_strategy, entry.random_seed) in shared
        definition = experiment_datasets.get_dataset_definition(entry.database_type)
        expected = (
            TEMPORAL_HOLDOUT
            if TEMPORAL_HOLDOUT in definition.protocols
            else STRATIFIED_HOLDOUT
        )
        assert entry.test_strategy == expected


def test_tabbert_runs_both_regimes_for_each_head():
    runs = experiment_grid.algorithm_runs_for(
        experiment_grid.BLOCK_TABBERT, experiment_grid.build_tabbert_grid()
    )
    algorithms = {algorithm for algorithm, _threshold, _count in runs}
    assert algorithms == {
        "tabbert-mlp [supervised-only]",
        "tabbert-mlp [mlm-before-finetune]",
        "tabbert-mlp [warmup-mlm-finetune]",
        "tabbert-lstm [supervised-only]",
        "tabbert-lstm [mlm-before-finetune]",
        "tabbert-lstm [warmup-mlm-finetune]",
    }
    # A cabeca linear saiu tambem do TabBERT.
    assert not any("linear" in algorithm for algorithm in algorithms)


def test_ieee_sequence_key_is_the_approximate_holder_not_card1():
    """G8-iii: card1 e informacao do cartao, nao do portador."""
    import transformer_sequence_windows

    for database_type in (
        "ieee-cis-completa",
        "ieee-cis-sem-identidade",
        "ieee-cis-transacional",
    ):
        holder_column, order_column = transformer_sequence_windows.SEQUENCE_KEYS[
            database_type
        ]
        assert holder_column == "approximate_holder_id"
        assert order_column == "TransactionDT"
    assert experiment_datasets.APPROXIMATE_HOLDER_COLUMNS == ("card1", "addr1")


def test_approximate_holder_id_combines_card_and_address():
    frame = pd.DataFrame(
        {
            "card1": [1000, 1000, 2000, 1000],
            "addr1": [204.0, 299.0, 204.0, 204.0],
            "TransactionDT": [86400, 86400, 86400, 172800],
        }
    )
    identifiers = experiment_datasets._build_approximate_holder_id(frame)
    # Mesmo card1 com addr1 diferente sao titulares diferentes; mesmo par em
    # momentos diferentes e o mesmo titular.
    assert identifiers.iloc[0] != identifiers.iloc[1]
    assert identifiers.iloc[0] != identifiers.iloc[2]
    assert identifiers.iloc[0] == identifiers.iloc[3]


def test_only_weightable_algorithms_sweep_class_weight_intensities():
    """G10 e C3.14: a ponderacao entra so onde o algoritmo a aceita."""
    runs = experiment_grid.algorithm_runs_for(
        experiment_grid.BLOCK_CLASSIC, experiment_grid.build_classic_grid()
    )
    labels = {algorithm for algorithm, _threshold, _count in runs}
    assert labels == {
        "logistic-regression [none]",
        "logistic-regression [sqrt-balanced]",
        "logistic-regression [balanced]",
        "naive-bayes",
        "knn",
    }
    # A regressao linear saiu: sua saida nao e probabilidade calibrada.
    assert not any("linear-regression" in label for label in labels)


def test_boosting_sweeps_the_same_three_intensities():
    """G10f: as duas familias precisam da mesma correcao relativa."""
    runs = experiment_grid.algorithm_runs_for(
        experiment_grid.BLOCK_BOOSTING, experiment_grid.build_boosting_grid()
    )
    labels = {algorithm for algorithm, _threshold, _count in runs}
    for algorithm in ("catboost", "xgboost"):
        for intensity in experiment_grid.CLASS_WEIGHT_INTENSITIES:
            assert f"{algorithm} [{intensity}]" in labels


def test_class_imbalance_ratio_uses_training_only():
    """R deve vir do treino da dobra, nunca do teste ou da base inteira."""
    train_target = pd.Series([0] * 27 + [1])
    ratio = pipeline.compute_class_imbalance_ratio(train_target, [0, 1])
    assert ratio == pytest.approx(27.0)
    assert pipeline.compute_class_imbalance_ratio(pd.Series([0, 0]), [0, 1]) is None


@pytest.mark.parametrize(
    "intensity,expected",
    [
        ("none", None),
        ("balanced", "balanced"),
    ],
)
def test_logistic_class_weight_control_and_full(intensity, expected):
    assert (
        pipeline.resolve_logistic_class_weight(intensity, 27.58, [0, 1]) == expected
    )


def test_logistic_class_weight_moderate_uses_square_root():
    weights = pipeline.resolve_logistic_class_weight("sqrt-balanced", 27.58, [0, 1])
    assert weights[0] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(27.58**0.5)
    assert weights[1] == pytest.approx(5.251, abs=1e-3)


def test_class_weight_intensity_is_part_of_the_already_recorded_key(tmp_path):
    """Sem isso, a execucao em sqrt-balanced seria pulada pela de none."""
    results = pd.DataFrame(
        [
            {
                "configuration_md5": "d" * 32,
                "algorithm": "logistic-regression",
                "execution_status": "success",
                "class_weight_intensity": "none",
            }
        ]
    )
    results.to_csv(tmp_path / "resultados_logistic-regression.csv", index=False)

    assert pipeline.is_experiment_already_recorded(
        tmp_path, "d" * 32, "logistic-regression", class_weight_intensity="none"
    )
    assert not pipeline.is_experiment_already_recorded(
        tmp_path, "d" * 32, "logistic-regression", class_weight_intensity="balanced"
    )


def test_auc_pr_is_recorded_alongside_roc_auc():
    """G11-iii: sob desbalanceamento a curva PR e mais informativa que a ROC."""
    assert "auc_pr" in pipeline.RESULT_CSV_COLUMNS
    roc_position = pipeline.RESULT_CSV_COLUMNS.index("roc_auc")
    assert pipeline.RESULT_CSV_COLUMNS.index("auc_pr") == roc_position + 1


def test_average_precision_matches_the_prevalence_baseline():
    """A linha de base da AUC-PR e a prevalencia, nao 0,5 como na ROC."""
    # Pontuacoes aleatorias sobre 1% de positivos: a precisao media deve
    # ficar na ordem da prevalencia, nao em 0,5.
    generator = np.random.default_rng(0)
    targets = np.zeros(2000, dtype=int)
    targets[:20] = 1
    generator.shuffle(targets)
    scores = generator.random(2000)
    average_precision = pipeline._average_precision(targets, scores, [0, 1])
    assert average_precision == pytest.approx(0.01, abs=0.02)

    # Pontuacao perfeita leva a AUC-PR a 1.
    assert pipeline._average_precision(targets, targets.astype(float), [0, 1]) == (
        pytest.approx(1.0)
    )


def test_average_precision_is_undefined_without_positives():
    targets = np.zeros(10, dtype=int)
    scores = np.linspace(0, 1, 10)
    assert np.isnan(pipeline._average_precision(targets, scores, [0, 1]))


def test_metrics_dictionary_carries_both_areas():
    targets = np.array([0, 0, 1, 1])
    predictions = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.6, 0.4, 0.9])
    computed = pipeline._linear_regression_metrics(
        targets, predictions, [0, 1], prediction_scores=scores
    )
    for metric in ("roc_auc", "auc_pr", "false_positive_rate", "accuracy"):
        assert metric in computed
    assert 0.0 <= computed["auc_pr"] <= 1.0


def test_result_schema_records_the_class_weighting():
    for required in ("class_weight_intensity", "class_weight_ratio"):
        assert required in pipeline.RESULT_CSV_COLUMNS


def test_application_only_home_credit_has_one_row_per_client():
    """Nova base: isola o efeito da juncao com previous_application."""
    definition = experiment_datasets.get_dataset_definition("eqm-ref-19-application")
    assert definition.source_files == "ref-19/application_data.csv"
    # Uma linha por cliente, agrupada mesmo assim para que o corte use a
    # mesma tabela de clientes da completa (item G16).
    assert definition.group_column == "SK_ID_CURR"
    assert "SK_ID_CURR" in experiment_datasets.columns_to_drop_after_partitioning(
        definition
    )
    # Sem sequencia: o cliente nao tem historico nesta visao.
    assert definition.sequence_keys is None


def test_sparkov_keeps_the_transaction_location():
    """A localizacao de uma transacao e atributo legitimo para fraude."""
    definition = experiment_datasets.get_dataset_definition("eqm-ref-20")
    dropped = experiment_datasets.columns_to_drop_after_partitioning(definition)
    for coordinate in ("lat", "long", "merch_lat", "merch_long"):
        assert coordinate not in dropped


def test_pca_is_declared_only_for_the_classic_block():
    """Boosting e Transformers nunca recebem PCA."""
    source = (PIPELINE_DIRECTORY / "prepare_lr_nb_knn_datasets.py").read_text(
        encoding="utf-8"
    )
    # A unica chamada de PCA vive no ramo classico, depois do else que separa
    # os modos nativo e de Transformer.
    classic_branch = source.split("elif uses_native_categorical_handling:", 1)[1]
    separator = "\n    else:\n"
    native_branch, encoded_branch = classic_branch.split(separator, 1)
    assert "_apply_pca_to_data_splits" not in native_branch
    assert "_apply_pca_to_data_splits" in encoded_branch
    assert "applies_pca_in_classic_block" in encoded_branch


# ---------------------------------------------------------------------------
# Codificacao supervisionada: desenho fatorial 2x2 (item C3.6)
# ---------------------------------------------------------------------------


def _categorical_training_frame(rows=4000, categories=8, seed=7):
    """Coluna categorica com associacao conhecida ao alvo, de 2% a 30%."""
    generator = np.random.default_rng(seed)
    rates = np.linspace(0.02, 0.30, categories)
    positions = generator.choice(categories, size=rows)
    target = (generator.random(rows) < rates[positions]).astype(int)
    return pd.DataFrame(
        {
            "cat": [f"cat_{position}" for position in positions],
            "alvo": target,
        }
    )


def _encode_training_column(frame, strategy, column="cat", target_column="alvo"):
    information = {
        "test_info": [
            {
                "train_data": frame,
                "validation_data": None,
                "test_data": frame.copy(),
            }
        ]
    }
    encoded = pipeline._encode_supervised_data_splits(
        information, strategy, target_column
    )
    return encoded["test_info"][0]["train_data"][column]


def test_encoding_axis_is_a_complete_two_by_two_factorial():
    """C3.6: o eixo antigo movia estatistica e mecanismo ao mesmo tempo.

    Com as quatro celulas presentes, variar uma coordenada por vez isola o
    efeito de cada fator; sem elas, nenhuma diferenca medida entre
    `ordered-target` e `woe` era atribuivel a um dos dois.
    """
    cells = {
        pipeline.parse_encoding_factors(strategy)
        for strategy in experiment_grid.CLASSIC_ENCODINGS
    }
    expected = {
        (statistic, control)
        for statistic in pipeline.ENCODING_STATISTICS
        for control in pipeline.ENCODING_LEAKAGE_CONTROLS
    }
    assert cells == expected
    assert len(experiment_grid.CLASSIC_ENCODINGS) == 4


def test_legacy_woe_name_still_means_log_odds_with_cross_fitting():
    """O apelido antigo tem de apontar para a celula identica do fatorial.

    `woe` sempre significou log-razao ajustada em cinco dobras, entao resultados
    gravados antes do fatorial continuam reproduziveis sob o novo nome.
    """
    assert pipeline.parse_encoding_factors("woe") == (
        pipeline.ENCODING_STATISTIC_LOG_ODDS,
        pipeline.ENCODING_CONTROL_K_FOLD,
    )


@pytest.mark.parametrize("statistic", pipeline.ENCODING_STATISTICS)
@pytest.mark.parametrize("control", pipeline.ENCODING_LEAKAGE_CONTROLS)
def test_a_singleton_category_is_never_encoded_from_its_own_label(
    statistic, control
):
    """C3.6: o teste direto de vazamento, valido nos dois mecanismos.

    Uma categoria que aparece uma unica vez no treino nao tem nenhuma OUTRA
    linha da qual estimar a estatistica. Se o valor que ela recebe reagir ao
    proprio rotulo, a linha esta vendo a si mesma -- que e exatamente o
    vazamento que os dois mecanismos existem para impedir.
    """
    strategy = f"{statistic}{pipeline.ENCODING_FACTOR_SEPARATOR}{control}"
    frame = _categorical_training_frame()
    singleton_position = len(frame) - 1
    frame.loc[singleton_position, "cat"] = "categoria_unica"

    frame.loc[singleton_position, "alvo"] = 0
    encoded_as_negative = _encode_training_column(frame, strategy)
    frame.loc[singleton_position, "alvo"] = 1
    encoded_as_positive = _encode_training_column(frame, strategy)

    assert encoded_as_negative.iloc[singleton_position] == pytest.approx(
        encoded_as_positive.iloc[singleton_position]
    )


@pytest.mark.parametrize("statistic", pipeline.ENCODING_STATISTICS)
def test_permutation_prefix_hides_a_rows_own_label_from_itself(statistic):
    """C3.6: no mecanismo de prefixo a particao nao depende dos rotulos.

    Por isso aqui se pode trocar o rotulo de UMA linha e exigir que a
    codificacao dela propria nao se mova -- uma verificacao mais forte do que a
    da categoria unica, porque vale para qualquer linha.
    """
    strategy = (
        f"{statistic}{pipeline.ENCODING_FACTOR_SEPARATOR}"
        f"{pipeline.ENCODING_CONTROL_PERMUTATION_PREFIX}"
    )
    frame = _categorical_training_frame()
    # Uma linha do ultimo bloco, que tem o prefixo mais longo e portanto o
    # maior numero de oportunidades de vazar.
    permutation = np.random.default_rng(pipeline.RANDOM_STATE).permutation(len(frame))
    position = int(permutation[-1])

    original_label = int(frame.loc[position, "alvo"])
    before = _encode_training_column(frame, strategy).iloc[position]
    frame.loc[position, "alvo"] = 1 - original_label
    after = _encode_training_column(frame, strategy).iloc[position]

    assert before == pytest.approx(after)


def test_both_statistics_share_the_same_leakage_control_partition():
    """C3.6: o mecanismo tem de ser literalmente o mesmo codigo nos dois lados.

    Sob o prefixo, o primeiro bloco nao tem prefixo algum e recebe o valor a
    priori. Esse conjunto de linhas e determinado pela permutacao, nao pela
    estatistica, entao tem de ser identico nas duas -- caso contrario o
    mecanismo voltaria a variar junto com a estatistica.
    """
    frame = _categorical_training_frame()
    prior_rows = {}
    for statistic in pipeline.ENCODING_STATISTICS:
        strategy = (
            f"{statistic}{pipeline.ENCODING_FACTOR_SEPARATOR}"
            f"{pipeline.ENCODING_CONTROL_PERMUTATION_PREFIX}"
        )
        prior_value = pipeline._encoding_prior_value(strategy, frame["alvo"])
        encoded = _encode_training_column(frame, strategy)
        prior_rows[statistic] = set(
            encoded.index[np.isclose(encoded.to_numpy(dtype=float), prior_value)]
        )

    statistics = list(pipeline.ENCODING_STATISTICS)
    assert prior_rows[statistics[0]] == prior_rows[statistics[1]]
    # E o bloco a priori tem de ser exatamente um decimo das linhas.
    expected_block_size = len(frame) // pipeline.PERMUTATION_PREFIX_BLOCK_COUNT
    assert len(prior_rows[statistics[0]]) == expected_block_size


def test_the_two_mechanisms_inject_different_amounts_of_noise():
    """C3.6: a razao de ser do fatorial, medida.

    O mecanismo nao e um detalhe de implementacao: o prefixo entrega ao modelo
    codificacoes de treino varias vezes mais ruidosas que o cross-fitting. Se
    esse efeito nao estivesse isolado num eixo proprio, ele seria lido como
    diferenca entre as estatisticas.
    """
    frame = _categorical_training_frame()
    noise = {}
    for statistic in pipeline.ENCODING_STATISTICS:
        for control in pipeline.ENCODING_LEAKAGE_CONTROLS:
            strategy = f"{statistic}{pipeline.ENCODING_FACTOR_SEPARATOR}{control}"
            encoded = _encode_training_column(frame, strategy)
            grouped = encoded.groupby(frame["cat"])
            # Ruido intra-categoria dividido pelo sinal entre categorias, para
            # que as escalas diferentes das duas estatisticas nao contaminem.
            noise[(statistic, control)] = grouped.std().mean() / grouped.mean().std()

    for statistic in pipeline.ENCODING_STATISTICS:
        prefix = noise[(statistic, pipeline.ENCODING_CONTROL_PERMUTATION_PREFIX)]
        cross_fitted = noise[(statistic, pipeline.ENCODING_CONTROL_K_FOLD)]
        assert prefix > 3 * cross_fitted, statistic

    # E o efeito do mecanismo tem de ser parecido nas duas estatisticas: e isso
    # que permite ler o fatorial como dois efeitos principais.
    prefix_noise = [
        noise[(statistic, pipeline.ENCODING_CONTROL_PERMUTATION_PREFIX)]
        for statistic in pipeline.ENCODING_STATISTICS
    ]
    assert max(prefix_noise) / min(prefix_noise) < 1.5


@pytest.mark.parametrize("alias", ["woe", "WOE", "weight_of_evidence", "weight-of-evidence"])
def test_encoding_aliases_collapse_into_one_configuration_identity(alias):
    """G9b: duas grafias da mesma celula seriam dados identicos preparados duas vezes.

    O apelido tem de ser resolvido DENTRO do configuration_md5, e nao apenas no
    codificador. Normalizar so na execucao faria `woe` e `log-odds:k-fold`
    virarem duas configuracoes distintas com o mesmo conteudo, inflando a
    contagem de pares exatamente como as bases republicadas inflavam.
    """
    arguments = dict(
        dataset="eqm-ref-22",
        fill_na_strategy="mean",
        test_strategy=STRATIFIED_HOLDOUT,
        prediction_column="isFraud",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        categorical_handling_mode="encoded",
        random_seed=42,
    )
    canonical = build_prepared_dataset_md5(
        **arguments, categorical_encoding="log-odds:k-fold"
    )
    assert build_prepared_dataset_md5(**arguments, categorical_encoding=alias) == canonical


def test_the_four_factorial_cells_are_four_distinct_configurations():
    arguments = dict(
        dataset="eqm-ref-22",
        fill_na_strategy="mean",
        test_strategy=STRATIFIED_HOLDOUT,
        prediction_column="isFraud",
        undersampling_technique=None,
        undersampling_parameters=None,
        auxiliary_data_is_test_data=False,
        categorical_handling_mode="encoded",
        random_seed=42,
    )
    identities = {
        build_prepared_dataset_md5(**arguments, categorical_encoding=strategy)
        for strategy in pipeline.FACTORIAL_CATEGORICAL_ENCODINGS
    }
    assert len(identities) == 4
    # E o braco de referencia nao pode colidir com a celula de mesma estatistica:
    # seu mecanismo e o prefixo por linha, que produz dados diferentes.
    reference = build_prepared_dataset_md5(
        **arguments, categorical_encoding="ordered-target"
    )
    assert reference not in identities


# ---------------------------------------------------------------------------
# Registro por split: selecao na validacao (item G12)
# ---------------------------------------------------------------------------


def _single_fold_execution_result(**split_metrics):
    """Uma dobra com as metricas informadas por split."""
    def metrics(value):
        return {
            name: value
            for name in (
                "f1", "f_beta_2", "accuracy", "precision", "recall",
                "roc_auc", "auc_pr",
            )
        }

    return {
        "algorithm": "logistic-regression",
        "undersampling_technique": None,
        "train_test_strategy": STRATIFIED_HOLDOUT,
        "results": [
            {
                "fold_index": FIRST_FOLD_INDEX,
                "split_sizes": {
                    "train_data": 70,
                    "validation_data": 15,
                    "test_data": 15,
                },
                "split_md5": {
                    "train_data": "md5-treino",
                    "validation_data": "md5-validacao",
                    "test_data": "md5-teste",
                },
                "split_results": {
                    f"{name}_data": {"metrics": metrics(value)}
                    for name, value in split_metrics.items()
                },
            }
        ],
    }


def _record(tmp_path, execution_result):
    destination = tmp_path / "resultados_logistic-regression.csv"
    pipeline.record_algorithm_result_to_csv(
        str(destination),
        "",
        execution_result,
        "none",
        "mean",
        1.0,
        configuration_md5="abc",
        database_type="eqm-ref-18",
    )
    return pd.read_csv(destination)


def test_every_evaluated_split_becomes_its_own_result_row(tmp_path):
    """G12: a melhor configuracao era escolhida na propria particao de teste.

    O gravador anterior escolhia UM split por dobra, com prioridade
    test > validation > train. Como os tres protocolos sempre produzem teste,
    nenhuma linha `split=validation` chegava ao CSV: as metricas de validacao
    eram calculadas e descartadas na escrita, e a fase de analise nao tinha em
    que selecionar sem olhar o teste.
    """
    recorded = _record(
        tmp_path,
        _single_fold_execution_result(train=0.91, validation=0.72, test=0.70),
    )

    assert list(recorded["split"]) == ["train", "validation", "test"]
    assert set(recorded["fold_index"]) == {FIRST_FOLD_INDEX}
    # Cada linha carrega a metrica do SEU split, nao a do teste repetida.
    by_split = dict(zip(recorded["split"], recorded["f1"]))
    assert by_split == pytest.approx(
        {"train": 0.91, "validation": 0.72, "test": 0.70}
    )


def test_validation_metrics_reach_the_csv_even_with_a_test_split(tmp_path):
    """O caso exato que falhava: teste presente escondia a validacao."""
    recorded = _record(
        tmp_path, _single_fold_execution_result(validation=0.72, test=0.70)
    )
    validation_rows = recorded.loc[recorded["split"] == "validation"]
    assert len(validation_rows) == 1
    assert float(validation_rows["f1"].iloc[0]) == pytest.approx(0.72)


def test_every_split_row_carries_the_three_dataset_md5s(tmp_path):
    """Rastreabilidade: a linha identifica os bytes dos tres splits da dobra.

    Os MD5 descrevem a particao, nao o split avaliado, entao as tres linhas
    repetem os mesmos valores -- e e isso que permite reconstruir de quais
    arquivos qualquer uma das linhas veio.
    """
    recorded = _record(
        tmp_path,
        _single_fold_execution_result(train=0.91, validation=0.72, test=0.70),
    )
    for column, expected in (
        ("train_dataset_md5", "md5-treino"),
        ("validation_dataset_md5", "md5-validacao"),
        ("test_dataset_md5", "md5-teste"),
    ):
        assert set(recorded[column]) == {expected}


def test_a_split_without_metrics_is_skipped_not_recorded_empty(tmp_path):
    """Protocolo sem validacao nao pode gerar uma linha de metricas vazias."""
    recorded = _record(tmp_path, _single_fold_execution_result(test=0.70))
    assert list(recorded["split"]) == ["test"]


def test_transformers_evaluate_validation_besides_test():
    """G12 nos Transformers, que antes nem calculavam a validacao.

    O treino fica de fora so nesta familia: nos blocos classico e de boosting
    suas metricas ja vinham calculadas e sao gravadas de graca, mas num
    Transformer exigiriam uma passada de inferencia inteira sobre o maior split
    para produzir uma medida que nao entra em selecao nem em relato.
    """
    import ast

    source = (PIPELINE_DIRECTORY / "run_transformer_prepared_experiments.py").read_text()
    tree = ast.parse(source)
    evaluated = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EVALUATED_SPLITS"
            for target in node.targets
        )
    )
    splits = tuple(element.value for element in evaluated.elts)
    assert splits == ("validation", "test")


# ---------------------------------------------------------------------------
# Ponderacao de classe como braco do balanceamento (itens G10 e C3.14)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "block,builder",
    [
        (experiment_grid.BLOCK_CLASSIC, experiment_grid.build_classic_grid),
        (experiment_grid.BLOCK_BOOSTING, experiment_grid.build_boosting_grid),
    ],
)
def test_weighted_arms_only_cover_configurations_without_resampling(block, builder):
    """Num treino reamostrado para 1:1, R ~ 1 e o peso repetiria o braco none."""
    entries = builder()
    unresampled = [entry for entry in entries if entry.undersampling_technique == "none"]
    assert 0 < len(unresampled) < len(entries)
    for algorithm, run_entries in experiment_grid.run_entries_for(block, entries):
        if algorithm.endswith("[sqrt-balanced]") or algorithm.endswith("[balanced]"):
            assert run_entries == unresampled
        elif algorithm.endswith("[none]"):
            assert run_entries == entries


@pytest.mark.parametrize("technique", ["random-undersampling", "RUS+SMOTE", "tomek-links"])
def test_weighting_is_rejected_on_resampled_configurations(technique):
    with pytest.raises(ValueError, match="balancing arm"):
        pipeline.validate_class_weight_arm("balanced", technique)
    pipeline.validate_class_weight_arm("none", technique)
    pipeline.validate_class_weight_arm("sqrt-balanced", None)
    pipeline.validate_class_weight_arm("balanced", "none")


# ---------------------------------------------------------------------------
# Limiar escolhido na validacao (item G11-i)
# ---------------------------------------------------------------------------


def test_f2_threshold_maximises_f2_on_the_given_scores():
    targets = np.array([0, 0, 0, 0, 1, 0, 1, 1])
    scores = np.array([0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.60, 0.90])
    threshold = decision_threshold.select_f2_threshold(targets, scores)

    def f2(cut):
        predicted = scores >= cut
        true_positive = (predicted & (targets == 1)).sum()
        precision = true_positive / predicted.sum()
        recall = true_positive / targets.sum()
        return 5 * precision * recall / (4 * precision + recall)

    assert f2(threshold) == pytest.approx(max(f2(cut) for cut in scores))
    assert threshold == pytest.approx(0.35)


def test_f2_threshold_is_undefined_without_both_classes():
    assert decision_threshold.select_f2_threshold([0, 0, 0], [0.1, 0.2, 0.3]) is None
    assert decision_threshold.select_f2_threshold([1, 1], [0.1, 0.2]) is None


def test_holdout_threshold_comes_from_its_own_validation():
    validation = (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.9]))
    [fold] = decision_threshold.select_fold_thresholds([validation], [True])
    assert fold.strategy == decision_threshold.THRESHOLD_STRATEGY_VALIDATION_F2
    assert fold.threshold == pytest.approx(0.3)


def test_k_fold_threshold_never_comes_from_the_evaluated_fold():
    """No k-fold a validacao E o conjunto avaliado: o limiar vem das outras dobras."""
    folds = [
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.8])),
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.7, 0.8])),
        (np.array([0, 0, 1, 1]), np.array([0.01, 0.02, 0.03, 0.04])),
    ]
    thresholds = decision_threshold.select_fold_thresholds(folds, [False] * 3)
    assert {fold.strategy for fold in thresholds} == {
        decision_threshold.THRESHOLD_STRATEGY_CROSS_FOLD_F2
    }
    # A terceira dobra tem scores em outra escala; se ela escolhesse o proprio
    # limiar, ele cairia abaixo de 0,05.
    assert thresholds[2].threshold == pytest.approx(0.7)

    changed = list(folds)
    changed[2] = (np.array([0, 0, 1, 1]), np.array([0.5, 0.6, 0.95, 0.99]))
    assert decision_threshold.select_fold_thresholds(changed, [False] * 3)[2] == thresholds[2]


def test_threshold_falls_back_to_fixed_without_both_validation_classes():
    validation = (np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    [fold] = decision_threshold.select_fold_thresholds([validation], [True])
    assert fold == decision_threshold.FoldThreshold(0.5, "fixed")


def _imbalanced_split(rows, seed):
    generator = np.random.default_rng(seed)
    target = (generator.random(rows) < 0.08).astype(int)
    signal = target * 1.5 + generator.normal(size=rows)
    return pd.DataFrame(
        {"x1": signal, "x2": generator.normal(size=rows), "alvo": target}
    )


@pytest.mark.parametrize(
    "with_test,expected_strategy",
    [
        (True, decision_threshold.THRESHOLD_STRATEGY_VALIDATION_F2),
        (False, decision_threshold.THRESHOLD_STRATEGY_CROSS_FOLD_F2),
    ],
)
def test_classic_execution_applies_the_selected_threshold(with_test, expected_strategy):
    fold_count = 1 if with_test else 3
    information = {
        "test_info": [
            {
                "train_data": _imbalanced_split(600, seed),
                "validation_data": _imbalanced_split(300, seed + 100),
                "test_data": _imbalanced_split(300, seed + 200) if with_test else None,
            }
            for seed in range(fold_count)
        ]
    }
    result = pipeline.execute_logistic_regression(information, "alvo")
    for fold in result["results"]:
        assert fold["threshold_strategy"] == expected_strategy
        threshold = fold["classification_threshold"]
        assert threshold != pytest.approx(0.5)
        evaluated = fold["split_results"]["test_data" if with_test else "validation_data"]
        predictions = evaluated["predictions"]
        expected_classes = (
            predictions["logistic_regression_probability"] >= threshold
        ).astype(int)
        assert (predictions["predicted_value"] == expected_classes).all()


def test_fixed_threshold_strategy_keeps_one_half():
    information = {
        "test_info": [
            {
                "train_data": _imbalanced_split(600, 1),
                "validation_data": _imbalanced_split(300, 2),
                "test_data": _imbalanced_split(300, 3),
            }
        ]
    }
    result = pipeline.execute_logistic_regression(
        information, "alvo", threshold_strategy="fixed"
    )
    fold = result["results"][0]
    assert fold["classification_threshold"] == 0.5
    assert fold["threshold_strategy"] == "fixed"


def test_selected_threshold_and_strategy_are_recorded_per_fold(tmp_path):
    execution_result = _single_fold_execution_result(validation=0.6, test=0.5)
    execution_result["results"][0]["classification_threshold"] = 0.137
    execution_result["results"][0]["threshold_strategy"] = "validation-f2"
    recorded = _record(tmp_path, execution_result)
    assert set(recorded["classification_threshold"]) == {0.137}
    assert set(recorded["threshold_strategy"]) == {"validation-f2"}


def test_fixed_threshold_results_do_not_satisfy_a_validation_run(tmp_path):
    """Uma execucao antiga em 0,5 nao pode pular a execucao com limiar escolhido."""
    results = pd.DataFrame(
        [
            {
                "configuration_md5": "e" * 32,
                "algorithm": "knn",
                "execution_status": "success",
                "classification_threshold": 0.5,
            }
        ]
    )
    results.to_csv(tmp_path / "resultados_knn.csv", index=False)
    assert pipeline.is_experiment_already_recorded(
        tmp_path, "e" * 32, "knn", threshold_strategy="fixed"
    )
    assert not pipeline.is_experiment_already_recorded(
        tmp_path, "e" * 32, "knn", threshold_strategy="validation-f2"
    )

    results["threshold_strategy"] = "cross-fold-validation-f2"
    results.to_csv(tmp_path / "resultados_knn.csv", index=False)
    assert pipeline.is_experiment_already_recorded(
        tmp_path, "e" * 32, "knn", threshold_strategy="validation-f2"
    )


# ---------------------------------------------------------------------------
# Particao identificada pelas linhas brutas (item G16)
# ---------------------------------------------------------------------------


def test_every_split_carries_a_partition_md5_and_no_internal_column():
    partitions = data_partitioning.build_partitions(
        _synthetic_frame(), None, _synthetic_definition(), STRATIFIED_HOLDOUT, 42
    )
    fold = partitions["test_info"][0]
    assert set(fold["partition_md5"]) == set(data_partitioning.SPLIT_NAMES)
    for split_name in data_partitioning.SPLIT_NAMES:
        assert data_partitioning.ROW_FINGERPRINT_COLUMN not in fold[split_name].columns


def test_partition_md5_identifies_the_set_of_raw_rows():
    """Nao depende da ordem nem da transformacao, so de quais linhas entraram."""
    frame = _synthetic_frame()
    fingerprints = data_partitioning.row_fingerprints(frame)
    same_rows = data_partitioning.partition_md5(fingerprints[::-1])
    assert data_partitioning.partition_md5(fingerprints) == same_rows

    changed = frame.copy()
    changed.loc[0, "feature"] += 1.0
    assert data_partitioning.partition_md5(
        data_partitioning.row_fingerprints(changed)
    ) != same_rows


def test_same_protocol_and_seed_give_the_same_partition_id():
    """Dois blocos que recomputam a particao chegam a mesma identidade."""
    first, second = (
        data_partitioning.build_partitions(
            _synthetic_frame(), None, _synthetic_definition(), STRATIFIED_K_FOLD, 42
        )
        for _ in range(2)
    )
    for left, right in zip(first["test_info"], second["test_info"]):
        assert data_partitioning.fold_partition_id(
            left["partition_md5"]
        ) == data_partitioning.fold_partition_id(right["partition_md5"])
    other_seed = data_partitioning.build_partitions(
        _synthetic_frame(), None, _synthetic_definition(), STRATIFIED_K_FOLD, 7
    )
    assert other_seed["test_info"][0]["partition_md5"] != first["test_info"][0]["partition_md5"]


def test_partition_id_is_recorded_in_the_result_row(tmp_path):
    execution_result = _single_fold_execution_result(test=0.5)
    execution_result["results"][0]["partition_md5"] = {
        "train_data": "a" * 32, "validation_data": "b" * 32, "test_data": "c" * 32,
    }
    recorded = _record(tmp_path, execution_result)
    assert set(recorded["partition_id"]) == {
        data_partitioning.fold_partition_id(execution_result["results"][0]["partition_md5"])
    }


def test_pairing_identities_are_recorded_in_the_result_row(tmp_path):
    """Item G16: treino efetivo e clientes do split chegam a linha de resultado."""
    execution_result = _single_fold_execution_result(test=0.5)
    fold = execution_result["results"][0]
    fold["train_population_md5"] = "d" * 32
    fold["unit_partition_md5"] = {"validation_data": "e" * 32, "test_data": "f" * 32}
    recorded = _record(tmp_path, execution_result)
    assert set(recorded["train_population_id"]) == {"d" * 32}
    assert set(recorded["unit_partition_id"]) == {
        data_partitioning.fold_partition_id(fold["unit_partition_md5"])
    }


def test_pairing_identities_are_empty_when_unknown(tmp_path):
    recorded = _record(tmp_path, _single_fold_execution_result(test=0.5))
    assert recorded["train_population_id"].fillna("").eq("").all()
    assert recorded["unit_partition_id"].fillna("").eq("").all()


# ---------------------------------------------------------------------------
# TabBERT sobre os protocolos compartilhados (item G16) e semente (G13)
# ---------------------------------------------------------------------------


def _interleaved_holders(row_count=600, holder_count=20):
    """Titulares com transacoes espalhadas no tempo, cruzando o corte temporal."""
    generator = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "holder": np.tile(np.arange(holder_count), row_count // holder_count),
            "event_time": np.arange(row_count),
            "feature": generator.normal(size=row_count),
            "label": (generator.random(row_count) < 0.1).astype(int),
        }
    )


def test_tabbert_windows_on_the_temporal_holdout_are_causal(monkeypatch):
    """No corte temporal compartilhado, o historico de um alvo de teste vem do passado."""
    import transformer_sequence_windows as windows_module

    definition = _synthetic_definition(
        group_column="holder", sequence_keys=("holder", "event_time")
    )
    partitions = data_partitioning.build_partitions(
        _interleaved_holders(), None, definition, TEMPORAL_HOLDOUT, 42,
        retained_columns=definition.sequence_keys,
    )
    fold = windows_module.assign_row_identity(partitions)["test_info"][0]
    for split_name in data_partitioning.SPLIT_NAMES:
        assert {"holder", "event_time"} <= set(fold[split_name].columns)

    monkeypatch.setitem(windows_module.SEQUENCE_KEYS, "sintetica", ("holder", "event_time"))
    index = windows_module.build_sequence_window_index(fold, "sintetica")
    rows = pd.concat(fold[name] for name in data_partitioning.SPLIT_NAMES).set_index("row_id")
    test_windows = index[index["is_target"] & (index["target_split"] == "test")]
    assert (test_windows["base_history_length"] > 0).any()
    rank = {"train": 0, "validation": 1, "test": 2}
    for sequence_index in test_windows["sequence_index"].unique():
        window = index[index["sequence_index"] == sequence_index]
        history = window[~window["is_target"] & (window["row_id"] != "[PAD]")]
        first_target = window[window["is_target"]]["row_id"].iloc[0]
        assert (history["source_split"].map(rank) <= rank["test"]).all()
        assert (
            rows.loc[history["row_id"], "event_time"] < rows.loc[first_target, "event_time"]
        ).all()


def test_tabbert_group_exclusive_split_follows_the_seed():
    """G13: com semente fixa, SEED_LIST repetiria os mesmos clientes."""
    import transformer_sequence_windows as windows_module

    frame = pd.DataFrame(
        {
            "SK_ID_CURR": np.repeat(np.arange(200), 3),
            "DAYS_DECISION": np.tile([-30, -20, -10], 200),
            "TARGET": np.repeat((np.arange(200) % 5 == 0).astype(int), 3),
        }
    )

    def test_clients(seed):
        split = windows_module.split_chronologically_by_identity(
            frame, "eqm-ref-19", classification_column_name="TARGET", random_seed=seed
        )
        return frozenset(split["test_info"][0]["test_data"]["SK_ID_CURR"])

    assert test_clients(42) == test_clients(42)
    assert test_clients(42) != test_clients(7)
