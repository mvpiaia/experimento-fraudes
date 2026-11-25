"""Verificacoes de sanidade das preparacoes e dos resultados (itens G15 e G17).

O erro do G15 passava pelas verificacoes P1-P9: os arquivos eram integros,
sem sobreposicao e distintos entre configuracoes. Faltava perguntar se cada
eixo da grade muda o que deveria mudar. Este modulo faz essas perguntas, sobre
o manifesto, os relatorios de preparacao (``preparation_report.py``) e os CSVs
de resultado, e monta as tabelas do apendice. O ponto de entrada continua
sendo ``validate_prepared_datasets.py``:

    P10 relatorio    -- presente, com o MD5 do manifesto, sem faltantes depois
                        da imputacao no bloco classico e com contadores iguais
                        aos dos CSVs gravados;
    P11 imputacao    -- configuracoes que diferem so na imputacao produzem
                        treinos diferentes, nas bases com faltantes;
    P12 codificacao  -- configuracoes que diferem so na codificacao produzem
                        blocos categoricos diferentes, antes da PCA;
    P13 triagem      -- o limiar do drop-incomplete-rows e o maior admissivel
                        pela regra, e as colunas triadas nao aparecem nos
                        arquivos;
    P14 PCA          -- componentes registrados onde a PCA se aplica e so la;
    P15 identidade   -- o configuration_md5 recalculado dos parametros do
                        manifesto bate com o gravado;
    R1 rastreio      -- cada resultado aponta para uma configuracao do
                        manifesto, com os mesmos parametros;
    R2 eixos         -- pares de resultados que diferem num unico eixo nao
                        tem metricas identicas bit a bit;
    R3 tamanho       -- train_df_size bate com o relatorio de preparacao;
    R4 contagens     -- execucoes previstas pela grade x sucesso, falha,
                        incompleta e ausente, em unidades e em linhas de
                        teste, com o motivo de cada execucao sem sucesso
                        (G17);
    P16 treino       -- configuracoes drop-incomplete-rows da mesma base de
                        origem, protocolo, semente e dobra treinam com as
                        mesmas linhas, e as dos pares por unidade com os
                        mesmos clientes; sem remocao, o treino efetivo e o da
                        particao (G16);
    R5 pareamento    -- resultados com partition_id e train_population_id,
                        base e variante com a mesma particao e, nos pares por
                        unidade, os mesmos clientes no teste e no treino da
                        remocao (G16).

    R6 estratos      -- nas bases e no protocolo do TabBERT, todos os
                        algoritmos avaliados nas mesmas transacoes (base,
                        protocolo, semente, dobra e split) tem as mesmas
                        contagens de transacoes e fraudes em cada estrato de
                        historico (item C3.19).

O apendice traz, alem das tabelas de faltantes, remocao e PCA, a tabela de
pareamento: por par, protocolo, semente e dobra, se o teste, o treino sem
remocao e o treino da remocao sao os mesmos nas bases comparadas.
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

import experiment_grid
import history_strata
import lr_nb_knn_specific as pipeline
import missing_screening
import prepare_lr_nb_knn_datasets as preparation
import preparation_report
from experiment_datasets import (
    DATASET_DEFINITIONS,
    get_dataset_definition,
    resolve_base_definition,
)
from prepared_dataset_storage import (
    CATEGORICAL_HANDLING_MODE_ENCODED,
    CATEGORICAL_HANDLING_MODE_NATIVE,
    build_prepared_dataset_md5,
    compute_file_content_md5,
    prepared_split_path,
    read_prepared_split_csv,
    normalize_categorical_encoding_strategy,
)


class CheckReport(Protocol):
    def add(self, check_id: str, scope: str, passed: bool, detail: str) -> None: ...


R2_METRICS = ("f1", "f_beta_2", "precision", "recall", "roc_auc", "auc_pr")
# Eixos testados no R2: coluna do resultado -> rotulo no apendice.
R2_AXES = {
    "fill_na_strategy": "imputacao",
    "categorical_encoding_strategy": "codificacao",
}
# Colunas que, alem do eixo testado, identificam um par comparavel.
R2_KEY_COLUMNS = (
    "dataset_id",
    "database_type",
    "categorical_encoding_strategy",
    "fill_na_strategy",
    "undersampling_technique",
    "random_seed",
    "test_data_size",
    "train_validation_test_strategy",
    "algorithm",
    "regime",
    "class_weight_intensity",
    "threshold_strategy",
    # A fracao, e nao label_budget: no drop-incomplete-rows o treino perde
    # fraudes, e a mesma fracao rotula menos fraudes que na imputacao.
    "label_fraction",
    "execution_seed",
    "evaluation_unit",
    "fold_index",
    "split",
)
# Parametros que o resultado e o manifesto gravam com o mesmo nome (R1).
R1_SHARED_PARAMETERS = (
    "dataset_id",
    "categorical_encoding_strategy",
    "fill_na_strategy",
    "undersampling_technique",
    "random_seed",
)
# Ordem dos parametros em build_prepared_dataset_md5, para o apendice (P15).
CONFIGURATION_HASH_PARAMETERS = (
    "dataset_id",
    "categorical_encoding_strategy (normalizada)",
    "fill_na_strategy",
    "test_strategy",
    "prediction_column_name",
    "undersampling_technique",
    "undersampling_parameter_values",
    "auxiliary_data_is_test_data",
    "random_seed",
    "PREPROCESSING_REVISION",
    "regra da triagem (so em drop-incomplete-rows)",
    "categorical_handling_mode",
    "parametros dos Transformers (so nos modos de Transformer)",
)
VERIFIED_HANDLING_MODES = (CATEGORICAL_HANDLING_MODE_ENCODED, CATEGORICAL_HANDLING_MODE_NATIVE)


def _text(value) -> str:
    """Normaliza ausencias: None, NaN, "" e "none" valem o mesmo."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "none"
    text = str(value).strip()
    return "none" if text.lower() in {"", "none", "null", "nan"} else text


def _definition(dataset: str):
    try:
        return get_dataset_definition(dataset)
    except (KeyError, ValueError):
        return None


def base_dataset_id(dataset: str) -> str:
    """Base de origem de uma variante de ablacao; a propria base nas demais."""
    definition = _definition(dataset)
    return resolve_base_definition(definition).dataset_id if definition else str(dataset)


def unit_pair_id(dataset: str) -> str | None:
    """Raiz do par por unidade (as duas Home Credit); None fora de um par."""
    definition = _definition(dataset)
    if definition is None:
        return None
    if definition.paired_dataset_id:
        return definition.paired_dataset_id
    is_pair_root = any(
        other.paired_dataset_id == definition.dataset_id
        for other in DATASET_DEFINITIONS.values()
    )
    return definition.dataset_id if is_pair_root else None


# ---------------------------------------------------------------------------
# Preparacoes
# ---------------------------------------------------------------------------


def _manifest_with_defaults(manifest: pd.DataFrame) -> pd.DataFrame:
    manifest = manifest.copy()
    for column in ("preparation_report_md5", "categorical_handling_mode", "fold_index"):
        if column not in manifest.columns:
            manifest[column] = ""
    manifest["categorical_handling_mode"] = (
        manifest["categorical_handling_mode"].fillna(CATEGORICAL_HANDLING_MODE_ENCODED)
        .astype(str).str.strip().str.lower()
    )
    return manifest


def load_fold_reports(manifest: pd.DataFrame, directory: Path) -> dict[tuple[str, int], dict]:
    """Relatorios de preparacao que existem e batem com o MD5 do manifesto."""
    reports = {}
    declared = manifest[manifest["preparation_report_md5"].fillna("").astype(str).str.strip() != ""]
    for (configuration_md5, fold_index), _rows in declared.groupby(["configuration_md5", "fold_index"]):
        path = preparation_report.preparation_report_path(directory, configuration_md5, int(fold_index))
        if path.is_file():
            reports[(configuration_md5, int(fold_index))] = json.loads(path.read_text(encoding="utf-8"))
    return reports


def check_preparation_reports(
    manifest: pd.DataFrame,
    directory: Path,
    reports: dict,
    report: CheckReport,
    configuration_limit: int = 0,
) -> None:
    """P10: relatorio presente, MD5 correto, sem faltantes residuais, contadores coerentes."""
    manifest = _manifest_with_defaults(manifest)
    verified = manifest[manifest["categorical_handling_mode"].isin(VERIFIED_HANDLING_MODES)]
    configurations = list(verified["configuration_md5"].dropna().unique())
    if configuration_limit > 0:
        configurations = configurations[:configuration_limit]
    for configuration_md5 in configurations:
        rows = verified[verified["configuration_md5"] == configuration_md5]
        target = rows["prediction_column_name"].iloc[0]
        handling_mode = rows["categorical_handling_mode"].iloc[0]
        problems = []
        for fold_index, fold_rows in rows.groupby("fold_index"):
            recorded_md5 = _text(fold_rows["preparation_report_md5"].iloc[0])
            if recorded_md5 == "none":
                problems.append(f"dobra {fold_index}: manifesto sem relatorio")
                continue
            path = preparation_report.preparation_report_path(directory, configuration_md5, int(fold_index))
            if not path.is_file():
                problems.append(f"dobra {fold_index}: {path.name} ausente")
                continue
            if compute_file_content_md5(path) != recorded_md5.lower():
                problems.append(f"dobra {fold_index}: MD5 do relatorio difere do manifesto")
                continue
            fold_report = reports[(configuration_md5, int(fold_index))]
            for row in fold_rows.itertuples(index=False):
                counts = fold_report["splits"].get(row.split, {})
                output = counts.get(preparation_report.STAGE_OUTPUT)
                if output is None:
                    problems.append(f"dobra {fold_index}/{row.split}: sem etapa de saida")
                    continue
                if handling_mode == CATEGORICAL_HANDLING_MODE_ENCODED and (
                    counts.get(preparation_report.STAGE_AFTER_IMPUTATION, {}).get("missing_cells", 0)
                    or output["missing_cells"]
                ):
                    problems.append(f"dobra {fold_index}/{row.split}: faltantes depois da imputacao")
                saved = read_prepared_split_csv(
                    prepared_split_path(directory, row.dataset_md5), usecols=[target]
                )
                saved_positives = int((saved[target] == 1).sum())
                if (len(saved), saved_positives) != (output["rows"], output["positives"]):
                    problems.append(
                        f"dobra {fold_index}/{row.split}: relatorio {output['rows']}/"
                        f"{output['positives']} x arquivo {len(saved)}/{saved_positives} "
                        "(linhas/positivos)"
                    )
        scope = f"{rows['dataset_id'].iloc[0]}/{configuration_md5[:8]}"
        report.add(
            "P10", scope, not problems,
            "relatorios coerentes com os arquivos" if not problems else "; ".join(problems[:5]),
        )


def _axis_groups(train_rows: pd.DataFrame, axis: str) -> list[tuple[tuple, pd.DataFrame]]:
    key = [
        "dataset_id", "categorical_handling_mode", "categorical_encoding_strategy",
        "fill_na_strategy", "undersampling_technique", "undersampling_parameter_values",
        "test_strategy", "random_seed", "fold_index",
    ]
    key = [column for column in key if column != axis and column in train_rows.columns]
    return list(train_rows.fillna("").groupby(key))


def check_axis_effects(manifest: pd.DataFrame, reports: dict, report: CheckReport) -> None:
    """P11 (imputacao) e P12 (codificacao): o eixo precisa mudar os dados."""
    manifest = _manifest_with_defaults(manifest)
    train_rows = manifest[manifest["split"] == "train"]
    for check_id, axis, applies in (
        ("P11", "fill_na_strategy", lambda d: d.has_missing_values),
        ("P12", "categorical_encoding_strategy", lambda d: d.has_categorical_attributes),
    ):
        problems: list[str] = []
        compared = 0
        for _key, group in _axis_groups(train_rows, axis):
            definition = _definition(group["dataset_id"].iloc[0])
            if definition is None or not applies(definition) or group[axis].nunique() < 2:
                continue
            fingerprint_by_value = {}
            for row in group.itertuples(index=False):
                if check_id == "P11":
                    fingerprint = _text(row.content_md5)
                else:
                    fold_report = reports.get((row.configuration_md5, int(row.fold_index)))
                    encoded = (fold_report or {}).get("encoded_categorical_md5") or {}
                    fingerprint = _text(encoded.get("train"))
                if fingerprint != "none":
                    fingerprint_by_value[getattr(row, axis)] = fingerprint
            for (value_a, md5_a), (value_b, md5_b) in combinations(sorted(fingerprint_by_value.items()), 2):
                compared += 1
                if md5_a == md5_b:
                    problems.append(
                        f"{group['dataset_id'].iloc[0]} dobra {group['fold_index'].iloc[0]}: "
                        f"{value_a} = {value_b}"
                    )
        label = R2_AXES[axis]
        report.add(
            check_id, "manifesto", not problems,
            f"{compared} pares de {label} com dados distintos"
            if not problems else f"eixo de {label} sem efeito: " + "; ".join(problems[:5]),
        )


def check_missing_screening(
    manifest: pd.DataFrame, directory: Path, reports: dict, report: CheckReport
) -> None:
    """P13: limiar admissivel e maximo; colunas triadas fora dos arquivos."""
    manifest = _manifest_with_defaults(manifest)
    drop_rows = manifest[manifest["fill_na_strategy"] == missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY]
    for configuration_md5, rows in drop_rows.groupby("configuration_md5"):
        problems = []
        for fold_index, fold_rows in rows.groupby("fold_index"):
            fold_report = reports.get((configuration_md5, int(fold_index)))
            screening = (fold_report or {}).get("missing_screening")
            if not screening:
                problems.append(f"dobra {fold_index}: sem triagem no relatorio")
                continue
            minimum = missing_screening.DROP_SCREENING_MIN_FRAUD_RETENTION * screening["positives_before"]
            admissible = [
                int(threshold)
                for threshold, kept in screening["positives_by_threshold"].items()
                if kept >= minimum
            ]
            if screening["positives_after"] < minimum:
                problems.append(f"dobra {fold_index}: preserva menos que o piso de fraudes")
            if admissible and screening["threshold_pct"] != max(admissible):
                problems.append(
                    f"dobra {fold_index}: limiar {screening['threshold_pct']} "
                    f"< maior admissivel {max(admissible)}"
                )
            screened = set(screening["screened_out_columns"])
            recorded = {
                column for column in _text(fold_rows["screened_out_columns"].iloc[0]).split(",")
                if column and column != "none"
            } if "screened_out_columns" in fold_rows.columns else screened
            if recorded != screened:
                problems.append(f"dobra {fold_index}: manifesto e relatorio divergem nas colunas")
            for row in fold_rows.itertuples(index=False):
                header = read_prepared_split_csv(
                    prepared_split_path(directory, row.dataset_md5), nrows=0
                ).columns
                leaked = sorted(screened.intersection(header))
                if leaked:
                    problems.append(f"dobra {fold_index}/{row.split}: colunas triadas presentes {leaked[:5]}")
        report.add(
            "P13", f"{rows['dataset_id'].iloc[0]}/{configuration_md5[:8]}", not problems,
            "triagem conforme a regra" if not problems else "; ".join(problems[:5]),
        )


def check_training_population(manifest: pd.DataFrame, report: CheckReport) -> None:
    """P16: treino efetivo igual onde o pareamento exige (item G16)."""
    if "train_population_md5" not in manifest.columns:
        report.add(
            "P16", "manifesto", True,
            "manifesto sem train_population_md5; verificacao nao se aplica",
        )
        return
    manifest = _manifest_with_defaults(manifest)
    train_rows = manifest[manifest["split"].astype(str).str.strip() == "train"].copy()
    train_rows["population"] = train_rows["train_population_md5"].map(_text)
    train_rows["partition"] = train_rows["partition_md5"].map(_text)
    train_rows["base_dataset"] = train_rows["dataset_id"].map(base_dataset_id)
    is_drop = train_rows["fill_na_strategy"] == missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY

    problems = []
    kept = train_rows[~is_drop & (train_rows["population"] != "none")]
    mismatched = kept[kept["population"] != kept["partition"]]
    for row in mismatched.head(5).itertuples(index=False):
        problems.append(
            f"{row.configuration_md5} dobra {row.fold_index}: sem remocao, mas o "
            "treino efetivo difere do treino da particao"
        )
    drop_rows = train_rows[is_drop]
    compared = 0
    for key, group in drop_rows.groupby(
        ["base_dataset", "test_strategy", "random_seed", "fold_index"], dropna=False
    ):
        compared += 1
        populations = set(group["population"])
        if "none" in populations or len(populations) > 1:
            datasets = ", ".join(sorted(set(group["dataset_id"])))
            problems.append(
                f"{key[0]}/{key[1]}/semente-{key[2]}/dobra-{key[3]}: "
                f"{len(populations)} treinos efetivos entre {datasets}"
            )
    report.add(
        "P16", "manifesto", not problems,
        f"{compared} grupos drop-incomplete-rows com o mesmo treino e "
        f"{len(kept)} treinos sem remocao iguais a particao"
        if not problems else "; ".join(problems[:5]),
    )
    _check_training_clients(train_rows, is_drop, report)


def _check_training_clients(
    train_rows: pd.DataFrame, is_drop: pd.Series, report: CheckReport
) -> None:
    """P16 nos pares por unidade: os mesmos clientes no treino efetivo."""
    paired = train_rows.assign(unit_pair=train_rows["dataset_id"].map(unit_pair_id))
    in_pair = paired["unit_pair"].notna()
    if not in_pair.any():
        return
    if "train_unit_population_md5" not in paired.columns:
        report.add(
            "P16", "pares por unidade", False,
            "manifesto sem train_unit_population_md5: o treino dos pares por "
            "unidade nao pode ser verificado",
        )
        return
    paired["clients"] = paired["train_unit_population_md5"].map(_text)
    paired["partition_clients"] = paired.get(
        "unit_partition_md5", pd.Series("", index=paired.index)
    ).map(_text)
    paired = paired[in_pair]
    is_drop = is_drop.loc[paired.index]
    problems = []
    kept = paired[~is_drop]
    for row in kept[kept["clients"] != kept["partition_clients"]].head(5).itertuples(index=False):
        problems.append(
            f"{row.configuration_md5} dobra {row.fold_index}: sem remocao, mas os "
            "clientes do treino efetivo diferem dos da particao"
        )
    compared = 0
    for key, group in paired[is_drop].groupby(
        ["unit_pair", "test_strategy", "random_seed", "fold_index"], dropna=False
    ):
        if group["dataset_id"].nunique() < 2:
            continue
        compared += 1
        clients = set(group["clients"])
        if "none" in clients or len(clients) > 1:
            datasets = ", ".join(sorted(set(group["dataset_id"])))
            problems.append(
                f"{key[0]}/{key[1]}/semente-{key[2]}/dobra-{key[3]}: "
                f"{len(clients)} conjuntos de clientes no treino entre {datasets}"
            )
    report.add(
        "P16", "pares por unidade", not problems,
        f"{compared} grupos drop-incomplete-rows com os mesmos clientes no treino"
        if not problems else "; ".join(problems[:5]),
    )


def check_pca_record(manifest: pd.DataFrame, reports: dict, report: CheckReport) -> None:
    """P14: componentes registrados onde a PCA se aplica, e so la."""
    manifest = _manifest_with_defaults(manifest)
    problems = []
    checked = 0
    first_rows = manifest.drop_duplicates(["configuration_md5", "fold_index"])
    for row in first_rows.itertuples(index=False):
        fold_report = reports.get((row.configuration_md5, int(row.fold_index)))
        definition = _definition(row.dataset_id)
        if fold_report is None or definition is None:
            continue
        checked += 1
        expects_pca = (
            row.categorical_handling_mode == CATEGORICAL_HANDLING_MODE_ENCODED
            and definition.applies_pca_in_classic_block
        )
        pca = fold_report.get("pca")
        scope = f"{row.dataset_id}/{row.configuration_md5[:8]} dobra {row.fold_index}"
        if expects_pca and not pca:
            problems.append(f"{scope}: PCA sem componentes registrados")
        elif not expects_pca and pca:
            problems.append(f"{scope}: componentes registrados sem PCA")
        elif pca and not 1 <= pca["pca_components"] <= pca["pca_input_columns"]:
            problems.append(f"{scope}: {pca['pca_components']} componentes de {pca['pca_input_columns']}")
    report.add(
        "P14", "relatorios", not problems,
        f"{checked} dobras com PCA registrada conforme o catalogo" if not problems else "; ".join(problems[:5]),
    )


def recompute_configuration_md5(row) -> str | None:
    """Recalcula a identidade a partir de uma linha do manifesto.

    Os modos de Transformer dependem de parametros que o manifesto nao grava
    (janelas, faixas numericas); para eles devolve None.
    """
    handling_mode = _text(getattr(row, "categorical_handling_mode", CATEGORICAL_HANDLING_MODE_ENCODED)).lower()
    if handling_mode not in VERIFIED_HANDLING_MODES:
        return None
    nullable = pipeline._parse_nullable_value
    return build_prepared_dataset_md5(
        row.dataset_id,
        nullable(row.categorical_encoding_strategy),
        nullable(row.fill_na_strategy),
        row.test_strategy,
        row.prediction_column_name,
        nullable(row.undersampling_technique),
        nullable(row.undersampling_parameter_values),
        pipeline._parse_bool(row.auxiliary_data_is_test_data),
        handling_mode,
        int(row.random_seed),
    )


def check_configuration_identity(manifest: pd.DataFrame, report: CheckReport) -> None:
    """P15: o configuration_md5 gravado e o recalculado dos parametros."""
    manifest = _manifest_with_defaults(manifest).fillna("")
    problems = []
    checked = 0
    for row in manifest.drop_duplicates("configuration_md5").itertuples(index=False):
        recomputed = recompute_configuration_md5(row)
        if recomputed is None:
            continue
        checked += 1
        if recomputed != str(row.configuration_md5).lower():
            problems.append(f"{row.dataset_id}: gravado {row.configuration_md5[:8]} x recalculado {recomputed[:8]}")
    report.add(
        "P15", "manifesto", not problems,
        f"{checked} identidades recalculadas conferem" if not problems else "; ".join(problems[:5]),
    )


# ---------------------------------------------------------------------------
# Resultados
# ---------------------------------------------------------------------------


def load_results(paths: list[Path]) -> pd.DataFrame:
    frames = [pd.read_csv(path, dtype=str).assign(result_file=path.name) for path in paths]
    results = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if "dataset_id" not in results.columns and "database_type" in results.columns:
        results["dataset_id"] = results["database_type"]
    return results


def _successful(results: pd.DataFrame) -> pd.DataFrame:
    status = results.get("execution_status", pd.Series("success", index=results.index))
    return results[status.fillna("").str.strip().str.lower() == "success"]


def check_history_strata(records: list[dict], report: CheckReport) -> None:
    """R6: mesmas contagens por estrato de historico entre algoritmos (item C3.19)."""
    if not records:
        report.add("R6", "estratos", True, "nenhum registro de estratos de historico")
        return
    divergent = history_strata.inconsistent_strata(records)
    if divergent.empty:
        report.add("R6", "estratos", True, f"{len(records)} registros com contagens coerentes")
        return
    for row in divergent.itertuples(index=False):
        report.add(
            "R6",
            f"estratos/{row.dataset_id}/{row.split}/estrato {row.stratum}",
            False,
            (
                f"contagens diferentes entre {row.algorithms} "
                f"(protocolo {row.test_strategy}, semente {row.random_seed}, dobra {row.fold_index})"
            ),
        )


def check_results_traceability(results: pd.DataFrame, manifest: pd.DataFrame, report: CheckReport) -> None:
    """R1: cada resultado aponta para uma configuracao do manifesto."""
    configurations = _manifest_with_defaults(manifest).drop_duplicates("configuration_md5").set_index("configuration_md5")
    orphans = sorted(set(results["configuration_md5"]) - set(configurations.index))
    divergent = []
    for row in results.drop_duplicates("configuration_md5").itertuples(index=False):
        if row.configuration_md5 not in configurations.index:
            continue
        recorded = configurations.loc[row.configuration_md5]
        for parameter in R1_SHARED_PARAMETERS:
            if parameter not in results.columns or parameter not in recorded.index:
                continue
            result_value, manifest_value = _text(getattr(row, parameter)), _text(recorded[parameter])
            if parameter == "categorical_encoding_strategy":
                result_value = _text(normalize_categorical_encoding_strategy(pipeline._parse_nullable_value(result_value)))
                manifest_value = _text(normalize_categorical_encoding_strategy(pipeline._parse_nullable_value(manifest_value)))
            if result_value != manifest_value:
                divergent.append(f"{row.configuration_md5[:8]} {parameter}: {result_value} x {manifest_value}")
    problems = [f"{len(orphans)} configuracoes fora do manifesto"] if orphans else []
    report.add(
        "R1", "resultados", not (orphans or divergent),
        f"{results['configuration_md5'].nunique()} configuracoes rastreadas"
        if not (orphans or divergent) else "; ".join(problems + divergent[:5]),
    )


def check_result_pairing(results: pd.DataFrame, report: CheckReport) -> None:
    """R5: identidades de pareamento nas linhas de resultado (item G16)."""
    successful = _successful(results).copy()
    required = ("partition_id", "train_population_id")
    absent = [column for column in required if column not in successful.columns]
    if absent:
        report.add("R5", "resultados", False, f"colunas ausentes: {', '.join(absent)}")
        return
    for column in (*required, "unit_partition_id", "train_unit_population_id"):
        if column not in successful.columns:
            successful[column] = ""
        successful[column] = successful[column].map(_text)
    for column in required:
        empty = int((successful[column] == "none").sum())
        report.add(
            "R5", f"resultados/{column}", empty == 0,
            "todas as linhas de sucesso identificadas"
            if empty == 0 else f"{empty} linhas de sucesso sem {column}",
        )

    successful["base_dataset"] = successful["dataset_id"].map(base_dataset_id)
    successful["unit_pair"] = successful["dataset_id"].map(unit_pair_id)
    fold_key = ["train_validation_test_strategy", "random_seed", "fold_index", "split"]
    is_drop = (
        successful["fill_na_strategy"] == missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY
    )
    problems = []
    variants = successful[successful["base_dataset"] != successful["dataset_id"]]
    for base in sorted(set(variants["base_dataset"])):
        family = successful[successful["base_dataset"] == base]
        for key, group in family.groupby(fold_key, dropna=False):
            if len(set(group["partition_id"])) > 1:
                problems.append(f"{base}/{'/'.join(map(str, key))}: particoes diferentes")
        drop_family = family[is_drop.loc[family.index]]
        for key, group in drop_family.groupby(fold_key, dropna=False):
            if len(set(group["train_population_id"])) > 1:
                problems.append(f"{base}/{'/'.join(map(str, key))}: treinos diferentes na remocao")
    paired = successful[successful["unit_pair"].notna()]
    for pair, family in paired.groupby("unit_pair"):
        for key, group in family.groupby(fold_key, dropna=False):
            units = set(group["unit_partition_id"])
            if "none" in units or len(units) > 1:
                problems.append(f"{pair}/{'/'.join(map(str, key))}: clientes diferentes")
        drop_family = family[is_drop.loc[family.index]]
        for key, group in drop_family.groupby(fold_key, dropna=False):
            if group["dataset_id"].nunique() < 2:
                continue
            clients = set(group["train_unit_population_id"])
            if "none" in clients or len(clients) > 1:
                problems.append(
                    f"{pair}/{'/'.join(map(str, key))}: clientes de treino diferentes na remocao"
                )
    report.add(
        "R5", "pares", not problems,
        "bases de ablacao e pares por unidade avaliados nas mesmas linhas e clientes"
        if not problems else "; ".join(problems[:5]),
    )


def _axis_applies(dataset: str, axis: str) -> bool:
    definition = _definition(dataset)
    if definition is None:
        return False
    if axis == "fill_na_strategy":
        return definition.has_missing_values
    return definition.has_categorical_attributes


def one_axis_pairs(results: pd.DataFrame, legacy: bool) -> pd.DataFrame:
    """Pares de resultados que diferem num unico eixo e se suas metricas coincidem."""
    successful = _successful(results).copy()
    metrics = [metric for metric in R2_METRICS if metric in successful.columns]
    if legacy:
        # CSVs antigos nao tem fold_index: a ordem das linhas dentro da
        # configuracao e do algoritmo faz as vezes da dobra.
        successful["fold_index"] = successful.groupby(["configuration_md5", "algorithm"]).cumcount().astype(str)
    has_content_md5 = "train_dataset_md5" in successful.columns and not legacy
    rows = []
    for axis, label in R2_AXES.items():
        if axis not in successful.columns:
            continue
        key = [column for column in R2_KEY_COLUMNS if column in successful.columns and column != axis]
        if "dataset_id" in key and "database_type" in key:
            key.remove("database_type")
        for _key_values, group in successful.fillna("").groupby(key):
            dataset = group["dataset_id"].iloc[0]
            if group[axis].nunique() < 2 or not _axis_applies(dataset, axis):
                continue
            by_value = group.drop_duplicates(axis).set_index(axis)
            for value_a, value_b in combinations(sorted(by_value.index), 2):
                metrics_a = pd.to_numeric(by_value.loc[value_a, metrics], errors="coerce").to_numpy(dtype=float)
                metrics_b = pd.to_numeric(by_value.loc[value_b, metrics], errors="coerce").to_numpy(dtype=float)
                identical = bool(np.array_equal(metrics_a, metrics_b, equal_nan=True))
                training_differs = (
                    by_value.loc[value_a, "train_dataset_md5"] != by_value.loc[value_b, "train_dataset_md5"]
                    if has_content_md5 else None
                )
                rows.append({
                    "dataset_id": dataset,
                    "axis": label,
                    "algorithm": group["algorithm"].iloc[0],
                    "value_a": value_a,
                    "value_b": value_b,
                    "identical_metrics": identical,
                    "training_differs": training_differs,
                })
    return pd.DataFrame(rows, columns=[
        "dataset_id", "axis", "algorithm", "value_a", "value_b", "identical_metrics", "training_differs",
    ])


def summarize_axis_effects(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(columns=["dataset_id", "axis", "algorithm", "pairs", "identical_pairs", "identical_pct"])
    summary = (
        pairs.groupby(["dataset_id", "axis", "algorithm"])
        .agg(pairs=("identical_metrics", "size"), identical_pairs=("identical_metrics", "sum"))
        .reset_index()
    )
    summary["identical_pct"] = (100 * summary["identical_pairs"] / summary["pairs"]).round(1)
    return summary


def check_one_axis_pairs(results: pd.DataFrame, report: CheckReport, legacy: bool) -> pd.DataFrame:
    """R2: nenhum eixo pode deixar as metricas identicas bit a bit.

    Com o MD5 de conteudo do treino, falha todo par identico cujos treinos
    diferem. Em CSVs antigos, sem esse MD5, falha o grupo base x eixo x
    algoritmo em que TODOS os pares sao identicos, que e o sintoma do G15.
    """
    pairs = one_axis_pairs(results, legacy)
    summary = summarize_axis_effects(pairs)
    suffix = " (legado: sem verificacao de conteudo)" if legacy else ""
    for (dataset, axis), group in summary.groupby(["dataset_id", "axis"]):
        if legacy:
            inert = group[(group["pairs"] >= 2) & (group["identical_pct"] == 100.0)]
            failed = not inert.empty
            detail_rows = inert
        else:
            suspicious = pairs[
                (pairs["dataset_id"] == dataset) & (pairs["axis"] == axis)
                & pairs["identical_metrics"] & (pairs["training_differs"] == True)  # noqa: E712
            ]
            failed = not suspicious.empty
            detail_rows = group[group["algorithm"].isin(suspicious["algorithm"])]
        overall = 100 * group["identical_pairs"].sum() / group["pairs"].sum()
        detail = f"{group['pairs'].sum()} pares, {overall:.1f}% identicos"
        if failed:
            detail += "; eixo sem efeito em " + ", ".join(
                f"{row.algorithm} ({row.identical_pct:.0f}%)" for row in detail_rows.itertuples()
            )
        report.add("R2", f"{dataset}/{axis}", not failed, detail + suffix)
    return summary


def check_training_sizes(results: pd.DataFrame, reports: dict, report: CheckReport) -> None:
    """R3: o treino que o algoritmo viu e o que a preparacao gravou."""
    problems = []
    checked = 0
    columns = {"configuration_md5", "fold_index", "train_df_size"}
    if not columns.issubset(results.columns):
        report.add("R3", "resultados", True, "resultados sem fold_index; verificacao nao se aplica")
        return
    for row in _successful(results).drop_duplicates(["configuration_md5", "fold_index"]).itertuples(index=False):
        fold_report = reports.get((row.configuration_md5, int(row.fold_index)))
        if fold_report is None:
            continue
        checked += 1
        expected = fold_report["splits"]["train"][preparation_report.STAGE_OUTPUT]["rows"]
        if int(float(row.train_df_size)) != expected:
            problems.append(f"{row.configuration_md5[:8]} dobra {row.fold_index}: {row.train_df_size} x {expected}")
    report.add(
        "R3", "resultados", not problems,
        f"{checked} dobras com o tamanho do treino do relatorio" if not problems else "; ".join(problems[:5]),
    )


# Blocos de execucao, na ordem do apendice.
RUN_BLOCKS = (
    experiment_grid.BLOCK_CLASSIC,
    experiment_grid.BLOCK_BOOSTING,
    experiment_grid.BLOCK_ANOMALY,
    experiment_grid.BLOCK_FT_TRANSFORMER,
    experiment_grid.BLOCK_TABBERT,
)
# Identidade de uma execucao: a configuracao preparada e o que o executor
# varia sobre ela (algoritmo, ponderacao, regime do TabBERT, fracao rotulada e
# semente de sorteio do DevNet).
RUN_KEY = (
    "configuration_md5", "algorithm", "class_weight_intensity", "regime",
    "label_fraction", "label_draw_seed",
)
RUN_STATES = ("sucesso", "falha", "incompleta", "ausente", "equivalente")
RUN_COUNT_GROUP = (
    "bloco", "dataset_id", "algorithm", "class_weight_intensity", "regime",
    "label_fraction", "undersampling_technique", "test_strategy",
)


def block_of_algorithm(algorithm: str) -> str | None:
    """Bloco que executa um algoritmo, pelo nome gravado no resultado."""
    algorithm = _text(algorithm)
    if algorithm in experiment_grid.CLASSIC_ALGORITHMS:
        return experiment_grid.BLOCK_CLASSIC
    if algorithm in experiment_grid.BOOSTING_ALGORITHMS:
        return experiment_grid.BLOCK_BOOSTING
    if algorithm in (
        *experiment_grid.ANOMALY_UNSUPERVISED_ALGORITHMS,
        *experiment_grid.ANOMALY_LABEL_BUDGET_ALGORITHMS,
    ):
        return experiment_grid.BLOCK_ANOMALY
    if algorithm.startswith("ft-transformer"):
        return experiment_grid.BLOCK_FT_TRANSFORMER
    if algorithm.startswith("tabbert"):
        return experiment_grid.BLOCK_TABBERT
    return None


def _fraction_text(value) -> str:
    """Fracao rotulada normalizada; vazia quando o eixo nao se aplica (-1 ou 0)."""
    text = _text(value)
    if text == "none":
        return ""
    try:
        fraction = float(text)
    except ValueError:
        return text
    return f"{fraction:g}" if fraction > 0 else ""


def _draw_seed_text(algorithm: str, value) -> str:
    # So o DevNet sorteia rotulos; nos Transformers execution_seed e a semente
    # de treino e nao identifica um braco.
    if _text(algorithm) not in experiment_grid.ANOMALY_LABEL_BUDGET_ALGORITHMS:
        return ""
    text = _text(value)
    try:
        return str(int(float(text)))
    except ValueError:
        return "" if text == "none" else text


_GRID_ENTRY_MD5_CACHE: dict[experiment_grid.GridEntry, str] = {}


def grid_entry_configuration_md5(entry) -> str:
    """configuration_md5 de uma entrada da grade, pela mesma rotina da preparacao.

    Monta os argumentos como ``prepare_grid_common.sh`` e chama
    ``resolve_configuration_identity``: a contagem prevista nao depende de
    reimplementar a regra do hash, inclusive nos modos de Transformer.
    """
    if entry in _GRID_ENTRY_MD5_CACHE:
        return _GRID_ENTRY_MD5_CACHE[entry]

    argv = [
        "prepare_lr_nb_knn_datasets.py",
        "--dataset", entry.dataset_id,
        "--prediction-column", entry.prediction_column,
        "--auxiliary-data-is-test-data", entry.auxiliary_data_is_test_data,
        "--categorical-handling-mode", entry.categorical_handling_mode,
        "--categorical-encoding", entry.categorical_encoding_strategy,
        "--fill-na-strategy", entry.fill_na_strategy,
        "--undersampling-technique", entry.undersampling_technique,
        "--undersampling-parameters", entry.undersampling_parameter_values,
        "--test-strategy", entry.test_strategy,
        "--random-seed", str(entry.random_seed),
    ]
    if entry.categorical_handling_mode in ("transformer-indexed", "sequence-tokenized"):
        argv += [
            "--numeric-normalization", entry.numeric_normalization,
            "--high-cardinality-strategy", entry.high_cardinality_strategy,
            "--high-cardinality-parameter", str(entry.high_cardinality_parameter),
        ]
    if entry.categorical_handling_mode == "sequence-tokenized":
        argv += [
            "--numeric-bins", str(experiment_grid.TABBERT_NUMERIC_BINS),
            "--window-length", str(experiment_grid.TABBERT_WINDOW_LENGTH),
            "--history-length", str(experiment_grid.TABBERT_HISTORY_LENGTH),
            "--window-stride", str(experiment_grid.TABBERT_WINDOW_STRIDE),
        ]
    saved_argv = sys.argv
    try:
        sys.argv = argv
        arguments = preparation.parse_arguments()
    finally:
        sys.argv = saved_argv
    md5 = preparation.resolve_configuration_identity(arguments)[0]
    _GRID_ENTRY_MD5_CACHE[entry] = md5
    return md5


def _block_runs(block: str):
    """(algoritmo, intensidade, regime, fracao, semente de sorteio, entradas) de um bloco."""
    entries = experiment_grid.GRID_BUILDERS[block]()
    if block == experiment_grid.BLOCK_ANOMALY:
        for algorithm, fraction, draw_seed in experiment_grid.anomaly_arms():
            yield (
                algorithm, experiment_grid.DEFAULT_CLASS_WEIGHT_INTENSITY, "",
                "" if fraction is None else f"{fraction:g}",
                "" if draw_seed is None else str(draw_seed), entries,
            )
        return
    for label, run_entries in experiment_grid.run_entries_for(block, entries):
        algorithm, _, suffix = label.partition(" [")
        suffix = suffix.rstrip("]")
        if block == experiment_grid.BLOCK_TABBERT:
            yield algorithm, experiment_grid.DEFAULT_CLASS_WEIGHT_INTENSITY, suffix, "", "", run_entries
        else:
            intensity = suffix or experiment_grid.DEFAULT_CLASS_WEIGHT_INTENSITY
            yield algorithm, intensity, "", "", "", run_entries


def expected_runs(blocks: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Execucoes que a grade preve, uma linha por execucao (G17).

    A grade ja multiplica as sementes de preparacao de SEED_LIST; no DevNet,
    cada fracao rotulada e cada semente de sorteio de LABEL_DRAW_SEEDS sao
    execucoes distintas.
    """
    blocks = RUN_BLOCKS if blocks is None else blocks
    rows = []
    for block in blocks:
        for algorithm, intensity, regime, fraction, draw_seed, entries in _block_runs(block):
            for entry in entries:
                rows.append({
                    "bloco": block,
                    "dataset_id": entry.dataset_id,
                    "configuration_md5": grid_entry_configuration_md5(entry),
                    "algorithm": algorithm,
                    "class_weight_intensity": intensity,
                    "regime": regime,
                    "label_fraction": fraction,
                    "label_draw_seed": draw_seed,
                    "categorical_encoding_strategy": entry.categorical_encoding_strategy,
                    "fill_na_strategy": entry.fill_na_strategy,
                    "undersampling_technique": entry.undersampling_technique,
                    "test_strategy": entry.test_strategy,
                    "random_seed": str(entry.random_seed),
                    "linhas_teste_previstas": experiment_grid.result_rows_for([entry]),
                })
    columns = [
        "bloco", "dataset_id", *RUN_KEY, "categorical_encoding_strategy", "fill_na_strategy",
        "undersampling_technique", "test_strategy", "random_seed", "linhas_teste_previstas",
    ]
    return pd.DataFrame(rows, columns=columns)


K_FOLD_PROTOCOL = "stratified-k-fold"


def _with_run_key(results: pd.DataFrame) -> pd.DataFrame:
    observed = results.copy()
    for column in ("class_weight_intensity", "regime", "label_fraction", "execution_seed",
                   "execution_status", "split", "evaluation_unit", "fold_index",
                   "failure_category", "failure_message"):
        if column not in observed.columns:
            observed[column] = ""
    observed["configuration_md5"] = observed["configuration_md5"].map(_text).str.lower()
    observed["algorithm"] = observed["algorithm"].map(_text)
    # Sem ponderacao gravada, a intensidade e a padrao ("none").
    observed["class_weight_intensity"] = observed["class_weight_intensity"].map(_text)
    observed["regime"] = observed["regime"].fillna("").astype(str).replace("nan", "")
    observed["label_fraction"] = observed["label_fraction"].map(_fraction_text)
    observed["label_draw_seed"] = [
        _draw_seed_text(algorithm, seed)
        for algorithm, seed in zip(observed["algorithm"], observed["execution_seed"])
    ]
    status = observed["execution_status"].fillna("").astype(str).str.strip().str.lower()
    # CSVs antigos sem status: toda linha gravada e de sucesso.
    observed["success"] = status.isin(["success", ""])
    observed["failure"] = status == "failure"
    split = observed["split"].fillna("").astype(str).str.strip()
    unit = observed["evaluation_unit"].fillna("").astype(str).str.strip()
    # Linha de avaliacao por registro; CSVs sem split contam todas as linhas.
    # No k-fold o teste da dobra e vazio: a dobra e avaliada na propria
    # validacao, com o limiar das validacoes das outras dobras
    # (cross-fold-validation-f2, decision_threshold.select_fold_thresholds).
    protocol = observed.get("train_validation_test_strategy", pd.Series("", index=observed.index))
    k_fold_validation = (
        protocol.fillna("").astype(str).str.strip().eq(K_FOLD_PROTOCOL)
        & split.isin(["validation", "validation_data"])
    )
    observed["test_row"] = (
        (split.isin(["", "nan", "test", "test_data"]) | k_fold_validation)
        & unit.isin(["", "nan", "row"])
    )
    return observed


def load_execution_log(path: Path | None) -> pd.DataFrame:
    """Le registro_execucoes.csv; vazio quando nao ha registro."""
    columns = ["configuration_md5", "algoritmo", "class_weight_intensity", "label_fraction",
               "regime", "tentativa", "codigo_saida", "categoria_saida", "timestamp_fim"]
    if path is None or not Path(path).is_file():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _log_attempts(log: pd.DataFrame, run) -> pd.DataFrame:
    """Linhas do registro que cobrem uma execucao prevista.

    Um comando pode cobrir varios algoritmos (boosting) ou fracoes (DevNet):
    os campos do registro sao listas separadas por espaco, e vazio cobre tudo.
    """
    if log.empty:
        return log
    candidates = log[log["configuration_md5"].str.lower() == run.configuration_md5]

    def covers(field: str, value: str) -> pd.Series:
        listed = candidates[field].fillna("").astype(str)
        return listed.map(lambda text: not text.strip() or value in {
            _fraction_text(item) if field == "label_fraction" else item for item in text.split()
        })

    if candidates.empty:
        return candidates
    mask = covers("algoritmo", run.algorithm)
    mask &= covers("class_weight_intensity", run.class_weight_intensity)
    mask &= covers("regime", run.regime)
    if run.label_fraction:
        mask &= covers("label_fraction", run.label_fraction)
    return candidates[mask]


def classify_runs(
    results: pd.DataFrame,
    expected: pd.DataFrame,
    execution_log: pd.DataFrame | None = None,
    twins: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Estado, motivo e linhas de teste de cada execucao prevista (G17).

    sucesso    todas as dobras previstas tem linha de teste de sucesso;
    falha      ha linha de falha (o motivo e failure_category);
    incompleta ha sucesso, mas faltam dobras;
    ausente    nenhuma linha (o motivo vem do registro do executor);
    equivalente gemea por eixo inerte cujo representante teve sucesso.
    """
    execution_log = load_execution_log(None) if execution_log is None else execution_log
    observed = _with_run_key(results)
    key = list(RUN_KEY)
    successes = (
        observed[observed["success"] & observed["test_row"]]
        .groupby(key)["fold_index"].nunique().rename("linhas_teste_sucesso")
    )
    failures = observed[observed["failure"]].groupby(key).agg(
        failure_category=("failure_category", "last"),
        failure_message=("failure_message", "last"),
    )
    runs = (
        expected.merge(successes.reset_index(), on=key, how="left")
        .merge(failures.reset_index(), on=key, how="left")
    )
    runs["linhas_teste_sucesso"] = runs["linhas_teste_sucesso"].fillna(0).astype(int)
    complete = runs["linhas_teste_sucesso"] >= runs["linhas_teste_previstas"]
    has_failure = runs["failure_category"].notna()
    runs["estado"] = np.select(
        [complete, has_failure, runs["linhas_teste_sucesso"] > 0],
        ["sucesso", "falha", "incompleta"],
        default="ausente",
    )
    reasons, messages, attempts = [], [], []
    for run in runs.itertuples(index=False):
        logged = _log_attempts(execution_log, run)
        attempts.append(len(logged))
        if run.estado == "sucesso":
            reasons.append("")
            messages.append("")
        elif run.estado == "falha":
            category = _text(run.failure_category)
            reasons.append("desconhecida" if category in ("none", "") else category)
            messages.append("" if _text(run.failure_message) == "none" else _text(run.failure_message))
        else:
            reasons.append(logged["categoria_saida"].iloc[-1] if not logged.empty else "nao-executada")
            messages.append("")
    runs["motivo"] = reasons
    runs["mensagem"] = messages
    runs["tentativas"] = attempts
    return _mark_equivalent_runs(runs, twins or {})


def _mark_equivalent_runs(runs: pd.DataFrame, twins: dict[str, str]) -> pd.DataFrame:
    """Gemeas por eixo inerte nao executam: herdam o sucesso do representante.

    A gemea vira ``equivalente`` somente quando a execucao do representante,
    com a mesma chave nos demais campos, teve sucesso.
    """
    if not twins or runs.empty:
        return runs
    other_key = [column for column in RUN_KEY if column != "configuration_md5"]
    successful = {
        (row["configuration_md5"], *(_text(row[column]) for column in other_key))
        for _, row in runs[runs["estado"] == "sucesso"].iterrows()
    }
    for index, row in runs.iterrows():
        representative = twins.get(row["configuration_md5"])
        if row["estado"] == "sucesso" or representative is None:
            continue
        if (representative, *(_text(row[column]) for column in other_key)) in successful:
            runs.at[index, "estado"] = "equivalente"
            runs.at[index, "motivo"] = f"gemea de {representative} (eixo de alta cardinalidade inerte)"
    return runs


def check_run_counts(
    results: pd.DataFrame,
    report: CheckReport,
    expected: pd.DataFrame | None = None,
    execution_log: pd.DataFrame | None = None,
    blocks: tuple[str, ...] | None = None,
    twins: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R4: execucoes previstas x sucesso, falha, incompleta e ausente (G17).

    Devolve a tabela agregada por bloco, base, algoritmo e eixos, em unidades e
    em linhas de teste, e a lista das execucoes que nao tiveram sucesso. Sem
    ``blocks``, so os blocos presentes nos resultados entram na grade prevista.
    """
    if expected is None:
        if blocks is None:
            present = {block_of_algorithm(value) for value in results.get("algorithm", pd.Series(dtype=str))}
            blocks = tuple(block for block in RUN_BLOCKS if block in present)
        expected = expected_runs(blocks)
    runs = classify_runs(results, expected, execution_log, twins)
    for state in RUN_STATES:
        runs[f"unidades_{state}"] = (runs["estado"] == state).astype(int)
        # As linhas que faltam herdam o estado da execucao; as de sucesso
        # contam como sucesso em qualquer estado.
        runs[f"linhas_{state}"] = np.where(
            runs["estado"] == state,
            runs["linhas_teste_previstas"] - runs["linhas_teste_sucesso"],
            0,
        )
    runs["linhas_sucesso"] = runs["linhas_teste_sucesso"]
    runs["unidades_previstas"] = 1
    runs["linhas_previstas"] = runs["linhas_teste_previstas"]
    count_columns = [
        "unidades_previstas", *(f"unidades_{state}" for state in RUN_STATES),
        "linhas_previstas", *(f"linhas_{state}" for state in RUN_STATES),
    ]
    group = [column for column in RUN_COUNT_GROUP if column in runs.columns]
    counts = (
        runs.groupby(group, dropna=False)[count_columns].sum().reset_index()
        if not runs.empty else pd.DataFrame(columns=[*group, *count_columns])
    )
    failures = runs[~runs["estado"].isin(("sucesso", "equivalente"))][[
        "bloco", "dataset_id", "configuration_md5", "algorithm", "class_weight_intensity",
        "regime", "label_fraction", "label_draw_seed", "undersampling_technique",
        "test_strategy", "random_seed", "estado", "motivo", "mensagem", "tentativas",
        "linhas_teste_previstas", "linhas_teste_sucesso",
    ]].reset_index(drop=True)
    totals = counts[count_columns].sum() if not counts.empty else pd.Series(0, index=count_columns)
    pending = int(totals["unidades_ausente"] + totals["unidades_incompleta"])
    # Modo legado: CSVs anteriores ao G17 nao dizem por que falharam nem
    # separam os splits.
    legacy = [
        note
        for column, note in (
            ("failure_category", "sem failure_category: motivo das falhas desconhecido"),
            ("split", "sem split: toda linha conta como teste"),
        )
        if column not in results.columns
    ]
    report.add(
        "R4", "grade", pending == 0,
        f"{int(totals['unidades_previstas'])} execucoes previstas, "
        f"{int(totals['unidades_sucesso'])} com sucesso, {int(totals['unidades_falha'])} com falha, "
        f"{int(totals['unidades_incompleta'])} incompletas, {int(totals['unidades_ausente'])} ausentes, "
        f"{int(totals['unidades_equivalente'])} equivalentes; "
        f"{int(totals['linhas_previstas'])} linhas de teste previstas, {int(totals['linhas_sucesso'])} obtidas"
        + (f" (legado: {'; '.join(legacy)})" if legacy else ""),
    )
    return counts, failures


# ---------------------------------------------------------------------------
# Apendice
# ---------------------------------------------------------------------------


def appendix_tables(manifest: pd.DataFrame, reports: dict) -> dict[str, pd.DataFrame]:
    """Faltantes, remocao de linhas e PCA por base, dobra e configuracao."""
    configurations = _manifest_with_defaults(manifest).drop_duplicates(["configuration_md5", "fold_index"])
    missing_rows, removal_rows, pca_rows = [], [], []
    for row in configurations.itertuples(index=False):
        fold_report = reports.get((row.configuration_md5, int(row.fold_index)))
        if fold_report is None:
            continue
        base = {
            "dataset_id": row.dataset_id,
            "categorical_handling_mode": row.categorical_handling_mode,
            "fill_na_strategy": row.fill_na_strategy,
            "categorical_encoding_strategy": row.categorical_encoding_strategy,
            "undersampling_technique": row.undersampling_technique,
            "test_strategy": row.test_strategy,
            "configuration_md5": row.configuration_md5,
            "fold_index": int(row.fold_index),
        }
        for split, counts in fold_report["splits"].items():
            # No k-fold o teste da dobra e vazio e so registra a etapa de saida.
            if not counts or preparation_report.STAGE_INPUT not in counts:
                continue
            missing_rows.append({
                **base, "split": split,
                "missing_cells_before": counts[preparation_report.STAGE_INPUT]["missing_cells"],
                "rows_with_missing_before": counts[preparation_report.STAGE_INPUT]["rows_with_missing"],
                "missing_cells_after": counts[preparation_report.STAGE_AFTER_IMPUTATION]["missing_cells"],
                "rows": counts[preparation_report.STAGE_OUTPUT]["rows"],
                "positives": counts[preparation_report.STAGE_OUTPUT]["positives"],
            })
        screening = fold_report.get("missing_screening")
        if screening:
            # Os contadores vem das etapas do proprio relatorio: numa base que
            # adota a decisao da base pareada, a triagem descreve a outra base.
            train_counts = fold_report["splits"].get("train", {})
            before = train_counts.get(preparation_report.STAGE_INPUT, {})
            after = train_counts.get(preparation_report.STAGE_AFTER_IMPUTATION, {})
            rows_before = before.get("rows", screening["rows_before"])
            rows_after = after.get("rows", screening["rows_after"])
            positives_before = before.get("positives", screening["positives_before"])
            positives_after = after.get("positives", screening["positives_after"])
            removal_rows.append({
                **base,
                "decision_dataset_id": screening.get("source_dataset_id") or row.dataset_id,
                "threshold_pct": screening["threshold_pct"],
                "screened_out_columns": len(screening["screened_out_columns"]),
                "rows_before": rows_before,
                "rows_after": rows_after,
                "positives_before": positives_before,
                "positives_after": positives_after,
                "positive_rate_before": positives_before / max(rows_before, 1),
                "positive_rate_after": positives_after / max(rows_after, 1),
                "train_clients_after": fold_report.get("train_unit_population_size"),
            })
        if fold_report.get("pca"):
            pca_rows.append({**base, **fold_report["pca"]})
    return {
        "appendix_missing_values.csv": pd.DataFrame(missing_rows),
        "appendix_row_removal.csv": pd.DataFrame(removal_rows),
        "appendix_pca_components.csv": pd.DataFrame(pca_rows),
        "appendix_pairing.csv": pairing_table(manifest, reports),
    }


PAIRED, NOT_PAIRED, NOT_APPLICABLE = "sim", "nao", "n/a"
PAIRING_COLUMNS = (
    "pair_type", "pair_root", "datasets", "test_strategy", "random_seed", "fold_index",
    "same_test", "same_train_without_removal", "same_train_with_removal",
    "removal_train_rows_positives_clients", "paired",
)


def _same(values) -> str:
    """"sim" se todos os valores existem e sao iguais; "n/a" sem valores."""
    values = [_text(value) for value in values]
    if not values:
        return NOT_APPLICABLE
    return PAIRED if "none" not in values and len(set(values)) == 1 else NOT_PAIRED


def _pair_groups(train_rows: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    """Pares do G16: (tipo, raiz, linhas), com ao menos duas bases presentes."""
    groups = []
    for kind, key in (("ablacao", base_dataset_id), ("unidade", unit_pair_id)):
        roots = train_rows["dataset_id"].map(key)
        for root, rows in train_rows[roots.notna()].groupby(roots[roots.notna()]):
            if rows["dataset_id"].nunique() >= 2:
                groups.append((kind, root, rows))
    return groups


def pairing_table(manifest: pd.DataFrame, reports: dict) -> pd.DataFrame:
    """Tabela de pareamento do apendice (item G16).

    Uma linha por par (variante de ablacao e sua base, ou par por unidade),
    protocolo, semente e dobra. Na ablacao, o criterio e o conjunto de linhas
    brutas (partition_md5 e train_population_md5); no par por unidade, o de
    clientes (unit_partition_md5 e train_unit_population_md5).
    """
    manifest = _manifest_with_defaults(manifest)
    for column in (
        "partition_md5", "unit_partition_md5", "train_population_md5",
        "train_unit_population_md5",
    ):
        if column not in manifest.columns:
            manifest[column] = ""
    split = manifest["split"].astype(str).str.strip()
    test_rows = manifest[split == "test"]
    train_rows = manifest[split == "train"]
    is_drop = train_rows["fill_na_strategy"] == missing_screening.DROP_INCOMPLETE_ROWS_STRATEGY
    fold_key = ["test_strategy", "random_seed", "fold_index"]
    rows = []
    for kind, root, pair_rows in _pair_groups(train_rows):
        test_column, train_column, population_column = (
            ("partition_md5", "partition_md5", "train_population_md5")
            if kind == "ablacao"
            else ("unit_partition_md5", "unit_partition_md5", "train_unit_population_md5")
        )
        datasets = sorted(pair_rows["dataset_id"].unique())
        pair_tests = test_rows[test_rows["dataset_id"].isin(datasets)]
        for key, group in pair_rows.groupby(fold_key, dropna=False):
            tests = pair_tests
            for column, value in zip(fold_key, key):
                tests = tests[tests[column] == value]
            kept = group[~is_drop.loc[group.index]]
            dropped = group[is_drop.loc[group.index]]
            drop_counts = []
            for dataset in datasets:
                dataset_rows = dropped[dropped["dataset_id"] == dataset]
                if dataset_rows.empty:
                    continue
                first = dataset_rows.iloc[0]
                fold_report = reports.get((first["configuration_md5"], int(first["fold_index"]))) or {}
                after = (
                    fold_report.get("splits", {}).get("train", {})
                    .get(preparation_report.STAGE_AFTER_IMPUTATION, {})
                )
                clients = fold_report.get("train_unit_population_size")
                drop_counts.append(
                    f"{dataset}={after.get('rows', '?')}/{after.get('positives', '?')}"
                    + (f"/{clients}" if clients is not None else "")
                )
            drop_same = (
                _same(dropped[population_column])
                if dropped["dataset_id"].nunique() >= 2 else NOT_APPLICABLE
            )
            row = {
                "pair_type": kind,
                "pair_root": root,
                "datasets": ", ".join(datasets),
                "test_strategy": key[0],
                "random_seed": key[1],
                "fold_index": int(key[2]),
                "same_test": _same(tests[test_column]),
                "same_train_without_removal": (
                    _same(kept[train_column]) if kept["dataset_id"].nunique() >= 2
                    else NOT_APPLICABLE
                ),
                "same_train_with_removal": drop_same,
                # linhas/fraudes/clientes do treino efetivo no braco de remocao
                "removal_train_rows_positives_clients": "; ".join(drop_counts),
            }
            row["paired"] = (
                NOT_PAIRED
                if NOT_PAIRED in (
                    row["same_test"], row["same_train_without_removal"], drop_same
                )
                else PAIRED
            )
            rows.append(row)
    return pd.DataFrame(rows, columns=list(PAIRING_COLUMNS))


def _markdown_table(frame: pd.DataFrame, limit: int = 60) -> str:
    if frame is None or frame.empty:
        return "_Sem dados._\n"
    shown = frame.head(limit)
    header = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "---|" * len(shown.columns)
    body = ["| " + " | ".join(map(str, values)) + " |" for values in shown.itertuples(index=False)]
    tail = [f"\n_{len(frame) - limit} linhas omitidas; ver o CSV._"] if len(frame) > limit else []
    return "\n".join([header, rule, *body, *tail]) + "\n"


APPENDIX_NOTES = {
    "appendix_inert_axes.csv": (
        "Nos Transformers, o grid varia a estratégia de alta cardinalidade "
        "(`minimum-frequency` 10 ou `hashing` 1024), que só atua em colunas "
        "categóricas de alta cardinalidade. Nas bases listadas, nenhuma coluna "
        "se enquadra: as duas estratégias geram os mesmos splits e o mesmo "
        "vocabulário (`content_md5` e `vocabulary_md5` iguais). Só o "
        "representante é executado; a gêmea conta como `equivalente` na R4 e "
        "herda os resultados por construção. Nessas bases, esse eixo não tem "
        "efeito e não deve ser lido como comparação entre estratégias."
    ),
}


def write_appendix(directory: Path, tables: dict[str, pd.DataFrame]) -> list[Path]:
    """Grava os CSVs e o resumo em Markdown, mesmo quando ha falhas."""
    directory.mkdir(parents=True, exist_ok=True)
    sections = [
        ("Faltantes antes e depois da imputação", "appendix_missing_values.csv"),
        ("Remoção de registros incompletos", "appendix_row_removal.csv"),
        ("Componentes retidos pela PCA", "appendix_pca_components.csv"),
        ("Pareamento de ablações e pares por cliente (P9, P16)", "appendix_pairing.csv"),
        ("Efeito de cada eixo nos resultados (R2)", "appendix_axis_effects.csv"),
        ("Execuções previstas e obtidas (R4)", "appendix_run_counts.csv"),
        ("Execuções sem sucesso e motivo (R4)", "appendix_run_failures.csv"),
        ("Configurações gêmeas por eixo inerte (P6)", "appendix_inert_axes.csv"),
    ]
    written = []
    # Todas as tabelas sao gravadas, vazias quando o modo nao as produz.
    for _title, name in sections:
        path = directory / name
        frame = tables.get(name)
        (frame if frame is not None else pd.DataFrame()).to_csv(path, index=False)
        written.append(path)
    lines = ["# Verificações de sanidade do pipeline\n"]
    lines.append("## Composição do `configuration_md5`\n")
    lines.extend(f"{position}. {parameter}" for position, parameter in enumerate(CONFIGURATION_HASH_PARAMETERS, 1))
    lines.append("")
    for title, name in sections:
        lines.append(f"## {title}\n")
        if name in APPENDIX_NOTES:
            lines.append(APPENDIX_NOTES[name] + "\n")
        lines.append(_markdown_table(tables.get(name)))
    summary = directory / "appendix_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    written.append(summary)
    return written
