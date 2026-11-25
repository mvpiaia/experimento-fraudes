#!/usr/bin/env bash
# Laco de preparacao compartilhado pelos quatro blocos (classico, boosting
# nativo, FT-Transformer e TabBERT).
#
# A grade nao e mais montada em lacos aninhados de bash: ela vem de
# experiment_grid.py, que consulta o catalogo de bases e so varia um eixo onde
# o fenomeno existe. Isso elimina as configuracoes identicas que a grade
# anterior produzia na ULB, onde 2 codificacoes x 4 imputacoes geravam 8
# copias do mesmo experimento (item G9b da revisao).
#
# O manifesto mantem configuration_md5 na coluna 11 e dataset_md5 na coluna
# 12, como antes, e ganha content_md5 na 13: o MD5 dos bytes do arquivo
# gravado, que e o que permite auditar depois se o arquivo lido na execucao e
# exatamente o preparado. A coluna 23, partition_md5, identifica o split pelas
# linhas brutas que ele contem: e igual entre blocos que avaliaram as mesmas
# linhas, e e o que torna o pareamento entre familias verificavel (G16). A
# coluna 26, feature_schema_md5, e o hash das colunas do arquivo (item G14); o
# JSON <dataset_md5>.columns.json ao lado de cada CSV traz o detalhe. As colunas
# 27 a 30 vem do relatorio de preparacao da dobra (item G15): o MD5 do
# <configuration_md5>.fold<k>.preparation.json e, no drop-incomplete-rows, o
# limiar escolhido, a regra e as colunas removidas pela triagem. As colunas 31
# e 32 sao as chaves de pareamento do item G16: train_population_md5, as
# linhas brutas do treino efetivo depois da remocao de incompletos, e
# unit_partition_md5, os clientes do split nas bases com unidade de avaliacao.
# A coluna 33, train_unit_population_md5, sao os clientes do treino efetivo:
# e o que pareia o braco de remocao do par da Home Credit (G16).
#
# Uso: prepare_grid_run <bloco> <diretorio-de-destino>

set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/data_directory_common.sh"

prepare_grid_run() {
  local block="$1"
  select_prepared_data_directory "${2:-}" || return 2
  require_locked_environment || return 3
  # Os mesmos bytes das bases publicadas (item G18).
  "${PYTHON_COMMAND:-python3}" "$(dirname "${BASH_SOURCE[0]}")/verify_datasets.py" || return 4
  # Sem o HD montado, a gravacao cairia no disco do sistema: wait_for_storage
  # recusa o sistema de arquivos raiz (salvo STORAGE_ALLOW_ROOT_FILESYSTEM,
  # para quem reproduz o trabalho sem HD) e espera o HD, de 2 em 2 minutos,
  # ate ele aceitar gravacao.
  wait_for_storage "${PREPARED_DATA_DIRECTORY}" "início da preparação (bloco ${block})" write \
    || return "${STORAGE_FAILURE_EXIT_CODE}"

  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
  local preparation_script="${script_dir}/prepare_lr_nb_knn_datasets.py"
  local grid_script="${script_dir}/experiment_grid.py"
  local python_command="${PYTHON_COMMAND:-python3}"
  local prepared_data_directory="${PREPARED_DATA_DIRECTORY}"
  local configuration_manifest="${CONFIGURATION_MANIFEST}"
  local max_prepared="${MAX_PREPARED_CONFIGURATIONS:-0}"
  local configuration_filter="${CONFIGURATION_MD5_FILTER:-}"
  # true recalcula e sobrescreve toda configuracao da grade. false (padrao)
  # reaproveita a linha e os arquivos de uma configuracao ja preparada.
  local force_recreate="${FORCE_RECREATE_PREPARED_DATASETS:-false}"

  local manifest_header
  manifest_header="$(printf '%s\t' \
    "database_type" \
    "dataset_filename" \
    "prediction_column_name" \
    "auxiliary_data_is_test_data" \
    "categorical_encoding_strategy" \
    "fill_na_strategy" \
    "undersampling_technique" \
    "undersampling_parameter_values" \
    "test_strategy" \
    "random_seed" \
    "configuration_md5" \
    "dataset_md5" \
    "content_md5" \
    "fold_index" \
    "split" \
    "categorical_handling_mode" \
    "numeric_normalization" \
    "high_cardinality_strategy" \
    "high_cardinality_parameter" \
    "vocabulary_md5" \
    "vocabulary_path" \
    "window_index_path" \
    "partition_md5" \
    "dataset_id" \
    "dataset_name" \
    "feature_schema_md5" \
    "preparation_report_md5" \
    "missing_screening_threshold" \
    "missing_screening_rule" \
    "screened_out_columns" \
    "train_population_md5" \
    "unit_partition_md5" \
    "train_unit_population_md5")"
  manifest_header="${manifest_header%$'\t'}"

  # O manifesto e compartilhado pelos quatro blocos: as linhas existentes sao
  # preservadas e cada configuracao nova e acrescentada ao final. Um manifesto
  # com outro cabecalho nao e misturado com o formato atual.
  if [[ -s "${configuration_manifest}" ]]; then
    local existing_header
    IFS= read -r existing_header < "${configuration_manifest}"
    if [[ "${existing_header}" != "${manifest_header}" ]]; then
      echo "Existing manifest has a different header: ${configuration_manifest}" >&2
      echo "Move it aside before preparing with the current format." >&2
      return 1
    fi
  else
    printf '%s\n' "${manifest_header}" > "${configuration_manifest}"
  fi

  # Arquivos de trabalho ficam ao lado do manifesto, no mesmo disco, para que
  # o mv que publica cada atualizacao seja atomico.
  local configuration_rows manifest_update
  configuration_rows="$(mktemp "${configuration_manifest}.rows.XXXXXX")"
  manifest_update="$(mktemp "${configuration_manifest}.tmp.XXXXXX")"
  trap 'rm -f "${configuration_rows}" "${manifest_update}"' EXIT

  local created=0 reused=0 failed=0

  finish_manifest() {
    rm -f "${configuration_rows}" "${manifest_update}"
    trap - EXIT
    echo "Prepared configurations: ${created}"
    echo "Reused configurations: ${reused}"
    echo "Failed configurations: ${failed}"
    echo "Configuration manifest: ${configuration_manifest}"
  }

  # Linhas ja preparadas para um configuration_md5 (coluna 11).
  manifest_has_configuration() {
    awk -F $'\t' -v md5="$1" 'NR > 1 && tolower($11) == md5 {found = 1; exit} END {exit !found}' \
      "${configuration_manifest}"
  }

  # Acrescenta as linhas de uma configuracao e grava o manifesto na hora, para
  # que uma interrupcao nao perca o que ja foi preparado. Linhas antigas do
  # mesmo configuration_md5 (caso de FORCE_RECREATE) sao substituidas.
  publish_configuration_rows() {
    awk -F $'\t' -v md5="$1" 'NR == 1 || tolower($11) != md5' \
      "${configuration_manifest}" > "${manifest_update}" \
      && cat -- "${configuration_rows}" >> "${manifest_update}" \
      && mv -- "${manifest_update}" "${configuration_manifest}"
  }

  local dataset_id dataset_name database_type dataset_filename prediction_column auxiliary_is_test
  local handling_mode encoding fill_na technique parameters protocol seed
  local normalization cardinality_strategy cardinality_parameter

  while IFS=$'\t' read -r \
    dataset_id \
    dataset_name \
    dataset_filename \
    prediction_column \
    auxiliary_is_test \
    handling_mode \
    encoding \
    fill_na \
    technique \
    parameters \
    protocol \
    seed \
    normalization \
    cardinality_strategy \
    cardinality_parameter; do
    [[ -n "${dataset_id}" ]] || continue
    database_type="${dataset_id}"

    local command=(
      "${python_command}"
      "${preparation_script}"
      --dataset "${database_type}"
      --prediction-column "${prediction_column}"
      --auxiliary-data-is-test-data "${auxiliary_is_test}"
      --categorical-handling-mode "${handling_mode}"
      --categorical-encoding "${encoding}"
      --fill-na-strategy "${fill_na}"
      --undersampling-technique "${technique}"
      --undersampling-parameters "${parameters}"
      --test-strategy "${protocol}"
      --random-seed "${seed}"
      --output-directory "${prepared_data_directory}"
    )
    if [[ "${handling_mode}" == "transformer-indexed" ]]; then
      command+=(
        --numeric-normalization "${normalization}"
        --high-cardinality-strategy "${cardinality_strategy}"
        --high-cardinality-parameter "${cardinality_parameter}"
      )
    elif [[ "${handling_mode}" == "sequence-tokenized" ]]; then
      # O TabBERT discretiza todo campo numerico em faixas e monta janelas
      # causais de 15 registros: 10 de historico e 5 alvos, com passo 5.
      command+=(
        --numeric-normalization "${normalization}"
        --high-cardinality-strategy "${cardinality_strategy}"
        --high-cardinality-parameter "${cardinality_parameter}"
        --numeric-bins "${TABBERT_NUMERIC_BINS:-10}"
        --window-length "${TABBERT_WINDOW_LENGTH:-15}"
        --history-length "${TABBERT_HISTORY_LENGTH:-10}"
        --window-stride "${TABBERT_WINDOW_STRIDE:-5}"
      )
    fi

    # O worker distribuido executa uma configuracao por scratch. A identidade
    # e calculada pelo mesmo entrypoint da preparacao, antes de qualquer carga
    # ou transformacao, portanto o filtro nao muda protocolo, semente ou dobra.
    local probe_output probe_md5=""
    if [[ -n "${configuration_filter}" || "${force_recreate}" != "true" ]]; then
      probe_output="$("${command[@]}" --print-configuration-md5-only 2>&1)" || probe_output=""
      while IFS= read -r probe_line; do
        if [[ "${probe_line}" == configuration_md5=* ]]; then
          probe_md5="${probe_line#configuration_md5=}"
        fi
      done <<< "${probe_output}"
    fi
    if [[ -n "${configuration_filter}" && "${probe_md5,,}" != "${configuration_filter,,}" ]]; then
      continue
    fi

    if [[ "${force_recreate}" != "true" ]]; then
      if [[ "${probe_md5}" =~ ^[[:xdigit:]]{32}$ ]] \
        && manifest_has_configuration "${probe_md5,,}"; then
        echo "Reusing configuration_md5=${probe_md5,,}: ${database_type}, encoding=${encoding}, fillna=${fill_na}, undersampling=${technique}, protocol=${protocol}, seed=${seed}"
        ((reused += 1))
        continue
      fi
    fi

    echo "Preparing ${database_type}: encoding=${encoding}, fillna=${fill_na}, undersampling=${technique}, protocol=${protocol}, seed=${seed}"

    local command_output="" command_status step_description
    step_description="preparação de ${database_type} (bloco ${block}, encoding=${encoding}, fillna=${fill_na}, undersampling=${technique}, protocol=${protocol}, seed=${seed})"
    # Uma configuracao interrompida pela perda de acesso e refeita desde o
    # inicio quando o HD volta; os arquivos parciais sao sobrescritos, e o
    # manifesto so recebe as linhas de uma configuracao concluida.
    while true; do
      command_status=0
      command_output="$("${command[@]}" 2>&1)" || command_status=$?
      if ((command_status == 0)) \
        || storage_directory_is_accessible "${prepared_data_directory}" write; then
        break
      fi
      echo "[$(storage_timestamp)] a preparação perdeu o acesso ao armazenamento (código ${command_status}): ${step_description}" >&2
      if ! wait_for_storage "${prepared_data_directory}" "${step_description}" write; then
        printf '%s\n' "${command_output}" >&2
        ((failed += 1))
        finish_manifest
        return "${STORAGE_FAILURE_EXIT_CODE}"
      fi
    done
    if ((command_status == 0)); then
      local configuration_md5=""
      local prepared_records=()
      # vocabulary_md5/path e window_index_path variam por dobra, entao sao
      # guardados indexados pelo indice da dobra.
      local -A vocabulary_md5_by_fold=() vocabulary_path_by_fold=() window_path_by_fold=()
      # Relatorio de preparacao e triagem do drop-incomplete-rows (item G15),
      # tambem por dobra.
      local -A report_md5_by_fold=() screening_threshold_by_fold=() screening_rule_by_fold=() screened_columns_by_fold=()
      while IFS= read -r output_line; do
        echo "${output_line}"
        if [[ "${output_line}" == configuration_md5=* ]]; then
          configuration_md5="${output_line#configuration_md5=}"
        elif [[ "${output_line}" == prepared_dataset=* ]]; then
          prepared_records+=("${output_line#prepared_dataset=}")
        elif [[ "${output_line}" == prepared_artifacts=* ]]; then
          local artifact_fold artifact_md5 artifact_vocabulary artifact_window
          IFS='|' read -r artifact_fold artifact_md5 artifact_vocabulary artifact_window \
            <<< "${output_line#prepared_artifacts=}"
          vocabulary_md5_by_fold["${artifact_fold}"]="${artifact_md5}"
          vocabulary_path_by_fold["${artifact_fold}"]="${artifact_vocabulary}"
          window_path_by_fold["${artifact_fold}"]="${artifact_window}"
        elif [[ "${output_line}" == preparation_report=* ]]; then
          local report_fold _report_dataset report_md5 report_threshold report_rule report_columns
          IFS='|' read -r report_fold _report_dataset report_md5 report_threshold report_rule report_columns \
            <<< "${output_line#preparation_report=}"
          report_md5_by_fold["${report_fold}"]="${report_md5}"
          screening_threshold_by_fold["${report_fold}"]="${report_threshold}"
          screening_rule_by_fold["${report_fold}"]="${report_rule}"
          screened_columns_by_fold["${report_fold}"]="${report_columns}"
        fi
      done <<< "${command_output}"

      if [[ ! "${configuration_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
        echo "Preparation did not return a valid configuration MD5." >&2
        ((failed += 1))
        continue
      fi
      if [[ ${#prepared_records[@]} -eq 0 ]]; then
        echo "Preparation did not return any dataset file records." >&2
        ((failed += 1))
        continue
      fi

      : > "${configuration_rows}"
      local prepared_record dataset_md5 fold_index split_name content_md5 _path partition_md5 feature_schema_md5 train_population_md5 unit_partition_md5 train_unit_population_md5
      for prepared_record in "${prepared_records[@]}"; do
        IFS="|" read -r dataset_md5 fold_index split_name content_md5 _path partition_md5 feature_schema_md5 train_population_md5 unit_partition_md5 train_unit_population_md5 <<< "${prepared_record}"
        if [[ ! "${dataset_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
          echo "Preparation returned an invalid dataset MD5: ${dataset_md5}" >&2
          continue
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
          "${database_type}" \
          "${dataset_filename}" \
          "${prediction_column}" \
          "${auxiliary_is_test}" \
          "${encoding}" \
          "${fill_na}" \
          "${technique}" \
          "${parameters}" \
          "${protocol}" \
          "${seed}" \
          "${configuration_md5,,}" \
          "${dataset_md5,,}" \
          "${content_md5,,}" \
          "${fold_index}" \
          "${split_name}" \
          "${handling_mode}" \
          "${normalization}" \
          "${cardinality_strategy}" \
          "${cardinality_parameter}" \
          "${vocabulary_md5_by_fold[${fold_index}]:-}" \
          "${vocabulary_path_by_fold[${fold_index}]:-}" \
          "${window_path_by_fold[${fold_index}]:-}" \
          "${partition_md5,,}" \
          "${dataset_id}" \
          "${dataset_name}" \
          "${feature_schema_md5,,}" \
          "${report_md5_by_fold[${fold_index}]:-}" \
          "${screening_threshold_by_fold[${fold_index}]:-}" \
          "${screening_rule_by_fold[${fold_index}]:-}" \
          "${screened_columns_by_fold[${fold_index}]:-}" \
          "${train_population_md5,,}" \
          "${unit_partition_md5,,}" \
          "${train_unit_population_md5,,}" >> "${configuration_rows}"
      done
      local published=false
      while true; do
        if publish_configuration_rows "${configuration_md5,,}"; then
          published=true
          break
        fi
        storage_directory_is_accessible "${prepared_data_directory}" write && break
        wait_for_storage "${prepared_data_directory}" "gravação do manifesto (configuration_md5 ${configuration_md5,,})" write \
          || break
      done
      if [[ "${published}" != true ]]; then
        echo "Could not update the manifest for configuration_md5=${configuration_md5,,}." >&2
        ((failed += 1))
        continue
      fi
      ((created += 1))
      if ((max_prepared > 0 && created >= max_prepared)); then
        finish_manifest
        return 0
      fi
    else
      printf '%s\n' "${command_output}" >&2
      echo "Dataset preparation failed: ${database_type}, encoding=${encoding}, fillna=${fill_na}, undersampling=${technique}, protocol=${protocol}, seed=${seed}" >&2
      ((failed += 1))
    fi
  done < <("${python_command}" "${grid_script}" --block "${block}")

  finish_manifest
}
