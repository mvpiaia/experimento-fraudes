#!/usr/bin/env bash
# Executa o TabBERT sobre as configuracoes ja preparadas.
#
# Uso: run_tabbert_prepared_experiments.sh <diretorio-de-dados> [all|mlp|lstm]
#
# Cada cabeca roda os TRES regimes de treino (item C3.19), gravados na coluna
# `regime` do resultado, com o mesmo total de epocas supervisionadas:
#   supervised-only      so o ajuste supervisionado (S, sem pre-treino);
#   mlm-before-finetune  MLM e depois ajuste supervisionado (PM->S);
#   warmup-mlm-finetune  aquecimento (1/3 das epocas), MLM e ajuste final
#                        (2/3), como sugerido pelo revisor (S->PM->S).
# O regime e parametro de execucao, nao de preparacao: nao entra no
# configuration_md5, e as duas execucoes leem os mesmos arquivos.
#
# A cabeca `linear` saiu do processo; restam `mlp` e `lstm`.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_storage_common.sh"
configure_prepared_storage "${1:-}" || exit 2
EXECUTION_LOG_BLOCK="tabbert"
shift

TRANSFORMER_SCRIPT="${SCRIPT_DIR}/run_transformer_prepared_experiments.py"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${SCRIPT_DIR}}"
LOCAL_STAGING_DIRECTORY="${LOCAL_STAGING_DIRECTORY:-${SCRIPT_DIR}/prepared_datasets}"
EXECUTION_DEVICE="${EXECUTION_DEVICE:-auto}"
FORCE_REPROCESS="${FORCE_REPROCESS:-false}"
# Limiar escolhido na validacao, maximizando F2 (item G11-i); fixed volta ao 0,5.
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-validation-f2}"
TABBERT_REGIMES="${TABBERT_REGIMES:-$("${PYTHON_COMMAND}" "${SCRIPT_DIR}/experiment_grid.py" --block tabbert --format tabbert-regimes)}"
HEAD_SELECTION="${1:-all}"

trap cleanup_runtime_manifest EXIT

case "${HEAD_SELECTION}" in
  all) models=("tabbert-mlp" "tabbert-lstm") ;;
  mlp) models=("tabbert-mlp") ;;
  lstm) models=("tabbert-lstm") ;;
  tabbert-mlp|tabbert-lstm) models=("${HEAD_SELECTION}") ;;
  *)
    echo "Unsupported head selection: ${HEAD_SELECTION}" >&2
    echo "Supported: all, mlp, lstm" >&2
    exit 2
    ;;
esac

copy_manifest_locally tabbert || exit $?

successful_execution_count=0
failed_execution_count=0
declare -A executed_configurations=()

exec 3< "${runtime_manifest}"
IFS=$'\t' read -r _header <&3
while IFS=$'\t' read -r \
  database_type \
  _dataset_filename \
  _prediction_column_name \
  _auxiliary_data_is_test_data \
  _categorical_encoding_strategy \
  _fill_na_strategy \
  _undersampling_technique \
  _undersampling_parameter_values \
  test_strategy \
  random_seed \
  configuration_md5 \
  _dataset_md5 \
  _content_md5 \
  _fold_index \
  _split_name \
  categorical_handling_mode \
  _rest <&3; do
  [[ -n "${database_type}" ]] || continue
  [[ "${categorical_handling_mode,,}" == "sequence-tokenized" ]] || continue
  # O corte proprio antigo saiu da grade: nao e pareavel com os outros blocos.
  # Suas linhas continuam no manifesto, mas so rodam com INCLUDE_LEGACY_SEQUENCE_SPLIT=true.
  if [[ "${test_strategy}" == "sequence-chronological" && "${INCLUDE_LEGACY_SEQUENCE_SPLIT:-false}" != "true" ]]; then
    continue
  fi

  if [[ ! "${configuration_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
    echo "Invalid or missing configuration_md5 for database_type=${database_type}. Rebuild the TabBERT manifest." >&2
    ((failed_execution_count += 1))
    continue
  fi
  configuration_md5="${configuration_md5,,}"
  [[ -z "${executed_configurations[${configuration_md5}]:-}" ]] || continue
  executed_configurations["${configuration_md5}"]=1

  for model in "${models[@]}"; do
    if ! plan_assigns "${model}" "${configuration_md5}"; then
      report_assigned_elsewhere "${model}" "${configuration_md5}"
      continue
    fi
    for regime in ${TABBERT_REGIMES}; do
      echo "Running configuration_md5=${configuration_md5}, database_type=${database_type}, model=${model}, regime=${regime}, seed=${random_seed}"

      command=(
        "${PYTHON_COMMAND}"
        "${TRANSFORMER_SCRIPT}"
        --model "${model}"
        --tabbert-regime "${regime}"
        --configuration-md5 "${configuration_md5}"
        --manifest "${runtime_manifest}"
        --prepared-data-directory "${PREPARED_DATA_DIRECTORY}"
        --local-staging-directory "${LOCAL_STAGING_DIRECTORY}"
        --device "${EXECUTION_DEVICE}"
        --threshold-strategy "${THRESHOLD_STRATEGY}"
        --seed "${random_seed}"
        --result-file "${RESULTS_DIRECTORY}/resultados_${model}.csv"
        --details-file "${RESULTS_DIRECTORY}/resultados_${model}_details.jsonl"
      )
      if [[ "${FORCE_REPROCESS}" == "true" ]]; then
        command+=(--force-reprocess)
      fi
      printf 'Executing command:'
      printf ' %q' "${command[@]}"
      printf '\n'

      set_execution_log_context "${model}" "" "" "${regime}"
      run_with_storage_recovery "${configuration_md5}" "${command[@]}"

      if ((command_exit_code != 0)); then
        echo "TabBERT experiment failed with exit code ${command_exit_code}: database_type=${database_type}, model=${model}, regime=${regime}" >&2
        ((failed_execution_count += 1))
        continue
      fi
      ((successful_execution_count += 1))
    done
  done
done
exec 3<&-

echo "Successful executions: ${successful_execution_count}"
echo "Failed executions: ${failed_execution_count}"
