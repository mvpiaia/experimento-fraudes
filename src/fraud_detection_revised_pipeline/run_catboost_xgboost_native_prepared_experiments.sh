#!/usr/bin/env bash
# Executa CatBoost/XGBoost sobre as configuracoes nativas ja preparadas.
#
# Uso: run_catboost_xgboost_native_prepared_experiments.sh <diretorio-de-dados> [all|catboost|xgboost]
#
# Filtra o manifesto pelas linhas com categorical_handling_mode=native. A
# ponderacao de classe e um braco do eixo de balanceamento (itens G10 e C3.14):
# as configuracoes SEM reamostragem rodam tres vezes por algoritmo, e as
# reamostradas (RUS), uma vez, em none:
#
#   none           auto_class_weights=None      / scale_pos_weight=1
#   sqrt-balanced  auto_class_weights=SqrtBalanced / scale_pos_weight=sqrt(R)
#   balanced       auto_class_weights=Balanced  / scale_pos_weight=R
#
# Sao os MESMOS niveis da regressao logistica (item G10f). R =
# negativos/positivos vem do treino natural de cada dobra. O limiar de decisao
# e escolhido na validacao, maximizando F2 (item G11-i; THRESHOLD_STRATEGY).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_storage_common.sh"
configure_prepared_storage "${1:-}" || exit 2
EXECUTION_LOG_BLOCK="boosting"
shift

BOOSTING_SCRIPT="${SCRIPT_DIR}/catboost_xgboost_specific.py"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${SCRIPT_DIR}}"
LOCAL_STAGING_DIRECTORY="${LOCAL_STAGING_DIRECTORY:-${SCRIPT_DIR}/prepared_datasets}"
ALGORITHM_SELECTION="${1:-all}"
FORCE_REPROCESS="${FORCE_REPROCESS:-false}"
CLASS_WEIGHT_INTENSITIES="${CLASS_WEIGHT_INTENSITIES:-none sqrt-balanced balanced}"
DEFAULT_CLASS_WEIGHT_INTENSITY="${DEFAULT_CLASS_WEIGHT_INTENSITY:-none}"
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-validation-f2}"

trap cleanup_runtime_manifest EXIT

supported_algorithm_selections=("all" "catboost" "xgboost")

validate_algorithm_selection() {
  local requested="$1" supported
  for supported in "${supported_algorithm_selections[@]}"; do
    [[ "${requested}" == "${supported}" ]] && return 0
  done
  echo "Unsupported algorithm selection: ${requested}" >&2
  echo "Supported selections: ${supported_algorithm_selections[*]}" >&2
  return 1
}

validate_algorithm_selection "${ALGORITHM_SELECTION}" || exit 2
copy_manifest_locally catboost_xgboost || exit $?

successful_execution_count=0
failed_execution_count=0
declare -A executed_configurations=()

exec 3< "${runtime_manifest}"
IFS=$'\t' read -r _header <&3
while IFS=$'\t' read -r \
  database_type \
  dataset_filename \
  prediction_column_name \
  auxiliary_data_is_test_data \
  categorical_encoding_strategy \
  fill_na_strategy \
  undersampling_technique \
  undersampling_parameter_values \
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
  [[ "${categorical_handling_mode,,}" == "native" ]] || continue
  [[ "${categorical_encoding_strategy,,}" == "none" ]] || continue

  if [[ ! "${configuration_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
    echo "Invalid or missing configuration_md5 for database_type=${database_type}. Rebuild the native prepared manifest." >&2
    ((failed_execution_count += 1))
    continue
  fi
  configuration_md5="${configuration_md5,,}"
  [[ -z "${executed_configurations[${configuration_md5}]:-}" ]] || continue
  executed_configurations["${configuration_md5}"]=1

  # Com plano de maquinas (item G18), so os algoritmos atribuidos a esta
  # maquina nesta configuracao; os dois juntos voltam a ser "all".
  configuration_algorithm="${ALGORITHM_SELECTION}"
  if [[ -n "${GRID_PLAN}" ]]; then
    planned_algorithms=()
    for candidate in catboost xgboost; do
      [[ "${ALGORITHM_SELECTION}" == "all" || "${ALGORITHM_SELECTION}" == "${candidate}" ]] || continue
      if plan_assigns "${candidate}" "${configuration_md5}"; then
        planned_algorithms+=("${candidate}")
      fi
    done
    if ((${#planned_algorithms[@]} == 0)); then
      report_assigned_elsewhere "${ALGORITHM_SELECTION}" "${configuration_md5}"
      continue
    fi
    configuration_algorithm="${planned_algorithms[0]}"
    if ((${#planned_algorithms[@]} == 2)); then
      configuration_algorithm="all"
    fi
  fi

  # Os bracos ponderados so rodam sobre configuracoes sem reamostragem.
  intensities="${DEFAULT_CLASS_WEIGHT_INTENSITY}"
  if [[ -z "${undersampling_technique}" || "${undersampling_technique,,}" == "none" ]]; then
    intensities="${CLASS_WEIGHT_INTENSITIES}"
  fi
  for class_weight_intensity in ${intensities}; do
  echo "Running native boosting configuration_md5=${configuration_md5}, database_type=${database_type}, algorithm=${configuration_algorithm}, class_weight=${class_weight_intensity}, fillna=${fill_na_strategy}, undersampling=${undersampling_technique}, protocol=${test_strategy}, seed=${random_seed}"

  command=(
    "${PYTHON_COMMAND}"
    "${BOOSTING_SCRIPT}"
    --algorithm "${configuration_algorithm}"
    --dataset-id "${database_type}"
    --dataset-filename "${dataset_filename}"
    --prediction-column-name "${prediction_column_name}"
    --auxiliary-data-is-test-data "${auxiliary_data_is_test_data}"
    --categorical-encoding-strategy "${categorical_encoding_strategy}"
    --categorical-handling-mode "${categorical_handling_mode}"
    --fill-na-strategy "${fill_na_strategy}"
    --undersampling-technique "${undersampling_technique}"
    --undersampling-parameter-values "${undersampling_parameter_values}"
    --test-strategy "${test_strategy}"
    --random-seed "${random_seed}"
    --class-weight-intensity "${class_weight_intensity}"
    --threshold-strategy "${THRESHOLD_STRATEGY}"
    --prepared-data-directory "${PREPARED_DATA_DIRECTORY}"
    --prepared-dataset-md5 "${configuration_md5}"
    --prepared-dataset-manifest "${runtime_manifest}"
    --results-directory "${RESULTS_DIRECTORY}"
    --local-staging-directory "${LOCAL_STAGING_DIRECTORY}"
  )
  if [[ "${FORCE_REPROCESS}" == "true" ]]; then
    command+=(--force-reprocess)
  fi
  printf 'Executing command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  if [[ "${ALGORITHM_SELECTION}" == "all" ]]; then
    set_execution_log_context "catboost xgboost" "${class_weight_intensity}"
  else
    set_execution_log_context "${ALGORITHM_SELECTION}" "${class_weight_intensity}"
  fi
  run_with_storage_recovery "${configuration_md5}" "${command[@]}"

  if ((command_exit_code != 0)); then
    echo "Native boosting experiment failed with exit code ${command_exit_code}: database_type=${database_type}, class_weight=${class_weight_intensity}, protocol=${test_strategy}" >&2
    ((failed_execution_count += 1))
    continue
  fi
  ((successful_execution_count += 1))
  done
done
exec 3<&-

echo "Successful executions: ${successful_execution_count}"
echo "Failed executions: ${failed_execution_count}"
