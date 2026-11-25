#!/usr/bin/env bash
# Executa os algoritmos classicos sobre as configuracoes ja preparadas.
#
# Uso: run_lr_nb_knn_specific_prepared_experiments.sh <diretorio-de-dados> [all|algoritmo ...]
#
# A ponderacao de classe na perda e um braco do eixo de balanceamento, ao lado
# de "sem reamostragem", RUS, RUS+SMOTE e Tomek Links (itens G10 e C3.14 da
# revisao). Por isso a regressao logistica roda tres vezes so nas
# configuracoes SEM reamostragem -- none, sqrt-balanced (sqrt(R)) e balanced
# (R), com R medido no treino natural da dobra -- e uma vez, em none, nas
# reamostradas, onde o treino ja e 1:1 e os pesos so repetiriam o none.
#
# Naive Bayes e KNN rodam so em "none": o primeiro estima densidades
# condicionais e o segundo decide por votacao entre vizinhos, e nenhum tem
# nocao de peso por classe.
#
# O limiar de decisao e escolhido na validacao, maximizando F2, e aplicado ao
# teste (item G11-i; THRESHOLD_STRATEGY=fixed volta ao 0,5).
#
# Intensidade e limiar sao parametros de execucao, nao de preparacao: nao
# entram no configuration_md5, mas entram na linha de resultado e na chave que
# decide se uma execucao ja foi registrada.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_storage_common.sh"
configure_prepared_storage "${1:-}" || exit 2
EXECUTION_LOG_BLOCK="classic"
shift

SPECIFIC_SCRIPT="${SCRIPT_DIR}/lr_nb_knn_specific.py"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${SCRIPT_DIR}}"
LOCAL_STAGING_DIRECTORY="${LOCAL_STAGING_DIRECTORY:-${SCRIPT_DIR}/prepared_datasets}"
EXECUTION_DEVICE="${EXECUTION_DEVICE:-auto}"
FORCE_REPROCESS="${FORCE_REPROCESS:-false}"
# Intensidades de ponderacao de classe varridas na regressao logistica.
# Naive Bayes e KNN nao tem nocao de peso por classe e rodam so em "none".
CLASS_WEIGHT_INTENSITIES="${CLASS_WEIGHT_INTENSITIES:-none sqrt-balanced balanced}"
DEFAULT_CLASS_WEIGHT_INTENSITY="${DEFAULT_CLASS_WEIGHT_INTENSITY:-none}"
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-validation-f2}"

trap cleanup_runtime_manifest EXIT

supported_algorithms=(
  "logistic-regression"
  "naive-bayes"
  "knn"
)

validate_algorithm() {
  local requested="$1" supported
  for supported in "${supported_algorithms[@]}"; do
    [[ "${requested}" == "${supported}" ]] && return 0
  done
  echo "Unsupported algorithm: ${requested}" >&2
  echo "Supported algorithms: ${supported_algorithms[*]}" >&2
  return 1
}

# Os bracos ponderados so rodam sobre configuracoes sem reamostragem.
intensities_for_algorithm() {
  local algorithm="$1" undersampling_technique="${2,,}"
  if [[ "${algorithm}" == "logistic-regression" \
    && ( -z "${undersampling_technique}" || "${undersampling_technique}" == "none" ) ]]; then
    echo "${CLASS_WEIGHT_INTENSITIES}"
  else
    echo "${DEFAULT_CLASS_WEIGHT_INTENSITY}"
  fi
}

if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
  algorithms=("${supported_algorithms[@]}")
else
  algorithms=("$@")
  for algorithm in "${algorithms[@]}"; do
    validate_algorithm "${algorithm}" || exit 2
  done
fi

copy_manifest_locally lr_nb_knn || exit $?

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
  [[ "${categorical_handling_mode,,}" == "encoded" ]] || continue

  if [[ ! "${configuration_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
    echo "Invalid or missing configuration_md5 for database_type=${database_type}. Rebuild the manifest." >&2
    ((failed_execution_count += 1))
    continue
  fi
  configuration_md5="${configuration_md5,,}"
  [[ -z "${executed_configurations[${configuration_md5}]:-}" ]] || continue
  executed_configurations["${configuration_md5}"]=1

  for algorithm in "${algorithms[@]}"; do
    if ! plan_assigns "${algorithm}" "${configuration_md5}"; then
      report_assigned_elsewhere "${algorithm}" "${configuration_md5}"
      continue
    fi
    for class_weight_intensity in $(intensities_for_algorithm "${algorithm}" "${undersampling_technique}"); do
      echo "Running configuration_md5=${configuration_md5}, database_type=${database_type}, algorithm=${algorithm}, class_weight=${class_weight_intensity}, encoding=${categorical_encoding_strategy}, fillna=${fill_na_strategy}, undersampling=${undersampling_technique}, protocol=${test_strategy}, seed=${random_seed}"

      command=(
        "${PYTHON_COMMAND}"
        "${SPECIFIC_SCRIPT}"
        --algorithm "${algorithm}"
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
        --execution-device "${EXECUTION_DEVICE}"
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

      set_execution_log_context "${algorithm}" "${class_weight_intensity}"
      run_with_storage_recovery "${configuration_md5}" "${command[@]}"

      if ((command_exit_code != 0)); then
        echo "Experiment failed with exit code ${command_exit_code}: database_type=${database_type}, algorithm=${algorithm}, class_weight=${class_weight_intensity}" >&2
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
