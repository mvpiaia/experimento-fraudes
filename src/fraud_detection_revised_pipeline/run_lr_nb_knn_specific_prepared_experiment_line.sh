#!/usr/bin/env bash
# Executa uma unica linha do manifesto, util para depurar uma configuracao.
#
# Uso: run_lr_nb_knn_specific_prepared_experiment_line.sh <diretorio-de-dados> <linha> [all|algoritmo ...]
# O cabecalho e a linha 1; escolha uma linha de dados a partir da 2.
#
# Como no executor completo, a regressao logistica varre as tres intensidades
# de ponderacao de classe (none, sqrt-balanced, balanced) so quando a linha nao
# tem reamostragem; Naive Bayes e KNN rodam so em "none", porque nao tem nocao
# de peso por classe. O limiar e escolhido na validacao (THRESHOLD_STRATEGY).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/data_directory_common.sh"
source "${SCRIPT_DIR}/execution_log_common.sh"
EXECUTION_LOG_BLOCK="classic"
select_prepared_data_directory "${1:-}" || exit 2
shift
SPECIFIC_SCRIPT="${SCRIPT_DIR}/lr_nb_knn_specific.py"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
EXECUTION_DEVICE="${EXECUTION_DEVICE:-auto}"
FORCE_REPROCESS="${FORCE_REPROCESS:-false}"
RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${SCRIPT_DIR}}"
LOCAL_STAGING_DIRECTORY="${LOCAL_STAGING_DIRECTORY:-${SCRIPT_DIR}/prepared_datasets}"
CLASS_WEIGHT_INTENSITIES="${CLASS_WEIGHT_INTENSITIES:-none sqrt-balanced balanced}"
DEFAULT_CLASS_WEIGHT_INTENSITY="${DEFAULT_CLASS_WEIGHT_INTENSITY:-none}"
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-validation-f2}"

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

if [[ $# -eq 0 || ! "${1}" =~ ^[0-9]+$ || "${1}" -lt 2 ]]; then
  echo "Usage: $0 <prepared-data-directory> <manifest-line-number> [all|algorithm ...]" >&2
  echo "The header is line 1; select a data line starting at 2." >&2
  exit 2
fi

manifest_line_number="$1"
shift

if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
  algorithms=("${supported_algorithms[@]}")
else
  algorithms=("$@")
  for algorithm in "${algorithms[@]}"; do
    validate_algorithm "${algorithm}" || exit 2
  done
fi

if [[ ! -f "${CONFIGURATION_MANIFEST}" ]]; then
  echo "Prepared dataset manifest not found: ${CONFIGURATION_MANIFEST}" >&2
  echo "Run prepare_lr_nb_knn_specific_experiment_datasets.sh ${PREPARED_DATA_DIRECTORY} first." >&2
  exit 1
fi

selected_manifest_line="$(sed -n "${manifest_line_number}p" "${CONFIGURATION_MANIFEST}")"
if [[ -z "${selected_manifest_line}" ]]; then
  echo "Manifest has no data at line ${manifest_line_number}: ${CONFIGURATION_MANIFEST}" >&2
  exit 1
fi

IFS=$'\t' read -r \
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
  _rest <<< "${selected_manifest_line}"

if [[ ! "${configuration_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
  echo "Invalid or missing configuration_md5 at manifest line ${manifest_line_number}." >&2
  exit 1
fi
configuration_md5="${configuration_md5,,}"

successful_execution_count=0
failed_execution_count=0

for algorithm in "${algorithms[@]}"; do
  for class_weight_intensity in $(intensities_for_algorithm "${algorithm}" "${undersampling_technique}"); do
    echo "Running manifest_line=${manifest_line_number}, configuration_md5=${configuration_md5}, database_type=${database_type}, algorithm=${algorithm}, class_weight=${class_weight_intensity}, protocol=${test_strategy}, seed=${random_seed}"

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
      --prepared-dataset-manifest "${CONFIGURATION_MANIFEST}"
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
    started_at="$(date --iso-8601=seconds)"
    command_exit_code=0
    "${command[@]}" || command_exit_code=$?
    record_execution_attempt "${configuration_md5}" "${started_at}" "${command_exit_code}"
    if ((command_exit_code == 0)); then
      ((successful_execution_count += 1))
    else
      echo "Experiment failed: manifest_line=${manifest_line_number}, algorithm=${algorithm}, class_weight=${class_weight_intensity}" >&2
      ((failed_execution_count += 1))
    fi
  done
done

echo "Successful executions: ${successful_execution_count}"
echo "Failed executions: ${failed_execution_count}"
