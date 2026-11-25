#!/usr/bin/env bash
# Executa os detectores de anomalia sobre as configuracoes nativas ja
# preparadas para o CatBoost/XGBoost (item G3 da revisao).
#
# Uso: run_anomaly_detection_prepared_experiments.sh <diretorio-de-dados> [all|isolation-forest|devnet]
#
# Nenhuma preparacao nova: o manifesto e filtrado pelas linhas com
# categorical_handling_mode=native e SEM reamostragem. RUS poria metade de
# fraudes no treino e destruiria a premissa de raridade de um detector de
# anomalias; o braco `none` preserva a prevalencia natural. As dobras sao as
# mesmas do boosting, entao os resultados se pareiam pelo partition_id.
#
# O DevNet roda uma vez por fracao de rotulos (LABEL_FRACTIONS, fracao das
# fraudes do treino que fica rotulada; 1%, 5% e 10%, como sugere a revisao) e
# por sorteio (LABEL_DRAW_SEEDS). A semente do sorteio
# e separada da semente de preparacao: a particao continua a da configuracao,
# pareavel com os demais blocos, e so muda quais fraudes ficam rotuladas.
#
# O limiar de decisao e escolhido na validacao, maximizando F2, como nos
# demais blocos (THRESHOLD_STRATEGY).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/run_storage_common.sh"
configure_prepared_storage "${1:-}" || exit 2
EXECUTION_LOG_BLOCK="anomaly"
shift

ANOMALY_SCRIPT="${SCRIPT_DIR}/anomaly_detection_specific.py"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${SCRIPT_DIR}}"
LOCAL_STAGING_DIRECTORY="${LOCAL_STAGING_DIRECTORY:-${SCRIPT_DIR}/prepared_datasets}"
ALGORITHM_SELECTION="${1:-all}"
FORCE_REPROCESS="${FORCE_REPROCESS:-false}"
THRESHOLD_STRATEGY="${THRESHOLD_STRATEGY:-validation-f2}"
# Fracoes e sementes de sorteio vem de experiment_grid.py, a mesma fonte da
# contagem de execucoes previstas (item G17); o ambiente ainda as substitui.
LABEL_FRACTIONS="${LABEL_FRACTIONS:-$("${PYTHON_COMMAND}" "${SCRIPT_DIR}/experiment_grid.py" --block anomaly --format label-fractions)}"
LABEL_DRAW_SEEDS="${LABEL_DRAW_SEEDS:-$("${PYTHON_COMMAND}" "${SCRIPT_DIR}/experiment_grid.py" --block anomaly --format label-draw-seeds)}"

trap cleanup_runtime_manifest EXIT

supported_algorithm_selections=("all" "isolation-forest" "devnet")

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
copy_manifest_locally anomaly_detection || exit $?

successful_execution_count=0
failed_execution_count=0
declare -A executed_configurations=()

# Uma execucao do script Python por configuracao e algoritmo.
run_anomaly_command() {
  local description="$1"
  shift
  echo "Running anomaly detection ${description}"
  local command=(
    "${PYTHON_COMMAND}"
    "${ANOMALY_SCRIPT}"
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
    --threshold-strategy "${THRESHOLD_STRATEGY}"
    --prepared-data-directory "${PREPARED_DATA_DIRECTORY}"
    --prepared-dataset-md5 "${configuration_md5}"
    --prepared-dataset-manifest "${runtime_manifest}"
    --results-directory "${RESULTS_DIRECTORY}"
    --local-staging-directory "${LOCAL_STAGING_DIRECTORY}"
    "$@"
  )
  if [[ "${FORCE_REPROCESS}" == "true" ]]; then
    command+=(--force-reprocess)
  fi
  printf 'Executing command:'
  printf ' %q' "${command[@]}"
  printf '\n'

  run_with_storage_recovery "${configuration_md5}" "${command[@]}"

  if ((command_exit_code != 0)); then
    echo "Anomaly detection experiment failed with exit code ${command_exit_code}: ${description}" >&2
    ((failed_execution_count += 1))
    return
  fi
  ((successful_execution_count += 1))
}

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
  [[ -z "${undersampling_technique}" || "${undersampling_technique,,}" == "none" ]] || continue

  if [[ ! "${configuration_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
    echo "Invalid or missing configuration_md5 for database_type=${database_type}. Rebuild the native prepared manifest." >&2
    ((failed_execution_count += 1))
    continue
  fi
  configuration_md5="${configuration_md5,,}"
  [[ -z "${executed_configurations[${configuration_md5}]:-}" ]] || continue
  executed_configurations["${configuration_md5}"]=1

  configuration_description="configuration_md5=${configuration_md5}, database_type=${database_type}, fillna=${fill_na_strategy}, protocol=${test_strategy}, seed=${random_seed}"

  if [[ "${ALGORITHM_SELECTION}" == "all" || "${ALGORITHM_SELECTION}" == "isolation-forest" ]] \
    && ! plan_assigns isolation-forest "${configuration_md5}"; then
    report_assigned_elsewhere isolation-forest "${configuration_md5}"
  elif [[ "${ALGORITHM_SELECTION}" == "all" || "${ALGORITHM_SELECTION}" == "isolation-forest" ]]; then
    run_anomaly_command "isolation-forest ${configuration_description}" \
      --algorithm isolation-forest
  fi

  if [[ "${ALGORITHM_SELECTION}" == "all" || "${ALGORITHM_SELECTION}" == "devnet" ]] \
    && ! plan_assigns devnet "${configuration_md5}"; then
    report_assigned_elsewhere devnet "${configuration_md5}"
  elif [[ "${ALGORITHM_SELECTION}" == "all" || "${ALGORITHM_SELECTION}" == "devnet" ]]; then
    # Uma unica chamada percorre todos os bracos sobre a mesma copia local da
    # configuracao; os ja gravados sao pulados individualmente.
    set_execution_log_context "devnet" "" "${LABEL_FRACTIONS}"
    # shellcheck disable=SC2086
    run_anomaly_command \
      "devnet label_fractions=[${LABEL_FRACTIONS}] draw_seeds=[${LABEL_DRAW_SEEDS}] ${configuration_description}" \
      --algorithm devnet \
      --label-fractions ${LABEL_FRACTIONS} \
      --label-draw-seeds ${LABEL_DRAW_SEEDS}
  fi
done
exec 3<&-

echo "Successful executions: ${successful_execution_count}"
echo "Failed executions: ${failed_execution_count}"
