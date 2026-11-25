#!/usr/bin/env bash
# Valida os dados canonicos, cria o plano e posiciona as configuracoes do
# Lenovo. Acer e Dell compartilham o black-hd; nenhum runner prepara dados.
# Uso: build_and_stage_execution_plan.sh <dados> <calibracoes> <plano>

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIRECTORY="${1:?usage: build_and_stage_execution_plan.sh <data-directory> <calibrations> <plan>}"
CALIBRATION_DIRECTORY="${2:?missing calibration directory}"
PLAN_DIRECTORY="${3:?missing plan directory}"
REFERENCE_HOST="${REFERENCE_HOST:-marcospiaia-Nitro-ANV15-41}"
LENOVO_HOST="${LENOVO_HOST:-marcospiaia-ThinkPad-E470}"
LENOVO_DESTINATION="${LENOVO_DESTINATION:-local}"
LENOVO_DATA_DIRECTORY="${LENOVO_DATA_DIRECTORY:-/mnt/network-transparent-hd/prepared}"
DRY_RUN="${DRY_RUN:-false}"

run() {
  if [[ "${DRY_RUN}" == true ]]; then printf 'DRY-RUN'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

manifest="${DATA_DIRECTORY%/}/prepared_dataset_configurations.tsv"
if [[ "${DRY_RUN}" != true && -e "${PLAN_DIRECTORY}" ]]; then
  previous_plan="${PLAN_DIRECTORY}.before-$(date '+%Y%m%d-%H%M%S')"
  mv -- "${PLAN_DIRECTORY}" "${previous_plan}"
  echo "Previous plan preserved at ${previous_plan}."
fi
run "${PYTHON_COMMAND:-python3}" "${SCRIPT_DIR}/validate_prepared_datasets.py" \
  --mode prepared --prepared-data-directory "${DATA_DIRECTORY}" --manifest "${manifest}"
run "${PYTHON_COMMAND:-python3}" "${SCRIPT_DIR}/grid_plan.py" allocate \
  --manifest "${manifest}" --prepared-data-directory "${DATA_DIRECTORY}" \
  --calibration-directory "${CALIBRATION_DIRECTORY}" --reference-host "${REFERENCE_HOST}" \
  --plan-directory "${PLAN_DIRECTORY}"
run "${PYTHON_COMMAND:-python3}" "${SCRIPT_DIR}/estimate_execution_time.py" \
  --plan "${PLAN_DIRECTORY}"
export PREPARED_DATA_DIRECTORY="${DATA_DIRECTORY}"
run \
  "${SCRIPT_DIR}/stage_machine_configurations.sh" "${PLAN_DIRECTORY}" "${LENOVO_HOST}" \
  "${LENOVO_DESTINATION}" "${LENOVO_DATA_DIRECTORY}"
