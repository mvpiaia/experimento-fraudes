#!/usr/bin/env bash
# Reune backups por lane, recusa duplicatas/lacunas no merge e valida os
# resultados contra o manifesto canonico.
# Uso: collect_and_validate_results.sh <dados> <plano> <coleta> <saida> [origem ...]

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIRECTORY="${1:?usage: collect_and_validate_results.sh <data> <plan> <collection> <output> [source ...]}"
PLAN_DIRECTORY="${2:?missing plan directory}"
COLLECTION_DIRECTORY="${3:?missing collection directory}"
OUTPUT_DIRECTORY="${4:?missing output directory}"
shift 4
sources=("$@")
if ((${#sources[@]} == 0)); then
  sources=(/mnt/network-black-hd/resultados /mnt/network-transparent-hd/resultados)
fi
DRY_RUN="${DRY_RUN:-false}"

run() {
  if [[ "${DRY_RUN}" == true ]]; then printf 'DRY-RUN'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

run mkdir -p "${COLLECTION_DIRECTORY}"
for source in "${sources[@]}"; do
  run rsync -a "${source%/}/" "${COLLECTION_DIRECTORY%/}/"
done
run "${PYTHON_COMMAND:-python3}" "${SCRIPT_DIR}/merge_machine_results.py" \
  --plan-directory "${PLAN_DIRECTORY}" --collection-directory "${COLLECTION_DIRECTORY}" \
  --output-directory "${OUTPUT_DIRECTORY}"

if [[ "${DRY_RUN}" == true ]]; then
  echo "DRY-RUN validate merged resultados_*.csv and *.jsonl against canonical manifest"
  exit 0
fi
shopt -s nullglob
results=("${OUTPUT_DIRECTORY}"/resultados_*.csv)
history=("${OUTPUT_DIRECTORY}"/*.jsonl)
shopt -u nullglob
((${#results[@]} > 0)) || { echo "No merged result CSVs in ${OUTPUT_DIRECTORY}." >&2; exit 4; }
validation=("${PYTHON_COMMAND:-python3}" "${SCRIPT_DIR}/validate_prepared_datasets.py" \
  --mode results --results "${results[@]}" \
  --manifest "${DATA_DIRECTORY%/}/prepared_dataset_configurations.tsv" \
  --prepared-data-directory "${DATA_DIRECTORY}")
((${#history[@]} == 0)) || validation+=(--history-strata "${history[@]}")
"${validation[@]}"
