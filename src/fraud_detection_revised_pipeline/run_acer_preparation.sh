#!/usr/bin/env bash
# Prepara a grade canonica somente no Acer, sem repartir a criacao das
# particoes entre maquinas. Uso: run_acer_preparation.sh <dados> <retry|force>

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIRECTORY="${1:?usage: run_acer_preparation.sh <data-directory> <retry|force>}"
MODE="${2:?missing mode: retry or force}"
PARTS_DIRECTORY="${PARTS_DIRECTORY:-${DATA_DIRECTORY%/}/partes}"
DRY_RUN="${DRY_RUN:-false}"

case "${MODE}" in
  retry) FORCE_RECREATE_PREPARED_DATASETS=false ;;
  force) FORCE_RECREATE_PREPARED_DATASETS=true ;;
  *) echo "Mode must be retry or force." >&2; exit 2 ;;
esac
export FORCE_RECREATE_PREPARED_DATASETS

run() {
  if [[ "${DRY_RUN}" == true ]]; then printf 'DRY-RUN'; printf ' %q' "$@"; printf '\n'; else "$@"; fi
}

run mkdir -p "${PARTS_DIRECTORY}"
if [[ "${MODE}" == force && "${DRY_RUN}" != true ]]; then
  backup="${PARTS_DIRECTORY}/before-force-$(date '+%Y%m%d-%H%M%S')"
  mkdir -p "${backup}"
  shopt -s nullglob
  manifests=("${PARTS_DIRECTORY}"/*.tsv "${DATA_DIRECTORY%/}/prepared_dataset_configurations.tsv")
  shopt -u nullglob
  ((${#manifests[@]} == 0)) || mv -- "${manifests[@]}" "${backup}/"
fi

status=0
prepare() {
  local manifest="$1" block="$2"
  shift 2
  run "${SCRIPT_DIR}/prepare_partition.sh" "${DATA_DIRECTORY}" \
    "${PARTS_DIRECTORY}/${manifest}.tsv" "${block}" "$@" || status=$?
}

# A ordem nao muda qualquer protocolo: cada chamada usa o mesmo codigo de
# preparacao e a mesma semente; apenas evita picos concorrentes no Acer.
prepare p1 classic home-credit-completa
prepare p2 classic ieee-cis-transacional
prepare p3 classic ieee-cis-completa ieee-cis-sem-identidade
prepare p4 classic sparkov sparkov-sem-titular
prepare p5a classic home-credit-somente-aplicacao ulb
prepare p5b boosting home-credit-completa ieee-cis-transacional
prepare p6a boosting ulb home-credit-somente-aplicacao ieee-cis-completa \
  ieee-cis-completa-sem-alta-cardinalidade ieee-cis-sem-identidade sparkov sparkov-sem-titular
prepare p6b ft-transformer ulb home-credit-completa home-credit-somente-aplicacao \
  ieee-cis-completa ieee-cis-sem-identidade ieee-cis-transacional sparkov sparkov-sem-titular
prepare p6c tabbert home-credit-completa ieee-cis-completa ieee-cis-sem-identidade \
  ieee-cis-transacional sparkov sparkov-sem-titular

if ((status != 0)); then
  echo "Preparation has failures; retry the same command before merging." >&2
  exit "${status}"
fi
run "${SCRIPT_DIR}/merge_partition_manifests.sh" "${DATA_DIRECTORY}" "${PARTS_DIRECTORY}"/*.tsv
run "${PYTHON_COMMAND:-python3}" "${SCRIPT_DIR}/validate_prepared_datasets.py" \
  --mode prepared --prepared-data-directory "${DATA_DIRECTORY}"
