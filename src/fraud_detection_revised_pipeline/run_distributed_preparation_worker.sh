#!/usr/bin/env bash
# Prepara e publica, uma por vez, as configuracoes atribuidas a esta maquina.
# Uso: run_distributed_preparation_worker.sh <plano> <retry|force>

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIRECTORY="${1:?usage: run_distributed_preparation_worker.sh <plan-directory> <retry|force>}"
MODE="${2:?missing mode: retry or force}"
HOST="${PREPARATION_HOST:-$(hostname)}"
PYTHON="${PYTHON_COMMAND:-python3}"
SCRATCH_ROOT="${PREPARATION_SCRATCH_ROOT:-${HOME}/preparation-scratch}"
SCRATCH_RESERVE_BYTES="${PREPARATION_SCRATCH_RESERVE_BYTES:-32212254720}"
MICRO_SHARD_BYTES="${PREPARATION_MICRO_SHARD_BYTES:-8589934592}"

case "${MODE}" in
  retry) force=false ;;
  force) force=true ;;
  *) echo "Mode must be retry or force." >&2; exit 2 ;;
esac

"${PYTHON}" "${SCRIPT_DIR}/distributed_preparation_plan.py" verify-worker \
  --plan-directory "${PLAN_DIRECTORY}" --host "${HOST}" || exit $?
SHARD_ROOT="$("${PYTHON}" "${SCRIPT_DIR}/distributed_preparation_plan.py" shard-root \
  --plan-directory "${PLAN_DIRECTORY}" --host "${HOST}")" || exit $?
ASSIGNMENTS="${PLAN_DIRECTORY%/}/preparation_assignments_${HOST}.tsv"
[[ -s "${ASSIGNMENTS}" ]] || { echo "No assignments for ${HOST}." >&2; exit 5; }
mkdir -p "${SCRATCH_ROOT}" "${SHARD_ROOT}"
scratch_free="$(df -B1 --output=avail "${SCRATCH_ROOT}" | tail -n 1 | tr -d ' ')"
scratch_required=$((SCRATCH_RESERVE_BYTES + MICRO_SHARD_BYTES))
if ((scratch_free < scratch_required)); then
  echo "Scratch has ${scratch_free} bytes free; ${scratch_required} required at ${SCRATCH_ROOT}." >&2
  exit 9
fi

# Falta de espaco no shard encerra a fila com este codigo: as configuracoes
# seguintes tambem nao teriam onde ser publicadas.
NO_SPACE_EXIT_CODE=10

# O manifesto tem linhas do md5 pedido (coluna 11)?
manifest_has_configuration() {
  awk -F $'\t' -v md5="${2,,}" \
    'NR > 1 && tolower($11) == md5 {found = 1; exit} END {exit !found}' "$1" 2>/dev/null
}

publish_scratch() {
  "${PYTHON}" "${SCRIPT_DIR}/publish_prepared_configuration.py" \
    --source-directory "$1" --manifest "$1/manifest.tsv" \
    --shard-directory "${SHARD_ROOT}" --configuration-md5 "$2"
}

# Para a fila quando o shard nao comporta a proxima configuracao.
require_configuration_space() {
  local code=0
  "${PYTHON}" "${SCRIPT_DIR}/distributed_preparation_plan.py" space \
    --plan-directory "${PLAN_DIRECTORY}" --host "${HOST}" \
    --block "$1" --dataset "$2" || code=$?
  if ((code == NO_SPACE_EXIT_CODE)); then
    echo "Stopping the queue: shard ${SHARD_ROOT} has no space for the next configuration." >&2
    exit "${NO_SPACE_EXIT_CODE}"
  fi
  return "${code}"
}

status=0
while IFS=$'\t' read -r block dataset cohort configuration_md5; do
  # Planos gerados por versoes anteriores usavam CRLF; o read do Bash deixa o
  # carriage return no ultimo campo. Aceita esses planos sem alterar os hashes.
  configuration_md5="${configuration_md5%$'\r'}"
  [[ "${block}" != block && -n "${configuration_md5}" ]] || continue
  published="${SHARD_ROOT%/}/manifests/${configuration_md5}.tsv"
  if [[ "${MODE}" == retry && -s "${published}" ]]; then
    if "${PYTHON}" "${SCRIPT_DIR}/publish_prepared_configuration.py" \
        --source-directory "${SHARD_ROOT%/}/files" --manifest "${published}" \
        --shard-directory "${SHARD_ROOT}" --configuration-md5 "${configuration_md5}" \
        --verify-only; then
      echo "Reusing published configuration ${configuration_md5} (${block}, ${dataset})."
      continue
    fi
    echo "Published configuration is invalid and will be rebuilt: ${configuration_md5}." >&2
    mv -- "${published}" "${published}.invalid-$(date '+%Y%m%d-%H%M%S')"
  fi
  require_configuration_space "${block}" "${dataset}" || { status=1; continue; }
  scratch="${SCRATCH_ROOT%/}/${configuration_md5}"
  # Uma configuracao preparada cuja publicacao falhou (por exemplo, por falta
  # de espaco) e publicada a partir do scratch, sem ser preparada de novo.
  if [[ "${MODE}" == retry ]] && manifest_has_configuration "${scratch}/manifest.tsv" "${configuration_md5}"; then
    publish_code=0
    publish_scratch "${scratch}" "${configuration_md5}" || publish_code=$?
    if ((publish_code == 0)); then
      echo "Published prepared scratch ${configuration_md5} (${block}, ${dataset}) without preparing again."
      rm -rf -- "${scratch}"
      continue
    fi
    if ((publish_code == NO_SPACE_EXIT_CODE)); then
      echo "Stopping the queue: no space to publish ${configuration_md5}; scratch kept." >&2
      exit "${NO_SPACE_EXIT_CODE}"
    fi
    echo "Scratch of ${configuration_md5} is not publishable; preparing again." >&2
  fi
  rm -rf -- "${scratch}"
  mkdir -p "${scratch}"
  manifest="${scratch}/manifest.tsv"
  echo "Preparing ${configuration_md5} on ${HOST}: ${block}, ${dataset}, cohort ${cohort}."
  if ! FORCE_RECREATE_PREPARED_DATASETS="${force}" \
      CONFIGURATION_MD5_FILTER="${configuration_md5}" \
      PREPARATION_SCRATCH_DATA_DIRECTORY="${scratch}" \
      STORAGE_ALLOW_ROOT_FILESYSTEM=true \
      "${SCRIPT_DIR}/prepare_partition.sh" "${scratch}" "${manifest}" "${block}" "${dataset}"; then
    echo "Preparation failed: ${configuration_md5}" >&2
    status=1
    continue
  fi
  # Sem linha do md5 pedido, o filtro nao achou a entrada na grade: o plano e o
  # preparador calculam identidades diferentes. Nao ha o que publicar.
  if ! manifest_has_configuration "${manifest}" "${configuration_md5}"; then
    echo "Identity mismatch: ${configuration_md5} (${block}, ${dataset}) produced no manifest rows." >&2
    rm -rf -- "${scratch}"
    status=1
    continue
  fi
  publish_code=0
  publish_scratch "${scratch}" "${configuration_md5}" || publish_code=$?
  if ((publish_code != 0)); then
    echo "Publication failed; scratch kept at ${scratch}." >&2
    if ((publish_code == NO_SPACE_EXIT_CODE)); then
      echo "Stopping the queue: shard ${SHARD_ROOT} is out of space." >&2
      exit "${NO_SPACE_EXIT_CODE}"
    fi
    # rsync pela rede nem sempre relata ENOSPC: confere o espaco de novo.
    require_configuration_space "${block}" "${dataset}" || true
    status=1
    continue
  fi
  rm -rf -- "${scratch}"
done < "${ASSIGNMENTS}"

exit "${status}"
