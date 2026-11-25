#!/usr/bin/env bash
# Envia para uma maquina as configuracoes preparadas que o plano lhe atribuiu
# (item G18). Roda no Acer, onde o HD da preparacao fica montado. Com os HDs da
# rede, e a etapa de posicionamento: com destino "local", copia as
# configuracoes do Lenovo do black-hd para o transparent-hd, que e local para
# ele (ambos montados no Acer). Com usuario@endereco, a copia vai por rsync
# sobre SSH, como antes.
#
# Se o acesso a um dos HDs cair durante a copia, o script espera (de 2 em 2
# minutos, sem limite) e repete a etapa interrompida; o rsync retoma o que ja
# foi copiado.
#
# Uso: stage_machine_configurations.sh <diretorio-do-plano> <host> <destino> <diretorio-remoto>
#
#   <host>             execution_host da maquina no plano (grid_plan.json)
#   <destino>          usuario@endereco para o SSH, ou "local" para copiar
#                      num diretorio desta maquina (testes)
#   <diretorio-remoto> o MACHINE_PREPARED_DATA_DIRECTORY daquela maquina
#
# Etapas: confere o espaco livre no destino; envia os arquivos de
# files_<host>.txt (rsync retoma copias interrompidas e nao reenvia o que ja
# esta la); grava manifest_<host>.tsv como o manifesto do destino; confere o
# content_md5 de cada CSV no destino com md5sum. Um arquivo divergente e
# removido e o script termina com falha. So depois da conferencia, copia o
# plano (grid_plan.json, assignments_<host>.tsv e lanes/<host>-*/, os
# processos daquela maquina) para <diretorio-remoto>/grid_plan, que e o
# GRID_PLAN a usar naquela maquina:
# a presenca do plano indica que os dados chegaram integros. Variavel:
# PREPARED_DATA_DIRECTORY (padrao: o diretorio desta maquina em
# machine_storage.conf).

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/data_directory_common.sh"

PLAN_DIRECTORY="${1:?usage: stage_machine_configurations.sh <plan> <host> <destination> <remote-directory>}"
HOST="${2:?missing host}"
DESTINATION="${3:?missing destination}"
REMOTE_DIRECTORY="${4:?missing remote directory}"
SOURCE_DIRECTORY="${PREPARED_DATA_DIRECTORY:-${MACHINE_PREPARED_DATA_DIRECTORY:-}}"
DISK_USE_LIMIT_PERCENT="${DISK_USE_LIMIT_PERCENT:-90}"
MANIFEST_NAME="prepared_dataset_configurations.tsv"
REMOTE_PLAN_NAME="grid_plan"

files_list="${PLAN_DIRECTORY}/files_${HOST}.txt"
host_manifest="${PLAN_DIRECTORY}/manifest_${HOST}.tsv"
[[ -n "${SOURCE_DIRECTORY}" ]] || { echo "Set PREPARED_DATA_DIRECTORY or machine_storage.conf." >&2; exit 2; }
[[ -f "${files_list}" && -f "${host_manifest}" ]] \
  || { echo "Host ${HOST} is not in the plan ${PLAN_DIRECTORY}." >&2; exit 5; }

# Repete um comando enquanto a falha vier da perda de acesso a um dos HDs.
with_storage_wait() {
  local description="$1" status
  shift
  while true; do
    "$@" && return 0
    status=$?
    local lost=false
    if ! storage_directory_is_accessible "${SOURCE_DIRECTORY}"; then
      wait_for_storage "${SOURCE_DIRECTORY}" "${description}" || return "${status}"
      lost=true
    fi
    if [[ "${DESTINATION}" == "local" ]] && ! storage_directory_is_accessible "${REMOTE_DIRECTORY}" write; then
      wait_for_storage "${REMOTE_DIRECTORY}" "${description}" write || return "${status}"
      lost=true
    fi
    [[ "${lost}" == true ]] || return "${status}"
  done
}

wait_for_storage "${SOURCE_DIRECTORY}" "cópia para ${HOST} (origem)" || exit 75
if [[ "${DESTINATION}" == "local" ]]; then
  wait_for_storage "${REMOTE_DIRECTORY}" "cópia para ${HOST} (destino)" write || exit 75
fi

remote() {
  if [[ "${DESTINATION}" == "local" ]]; then
    bash -c "$1"
  else
    ssh -o BatchMode=yes "${DESTINATION}" "$1"
  fi
}

remote_path() {
  if [[ "${DESTINATION}" == "local" ]]; then
    echo "${REMOTE_DIRECTORY}/$1"
  else
    echo "${DESTINATION}:${REMOTE_DIRECTORY}/$1"
  fi
}

# Manifesto sem content_md5 nao permite conferir a copia.
content_column="$(head -n 1 "${host_manifest}" | tr '\t' '\n' | grep -nx content_md5 | cut -d: -f1)"
dataset_column="$(head -n 1 "${host_manifest}" | tr '\t' '\n' | grep -nx dataset_md5 | cut -d: -f1)"
if [[ -z "${content_column}" || -z "${dataset_column}" ]]; then
  echo "The manifest has no content_md5 column; prepare the configurations again." >&2
  exit 6
fi

required_bytes=0
while IFS= read -r name; do
  [[ -n "${name}" ]] || continue
  size="$(stat -c %s -- "${SOURCE_DIRECTORY}/${name}")" || { echo "Missing source file: ${name}" >&2; exit 7; }
  required_bytes=$((required_bytes + size))
done < "${files_list}"

remote "mkdir -p '${REMOTE_DIRECTORY}'" || exit 8
present_bytes="$(remote "du -sb '${REMOTE_DIRECTORY}' | cut -f1" || echo 0)"
available_bytes="$(remote "df -B1 --output=avail '${REMOTE_DIRECTORY}' | tail -n 1" | tr -d ' ')"
needed=$((required_bytes - ${present_bytes:-0}))
if ((needed > available_bytes * DISK_USE_LIMIT_PERCENT / 100)); then
  echo "Not enough space on ${HOST}: ${needed} bytes needed, ${available_bytes} available." >&2
  exit 9
fi

echo "Copying $(wc -l < "${files_list}") files (${required_bytes} bytes) to ${HOST}."
rsync_options=(-a --partial)
[[ "${DESTINATION}" == "local" ]] || rsync_options+=(-e "ssh -o BatchMode=yes")
with_storage_wait "cópia das configurações para ${HOST}" \
  rsync "${rsync_options[@]}" --files-from="${files_list}" \
  "${SOURCE_DIRECTORY}/" "$(remote_path "")" || exit 10
with_storage_wait "cópia do manifesto para ${HOST}" \
  rsync "${rsync_options[@]}" "${host_manifest}" "$(remote_path "${MANIFEST_NAME}")" || exit 10

checksums="$(awk -F '\t' -v content="${content_column}" -v dataset="${dataset_column}" \
  'NR > 1 && $content != "" {print tolower($content) " " tolower($dataset)}' "${host_manifest}")"
# O content_md5 e o do CSV descomprimido: um split .csv.zst e conferido pelo
# zstd -dc. A saida lista os arquivos divergentes ou ausentes.
verify_script='while read -r content dataset; do
  if [ -f "$dataset.csv" ]; then name="$dataset.csv"; got="$(md5sum < "$name")"
  elif [ -f "$dataset.csv.zst" ]; then name="$dataset.csv.zst"; got="$(zstd -dcq -- "$name" | md5sum)"
  else echo "$dataset.csv"; continue; fi
  [ "${got%% *}" = "$content" ] || echo "$name"
done'
failed="$(printf '%s\n' "${checksums}" | remote "cd '${REMOTE_DIRECTORY}' && ${verify_script}")"
if [[ -n "${failed}" ]]; then
  while IFS= read -r name; do
    echo "Checksum mismatch on ${HOST}: ${name}; removing the copy." >&2
    remote "rm -f -- '${REMOTE_DIRECTORY}/${name}'"
  done <<< "${failed}"
  exit 11
fi
echo "Staged and verified $(printf '%s\n' "${checksums}" | wc -l) prepared files on ${HOST}."

# O plano vai por ultimo: os runners daquela maquina so o encontram depois
# de os dados terem sido conferidos.
# Remove somente o plano publicado anteriormente, depois de os dados terem
# sido conferidos. Assim, lanes de uma calibracao antiga nao sobrevivem a uma
# nova alocacao. Os dados preparados e seus hashes nao sao tocados.
remote "rm -rf -- '${REMOTE_DIRECTORY}/${REMOTE_PLAN_NAME}' && mkdir -p '${REMOTE_DIRECTORY}/${REMOTE_PLAN_NAME}/lanes'" || exit 12
with_storage_wait "cópia do plano para ${HOST}" \
  rsync "${rsync_options[@]}" \
  "${PLAN_DIRECTORY}/grid_plan.json" "${PLAN_DIRECTORY}/assignments_${HOST}.tsv" \
  "$(remote_path "${REMOTE_PLAN_NAME}/")" || exit 12
shopt -s nullglob
host_lanes=("${PLAN_DIRECTORY}/lanes/${HOST}-"*)
shopt -u nullglob
if ((${#host_lanes[@]} > 0)); then
  with_storage_wait "cópia dos processos de ${HOST}" \
    rsync "${rsync_options[@]}" "${host_lanes[@]}" "$(remote_path "${REMOTE_PLAN_NAME}/lanes/")" || exit 12
fi
echo "Plan copied: on ${HOST}, run with GRID_PLAN=${REMOTE_DIRECTORY}/${REMOTE_PLAN_NAME}"
