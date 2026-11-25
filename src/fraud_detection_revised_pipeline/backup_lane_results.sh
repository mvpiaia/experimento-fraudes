#!/usr/bin/env bash
# Copia os resultados dos processos desta maquina para um HD da rede.
#
# Uso: backup_lane_results.sh <diretorio-de-backup> [diretorio-de-processos]
#
#   <diretorio-de-backup>     no HD que NAO e local desta maquina, p.ex.
#                             /mnt/network-transparent-hd/resultados no Acer e
#                             /mnt/network-black-hd/resultados no Lenovo e no Dell
#   [diretorio-de-processos]  padrao: lanes/ ao lado dos scripts
#
# Os runners gravam os CSVs no disco local (nunca ao vivo na rede); este
# script, agendado pelo cron, leva uma copia de cada processo para
# <diretorio-de-backup>/<processo>/. Essa arvore e tambem a coleta que o
# merge_machine_results.py une no fim.
#
# A copia e segura com o processo gravando: cada arquivo e copiado primeiro
# para um temporario local, do qual sai uma eventual ultima linha incompleta;
# depois vai para o HD com nome temporario e so entao e renomeado. O arquivo do
# processo nunca e alterado. Se o HD estiver fora, o script espera (de 2 em 2
# minutos, sem limite) e repete a copia.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/data_directory_common.sh"

BACKUP_ROOT="${1:?usage: backup_lane_results.sh <backup-directory> [lanes-directory]}"
LANES_ROOT="${2:-${SCRIPT_DIR}/lanes}"
BACKUP_PATTERNS=("resultados_*.csv" "resultados_*.jsonl" "registro_execucoes.csv")

# copy_snapshot <origem> <destino-local>: copia so as linhas completas.
copy_snapshot() {
  local source="$1" target="$2"
  cp -- "${source}" "${target}" || return 1
  if [[ -s "${target}" && -n "$(tail -c 1 -- "${target}")" ]]; then
    # Sem quebra de linha no fim, a ultima linha estava sendo gravada: sai da
    # copia e entra no proximo backup.
    truncate -s "$(($(stat -c %s -- "${target}") - $(tail -n 1 -- "${target}" | wc -c)))" -- "${target}"
  fi
}

publish_file() {
  local snapshot="$1" destination="$2"
  local temporary="${destination%/*}/.${destination##*/}.tmp.$$"
  cp -- "${snapshot}" "${temporary}" && mv -f -- "${temporary}" "${destination}"
  local status=$?
  ((status == 0)) || rm -f -- "${temporary}" 2>/dev/null
  return "${status}"
}

shopt -s nullglob
lane_directories=("${LANES_ROOT}"/*/results)
if ((${#lane_directories[@]} == 0)); then
  echo "No lane results under ${LANES_ROOT}."
  exit 0
fi

workspace="$(mktemp -d)"
trap 'rm -rf -- "${workspace}"' EXIT
copied=0
for results_directory in "${lane_directories[@]}"; do
  lane="$(basename "$(dirname "${results_directory}")")"
  destination_directory="${BACKUP_ROOT%/}/${lane}"
  for pattern in "${BACKUP_PATTERNS[@]}"; do
    for source in "${results_directory}"/${pattern}; do
      name="$(basename "${source}")"
      snapshot="${workspace}/${lane}.${name}"
      copy_snapshot "${source}" "${snapshot}" || { echo "Could not read ${source}" >&2; continue; }
      while true; do
        if storage_directory_is_accessible "${destination_directory}" write \
          && publish_file "${snapshot}" "${destination_directory}/${name}"; then
          ((copied += 1))
          break
        fi
        if storage_directory_is_accessible "${BACKUP_ROOT}" write; then
          echo "Could not write ${destination_directory}/${name}, although the drive is accessible." >&2
          break
        fi
        wait_for_storage "${BACKUP_ROOT}" "backup de ${lane}/${name}" write || exit "${STORAGE_FAILURE_EXIT_CODE}"
      done
    done
  done
done
echo "[$(storage_timestamp)] backup: ${copied} arquivos de ${#lane_directories[@]} processos em ${BACKUP_ROOT}"
