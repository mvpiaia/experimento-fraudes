#!/usr/bin/env bash
# Traz para o Acer os resultados de uma maquina (item G18), pela rede local.
#
# Uso: collect_machine_results.sh <host> <destino> <diretorio-remoto-de-resultados> <diretorio-de-coleta>
#
#   <destino>  usuario@endereco para o SSH, ou "local" (testes)
#
# Os arquivos vao para <diretorio-de-coleta>/<host>/, um diretorio por
# maquina: a coleta nunca sobrescreve os resultados de uma maquina com os de
# outra. A juncao e feita depois por merge_machine_results.py.

set -uo pipefail

HOST="${1:?usage: collect_machine_results.sh <host> <destination> <remote-results> <collection>}"
DESTINATION="${2:?missing destination}"
REMOTE_RESULTS="${3:?missing remote results directory}"
COLLECTION="${4:?missing collection directory}"

target="${COLLECTION}/${HOST}"
mkdir -p "${target}"
source_path="${REMOTE_RESULTS%/}/"
rsync_options=(-a --include='resultados_*.csv' --include='resultados_*_details.jsonl' --include='resultados_*_history_strata.jsonl' --include='resultados_*_test_scores.jsonl' --exclude='*')
if [[ "${DESTINATION}" != "local" ]]; then
  source_path="${DESTINATION}:${source_path}"
  rsync_options+=(-e "ssh -o BatchMode=yes")
fi
rsync "${rsync_options[@]}" "${source_path}" "${target}/" || exit 10
echo "Collected $(find "${target}" -maxdepth 1 -name 'resultados_*.csv' | wc -l) result files from ${HOST} into ${target}."
