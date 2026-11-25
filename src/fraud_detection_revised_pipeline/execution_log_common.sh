#!/usr/bin/env bash
# Registro de execucoes, compartilhado pelos executores (item G17).
#
# Cada tentativa de comando de experimento acrescenta uma linha a
# registro_execucoes.csv, no diretorio de resultados. O registro existe para o
# caso que o Python nao consegue gravar: um processo morto pelo kernel (OOM
# killer, codigo 137) nao chega ao `except` e nao deixa linha de resultado. Sem
# o codigo de saida, essa execucao viraria uma "ausencia" sem explicacao na
# tabela de execucoes previstas x realizadas (preparation_sanity_checks.R4).
#
# O executor define EXECUTION_LOG_BLOCK uma vez e chama
# set_execution_log_context antes de cada comando; record_execution_attempt
# grava a linha depois que o comando termina.

EXECUTION_LOG_HEADER="timestamp_inicio,timestamp_fim,bloco,configuration_md5,algoritmo,class_weight_intensity,label_fraction,regime,tentativa,codigo_saida,categoria_saida"
EXECUTION_LOG_MEMORY_EXIT_CODE=137

execution_log_block="${EXECUTION_LOG_BLOCK:-}"
execution_log_algorithm=""
execution_log_intensity=""
execution_log_label_fraction=""
execution_log_regime=""

execution_log_path() {
  echo "${EXECUTION_LOG_FILE:-${RESULTS_DIRECTORY:-.}/registro_execucoes.csv}"
}

# Uso: set_execution_log_context ALGORITMO [INTENSIDADE] [FRACAO] [REGIME]
# Varios algoritmos ou fracoes num mesmo comando vao separados por espaco.
set_execution_log_context() {
  execution_log_algorithm="${1:-}"
  execution_log_intensity="${2:-}"
  execution_log_label_fraction="${3:-}"
  execution_log_regime="${4:-}"
}

execution_exit_category() {
  local exit_code="$1"
  if ((exit_code == 0)); then
    echo "sucesso"
  elif ((exit_code == EXECUTION_LOG_MEMORY_EXIT_CODE)); then
    echo "memoria"
  elif ((exit_code == ${STORAGE_FAILURE_EXIT_CODE:-75})); then
    echo "armazenamento"
  else
    echo "erro-python"
  fi
}

# Uso: record_execution_attempt CONFIGURATION_MD5 INICIO CODIGO_SAIDA
record_execution_attempt() {
  local configuration_md5="$1"
  local started_at="$2"
  local exit_code="$3"
  local log_path
  log_path="$(execution_log_path)"
  mkdir -p "$(dirname "${log_path}")"
  if [[ ! -s "${log_path}" ]]; then
    echo "${EXECUTION_LOG_HEADER}" > "${log_path}"
  fi
  local key="${execution_log_block:-${EXECUTION_LOG_BLOCK:-}},${configuration_md5},${execution_log_algorithm},${execution_log_intensity},${execution_log_label_fraction},${execution_log_regime}"
  # A tentativa conta as linhas anteriores da mesma execucao no registro.
  local attempt
  attempt="$(awk -F, -v key="${key}" 'NR > 1 && ($3 "," $4 "," $5 "," $6 "," $7 "," $8) == key { count++ } END { print count + 1 }' "${log_path}")"
  echo "${started_at},$(date --iso-8601=seconds),${key},${attempt},${exit_code},$(execution_exit_category "${exit_code}")" >> "${log_path}"
}
