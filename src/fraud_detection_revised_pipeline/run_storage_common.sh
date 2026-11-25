#!/usr/bin/env bash
# Recuperacao do armazenamento, compartilhada pelos executores.
#
# Os arquivos preparados vivem num HD (local por USB ou compartilhado pela
# rede). Quando o acesso cai no meio de uma execucao, o Python levanta
# PreparedDatasetStorageError e sai com codigo 75; estas funcoes tratam esse
# codigo como sinal para esperar o armazenamento voltar (wait_for_storage, em
# data_directory_common.sh: uma tentativa a cada 2 minutos, sem limite) e
# repetir exatamente o mesmo comando. Com UUID configurado, o disco tambem e
# remontado via udisksctl antes de cada espera. O contrato do codigo 75 e
# deliberado e nao deve ser alterado.
#
# O codigo 75 tambem sinaliza um arquivo preparado ausente ou um relatorio
# divergente, que esperar nao resolve: com o armazenamento acessivel, o passo
# e repetido uma vez e, se falhar de novo, vira falha comum.
#
# Cada executor chama configure_prepared_storage com o diretorio recebido como
# primeiro argumento antes de usar qualquer outra funcao daqui.

set -uo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/data_directory_common.sh"
source "$(dirname "${BASH_SOURCE[0]}")/execution_log_common.sh"

STORAGE_RECOVERY_TIMEOUT_SECONDS="${STORAGE_RECOVERY_TIMEOUT_SECONDS:-90}"

# Define o diretorio preparado e o disco que a recuperacao deve remontar. O
# UUID vem de STORAGE_DEVICE_UUID, de MACHINE_STORAGE_DEVICE_UUID (no
# machine_storage.conf, para o diretorio da maquina) ou do disco montado no
# diretorio. Sem UUID, o diretorio da maquina e tratado como local: nao ha
# disco a remontar (item G18; E470 e Dell rodam sobre a copia local).
configure_prepared_storage() {
  select_prepared_data_directory "${1:-}" || return 2
  require_locked_environment || return 3
  load_grid_plan || return 5
  STORAGE_IS_LOCAL=false
  local is_machine_directory=false
  if [[ -n "${MACHINE_PREPARED_DATA_DIRECTORY:-}" \
    && "${PREPARED_DATA_DIRECTORY}" == "${MACHINE_PREPARED_DATA_DIRECTORY}" ]]; then
    is_machine_directory=true
  fi
  if [[ -z "${STORAGE_DEVICE_UUID:-}" && "${is_machine_directory}" == true ]]; then
    STORAGE_DEVICE_UUID="${MACHINE_STORAGE_DEVICE_UUID:-}"
  fi
  if [[ -z "${STORAGE_DEVICE_UUID:-}" ]]; then
    STORAGE_DEVICE_UUID="$(
      timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" \
        findmnt -rn --mountpoint "${PREPARED_DATA_DIRECTORY}" -o UUID 2>/dev/null
    )" || STORAGE_DEVICE_UUID=""
  fi
  if [[ -z "${STORAGE_DEVICE_UUID}" && "${is_machine_directory}" == true ]]; then
    STORAGE_IS_LOCAL=true
    STORAGE_DEVICE_PATH=""
  elif [[ -z "${STORAGE_DEVICE_UUID}" ]]; then
    echo "Could not determine the disk UUID for ${PREPARED_DATA_DIRECTORY}." >&2
    echo "Mount the external HD there, set STORAGE_DEVICE_UUID or configure machine_storage.conf." >&2
    return 1
  else
    STORAGE_DEVICE_PATH="${STORAGE_DEVICE_PATH:-/dev/disk/by-uuid/${STORAGE_DEVICE_UUID}}"
  fi
  # Nenhum passo comeca sem o armazenamento: espera ate ele responder.
  wait_for_storage "${PREPARED_DATA_DIRECTORY}" "início da execução" || return "${STORAGE_FAILURE_EXIT_CODE}"
}

# Chamado por wait_for_storage antes de cada espera: so ha o que remontar
# quando o diretorio esta num disco com UUID.
storage_recovery_hook() {
  [[ "${STORAGE_IS_LOCAL:-false}" == true ]] && return 1
  recover_external_storage
}

# Descricao do passo em curso, para as mensagens de espera.
storage_step_description() {
  local description="bloco ${execution_log_block:-${EXECUTION_LOG_BLOCK:-?}}"
  [[ -n "${execution_log_algorithm:-}" ]] && description+=", algoritmo ${execution_log_algorithm}"
  [[ -n "${execution_log_regime:-}" ]] && description+=", regime ${execution_log_regime}"
  [[ -n "${1:-}" ]] && description+=", configuration_md5 $1"
  echo "${description}"
}

# Plano de divisao da grade entre maquinas (item G18). Com GRID_PLAN
# apontando para o diretorio do plano (grid_plan.py allocate), o runner so
# executa os pares (algoritmo, configuracao) atribuidos a esta maquina e
# interrompe se o host nao estiver no plano. Sem GRID_PLAN, roda tudo.
#
# Com GRID_LANE (um processo da maquina no plano, p.ex. dell-cpu2), o runner
# so executa os pares daquele processo e usa os artefatos dele: o manifesto do
# processo e, salvo se definidos, um diretorio de resultados e um de staging
# proprios, em lanes/<processo>/ ao lado dos scripts. Dois processos nunca
# gravam no mesmo CSV nem limpam o staging um do outro.
GRID_PLAN="${GRID_PLAN:-}"
GRID_LANE="${GRID_LANE:-}"
grid_plan_assignments=""

load_grid_plan() {
  if [[ -z "${GRID_PLAN}" ]]; then
    if [[ -n "${GRID_LANE}" ]]; then
      echo "GRID_LANE=${GRID_LANE} requires GRID_PLAN." >&2
      return 5
    fi
    return 0
  fi
  local lane_arguments=()
  [[ -n "${GRID_LANE}" ]] && lane_arguments=(--lane "${GRID_LANE}")
  grid_plan_assignments="$(mktemp "/tmp/grid_plan_assignments.XXXXXX.tsv")"
  "${PYTHON_COMMAND:-python3}" "$(dirname "${BASH_SOURCE[0]}")/grid_plan.py" assignments \
    --plan-directory "${GRID_PLAN}" "${lane_arguments[@]}" --output "${grid_plan_assignments}" || return 5
  if [[ -n "${GRID_LANE}" ]]; then
    local lane_root
    lane_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lanes/${GRID_LANE}"
    CONFIGURATION_MANIFEST="${GRID_PLAN%/}/lanes/${GRID_LANE}/manifest.tsv"
    RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${lane_root}/results}"
    LOCAL_STAGING_DIRECTORY="${LOCAL_STAGING_DIRECTORY:-${lane_root}/staging}"
    EXECUTION_LOG_FILE="${EXECUTION_LOG_FILE:-${RESULTS_DIRECTORY}/registro_execucoes.csv}"
    mkdir -p "${RESULTS_DIRECTORY}" "${LOCAL_STAGING_DIRECTORY}" || return 5
    echo "Lane ${GRID_LANE}: manifest ${CONFIGURATION_MANIFEST}, results ${RESULTS_DIRECTORY}, staging ${LOCAL_STAGING_DIRECTORY}"
  fi
}

# Os CSVs de resultado nunca sao gravados ao vivo num HD da rede: varios
# processos acrescentando linhas numa montagem CIFS soft arriscam linhas
# truncadas ou um arquivo perdido numa queda do Wi-Fi. O backup periodico
# (backup_lane_results.sh) copia os resultados para a rede.
NETWORK_FILESYSTEM_TYPES=" cifs smb3 smbfs nfs nfs4 fuse.sshfs "

require_local_results_directory() {
  local directory="${RESULTS_DIRECTORY:-.}" filesystem
  mkdir -p "${directory}" 2>/dev/null || true
  filesystem="$(timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" findmnt -n -o FSTYPE -T "${directory}" 2>/dev/null)"
  if [[ -n "${filesystem}" && "${NETWORK_FILESYSTEM_TYPES}" == *" ${filesystem} "* ]]; then
    echo "Results directory is on a network mount (${filesystem}): ${directory}" >&2
    echo "Write results to a local directory and back them up with backup_lane_results.sh." >&2
    return 1
  fi
}

# plan_assigns <algoritmo> <configuration_md5>: verdadeiro se o par e desta
# maquina, ou se nao ha plano.
plan_assigns() {
  [[ -n "${GRID_PLAN}" ]] || return 0
  grep -qxF "$1"$'\t'"${2,,}" "${grid_plan_assignments}"
}

report_assigned_elsewhere() {
  echo "Skipping algorithm=$1 configuration_md5=$2: assigned to another machine in ${GRID_PLAN}."
}

runtime_manifest=""

cleanup_runtime_manifest() {
  if [[ -n "${runtime_manifest}" && -f "${runtime_manifest}" ]]; then
    rm -f -- "${runtime_manifest}"
  fi
  if [[ -n "${grid_plan_assignments}" && -f "${grid_plan_assignments}" ]]; then
    rm -f -- "${grid_plan_assignments}"
  fi
}

is_storage_file_readable() {
  timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" head -c 1 -- "$1" >/dev/null 2>&1
}

is_expected_storage_mounted() {
  if [[ "${STORAGE_IS_LOCAL:-false}" == true ]]; then
    [[ -d "${PREPARED_DATA_DIRECTORY}" ]]
    return
  fi
  local mounted_uuid
  mounted_uuid="$(
    timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" \
      findmnt -rn -T "${PREPARED_DATA_DIRECTORY}" -o UUID 2>/dev/null
  )" || return 1
  [[ "${mounted_uuid,,}" == "${STORAGE_DEVICE_UUID,,}" ]]
}

is_storage_device_mounted() {
  local resolved_device_path
  resolved_device_path="$(readlink -f -- "${STORAGE_DEVICE_PATH}")" || return 1
  timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" \
    findmnt -rn -S "${resolved_device_path}" >/dev/null 2>&1
}

recover_external_storage() {
  if [[ "${STORAGE_IS_LOCAL:-false}" == true ]]; then
    echo "Local prepared data directory ${PREPARED_DATA_DIRECTORY}: there is no disk to remount." >&2
    return 1
  fi
  if [[ ! -e "${STORAGE_DEVICE_PATH}" ]]; then
    echo "External HD with UUID ${STORAGE_DEVICE_UUID} is not connected yet." >&2
    return 1
  fi
  if ! command -v udisksctl >/dev/null 2>&1; then
    echo "udisksctl is required to remount ${STORAGE_DEVICE_PATH}." >&2
    return 1
  fi

  if is_storage_device_mounted; then
    echo "Removing the unavailable mapping for ${STORAGE_DEVICE_PATH}." >&2
    timeout "${STORAGE_RECOVERY_TIMEOUT_SECONDS}" \
      udisksctl unmount --block-device "${STORAGE_DEVICE_PATH}" \
      >/dev/null 2>&1 || return 1
  fi

  echo "Mounting external HD ${STORAGE_DEVICE_PATH}." >&2
  timeout "${STORAGE_RECOVERY_TIMEOUT_SECONDS}" \
    udisksctl mount --block-device "${STORAGE_DEVICE_PATH}" \
    >/dev/null 2>&1 || return 1

  if ! is_expected_storage_mounted; then
    echo "The external HD was not mounted at ${PREPARED_DATA_DIRECTORY}." >&2
    return 1
  fi
  echo "External HD restored at ${PREPARED_DATA_DIRECTORY}." >&2
}

# wait_for_storage_file <arquivo> <descricao>: espera o arquivo ficar legivel.
# Com o diretorio acessivel e o arquivo inexistente, nao ha o que esperar:
# devolve 1.
wait_for_storage_file() {
  local file_path="$1" description="$2"
  local directory
  directory="$(dirname -- "${file_path}")"
  while ! is_storage_file_readable "${file_path}"; do
    if storage_directory_is_accessible "${directory}"; then
      if [[ ! -e "${file_path}" ]]; then
        echo "${description} does not exist: ${file_path}" >&2
        return 1
      fi
      # Diretorio responde, mas o arquivo nao: a rede esta instavel.
      echo "[$(storage_timestamp)] ${description} sem leitura: ${file_path}; nova tentativa em ${STORAGE_RETRY_SECONDS} s" >&2
      sleep "${STORAGE_RETRY_SECONDS}"
      continue
    fi
    wait_for_storage "${directory}" "leitura de ${description}" || return $?
  done
}

copy_manifest_locally() {
  # Primeiro passo comum a todos os runners depois de definirem
  # RESULTS_DIRECTORY: nenhuma configuracao roda com resultados na rede.
  require_local_results_directory || return 6
  runtime_manifest="$(mktemp "/tmp/${1:-prepared}_manifest.XXXXXX.tsv")"
  local status
  while true; do
    wait_for_storage_file "${CONFIGURATION_MANIFEST}" "manifesto de configurações preparadas"
    status=$?
    ((status == 0)) || return "${status}"
    if timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" \
      cp -- "${CONFIGURATION_MANIFEST}" "${runtime_manifest}" \
      && [[ -s "${runtime_manifest}" ]]; then
      return 0
    fi
    echo "[$(storage_timestamp)] não foi possível copiar o manifesto; nova tentativa em ${STORAGE_RETRY_SECONDS} s" >&2
    sleep "${STORAGE_RETRY_SECONDS}"
  done
}

# Lista os dataset_md5 (coluna 12) de um configuration_md5 (coluna 11).
configuration_dataset_md5_list() {
  awk -F $'\t' -v configuration_md5="$1" \
    'NR > 1 && tolower($11) == tolower(configuration_md5) {print $12}' \
    "${runtime_manifest}"
}

# Split preparado de um dataset_md5: o CSV, senao o .csv.zst
# (compact-prepared-storage). Sem nenhum dos dois, devolve o caminho do CSV,
# que nao existe, para as mensagens e checagens abaixo.
prepared_split_file() {
  local base="${PREPARED_DATA_DIRECTORY}/${1,,}"
  if [[ ! -f "${base}.csv" && -f "${base}.csv.zst" ]]; then
    printf '%s\n' "${base}.csv.zst"
  else
    printf '%s\n' "${base}.csv"
  fi
}

configuration_files_are_available() {
  local dataset_md5 matched=0
  while IFS= read -r dataset_md5; do
    [[ -n "${dataset_md5}" ]] || continue
    ((matched += 1))
    [[ "${dataset_md5}" =~ ^[[:xdigit:]]{32}$ ]] || return 1
    is_storage_file_readable "$(prepared_split_file "${dataset_md5}")" || return 1
  done < <(configuration_dataset_md5_list "$1")
  ((matched > 0))
}

configuration_manifest_is_complete() {
  local dataset_md5 matched=0
  while IFS= read -r dataset_md5; do
    [[ -n "${dataset_md5}" ]] || continue
    ((matched += 1))
    [[ "${dataset_md5}" =~ ^[[:xdigit:]]{32}$ ]] || return 1
    [[ -f "$(prepared_split_file "${dataset_md5}")" ]] || return 1
  done < <(configuration_dataset_md5_list "$1")
  ((matched > 0))
}

report_unavailable_configuration_files() {
  local dataset_md5 matched=0
  while IFS= read -r dataset_md5; do
    [[ -n "${dataset_md5}" ]] || continue
    ((matched += 1))
    if [[ ! "${dataset_md5}" =~ ^[[:xdigit:]]{32}$ ]]; then
      echo "Invalid dataset_md5 '${dataset_md5}' in the manifest for configuration $1." >&2
      continue
    fi
    local split_file
    split_file="$(prepared_split_file "${dataset_md5}")"
    if [[ ! -f "${split_file}" ]]; then
      echo "Manifest references a missing prepared file: ${split_file} (nor .csv.zst)" >&2
    elif ! is_storage_file_readable "${split_file}"; then
      echo "Prepared file exists but cannot be read: ${split_file}" >&2
    fi
  done < <(configuration_dataset_md5_list "$1")
  if ((matched == 0)); then
    echo "No manifest rows were found for configuration $1." >&2
  fi
}

# wait_for_configuration_files <configuration_md5>: espera os arquivos da
# configuracao. Com o armazenamento acessivel e a configuracao incompleta no
# manifesto, nao ha o que esperar: devolve 1.
wait_for_configuration_files() {
  local requested="$1" status
  while ! configuration_files_are_available "${requested}"; do
    if storage_directory_is_accessible "${PREPARED_DATA_DIRECTORY}"; then
      if ! configuration_manifest_is_complete "${requested}"; then
        echo "The prepared data directory is accessible, but the prepared configuration is incomplete." >&2
        report_unavailable_configuration_files "${requested}"
        return 1
      fi
      echo "[$(storage_timestamp)] arquivos da configuração ${requested} sem leitura; nova tentativa em ${STORAGE_RETRY_SECONDS} s" >&2
      sleep "${STORAGE_RETRY_SECONDS}"
      continue
    fi
    wait_for_storage "${PREPARED_DATA_DIRECTORY}" "$(storage_step_description "${requested}")"
    status=$?
    ((status == 0)) || return "${status}"
  done
}

# Executa um comando repetindo-o enquanto a falha for de armazenamento.
# Define command_exit_code para o chamador inspecionar. Cada tentativa vira
# uma linha do registro de execucoes (execution_log_common.sh). Um passo
# interrompido pela perda de acesso e repetido desde o inicio quando o acesso
# volta; o executor so grava linha de falha quando a falha nao e de acesso.
run_with_storage_recovery() {
  local configuration_md5="$1"
  shift
  local started_at accessible_storage_failures=0 status
  while true; do
    command_exit_code=0
    started_at="$(date --iso-8601=seconds)"
    "$@" || command_exit_code=$?
    record_execution_attempt "${configuration_md5}" "${started_at}" "${command_exit_code}"
    if ((command_exit_code == 0)); then
      return 0
    fi
    if storage_directory_is_accessible "${PREPARED_DATA_DIRECTORY:-.}" \
      && configuration_files_are_available "${configuration_md5}"; then
      if ((command_exit_code != STORAGE_FAILURE_EXIT_CODE)); then
        return 1
      fi
      # Codigo 75 com o armazenamento acessivel: arquivo lateral ausente ou
      # relatorio divergente. Repete uma vez; na segunda, e falha comum.
      ((accessible_storage_failures += 1))
      if ((accessible_storage_failures >= 2)); then
        echo "[$(storage_timestamp)] a falha não é de acesso: o armazenamento responde, mas um arquivo da configuração ${configuration_md5} está ausente ou diverge do manifesto." >&2
        report_unavailable_configuration_files "${configuration_md5}"
        return 1
      fi
      echo "[$(storage_timestamp)] código ${command_exit_code} com o armazenamento acessível; repetindo uma vez o passo: $(storage_step_description "${configuration_md5}")" >&2
      continue
    fi
    accessible_storage_failures=0
    echo "[$(storage_timestamp)] o passo perdeu o acesso ao armazenamento (código ${command_exit_code}): $(storage_step_description "${configuration_md5}")" >&2
    wait_for_storage "${PREPARED_DATA_DIRECTORY:-.}" "$(storage_step_description "${configuration_md5}")"
    status=$?
    if ((status == 0)); then
      wait_for_configuration_files "${configuration_md5}"
      status=$?
    fi
    if ((status != 0)); then
      command_exit_code="${status}"
      return 1
    fi
  done
}
