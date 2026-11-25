#!/usr/bin/env bash
# Escolha do diretorio dos arquivos preparados, compartilhada pelos scripts de
# preparacao e de execucao.
#
# O diretorio e o primeiro argumento obrigatorio de todos eles e so pode ser
# um dos HDs externos abaixo. Ele define tanto onde a preparacao grava quanto
# de onde a execucao le; o manifesto fica dentro dele, salvo se
# CONFIGURATION_MANIFEST for definido explicitamente.

SUPPORTED_PREPARED_DATA_DIRECTORIES=(
  "/media/marcospiaia/dayasets"
  "/media/marcospiaia/new-datasets"
)

# Configuracao da maquina (item G18): a reexecucao roda em tres maquinas, e
# cada uma tem o proprio diretorio de dados preparados. O arquivo
# machine_storage.conf, fora do controle de versao, define
# MACHINE_PREPARED_DATA_DIRECTORY e, quando o diretorio esta num disco que
# pode ser remontado, MACHINE_STORAGE_DEVICE_UUID. Ver
# machine_storage.conf.example.
MACHINE_STORAGE_CONFIG="${MACHINE_STORAGE_CONFIG:-$(dirname "${BASH_SOURCE[0]}")/machine_storage.conf}"
if [[ -f "${MACHINE_STORAGE_CONFIG}" ]]; then
  # shellcheck disable=SC1090
  source "${MACHINE_STORAGE_CONFIG}"
fi
if [[ -n "${MACHINE_PREPARED_DATA_DIRECTORY:-}" ]]; then
  MACHINE_PREPARED_DATA_DIRECTORY="${MACHINE_PREPARED_DATA_DIRECTORY%/}"
  SUPPORTED_PREPARED_DATA_DIRECTORIES+=("${MACHINE_PREPARED_DATA_DIRECTORY}")
fi
# O worker distribuido prepara uma configuracao de cada vez num scratch local.
# A excecao e explicita e vale somente para o diretorio exato informado pelo
# proprio worker; subdiretorios arbitrarios de /home continuam recusados.
if [[ -n "${PREPARATION_SCRATCH_DATA_DIRECTORY:-}" ]]; then
  PREPARATION_SCRATCH_DATA_DIRECTORY="${PREPARATION_SCRATCH_DATA_DIRECTORY%/}"
  SUPPORTED_PREPARED_DATA_DIRECTORIES+=("${PREPARATION_SCRATCH_DATA_DIRECTORY}")
fi

# Verificacao informativa do ambiente (item G18): divergencias de bibliotecas
# apenas gravam environment_locked=false. Commit, arvore Git e versao do Python
# nao bloqueiam preparacao nem execucao.
require_locked_environment() {
  "${PYTHON_COMMAND:-python3}" \
    "$(dirname "${BASH_SOURCE[0]}")/execution_provenance.py" --check
}

prepared_data_directory_usage() {
  echo "The first argument must be the prepared data directory, one of:" >&2
  printf '  %s\n' "${SUPPORTED_PREPARED_DATA_DIRECTORIES[@]}" >&2
}

# Valida o diretorio pedido e define PREPARED_DATA_DIRECTORY e
# CONFIGURATION_MANIFEST.
select_prepared_data_directory() {
  local requested="${1:-}" supported
  requested="${requested%/}"
  for supported in "${SUPPORTED_PREPARED_DATA_DIRECTORIES[@]}"; do
    if [[ "${requested}" == "${supported}" ]]; then
      PREPARED_DATA_DIRECTORY="${requested}"
      CONFIGURATION_MANIFEST="${CONFIGURATION_MANIFEST:-${PREPARED_DATA_DIRECTORY}/prepared_dataset_configurations.tsv}"
      return 0
    fi
  done
  if [[ -n "${requested}" ]]; then
    echo "Unsupported prepared data directory: ${requested}" >&2
  fi
  prepared_data_directory_usage
  return 1
}

# --- Espera pelo armazenamento (HDs da rede) -------------------------------
# Os dados preparados ficam em HDs compartilhados pela rede (Wi-Fi). Quando o
# acesso cai, nenhum script desiste: wait_for_storage confere o diretorio a cada
# STORAGE_RETRY_SECONDS (padrao de 2 minutos), sem limite de tentativas
# (STORAGE_RETRY_ATTEMPTS=0), imprime cada tentativa que falha e, quando o
# acesso volta, avisa e devolve o controle para que o passo interrompido seja
# executado de novo desde o inicio. Os scripts nao montam a rede por conta
# propria (mount exige root): a montagem e do /etc/fstab, e o CIFS se reconecta
# sozinho quando o servidor volta.
STORAGE_RETRY_SECONDS="${STORAGE_RETRY_SECONDS:-120}"
STORAGE_RETRY_ATTEMPTS="${STORAGE_RETRY_ATTEMPTS:-0}"
STORAGE_CHECK_TIMEOUT_SECONDS="${STORAGE_CHECK_TIMEOUT_SECONDS:-10}"
STORAGE_FAILURE_EXIT_CODE="${STORAGE_FAILURE_EXIT_CODE:-75}"
# Um diretorio no sistema de arquivos raiz e, quase sempre, o ponto de
# montagem de um HD que caiu: gravar ali encheria o disco do sistema. So quem
# reproduz o trabalho sem HD (diretorio local) libera isso.
STORAGE_ALLOW_ROOT_FILESYSTEM="${STORAGE_ALLOW_ROOT_FILESYSTEM:-${MACHINE_STORAGE_ALLOW_ROOT_FILESYSTEM:-false}}"

storage_timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

storage_elapsed_text() {
  local seconds="$1"
  printf '%dh%02dmin%02ds' $((seconds / 3600)) $((seconds % 3600 / 60)) $((seconds % 60))
}

# storage_directory_is_accessible <diretorio> [write]: o diretorio responde
# dentro do tempo limite, nao esta no sistema de arquivos raiz (salvo
# liberacao) e, com "write", aceita a criacao de um arquivo.
storage_directory_is_accessible() {
  local directory="$1" mode="${2:-read}"
  if [[ "${mode}" == write && ! -e "${directory}" ]]; then
    # So cria o diretorio quando o pai esta acessivel: com o HD desmontado, o
    # mkdir criaria a pasta no disco do sistema, sob o ponto de montagem.
    local parent
    parent="$(dirname -- "${directory}")"
    [[ "${parent}" != "${directory}" ]] || return 1
    storage_directory_is_accessible "${parent}" write || return 1
    timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" mkdir -p -- "${directory}" 2>/dev/null || return 1
  fi
  timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" stat -- "${directory}/." >/dev/null 2>&1 || return 1
  if [[ "${STORAGE_ALLOW_ROOT_FILESYSTEM}" != true ]]; then
    local mount_target
    mount_target="$(timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" findmnt -n -o TARGET -T "${directory}" 2>/dev/null)" \
      || return 1
    [[ -n "${mount_target}" && "${mount_target}" != "/" ]] || return 1
  fi
  if [[ "${mode}" == write ]]; then
    local probe="${directory}/.storage_probe.$(hostname).$$"
    timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" touch -- "${probe}" 2>/dev/null || return 1
    timeout "${STORAGE_CHECK_TIMEOUT_SECONDS}" rm -f -- "${probe}" 2>/dev/null || return 1
  fi
  return 0
}

# wait_for_storage <diretorio> <descricao do passo> [write]
# Volta 0 quando o diretorio esta acessivel (na hora ou depois de esperar) e
# STORAGE_FAILURE_EXIT_CODE se STORAGE_RETRY_ATTEMPTS > 0 se esgotar. Antes de
# cada espera chama storage_recovery_hook, se o script o definir (o remonte
# por UUID dos executores). STORAGE_WAITED fica true quando houve espera.
wait_for_storage() {
  local directory="$1" step="$2" mode="${3:-read}"
  local attempt=0 started_at
  STORAGE_WAITED=false
  started_at="$(date +%s)"
  until storage_directory_is_accessible "${directory}" "${mode}"; do
    ((attempt += 1))
    STORAGE_WAITED=true
    if declare -F storage_recovery_hook >/dev/null; then
      storage_recovery_hook || true
      storage_directory_is_accessible "${directory}" "${mode}" && break
    fi
    if ((STORAGE_RETRY_ATTEMPTS > 0 && attempt >= STORAGE_RETRY_ATTEMPTS)); then
      echo "[$(storage_timestamp)] armazenamento indisponível após ${attempt} tentativas ($(storage_elapsed_text $(($(date +%s) - started_at)))): ${directory} — passo: ${step}. Encerrando." >&2
      return "${STORAGE_FAILURE_EXIT_CODE}"
    fi
    echo "[$(storage_timestamp)] armazenamento indisponível (tentativa ${attempt}): ${directory} — passo: ${step}; nova tentativa em ${STORAGE_RETRY_SECONDS} s" >&2
    sleep "${STORAGE_RETRY_SECONDS}"
  done
  if [[ "${STORAGE_WAITED}" == true ]]; then
    echo "[$(storage_timestamp)] acesso restabelecido a ${directory} após $(storage_elapsed_text $(($(date +%s) - started_at))); repetindo o passo: ${step}" >&2
  fi
  return 0
}
