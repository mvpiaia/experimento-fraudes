#!/usr/bin/env bash
# Abre uma sessao tmux/cgroup para cada lane da maquina atual.
# Uso: run_machine_lanes.sh <dados> <plano> <retry|force>

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIRECTORY="${1:?usage: run_machine_lanes.sh <data-directory> <plan-directory> <retry|force>}"
PLAN_DIRECTORY="${2:?missing plan directory}"
MODE="${3:?missing mode: retry or force}"
EXECUTION_HOST="${EXECUTION_HOST:-$(hostname)}"
VENV_DIRECTORY="${VENV_DIRECTORY:-${VIRTUAL_ENV:-${HOME}/venv-mestrado}}"
LOG_DIRECTORY="${LOG_DIRECTORY:-${HOME}/logs_execucao}"
DRY_RUN="${DRY_RUN:-false}"

case "${MODE}" in
  retry) force=false ;;
  force) force=true ;;
  *) echo "Mode must be retry or force." >&2; exit 2 ;;
esac

shopt -s nullglob
lane_directories=("${PLAN_DIRECTORY%/}/lanes/${EXECUTION_HOST}-"*)
shopt -u nullglob
((${#lane_directories[@]} > 0)) || {
  echo "No lanes for ${EXECUTION_HOST} in ${PLAN_DIRECTORY}." >&2
  exit 5
}
# Variaveis que reduzem a grade (experiment_grid.py) sao lidas do ambiente sem
# aviso, e o tmux passa o ambiente global do servidor a toda sessao nova: no
# Dell, uma sobra de medicao (LABEL_FRACTIONS=0.05) fez o DevNet da fase 1
# rodar 1 de 15 bracos, e CLASS_WEIGHT_INTENSITIES=none tirou os bracos
# ponderados da regressao logistica e do boosting. Lanes de um plano rodam a
# grade inteira.
TMUX_COMMAND="${TMUX_COMMAND:-tmux}"
GRID_REDUCTION_PATTERN='^(LABEL_FRACTIONS|LABEL_DRAW_SEEDS|SEED_LIST|DATASET_FILTER|PROTOCOL_FILTER|UNDERSAMPLING_FILTER|FILL_NA_FILTER|CLASS_WEIGHT_INTENSITIES|DEFAULT_CLASS_WEIGHT_INTENSITY|TABBERT_REGIMES|THRESHOLD_STRATEGY|INCLUDE_LEGACY_SEQUENCE_SPLIT)='
reductions="$(
  { env; "${TMUX_COMMAND}" show-environment -g 2>/dev/null; } \
    | grep -E "${GRID_REDUCTION_PATTERN}" | sort -u
)"
if [[ -n "${reductions}" && "${ALLOW_GRID_REDUCTION:-false}" != true ]]; then
  echo "Grid-reducing variables in the shell or the tmux global environment:" >&2
  echo "${reductions}" >&2
  echo "Unset them (tmux set-environment -g -u NAME) or set ALLOW_GRID_REDUCTION=true." >&2
  exit 6
fi
mkdir -p "${LOG_DIRECTORY}"

for lane_directory in "${lane_directories[@]}"; do
  lane="$(basename "${lane_directory}")"
  if tmux has-session -t "=${lane}" 2>/dev/null; then
    echo "Session ${lane} already exists; leaving it untouched."
    continue
  fi
  printf -v inner 'source %q/bin/activate && cd %q && FORCE_REPROCESS=%q %q %q %q %q' \
    "${VENV_DIRECTORY}" "${SCRIPT_DIR}" "${force}" "${SCRIPT_DIR}/run_lane.sh" \
    "${DATA_DIRECTORY}" "${PLAN_DIRECTORY}" "${lane}"
  scope=(systemd-run --user --scope --unit="lane-${lane}" -p OOMPolicy=continue bash -lc "${inner}")
  printf -v command '%q ' "${scope[@]}"
  command+="2>&1 | tee $(printf '%q' "${LOG_DIRECTORY}/${lane}.log")"
  if [[ "${DRY_RUN}" == true ]]; then
    printf 'DRY-RUN tmux new-session -d -s %q %q\n' "${lane}" "${command}"
  else
    tmux new-session -d -s "${lane}" "${command}"
  fi
done
