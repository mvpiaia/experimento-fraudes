#!/usr/bin/env bash
# Executa um processo (lane) do plano da grade: todos os runners em sequencia,
# cada um pulando os pares que nao sao daquele processo.
#
# Uso: run_lane.sh <diretorio-de-dados> <diretorio-do-plano> <processo>
#
#   <processo>  nome no plano, p.ex. marcospiaia-Latitude-3490-cpu2 ou
#               marcospiaia-Nitro-ANV15-41-gpu1 (ls <plano>/lanes)
#
# Um processo de GPU roda o FT-Transformer e o TabBERT; um de CPU roda o bloco
# classico, o boosting e a deteccao de anomalias. Os resultados e o staging
# ficam em lanes/<processo>/, no disco local. Rodar de novo o mesmo comando
# retoma o processo: o que ja tem sucesso e pulado. Com LANE_DRY_RUN=true, so
# imprime os comandos.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIRECTORY="${1:?usage: run_lane.sh <data-directory> <plan-directory> <lane>}"
PLAN_DIRECTORY="${2:?missing plan directory}"
LANE="${3:?missing lane}"

if [[ ! -d "${PLAN_DIRECTORY%/}/lanes/${LANE}" ]]; then
  echo "Lane ${LANE} is not in ${PLAN_DIRECTORY}. Lanes in the plan:" >&2
  ls "${PLAN_DIRECTORY%/}/lanes" >&2
  exit 5
fi

if [[ "${LANE}" == *-gpu* ]]; then
  RUNNERS=(run_ft_transformer_prepared_experiments.sh run_tabbert_prepared_experiments.sh)
else
  RUNNERS=(
    run_lr_nb_knn_specific_prepared_experiments.sh
    run_catboost_xgboost_native_prepared_experiments.sh
    run_anomaly_detection_prepared_experiments.sh
  )
fi

status=0
for runner in "${RUNNERS[@]}"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${LANE}: ${runner}"
  if [[ "${LANE_DRY_RUN:-false}" == true ]]; then
    echo "GRID_PLAN=${PLAN_DIRECTORY} GRID_LANE=${LANE} ${SCRIPT_DIR}/${runner} ${DATA_DIRECTORY}"
    continue
  fi
  GRID_PLAN="${PLAN_DIRECTORY}" GRID_LANE="${LANE}" "${SCRIPT_DIR}/${runner}" "${DATA_DIRECTORY}" || status=$?
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] ${LANE}: concluido (codigo ${status})"
exit "${status}"
