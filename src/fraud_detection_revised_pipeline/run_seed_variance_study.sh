#!/usr/bin/env bash
# Estudo de variancia por semente de preparacao (item G13 da revisao).
#
# Uso: run_seed_variance_study.sh <diretorio-de-dados>
#
# A grade principal roda so com a semente 42. Rodar tudo com varias sementes
# multiplicaria um custo que ja e de dias; em vez disso, este script prepara e
# executa um CORTE reduzido da grade com varias sementes e resume a dispersao
# entre elas. Se a dispersao for pequena diante das diferencas relatadas no
# Capitulo 5, os resultados com a semente 42 se sustentam.
#
# Manifesto e resultados sao proprios do estudo: nada da grade principal e
# lido ou alterado. Os runners percorrem o manifesto inteiro, entao um
# manifesto separado e o que restringe a execucao ao corte.
#
# Variaveis (padrao entre parenteses):
#   STUDY_DATASETS       bases do corte ("ulb ieee-cis-completa")
#   SEED_LIST            sementes de preparacao ("42 7 2026 13 101")
#   PROTOCOL_FILTER      protocolo ("stratified-holdout")
#   UNDERSAMPLING_FILTER reamostragem ("none", o braco que IF e DevNet usam)
#   FILL_NA_FILTER       imputacao ("median"; bases sem ausentes mantem "none")
#   STUDY_BLOCKS         blocos ("classic boosting anomaly"; aceita tambem
#                        "ft-transformer", de custo bem maior)
#   LABEL_DRAW_SEEDS     sorteios do DevNet ("0": a variancia estudada aqui e
#                        a da preparacao, nao a do sorteio dos rotulos)
#   CONFIGURATION_MANIFEST  (<diretorio>/seed_variance_manifest.tsv)
#   RESULTS_DIRECTORY       (seed_variance_results/ ao lado do script)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/data_directory_common.sh"

# Lido antes de select_prepared_data_directory, que preencheria o manifesto
# principal como padrao.
REQUESTED_MANIFEST="${CONFIGURATION_MANIFEST:-}"
select_prepared_data_directory "${1:-}" || exit 2
PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"

STUDY_DATASETS="${STUDY_DATASETS:-ulb ieee-cis-completa}"
STUDY_BLOCKS="${STUDY_BLOCKS:-classic boosting anomaly}"
export SEED_LIST="${SEED_LIST:-42 7 2026 13 101}"
export PROTOCOL_FILTER="${PROTOCOL_FILTER:-stratified-holdout}"
export UNDERSAMPLING_FILTER="${UNDERSAMPLING_FILTER:-none}"
export FILL_NA_FILTER="${FILL_NA_FILTER:-median}"
export LABEL_DRAW_SEEDS="${LABEL_DRAW_SEEDS:-0}"
export CONFIGURATION_MANIFEST="${REQUESTED_MANIFEST:-${PREPARED_DATA_DIRECTORY}/seed_variance_manifest.tsv}"
export RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${SCRIPT_DIR}/seed_variance_results}"
mkdir -p "${RESULTS_DIRECTORY}"

study_uses_block() {
  [[ " ${STUDY_BLOCKS} " == *" $1 "* ]]
}

echo "Seed variance study: datasets=[${STUDY_DATASETS}] seeds=[${SEED_LIST}] blocks=[${STUDY_BLOCKS}]"
echo "Manifest: ${CONFIGURATION_MANIFEST}"
echo "Results: ${RESULTS_DIRECTORY}"

# Preparacao: uma base por vez, porque o DATASET_FILTER aceita um valor so.
# IF e DevNet consomem as configuracoes nativas do boosting.
for dataset in ${STUDY_DATASETS}; do
  export DATASET_FILTER="${dataset}"
  if study_uses_block classic; then
    "${SCRIPT_DIR}/prepare_lr_nb_knn_specific_experiment_datasets.sh" "${PREPARED_DATA_DIRECTORY}" \
      || echo "Classic preparation reported failures for ${dataset}" >&2
  fi
  if study_uses_block boosting || study_uses_block anomaly; then
    "${SCRIPT_DIR}/prepare_catboost_xgboost_native_experiment_datasets.sh" "${PREPARED_DATA_DIRECTORY}" \
      || echo "Native preparation reported failures for ${dataset}" >&2
  fi
  if study_uses_block ft-transformer; then
    "${SCRIPT_DIR}/prepare_ft_transformer_experiment_datasets.sh" "${PREPARED_DATA_DIRECTORY}" \
      || echo "FT-Transformer preparation reported failures for ${dataset}" >&2
  fi
done
unset DATASET_FILTER

# Execucao: o manifesto do estudo so contem o corte.
if study_uses_block classic; then
  "${SCRIPT_DIR}/run_lr_nb_knn_specific_prepared_experiments.sh" "${PREPARED_DATA_DIRECTORY}" all
fi
if study_uses_block boosting; then
  "${SCRIPT_DIR}/run_catboost_xgboost_native_prepared_experiments.sh" "${PREPARED_DATA_DIRECTORY}" all
fi
if study_uses_block anomaly; then
  "${SCRIPT_DIR}/run_anomaly_detection_prepared_experiments.sh" "${PREPARED_DATA_DIRECTORY}" all
fi
if study_uses_block ft-transformer; then
  "${SCRIPT_DIR}/run_ft_transformer_prepared_experiments.sh" "${PREPARED_DATA_DIRECTORY}" all
fi

"${PYTHON_COMMAND}" "${SCRIPT_DIR}/seed_variance_analysis.py" \
  --results-directory "${RESULTS_DIRECTORY}" \
  --output "${RESULTS_DIRECTORY}/seed_variance_summary.csv"
