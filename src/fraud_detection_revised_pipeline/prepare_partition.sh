#!/usr/bin/env bash
# Prepara uma parte da grade (um bloco e uma lista de bases) num manifesto
# proprio, para que varias partes rodem em paralelo no mesmo diretorio.
#
# Uso: prepare_partition.sh <diretorio-de-dados> <manifesto-da-parte> <bloco> <base>...
#
#   <bloco>  classic | boosting | ft-transformer | tabbert
#            (a deteccao de anomalias nao prepara nada: usa as configuracoes
#            nativas sem reamostragem do bloco boosting)
#
# Os arquivos preparados vao para o diretorio de dados, como sempre; so o
# manifesto e separado, porque dois processos publicando o mesmo manifesto
# perderiam linhas. Se o manifesto da parte ainda nao existe, ele comeca como
# copia do manifesto principal do diretorio: as configuracoes ja preparadas
# (por exemplo, no teste rapido) sao reaproveitadas, e nao refeitas. Rodar de
# novo o mesmo comando retoma a parte, refazendo so o que falhou ou faltou.
# No fim, merge_partition_manifests.sh junta as partes no manifesto principal.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIRECTORY="${1:?usage: prepare_partition.sh <data-directory> <partition-manifest> <block> <dataset>...}"
PARTITION_MANIFEST="${2:?missing partition manifest}"
BLOCK="${3:?missing block}"
shift 3
(($# > 0)) || { echo "List at least one dataset." >&2; exit 2; }

case "${BLOCK}" in
  classic) PREPARE_SCRIPT="prepare_lr_nb_knn_specific_experiment_datasets.sh" ;;
  boosting) PREPARE_SCRIPT="prepare_catboost_xgboost_native_experiment_datasets.sh" ;;
  ft-transformer) PREPARE_SCRIPT="prepare_ft_transformer_experiment_datasets.sh" ;;
  tabbert) PREPARE_SCRIPT="prepare_tabbert_experiment_datasets.sh" ;;
  *) echo "Unsupported block: ${BLOCK} (classic, boosting, ft-transformer, tabbert)" >&2; exit 2 ;;
esac

MAIN_MANIFEST="${DATA_DIRECTORY%/}/prepared_dataset_configurations.tsv"
if [[ ! -s "${PARTITION_MANIFEST}" && -s "${MAIN_MANIFEST}" ]]; then
  cp -- "${MAIN_MANIFEST}" "${PARTITION_MANIFEST}" || exit 3
fi

status=0
for dataset in "$@"; do
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] parte ${BLOCK}: preparando ${dataset}"
  DATASET_FILTER="${dataset}" CONFIGURATION_MANIFEST="${PARTITION_MANIFEST}" \
    "${SCRIPT_DIR}/${PREPARE_SCRIPT}" "${DATA_DIRECTORY}" || status=$?
done
echo "[$(date '+%Y-%m-%d %H:%M:%S')] parte ${BLOCK} concluida (${*}); manifesto: ${PARTITION_MANIFEST}"
exit "${status}"
