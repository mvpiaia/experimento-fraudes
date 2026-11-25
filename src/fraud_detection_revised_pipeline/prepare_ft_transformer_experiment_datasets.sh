#!/usr/bin/env bash
# Prepara a grade do FT-Transformer (modo transformer-indexed).
#
# A ausencia e resolvida pelo proprio pre-processamento do modelo (media do
# treinamento mais indicador), entao fill_na_strategy e sempre model-managed.
# O FT-Transformer usa os MESMOS protocolos de avaliacao dos demais blocos,
# para que suas metricas sejam pareaveis sobre as mesmas particoes.
#
# Uso: prepare_ft_transformer_experiment_datasets.sh <diretorio-de-destino>
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/prepare_grid_common.sh"
prepare_grid_run ft-transformer "${1:-}"
