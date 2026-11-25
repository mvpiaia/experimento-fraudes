#!/usr/bin/env bash
# Prepara a grade dos algoritmos classicos (regressoes linear e logistica,
# Naive Bayes e KNN), no modo de codificacao categorica supervisionada.
#
# Diferencas em relacao a versao anterior:
#   * as republicacoes legadas da ULB nao existem mais;
#   * a PCA passou a ser propriedade da base, nao um eixo da grade;
#   * test_data_size sumiu; o protocolo e descrito so por test_strategy;
#   * entrou a linha de base "sem reamostragem" e o RUS isolado, que da as
#     familias classica e de boosting uma estrategia comum;
#   * a grade vem de experiment_grid.py e nao repete configuracoes identicas.
#
# Uso: prepare_lr_nb_knn_specific_experiment_datasets.sh <diretorio-de-destino>
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/prepare_grid_common.sh"
prepare_grid_run classic "${1:-}"
