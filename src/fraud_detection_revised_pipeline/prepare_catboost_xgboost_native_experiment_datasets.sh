#!/usr/bin/env bash
# Prepara a grade de CatBoost/XGBoost com tratamento nativo de categorias.
#
# Aqui fill_na_strategy="native" significa deixar o proprio estimador tratar a
# ausencia, e nao remover linhas; "none" fica so para as bases sem faltantes.
# Sao coisas diferentes e por isso tem nomes diferentes nos dois blocos (item
# G16 da revisao). A reamostragem usa a mesma
# razao 1:1 do RUS disponivel no bloco classico, de modo que as duas familias
# tenham uma estrategia comum (item G10f).
#
# Uso: prepare_catboost_xgboost_native_experiment_datasets.sh <diretorio-de-destino>
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/prepare_grid_common.sh"
prepare_grid_run boosting "${1:-}"
