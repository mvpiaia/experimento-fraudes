#!/usr/bin/env bash
# Prepara a grade do TabBERT (modo sequence-tokenized).
#
# O TabBERT usa os MESMOS protocolos e a mesma semente dos outros blocos, para
# ser pareado com eles (item G16): temporal-holdout onde a base tem ordem
# temporal, e stratified-holdout agrupado por cliente na Home Credit. As
# janelas causais usam como historico os registros anteriores do titular nos
# splits anteriores; titular e ordem ficam nos arquivos para monta-las, mas o
# esquema de atributos os exclui do modelo.
#
# Só entram as bases que tem titular e ordem. A ULB fica de fora: suas linhas
# sao transacoes anonimas, sem portador, e nao ha sequencia a modelar.
# `home-credit-completa-adjusted`, presente na grade anterior, tambem saiu, porque
# acrescentava uma linha sintetica por cliente -- manipulacao da base.
#
# Na IEEE-CIS o titular deixou de ser `card1`, que e informacao do cartao e
# nao do portador, e passou a ser a combinacao `card1` + `addr1` (item G8-iii).
#
# Nao ha eixo de reamostragem: RUS e SMOTE quebrariam as janelas, retirando ou
# inventando posicoes no meio do historico de um titular.
#
# Uso: prepare_tabbert_experiment_datasets.sh <diretorio-de-destino>
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/prepare_grid_common.sh"
prepare_grid_run tabbert "${1:-}"
