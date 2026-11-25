#!/usr/bin/env bash
# Benchmark de inferencia (item G18): roda so no Acer, para que os tempos de
# todos os algoritmos e bases venham da mesma maquina.
#
# Uso: run_inference_benchmark.sh <diretorio-de-dados> [algoritmo ...]
#
# Para cada algoritmo e base, ajusta o modelo uma vez numa configuracao de
# referencia e grava em resultados_benchmark_inferencia.csv o tempo por 1.000
# transacoes (lote) e a latencia unitaria (p50, p95 e IC do p95). Variaveis:
#   RESULTS_DIRECTORY     (ao lado do script)
#   LATENCY_SAMPLE_SIZE   (1000; ver benchmark_inference.py)
#   BATCH_REPEATS         (5)

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/data_directory_common.sh"
select_prepared_data_directory "${1:-}" || exit 2
shift
require_locked_environment || exit 3

PYTHON_COMMAND="${PYTHON_COMMAND:-python3}"
RESULTS_DIRECTORY="${RESULTS_DIRECTORY:-${SCRIPT_DIR}}"
ALGORITHMS=("${@:-all}")

"${PYTHON_COMMAND}" "${SCRIPT_DIR}/benchmark_inference.py" \
  --prepared-data-directory "${PREPARED_DATA_DIRECTORY}" \
  --prepared-dataset-manifest "${CONFIGURATION_MANIFEST}" \
  --results-directory "${RESULTS_DIRECTORY}" \
  --algorithm "${ALGORITHMS[@]}" \
  --latency-sample-size "${LATENCY_SAMPLE_SIZE:-1000}" \
  --batch-repeats "${BATCH_REPEATS:-5}"
