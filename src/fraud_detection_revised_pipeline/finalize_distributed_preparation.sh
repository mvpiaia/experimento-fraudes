#!/usr/bin/env bash
# Reune os shards no black-hd e valida integralmente antes da execucao.
# Uso: finalize_distributed_preparation.sh <plano> <diretorio-canonico>

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLAN_DIRECTORY="${1:?usage: finalize_distributed_preparation.sh <plan> <canonical-directory>}"
CANONICAL_DIRECTORY="${2:?missing canonical directory}"
PYTHON="${PYTHON_COMMAND:-python3}"

"${PYTHON}" "${SCRIPT_DIR}/merge_distributed_preparation.py" \
  --plan-directory "${PLAN_DIRECTORY}" --destination "${CANONICAL_DIRECTORY}" || exit $?
"${PYTHON}" "${SCRIPT_DIR}/validate_prepared_datasets.py" --mode prepared \
  --prepared-data-directory "${CANONICAL_DIRECTORY}" || exit $?
"${PYTHON}" "${SCRIPT_DIR}/validate_prepared_datasets.py" --mode partition \
  --prepared-data-directory "${CANONICAL_DIRECTORY}" || exit $?
touch "${CANONICAL_DIRECTORY%/}/.distributed-preparation-validated"
echo "Distributed preparation validated and ready: ${CANONICAL_DIRECTORY}"
