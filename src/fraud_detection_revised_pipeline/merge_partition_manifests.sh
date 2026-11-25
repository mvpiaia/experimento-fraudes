#!/usr/bin/env bash
# Junta os manifestos das partes preparadas em paralelo (prepare_partition.sh)
# no manifesto principal do diretorio de dados.
#
# Uso: merge_partition_manifests.sh <diretorio-de-dados> <manifesto-da-parte>...
#
# Todas as partes precisam ter o cabecalho do manifesto principal. Linhas
# repetidas (configuracoes reaproveitadas, que estao em varias partes) entram
# uma vez. O manifesto principal anterior e guardado ao lado, com a data.

set -uo pipefail

DATA_DIRECTORY="${1:?usage: merge_partition_manifests.sh <data-directory> <partition-manifest>...}"
shift
(($# > 0)) || { echo "List the partition manifests." >&2; exit 2; }
MAIN_MANIFEST="${DATA_DIRECTORY%/}/prepared_dataset_configurations.tsv"

header="$(head -n 1 -- "$1")"
for manifest in "$@"; do
  [[ -s "${manifest}" ]] || { echo "Empty or missing partition manifest: ${manifest}" >&2; exit 3; }
  if [[ "$(head -n 1 -- "${manifest}")" != "${header}" ]]; then
    echo "Partition manifest with a different header: ${manifest}" >&2
    exit 4
  fi
done

merged="$(mktemp "${MAIN_MANIFEST}.merge.XXXXXX")"
{
  printf '%s\n' "${header}"
  for manifest in "$@"; do tail -n +2 -- "${manifest}"; done | awk 'NF && !seen[$0]++'
} > "${merged}" || exit 5

# Uma configuracao com linhas diferentes em duas partes (preparada de novo com
# FORCE_RECREATE numa delas) e ambigua: para, em vez de escolher.
duplicated="$(awk -F '\t' 'NR > 1 {key = tolower($11) "|" $14 "|" $15; count[key]++} END {for (k in count) if (count[k] > 1) print k}' "${merged}")"
if [[ -n "${duplicated}" ]]; then
  echo "Configurations with conflicting rows across partitions (configuration_md5|fold|split):" >&2
  printf '%s\n' "${duplicated}" >&2
  rm -f -- "${merged}"
  exit 6
fi

if [[ -s "${MAIN_MANIFEST}" ]]; then
  cp -- "${MAIN_MANIFEST}" "${MAIN_MANIFEST%.tsv}.$(date '+%Y%m%d-%H%M%S').tsv" || exit 7
fi
mv -- "${merged}" "${MAIN_MANIFEST}" || exit 7
echo "Manifesto principal: ${MAIN_MANIFEST} ($(($(wc -l < "${MAIN_MANIFEST}") - 1)) linhas, $(awk -F '\t' 'NR > 1 {print tolower($11)}' "${MAIN_MANIFEST}" | sort -u | wc -l) configuracoes)"
