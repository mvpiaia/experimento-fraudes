#!/usr/bin/env bash
# Espera a deduplicacao de um shard, interrompe a compactacao antes da
# compressao e inicia o retry do worker dono do shard por SSH.
#
# Uso: retry_after_dedupe.sh <log da compactacao> <sessao tmux da compactacao>
#        <diretorio do shard> <usuario@host do worker> <diretorio do plano>
#
# A compactacao deve ter sido iniciada com --dedupe-only seguido da compressao,
# gravando em <log>. Um retry nunca roda durante a compactacao do proprio shard:
# ele confere cada configuracao publicada e refaria uma pega no meio da troca
# de .csv para .csv.zst. A compressao interrompida e retomada depois do retry.
set -u
if (($# != 5)); then
  sed -n '5,6p' "$0" >&2
  exit 2
fi
LOG="$1"
SESSION="$2"
SHARD="${3%/}"
WORKER="$4"
PREP_PLAN="$5"
PIPELINE="$(cd "$(dirname "$0")" && pwd)"

say() { echo "[$(date '+%F %T')] $*"; }

say "aguardando o fim da deduplicacao em ${LOG}"
until grep -q '^dedupe:' "${LOG}"; do
  if ! tmux has-session -t "${SESSION}" 2>/dev/null; then
    say "a compactacao terminou sem o resumo da deduplicacao; nada foi iniciado"
    exit 1
  fi
  sleep 60
done
say "$(grep '^dedupe:' "${LOG}" | tail -1)"

say "interrompendo a compactacao antes da compressao"
tmux kill-session -t "${SESSION}" 2>/dev/null
while pgrep -f "compact_prepared_storage.py ${SHARD}" >/dev/null; do sleep 5; done
echo "[$(date '+%F %T')] compressao interrompida por retry_after_dedupe.sh; retomar depois do retry" >> "${LOG}"
say "compactacao parada"

if ssh -o BatchMode=yes "${WORKER}" 'tmux has-session -t preparacao-distribuida 2>/dev/null'; then
  say "${WORKER} ja tem a sessao preparacao-distribuida; nada foi iniciado"
  exit 1
fi
ssh -o BatchMode=yes "${WORKER}" "tmux new-session -d -s preparacao-distribuida \
  \"source ~/venv-mestrado/bin/activate && cd ${PIPELINE} && \
  ./run_distributed_preparation_worker.sh ${PREP_PLAN} retry 2>&1 | tee -a ~/preparacao-distribuida.log\"" \
  && say "retry iniciado em ${WORKER} (tmux preparacao-distribuida)" \
  || { say "falha ao iniciar o retry em ${WORKER}"; exit 1; }
