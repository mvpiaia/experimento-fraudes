"""Gera o relatorio publico em HTML, pronto para deploy no Firebase Hosting.

O relatorio e gerado, e nao escrito a mao, porque quase todos os numeros vem
do proprio codigo: o catalogo de bases, a grade experimental e a estimativa de
tempo sao lidos de ``experiment_datasets``, ``experiment_grid`` e
``estimate_execution_time``. Assim o documento nao desatualiza quando a grade
muda.

    python3 build_public_report.py
    python3 build_public_report.py --output-directory ../../public

O resultado e um unico ``index.html`` autocontido: sem CSS, fonte ou script
externo, para funcionar em qualquer hospedagem estatica e sem rastrear quem o
le.
"""

from __future__ import annotations

import argparse
import html
from collections import defaultdict
from datetime import date
from pathlib import Path

import estimate_execution_time as estimator
import experiment_datasets
import experiment_grid
import lr_nb_knn_specific as pipeline
import review_items
from review_items import (
    SEVERITY_CRITICAL,
    SEVERITY_IMPORTANT,
    SEVERITY_MINOR,
    STATUS_DECISION,
    STATUS_PARTIAL,
    STATUS_RESOLVED,
    STATUS_THESIS,
)


REPORT_TITLE = "Pipeline Experimental Revisado"
REPORT_SUBTITLE = (
    "Resposta ao Relatório de Revisão da primeira versão da dissertação"
)
THESIS_TITLE = (
    "Identificação de Anomalias em Transações Financeiras: "
    "avaliação comparativa de classificadores em múltiplas bases"
)
AUTHOR = "Marcos Vinícius Piaia"
ADVISORS = "Prof. Dr. Zanoni Dias · Coorientador: Prof. Dr. Hélio Pedrini"
INSTITUTION = "Universidade Estadual de Campinas — Instituto de Computação"

STATUS_LABELS = {
    STATUS_RESOLVED: "Resolvido no pipeline",
    STATUS_PARTIAL: "Parcial",
    STATUS_THESIS: "Dissertação",
    STATUS_DECISION: "Aguarda decisão",
}
SEVERITY_ORDER = {SEVERITY_CRITICAL: 0, SEVERITY_IMPORTANT: 1, SEVERITY_MINOR: 2}

# Prevalencia da classe positiva em cada base, usada para explicar por que F1,
# F2 e precisao nao sao comparaveis entre bases.
DATASET_PREVALENCE = {
    "ulb": "0,173%",
    "home-credit-completa": "8,7%",
    "home-credit-somente-aplicacao": "8,07%",
    "sparkov": "0,579% (treino) / 0,386% (teste)",
    "ieee-cis-completa": "7,85%",
    "ieee-cis-sem-identidade": "7,85%",
    "ieee-cis-transacional": "3,50%",
    "sparkov-sem-titular": "0,579% (treino) / 0,386% (teste)",
    "ieee-cis-completa-sem-alta-cardinalidade": "7,85%",
}
DATASET_SHORT_NAME = {
    "ulb": "ULB",
    "home-credit-completa": "Home Credit (com junção)",
    "home-credit-somente-aplicacao": "Home Credit (só aplicações)",
    "sparkov": "Sparkov",
    "ieee-cis-completa": "IEEE-CIS (identidade)",
    "ieee-cis-sem-identidade": "IEEE-CIS (sem identidade)",
    "ieee-cis-transacional": "IEEE-CIS (transações)",
    "sparkov-sem-titular": "Sparkov (sem titular)",
    "ieee-cis-completa-sem-alta-cardinalidade": "IEEE-CIS (sem alta cardinalidade)",
}
DATASET_DOMAIN = {
    "ulb": "Fraude em cartão",
    "home-credit-completa": "Risco de crédito (controle)",
    "home-credit-somente-aplicacao": "Risco de crédito (controle)",
    "sparkov": "Fraude em cartão (sintética)",
    "ieee-cis-completa": "Fraude em cartão",
    "ieee-cis-sem-identidade": "Fraude em cartão",
    "ieee-cis-transacional": "Fraude em cartão",
    "sparkov-sem-titular": "Fraude em cartão (sintética)",
    "ieee-cis-completa-sem-alta-cardinalidade": "Fraude em cartão",
}

BLOCK_LABELS = {
    experiment_grid.BLOCK_CLASSIC: "Clássicos",
    experiment_grid.BLOCK_BOOSTING: "Boosting nativo",
    experiment_grid.BLOCK_FT_TRANSFORMER: "FT-Transformer",
    experiment_grid.BLOCK_TABBERT: "TabBERT",
}

# Script de preparacao e executor de cada bloco. O executor so le as linhas do
# manifesto cujo categorical_handling_mode e o do seu bloco.
BLOCK_SCRIPTS = {
    experiment_grid.BLOCK_CLASSIC: (
        "prepare_lr_nb_knn_specific_experiment_datasets.sh",
        "run_lr_nb_knn_specific_prepared_experiments.sh",
    ),
    experiment_grid.BLOCK_BOOSTING: (
        "prepare_catboost_xgboost_native_experiment_datasets.sh",
        "run_catboost_xgboost_native_prepared_experiments.sh",
    ),
    experiment_grid.BLOCK_FT_TRANSFORMER: (
        "prepare_ft_transformer_experiment_datasets.sh",
        "run_ft_transformer_prepared_experiments.sh",
    ),
    experiment_grid.BLOCK_TABBERT: (
        "prepare_tabbert_experiment_datasets.sh",
        "run_tabbert_prepared_experiments.sh",
    ),
}
BLOCK_PARTITIONS = {
    experiment_grid.BLOCK_CLASSIC: "Os três protocolos; partições comuns",
    experiment_grid.BLOCK_BOOSTING: "Os três protocolos; as mesmas partições dos clássicos",
    experiment_grid.BLOCK_FT_TRANSFORMER: "Protocolos de holdout; as mesmas partições dos demais",
    experiment_grid.BLOCK_TABBERT: "<code>temporal-holdout</code> (Home Credit: <code>stratified-holdout</code> por cliente); as mesmas partições dos demais",
}


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


# ---------------------------------------------------------------------------
# Blocos de conteudo gerados a partir do codigo
# ---------------------------------------------------------------------------


def _dataset_table_row(definition) -> str:
    database_type = definition.database_type
    protocols = ", ".join(
        protocol.replace("stratified-", "estratificado ").replace(
            "temporal-holdout", "temporal"
        )
        for protocol in definition.protocols
    )
    evaluation_unit = (
        f"por cliente (média de <code>{escape(definition.evaluation_unit_column)}</code>) e por registro"
        if definition.evaluation_unit_column
        else "por registro"
    )
    return (
        "<tr>"
        f"<td><code>{escape(database_type)}</code><br>"
        f"<span class='muted'>{escape(DATASET_SHORT_NAME[database_type])}</span></td>"
        f"<td>{escape(DATASET_DOMAIN[database_type])}</td>"
        f"<td class='num'>{estimator.DATASET_ROW_COUNT[database_type]:,}</td>".replace(
            ",", "."
        )
        + f"<td class='num'>{escape(DATASET_PREVALENCE[database_type])}</td>"
        f"<td>{'sim' if definition.applies_pca_in_classic_block else '<strong>não</strong>'}</td>"
        f"<td>{escape(definition.group_column or '—')}</td>"
        f"<td>{escape(definition.temporal_order_column or '—')}</td>"
        f"<td>{'sim' if definition.sequence_keys else 'não'}</td>"
        f"<td class='small'>{escape(protocols)}</td>"
        f"<td class='small'>{evaluation_unit}</td>"
        "</tr>"
    )


def build_dataset_table() -> str:
    """Bases de fraude primeiro; as de controle em secao propria (item G7-c).

    O dominio vem do catalogo, a mesma fonte que separa as bases de controle
    dos agregados de fraude no dashboard.
    """
    sections = (
        (experiment_datasets.DOMAIN_FRAUD, "Bases de fraude"),
        (
            experiment_datasets.DOMAIN_OUT_OF_DOMAIN_CONTROL,
            "Controle fora do domínio: relatado à parte, nunca nas médias de fraude",
        ),
    )
    rows = []
    for domain, title in sections:
        definitions = [
            definition
            for definition in experiment_datasets.DATASET_DEFINITIONS.values()
            if definition.domain == domain
        ]
        if not definitions:
            continue
        rows.append(f"<tr><th colspan='10'>{escape(title)}</th></tr>")
        rows.extend(_dataset_table_row(definition) for definition in definitions)
    return (
        "<table><thead><tr>"
        "<th>Base</th><th>Domínio</th><th>Linhas</th><th>Prevalência</th>"
        "<th>PCA</th><th>Agrupamento</th><th>Ordem temporal</th>"
        "<th>TabBERT</th><th>Protocolos</th><th>Avaliação</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_dropped_columns_list() -> str:
    items = []
    for definition in experiment_datasets.DATASET_DEFINITIONS.values():
        dropped = experiment_datasets.columns_to_drop_after_partitioning(definition)
        if not dropped:
            continue
        columns = ", ".join(f"<code>{escape(column)}</code>" for column in dropped)
        items.append(
            f"<li><strong>{escape(DATASET_SHORT_NAME[definition.database_type])}</strong>"
            f" <span class='muted'>({escape(definition.database_type)})</span>: {columns}</li>"
        )
    return "<ul class='spaced'>" + "".join(items) + "</ul>"


def build_grid_table() -> str:
    rows = []
    total_configurations = 0
    total_runs = 0
    for block in experiment_grid.SUPPORTED_BLOCKS:
        entries = experiment_grid.GRID_BUILDERS[block]()
        runs = experiment_grid.algorithm_runs_for(block, entries)
        run_count = sum(count for _algorithm, _threshold, count in runs)
        algorithms = ", ".join(
            f"<code>{escape(algorithm)}</code>"
            for algorithm, _threshold, _count in runs
        )
        total_configurations += len(entries)
        total_runs += run_count
        rows.append(
            "<tr>"
            f"<td><strong>{escape(BLOCK_LABELS[block])}</strong></td>"
            f"<td class='num'>{len(entries)}</td>"
            f"<td class='num'>{experiment_grid.result_rows_for(entries)}</td>"
            f"<td class='num'>{run_count}</td>"
            f"<td class='small'>{algorithms}</td>"
            "</tr>"
        )
    rows.append(
        "<tr class='total'>"
        "<td><strong>Total</strong></td>"
        f"<td class='num'><strong>{total_configurations}</strong></td>"
        "<td class='num'>—</td>"
        f"<td class='num'><strong>{total_runs}</strong></td>"
        "<td></td></tr>"
    )
    return (
        "<table><thead><tr>"
        "<th>Bloco</th><th>Configurações preparadas</th>"
        "<th>Dobras por algoritmo</th><th>Execuções</th><th>Algoritmos</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_grid_per_dataset_table() -> str:
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    for block in experiment_grid.SUPPORTED_BLOCKS:
        for entry in experiment_grid.GRID_BUILDERS[block]():
            counts[entry.database_type][block] = (
                counts[entry.database_type].get(block, 0) + 1
            )
    rows = []
    for database_type in experiment_datasets.DATASET_DEFINITIONS:
        cells = "".join(
            f"<td class='num'>{counts[database_type].get(block, 0) or '—'}</td>"
            for block in experiment_grid.SUPPORTED_BLOCKS
        )
        total = sum(counts[database_type].values())
        rows.append(
            f"<tr><td><code>{escape(database_type)}</code></td>{cells}"
            f"<td class='num'><strong>{total}</strong></td></tr>"
        )
    headers = "".join(
        f"<th>{escape(BLOCK_LABELS[block])}</th>"
        for block in experiment_grid.SUPPORTED_BLOCKS
    )
    return (
        f"<table><thead><tr><th>Base</th>{headers}<th>Total</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_execution_routing_table() -> str:
    rows = []
    for block in experiment_grid.SUPPORTED_BLOCKS:
        entries = experiment_grid.GRID_BUILDERS[block]()
        handling_modes = sorted({entry.categorical_handling_mode for entry in entries})
        prepare_script, run_script = BLOCK_SCRIPTS[block]
        algorithms = "<br>".join(
            f"<code>{escape(algorithm)}</code>"
            for algorithm, _threshold, _count in experiment_grid.algorithm_runs_for(block, entries)
        )
        rows.append(
            "<tr>"
            f"<td><strong>{escape(BLOCK_LABELS[block])}</strong></td>"
            f"<td>{', '.join(f'<code>{escape(mode)}</code>' for mode in handling_modes)}</td>"
            f"<td class='small'><code>{escape(prepare_script)}</code><br>"
            f"<code>{escape(run_script)}</code></td>"
            f"<td class='small'>{BLOCK_PARTITIONS[block]}</td>"
            f"<td class='small'>{algorithms}</td>"
            f"<td class='num'>{len(entries)}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Bloco</th><th><code>categorical_handling_mode</code></th>"
        "<th>Preparação / execução</th><th>Partição</th>"
        "<th>Algoritmos</th><th>Configurações</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_execution_combinations_table() -> str:
    """Base x bloco: configuracoes preparadas e execucoes (configuracao/algoritmo)."""
    entries_by_cell: dict[tuple[str, str], list] = defaultdict(list)
    for block in experiment_grid.SUPPORTED_BLOCKS:
        for entry in experiment_grid.GRID_BUILDERS[block]():
            entries_by_cell[(entry.database_type, block)].append(entry)

    def run_count(block: str, entries: list) -> int:
        return sum(
            count for _algorithm, _threshold, count
            in experiment_grid.algorithm_runs_for(block, entries)
        )

    rows = []
    block_totals = {block: [0, 0] for block in experiment_grid.SUPPORTED_BLOCKS}
    for database_type in experiment_datasets.DATASET_DEFINITIONS:
        cells = []
        dataset_runs = 0
        for block in experiment_grid.SUPPORTED_BLOCKS:
            entries = entries_by_cell.get((database_type, block), [])
            if not entries:
                cells.append("<td class='num muted'>—</td>")
                continue
            runs = run_count(block, entries)
            dataset_runs += runs
            block_totals[block][0] += len(entries)
            block_totals[block][1] += runs
            cells.append(f"<td class='num'>{len(entries)} / {runs}</td>")
        rows.append(
            f"<tr><td><code>{escape(database_type)}</code><br>"
            f"<span class='muted small'>{escape(DATASET_SHORT_NAME.get(database_type, ''))}</span></td>"
            f"{''.join(cells)}<td class='num'><strong>{dataset_runs}</strong></td></tr>"
        )
    total_cells = "".join(
        f"<td class='num'><strong>{configurations} / {runs}</strong></td>"
        for configurations, runs in block_totals.values()
    )
    grand_total = sum(runs for _configurations, runs in block_totals.values())
    rows.append(
        f"<tr class='total'><td><strong>Total</strong></td>{total_cells}"
        f"<td class='num'><strong>{grand_total}</strong></td></tr>"
    )
    headers = "".join(
        f"<th>{escape(BLOCK_LABELS[block])}</th>"
        for block in experiment_grid.SUPPORTED_BLOCKS
    )
    return (
        f"<table><thead><tr><th>Base</th>{headers}<th>Execuções</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_time_table() -> str:
    rows_data, block_totals = estimator.build_estimate()
    by_algorithm: dict[str, float] = defaultdict(float)
    by_algorithm_folds: dict[str, int] = defaultdict(int)
    for row in rows_data:
        by_algorithm[row["algoritmo"]] += row["total_s"]
        by_algorithm_folds[row["algoritmo"]] += row["dobras"]

    rows = []
    for algorithm, total in sorted(by_algorithm.items(), key=lambda item: -item[1]):
        rows.append(
            "<tr>"
            f"<td><code>{escape(algorithm)}</code></td>"
            f"<td class='num'>{by_algorithm_folds[algorithm]}</td>"
            f"<td class='num'><strong>{escape(estimator._format_duration(total))}</strong></td>"
            f"<td class='num muted'>{escape(estimator._format_duration(total * estimator.UNCERTAINTY_LOW))}"
            f" – {escape(estimator._format_duration(total * estimator.UNCERTAINTY_HIGH))}</td>"
            "</tr>"
        )
    grand_total = sum(block_totals.values())
    rows.append(
        "<tr class='total'><td><strong>Total sequencial</strong></td><td></td>"
        f"<td class='num'><strong>{escape(estimator._format_duration(grand_total))}</strong></td>"
        f"<td class='num muted'>{escape(estimator._format_duration(grand_total * estimator.UNCERTAINTY_LOW))}"
        f" – {escape(estimator._format_duration(grand_total * estimator.UNCERTAINTY_HIGH))}</td></tr>"
    )
    return (
        "<table><thead><tr><th>Algoritmo</th><th>Dobras</th>"
        "<th>Estimativa</th><th>Faixa</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_bottleneck_table() -> str:
    rows_data, _ = estimator.build_estimate()
    rows = []
    for row in sorted(rows_data, key=lambda item: -item["total_s"])[:8]:
        rows.append(
            "<tr>"
            f"<td><code>{escape(row['algoritmo'])}</code></td>"
            f"<td><code>{escape(row['base'])}</code></td>"
            f"<td class='num'>{row['dobras']}</td>"
            f"<td class='num'>{row['s_por_dobra']:,.0f} s</td>".replace(",", ".")
            + f"<td class='num'><strong>{escape(estimator._format_duration(row['total_s']))}</strong></td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Algoritmo</th><th>Base</th><th>Dobras</th>"
        "<th>s / dobra</th><th>Total</th></tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def build_review_table() -> str:
    ordered = sorted(
        review_items.REVIEW_ITEMS,
        key=lambda item: (
            {STATUS_RESOLVED: 0, STATUS_PARTIAL: 1, STATUS_DECISION: 2, STATUS_THESIS: 3}[
                item.status
            ],
            SEVERITY_ORDER[item.severity],
            item.identifier,
        ),
    )
    rows = []
    for item in ordered:
        evidence = (
            "<ul class='evidence'>"
            + "".join(f"<li><code>{escape(entry)}</code></li>" for entry in item.evidence)
            + "</ul>"
            if item.evidence
            else ""
        )
        rows.append(
            f"<tr data-status='{escape(item.status)}' data-severity='{escape(item.severity)}'>"
            f"<td class='identifier'><strong>{escape(item.identifier)}</strong>"
            f"<br><span class='badge severity-{escape(item.severity.lower())}'>"
            f"{escape(item.severity)}</span></td>"
            f"<td><strong>{escape(item.title)}</strong>"
            f"<p class='problem'>{escape(item.problem)}</p></td>"
            f"<td><span class='badge status-{escape(item.status)}'>"
            f"{escape(STATUS_LABELS[item.status])}</span>"
            f"<p>{escape(item.response)}</p>{evidence}</td>"
            "</tr>"
        )
    return (
        "<table class='review'><thead><tr><th>Item</th>"
        "<th>O que a revisão apontou</th><th>Como foi endereçado</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def build_status_summary() -> str:
    counts = review_items.status_counts()
    total = len(review_items.REVIEW_ITEMS)
    cards = []
    for status in (STATUS_RESOLVED, STATUS_PARTIAL, STATUS_DECISION, STATUS_THESIS):
        count = counts.get(status, 0)
        if not count:
            continue
        cards.append(
            f"<div class='card status-{escape(status)}'>"
            f"<div class='card-number'>{count}</div>"
            f"<div class='card-label'>{escape(STATUS_LABELS[status])}</div></div>"
        )
    cards.append(
        f"<div class='card'><div class='card-number'>{total}</div>"
        "<div class='card-label'>Itens analisados</div></div>"
    )
    return "<div class='cards'>" + "".join(cards) + "</div>"


def build_prevalence_table() -> str:
    rows = []
    for database_type, prevalence in DATASET_PREVALENCE.items():
        rows.append(
            "<tr>"
            f"<td><code>{escape(database_type)}</code></td>"
            f"<td>{escape(DATASET_SHORT_NAME[database_type])}</td>"
            f"<td class='num'>{escape(prevalence)}</td>"
            f"<td>{escape(DATASET_DOMAIN[database_type])}</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Base</th><th>Nome</th><th>Prevalência</th>"
        "<th>Domínio</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


# ---------------------------------------------------------------------------
# Texto fixo
# ---------------------------------------------------------------------------

STYLE = """
:root {
  --bg: #ffffff;
  --surface: #f7f8fa;
  --surface-strong: #eef0f4;
  --text: #15181d;
  --text-muted: #5b6472;
  --border: #d9dde4;
  --accent: #1b4f9c;
  --accent-soft: #e8effb;
  --ok: #1a7f52;
  --ok-soft: #e3f4ec;
  --warn: #9a6400;
  --warn-soft: #fdf2dc;
  --info: #5b6472;
  --info-soft: #eef0f4;
  --crit: #a32c2c;
  --crit-soft: #fbe9e9;
  --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #11141a;
    --surface: #171b22;
    --surface-strong: #1e232c;
    --text: #e6e9ef;
    --text-muted: #9aa4b2;
    --border: #2a303a;
    --accent: #7aa9ef;
    --accent-soft: #1a2535;
    --ok: #56c694;
    --ok-soft: #14291f;
    --warn: #e0ad53;
    --warn-soft: #2a2213;
    --info: #9aa4b2;
    --info-soft: #1e232c;
    --crit: #ef8a8a;
    --crit-soft: #2c1818;
  }
}
:root[data-theme="dark"] {
  --bg: #11141a;
  --surface: #171b22;
  --surface-strong: #1e232c;
  --text: #e6e9ef;
  --text-muted: #9aa4b2;
  --border: #2a303a;
  --accent: #7aa9ef;
  --accent-soft: #1a2535;
  --ok: #56c694;
  --ok-soft: #14291f;
  --warn: #e0ad53;
  --warn-soft: #2a2213;
  --info: #9aa4b2;
  --info-soft: #1e232c;
  --crit: #ef8a8a;
  --crit-soft: #2c1818;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
    "Helvetica Neue", Arial, sans-serif;
  font-size: 16px;
  line-height: 1.65;
  -webkit-text-size-adjust: 100%;
}
.layout { display: flex; gap: 0; align-items: flex-start; }
nav.toc {
  position: sticky; top: 0; flex: 0 0 258px; max-height: 100vh;
  overflow-y: auto; padding: 32px 20px 48px; border-right: 1px solid var(--border);
  background: var(--surface); font-size: 14px;
}
nav.toc h2 {
  font-size: 11px; text-transform: uppercase; letter-spacing: .08em;
  color: var(--text-muted); margin: 0 0 12px;
}
nav.toc a {
  display: block; padding: 5px 0; color: var(--text-muted);
  text-decoration: none; border-left: 2px solid transparent; padding-left: 10px;
}
nav.toc a:hover { color: var(--accent); border-left-color: var(--accent); }
main { flex: 1 1 auto; min-width: 0; padding: 40px 32px 96px; max-width: 1080px; }
header.masthead { border-bottom: 2px solid var(--border); padding-bottom: 24px; margin-bottom: 36px; }
header.masthead .institution {
  font-size: 13px; color: var(--text-muted); text-transform: uppercase;
  letter-spacing: .06em; margin: 0 0 14px;
}
header.masthead h1 { font-size: 34px; line-height: 1.2; margin: 0 0 8px; letter-spacing: -.02em; }
header.masthead .subtitle { font-size: 18px; color: var(--text-muted); margin: 0 0 18px; }
header.masthead .meta { font-size: 14px; color: var(--text-muted); margin: 0; }
h2 {
  font-size: 25px; margin: 52px 0 14px; padding-top: 10px; letter-spacing: -.01em;
  border-top: 1px solid var(--border);
}
h2:first-of-type { border-top: none; }
h3 { font-size: 19px; margin: 32px 0 10px; }
h4 { font-size: 16px; margin: 24px 0 8px; color: var(--text-muted); }
p { margin: 0 0 14px; }
ul, ol { margin: 0 0 14px; padding-left: 22px; }
li { margin-bottom: 6px; }
ul.spaced li { margin-bottom: 10px; }
code {
  font-family: var(--mono); font-size: .88em; background: var(--surface-strong);
  padding: 1px 5px; border-radius: 4px; word-break: break-word;
}
pre {
  background: var(--surface-strong); border: 1px solid var(--border);
  border-radius: 8px; padding: 14px 16px; overflow-x: auto; font-size: 13px;
  line-height: 1.55; font-family: var(--mono);
}
pre code { background: none; padding: 0; font-size: inherit; }
table {
  width: 100%; border-collapse: collapse; margin: 18px 0 24px; font-size: 14.5px;
  display: block; overflow-x: auto;
}
thead th {
  text-align: left; background: var(--surface-strong); padding: 9px 11px;
  border-bottom: 2px solid var(--border); font-weight: 600; white-space: nowrap;
}
tbody td { padding: 9px 11px; border-bottom: 1px solid var(--border); vertical-align: top; }
tbody tr:hover { background: var(--surface); }
td.num, th.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
td.small, .small { font-size: 13px; }
tr.total td { background: var(--surface-strong); border-bottom: none; }
.muted { color: var(--text-muted); }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin: 20px 0 28px; }
.card {
  flex: 1 1 150px; background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 16px 18px;
}
.card-number { font-size: 30px; font-weight: 650; line-height: 1.1; }
.card-label { font-size: 13px; color: var(--text-muted); margin-top: 2px; }
.card.status-resolvido { background: var(--ok-soft); border-color: var(--ok); }
.card.status-resolvido .card-number { color: var(--ok); }
.card.status-parcial { background: var(--warn-soft); border-color: var(--warn); }
.card.status-parcial .card-number { color: var(--warn); }
.card.status-dissertacao { background: var(--info-soft); }
.badge {
  display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px;
  border-radius: 999px; text-transform: uppercase; letter-spacing: .04em;
  white-space: nowrap;
}
.status-resolvido { background: var(--ok-soft); color: var(--ok); }
.status-parcial { background: var(--warn-soft); color: var(--warn); }
.status-dissertacao { background: var(--info-soft); color: var(--info); }
.status-decisao { background: var(--accent-soft); color: var(--accent); }
.severity-critico { background: var(--crit-soft); color: var(--crit); }
.severity-importante { background: var(--warn-soft); color: var(--warn); }
.severity-menor { background: var(--info-soft); color: var(--info); }
table.review td { font-size: 14px; }
table.review td.identifier { white-space: nowrap; width: 94px; }
table.review p { margin: 6px 0 0; }
table.review p.problem { color: var(--text-muted); }
ul.evidence { margin: 8px 0 0; padding-left: 16px; font-size: 12.5px; }
ul.evidence li { margin-bottom: 3px; color: var(--text-muted); }
.callout {
  background: var(--accent-soft); border-left: 3px solid var(--accent);
  border-radius: 0 8px 8px 0; padding: 14px 18px; margin: 20px 0;
}
.callout.warn { background: var(--warn-soft); border-left-color: var(--warn); }
.callout.ok { background: var(--ok-soft); border-left-color: var(--ok); }
.callout p:last-child { margin-bottom: 0; }
.filters { display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0 20px; }
.filters button {
  font: inherit; font-size: 13px; padding: 6px 14px; border-radius: 999px;
  border: 1px solid var(--border); background: var(--surface); color: var(--text);
  cursor: pointer;
}
.filters button[aria-pressed="true"] { background: var(--accent); color: #fff; border-color: var(--accent); }
footer {
  margin-top: 64px; padding-top: 20px; border-top: 1px solid var(--border);
  font-size: 13px; color: var(--text-muted);
}
@media (max-width: 900px) {
  .layout { flex-direction: column; }
  nav.toc {
    position: static; flex: 1 1 auto; width: 100%; max-height: none;
    border-right: none; border-bottom: 1px solid var(--border); padding: 20px 16px;
  }
  main { padding: 24px 16px 64px; }
  header.masthead h1 { font-size: 26px; }
}
@media print {
  nav.toc, .filters { display: none; }
  main { max-width: none; padding: 0; }
  table { display: table; }
  h2 { page-break-after: avoid; }
  tr { page-break-inside: avoid; }
}
"""

FILTER_SCRIPT = """
(function () {
  var buttons = document.querySelectorAll('.filters button');
  var rows = document.querySelectorAll('table.review tbody tr');
  buttons.forEach(function (button) {
    button.addEventListener('click', function () {
      var wanted = button.getAttribute('data-filter');
      buttons.forEach(function (other) {
        other.setAttribute('aria-pressed', other === button ? 'true' : 'false');
      });
      rows.forEach(function (row) {
        var matches = wanted === 'todos' || row.getAttribute('data-status') === wanted;
        row.style.display = matches ? '' : 'none';
      });
    });
  });
})();
"""

SECTIONS = [
    ("resumo", "Resumo executivo"),
    ("mudancas", "O que mudou"),
    ("bases", "As bases e o que roda em cada uma"),
    ("protocolos", "Protocolos de avaliação"),
    ("grade", "Grade experimental"),
    ("validacao", "Validação contra vazamento"),
    ("comparacoes", "Comparações possíveis"),
    ("observacoes", "Como cada observação foi endereçada"),
    ("tempo", "Estimativa de tempo"),
    ("pendencias", "Pendências"),
    ("distribuicao", "Distribuição da execução"),
]


def build_navigation() -> str:
    links = "".join(
        f"<a href='#{escape(anchor)}'>{escape(title)}</a>"
        for anchor, title in SECTIONS
    )
    return f"<nav class='toc'><h2>Conteúdo</h2>{links}</nav>"


def build_encoding_factorial_section() -> str:
    """A secao que detalha o fatorial 2x2 da codificacao (item C3.6).

    Os nomes das celulas e as contagens de dobras/blocos vem do codigo, para que
    o texto nao possa divergir da grade que sera executada. As medidas de ruido
    sao reproduzidas por
    `tests/unit/test_revised_pipeline.py::test_the_two_mechanisms_inject_different_amounts_of_noise`.
    """
    cells = "".join(
        f"<tr><td><code>{html.escape(strategy)}</code></td>"
        f"<td>{html.escape(statistic)}</td><td>{html.escape(control)}</td></tr>"
        for strategy in pipeline.FACTORIAL_CATEGORICAL_ENCODINGS
        for statistic, control in [pipeline.parse_encoding_factors(strategy)]
    )
    return f"""
<h4>4.1 Codificação categórica: de um eixo confundido para um fatorial 2×2</h4>

<div class="callout warn">
<p><strong>O problema.</strong> A grade anterior tinha um único eixo de
codificação, com dois valores: <code>ordered-target</code> e <code>woe</code>.
Eles não diferiam em uma coisa, e sim em duas ao mesmo tempo:</p>
<ul>
<li>a <strong>estatística</strong> que resume a categoria — média do alvo
suavizada contra log-razão de chances;</li>
<li>o <strong>mecanismo</strong> que impede a linha de ver o próprio rótulo —
prefixo de uma permutação contra <em>cross-fitting</em> de cinco dobras.</li>
</ul>
<p>Nenhuma diferença medida entre os dois braços podia ser atribuída a um dos
fatores. Uma vantagem do <code>woe</code> em F1 era compatível com "a log-razão
é a melhor estatística" e com "cinco dobras são um controle melhor que uma
permutação", e o desenho não separava as duas leituras.</p>
</div>

<p>O eixo foi substituído por duas coordenadas independentes. O nome da
estratégia declara a célula, no formato
<code>&lt;estatística&gt;:&lt;mecanismo&gt;</code>:</p>

<table>
<thead><tr><th>Célula</th><th>Estatística</th><th>Mecanismo</th></tr></thead>
<tbody>{cells}</tbody>
</table>

<p>Duas células já existiam sob outro nome: <code>log-odds:k-fold</code> é
exatamente o antigo <code>woe</code> — mesma estatística, mesmo mecanismo, mesmo
número de dobras. O apelido continua sendo aceito, e é resolvido
<strong>dentro do <code>configuration_md5</code></strong>, não apenas no
codificador: se a normalização acontecesse só na execução, <code>woe</code> e
<code>log-odds:k-fold</code> seriam duas configurações distintas com conteúdo
idêntico, inflando a contagem de pares do mesmo modo que as bases republicadas
inflavam (item G9b). As outras duas células são novas, e são elas que fecham a
conta.</p>

<h4>4.2 Por que o mecanismo precisou ser reimplementado</h4>

<p>Preencher as células que faltavam não era uma questão de nomear
combinações. O <code>WOEEncoder</code> <strong>não é um codificador
ordenado</strong>: ajustá-lo sobre uma moldura permutada não produz estatística
de prefixo alguma — devolve o WoE do treino inteiro para todas as linhas, que é
precisamente o vazamento que o mecanismo deveria impedir. A célula
<code>log-odds:permutation-prefix</code> não existia como opção de biblioteca.</p>

<p>O mecanismo foi então implementado <strong>uma única vez, por fora do
codificador</strong>, em
<code>_encode_training_rows_with_leakage_control</code>. A estatística entra
pelo objeto codificador; o mecanismo entra pelo particionamento das linhas. É
literalmente o mesmo código nas duas estatísticas, o que é a condição para que o
fatorial seja legível.</p>

<ul class="spaced">
<li><strong><code>k-fold</code></strong> — <code>StratifiedKFold</code> com
{pipeline.CROSS_FITTING_FOLD_COUNT} dobras: cada dobra é codificada por um
ajuste feito no complemento dela.</li>
<li><strong><code>permutation-prefix</code></strong> — uma permutação
determinística dividida em {pipeline.PERMUTATION_PREFIX_BLOCK_COUNT} blocos
expansivos: o bloco <em>k</em> é codificado por um ajuste feito apenas nos
blocos anteriores. O bloco inicial, que não tem prefixo algum, recebe o valor a
priori da estatística — a prevalência do treino para a média suavizada, zero
para a log-razão, que é o valor que cada uma assume na ausência de qualquer
observação.</li>
</ul>

<p>O esquema ordenado original encodifica cada linha com as linhas que a
precedem, o que exigiria um reajuste por linha. Os blocos expansivos reduzem o
custo a um reajuste por bloco preservando a propriedade essencial: nenhuma linha
participa da estatística que a codifica. Com
{pipeline.PERMUTATION_PREFIX_BLOCK_COUNT} blocos, 10% das linhas ficam sem
prefixo, contra 20% se fossem cinco.</p>

<h4>4.3 A medição que justifica o desenho</h4>

<p>Sobre uma coluna categórica sintética de 8 categorias com taxa de alvo
conhecida (2% a 30%), medindo o ruído injetado nas codificações
<strong>de treino</strong> — dispersão intra-categoria dividida pela dispersão
entre categorias, para que as escalas diferentes das duas estatísticas não
contaminem a comparação:</p>

<table>
<thead><tr><th>Célula</th><th>Ruído / sinal</th></tr></thead>
<tbody>
<tr><td><code>smoothed-mean:k-fold</code></td><td>0,053</td></tr>
<tr><td><code>log-odds:k-fold</code></td><td>0,062</td></tr>
<tr><td><code>smoothed-mean:permutation-prefix</code></td><td><strong>0,318</strong></td></tr>
<tr><td><code>log-odds:permutation-prefix</code></td><td><strong>0,298</strong></td></tr>
<tr><td class="muted"><code>ordered-target</code> (referência, prefixo por linha)</td>
<td class="muted">0,280</td></tr>
</tbody>
</table>

<p>O resultado é o argumento inteiro em uma tabela. O <strong>mecanismo</strong>
move o ruído de ~0,06 para ~0,31, um fator de cinco. A
<strong>estatística</strong>, com o mecanismo fixo, move-o de 0,053 para 0,062 —
um fator de 1,2. O efeito dominante do eixo antigo era o fator que a
dissertação não estava medindo.</p>

<p>Dois fatos adicionais, ambos relevantes para a redação:</p>

<ul class="spaced">
<li>O ruído do mecanismo de prefixo é praticamente <strong>o mesmo nas duas
estatísticas</strong> (0,318 e 0,298). É essa simetria que permite ler o
fatorial como dois efeitos principais, em vez de uma interação intratável.</li>
<li>A aproximação por blocos <strong>cerca</strong> a formulação publicada
(0,318 e 0,298 contra 0,280), o que indica que ela reproduz o mecanismo em vez
de criar um terceiro regime. O braço <code>ordered-target</code> continua
disponível no código justamente para que essa distância seja mensurável, mas
está <strong>fora da grade</strong>: seu prefixo por linha não tem equivalente
para a log-razão, e incluí-lo reintroduziria o confundimento.</li>
</ul>

<div class="callout warn">
<p><strong>Sobre a implementação de referência.</strong> O
<code>category_encoders.CatBoostEncoder</code> usa <strong>uma</strong>
permutação; o CatBoost real usa várias e tira a média, justamente para atenuar o
problema que a medição mostra. O ruído do prefixo por linha é fortemente
heterocedástico: no primeiro decil da permutação ele vale 0,675, no último
0,019 — uma razão de 35×, porque as primeiras linhas não têm prefixo do qual
estimar. O braço <code>ordered-target</code> da grade anterior era, portanto,
uma versão enfraquecida da técnica publicada, o que tornava a atribuição ainda
menos defensável.</p>
</div>

<h4>4.4 Quais comparações permanecem válidas</h4>

<p>O fatorial muda o que se pode afirmar. O que segue valendo, o que passou a
valer e o que continua fora de alcance:</p>

<table>
<thead><tr><th>Comparação</th><th>Situação</th><th>Por quê</th></tr></thead>
<tbody>
<tr>
<td><code>smoothed-mean</code> × <code>log-odds</code>, com o mecanismo fixo</td>
<td><strong>Válida — nova</strong></td>
<td>É o efeito da estatística, isolado. Duas leituras independentes (uma com
<code>k-fold</code>, outra com <code>permutation-prefix</code>), o que permite
verificar se a conclusão se sustenta sob os dois mecanismos.</td>
</tr>
<tr>
<td><code>k-fold</code> × <code>permutation-prefix</code>, com a estatística fixa</td>
<td><strong>Válida — nova</strong></td>
<td>É o efeito do mecanismo, isolado. Também com duas leituras independentes.</td>
</tr>
<tr>
<td>Interação entre os dois fatores</td>
<td><strong>Válida — nova</strong></td>
<td>Com as quatro células preenchidas, dá-se para perguntar se a melhor
estatística depende do mecanismo. No eixo antigo essa pergunta não existia.</td>
</tr>
<tr>
<td>Antigo <code>ordered-target</code> × antigo <code>woe</code></td>
<td><strong>Não se faz mais</strong></td>
<td>Era a comparação confundida. Qualquer diferença media os dois fatores
somados, com o mecanismo dominando. Os resultados dessa comparação, se
existirem em execuções anteriores, não sustentam afirmação sobre a
estatística.</td>
</tr>
<tr>
<td><code>log-odds:k-fold</code> contra resultados antigos de <code>woe</code></td>
<td><strong>Válida</strong></td>
<td>Mesma estatística, mesmo mecanismo, mesmo número de dobras: é a mesma
célula sob outro nome, e o apelido garante a reprodutibilidade.</td>
</tr>
<tr>
<td><code>ordered-target</code> (referência) × <code>smoothed-mean:permutation-prefix</code></td>
<td><strong>Válida, com escopo restrito</strong></td>
<td>Mede apenas a distância entre o prefixo por linha e a aproximação por
blocos, com a estatística fixa. Não é uma comparação entre estatísticas e não
entra nas conclusões sobre codificação.</td>
</tr>
<tr>
<td>Qualquer comparação do eixo de codificação na <code>ulb</code></td>
<td><strong>Não se aplica</strong></td>
<td>A ULB não tem atributo categórico; a codificação é <code>none</code> e o
eixo não varia. O fatorial vive nas bases 19, 20, 22, 22-noid e
22-trans.</td>
</tr>
<tr>
<td>Comparação do eixo de codificação com CatBoost, XGBoost ou os Transformers</td>
<td><strong>Não se aplica</strong></td>
<td>Nenhum dos dois blocos usa codificação supervisionada: o boosting trata
categóricas nativamente e os Transformers as tokenizam. O fatorial é um eixo
exclusivo do bloco clássico, e por isso seu custo não toca os 60,3 dias do
boosting.</td>
</tr>
</tbody></table>

<p>As comparações <strong>pareadas</strong> continuam exigindo que todo o resto
esteja fixo — base, imputação, balanceamento, protocolo, semente, algoritmo e
intensidade de ponderação. A dispersão entre as cinco dobras do
<code>stratified-k-fold</code> é o que decide se uma diferença entre células é
sinal ou ruído, e é especialmente necessária aqui: o eixo do mecanismo mexe
justamente na variância das entradas do modelo.</p>

<div class="callout">
<p><strong>Custo.</strong> O eixo passou de 2 para 4 valores nas cinco bases com
atributos categóricos, levando a grade clássica de 452 para
{len(experiment_grid.build_classic_grid())} configurações. Em tempo de execução
são cerca de 4,5 dias, contra 2,2 do eixo anterior, mais 1,3 dia de preparação —
um acréscimo de aproximadamente 5,6% sobre o total sequencial estimado. O bloco
de boosting, que domina o custo, não é afetado.</p>
</div>
"""


def build_html() -> str:
    counts = review_items.status_counts()
    resolved = counts.get(STATUS_RESOLVED, 0)
    partial = counts.get(STATUS_PARTIAL, 0)
    total_items = len(review_items.REVIEW_ITEMS)
    generation_date = date.today().strftime("%d/%m/%Y")

    classic_entries = experiment_grid.build_classic_grid()
    tabbert_entries = experiment_grid.build_tabbert_grid()
    _rows, block_totals = estimator.build_estimate()
    grand_total = sum(block_totals.values())

    return f"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(REPORT_TITLE)}</title>
<meta name="description" content="{escape(REPORT_SUBTITLE)} — {escape(AUTHOR)}">
<meta name="author" content="{escape(AUTHOR)}">
<meta property="og:title" content="{escape(REPORT_TITLE)}">
<meta property="og:description" content="{escape(REPORT_SUBTITLE)}">
<meta property="og:type" content="article">
<style>{STYLE}</style>
</head>
<body>
<div class="layout">
{build_navigation()}
<main>

<header class="masthead">
  <p class="institution">{escape(INSTITUTION)}</p>
  <h1>{escape(REPORT_TITLE)}</h1>
  <p class="subtitle">{escape(REPORT_SUBTITLE)}</p>
  <p class="meta">
    <strong>{escape(AUTHOR)}</strong> · {escape(ADVISORS)}<br>
    Dissertação: <em>{escape(THESIS_TITLE)}</em><br>
    Relatório de revisão analisado: 01/10/2026 · Este documento: {escape(generation_date)}
  </p>
</header>

<section id="resumo">
<h2>Resumo executivo</h2>

<p>O <em>Relatório de Revisão</em> da primeira versão da dissertação apontou
problemas que comprometiam a validade de parte dos resultados: vazamento de
dados, protocolos que ignoravam a dimensão temporal, indícios de erro no
<em>pipeline</em>, reamostragem que descartava quase todas as transações
legítimas do treinamento e limiar de decisão arbitrário.</p>

<p>Este documento descreve o <em>pipeline</em> experimental reimplementado para
responder a essas observações, e registra, item a item, o que foi feito. O
código antigo permanece intacto; o novo vive em uma pasta própria e é
autocontido.</p>

{build_status_summary()}

<p>Dos {total_items} itens analisados, <strong>{resolved} estão resolvidos no
código</strong>, com teste automatizado ou verificação de dados que os protege,
e <strong>{partial} são parciais</strong> — o <em>pipeline</em> faz a parte que
lhe cabe e o restante pertence à fase de análise ou à redação. Os demais são de
texto, estrutura ou formatação e não dependem de código.</p>

<div class="callout">
<p>Além dos pontos mencionados, acredita-se também que dois outros pontos de
atenção foram encontrados com relação aos tratamentos da base ULB. Nela
encontramos 1.081 linhas idênticas em todas as 31 colunas, no entanto, com
partição aleatória, cópias do mesmo registro caíam em treino e teste e o modelo
acertava por memorização. E nada garantia que o arquivo lido na execução fosse o
mesmo que a preparação gravou. Ainda que este ponto sobre a memorização tenha
sido levantado em outros cenários na revisão, incluí esta nota indicando a
correção da base e também a introdução dos testes, validando as partições.</p>
</div>
</section>

<section id="mudancas">
<h2>O que mudou</h2>

<table>
<thead><tr><th>Decisão</th><th>Antes</th><th>Agora</th><th>Item</th></tr></thead>
<tbody>
<tr><td>Bases</td><td>8 configurações de 4 fontes, com <code>eqm-ref-21</code> e
<code>ulb-pca</code> duplicando a ULB</td>
<td>7 configurações de 4 fontes. Perduraram: <code>ulb</code> (ULB),
<code>home-credit-completa</code> (Home Credit com junção), <code>sparkov</code>
(Sparkov), <code>ieee-cis-completa</code>, <code>ieee-cis-sem-identidade</code> e
<code>ieee-cis-transacional</code> (as três visões da IEEE-CIS). Entrou
<code>home-credit-somente-aplicacao</code>, a Home Credit sem a junção.
<code>eqm-ref-21</code>, <code>eqm-ref-21-pca</code> e
<code>ulb-pca</code> foram eliminadas</td><td>G9</td></tr>
<tr><td>PCA</td><td>Eixo implícito da grade, confundido com a base</td>
<td>Como é uma propriedade de cada base, para fins de redução de processamento,
manteve-se o PCA fora da ULB e ele foi utilizado nas demais bases
<strong>apenas para os algoritmos clássicos</strong>. Boosting e Transformers
nunca o recebem</td><td>C5.31</td></tr>
<tr><td>Reserva (<code>test_data_size</code>)</td><td>Eixo da grade, inerte em 2 das 4 políticas</td>
<td>Considerando o efeito como uma ideia ruim do processo, mesmo levando em
conta o tempo de processamento, o experimento removeu essa parte do processo:
o eixo saiu da identidade, do manifesto e dos resultados</td><td>G9</td></tr>
<tr><td>Protocolos</td><td>4 políticas, uma descartando 20–40% dos dados</td>
<td>3 protocolos explícitos, nenhum descartando dados</td><td>G9</td></tr>
<tr><td>Particionamento</td><td>Por linha; cliente/cartão atravessava treino e teste</td>
<td>Por grupo nas bases que têm titular: <code>home-credit-completa</code> por
<code>SK_ID_CURR</code>, <code>sparkov</code> por <code>cc_num</code> e as
três visões da IEEE-CIS pelo titular aproximado (<code>card1</code> +
<code>addr1</code>). <code>ulb</code> e
<code>home-credit-somente-aplicacao</code> não precisam: a primeira não tem portador,
a segunda tem uma linha por cliente. Ver detalhamento por família de algoritmos
abaixo</td><td>G7, G8</td></tr>
<tr><td>Protocolo temporal</td><td>Inexistente (a "divisão aleatória" era temporal por acidente)</td>
<td><code>temporal-holdout</code> explícito</td><td>G8</td></tr>
<tr><td>Identificadores</td><td><code>cc_num</code>, <code>trans_num</code>,
<code>TransactionID</code>, <code>Time</code> entravam como atributos</td>
<td>Removidos de todas as bases, antes do ajuste</td><td>G14</td></tr>
<tr><td>Padronização</td><td>Só nas configurações com PCA</td>
<td><strong>Clássicos:</strong> padronização em todas as configurações,
ajustada no treino de cada partição. Na <code>ulb</code> isso corrige o
caso em que <code>Amount</code> (até ~25.691) dominava as distâncias
euclidianas contra componentes de variância unitária; na
<code>sparkov</code>, coloca na mesma escala hora, idade, valor, coordenadas
e distância. <strong>Boosting:</strong> nenhuma padronização — árvores são
invariantes a transformações monótonas de cada atributo.
<strong>Transformers:</strong> normalização própria do modelo
(<code>quantile</code> ou <code>standard</code> no FT-Transformer,
discretização em faixas no TabBERT), nunca a padronização do bloco
clássico</td><td>G14</td></tr>
<tr><td>Linha de base sem reamostragem</td><td>Não existia</td>
<td>Braço <code>none</code> nos blocos clássico e de boosting</td><td>G10</td></tr>
<tr><td>Remoção de incompletos</td><td>Removia linhas do <strong>teste</strong> também</td>
<td>Remove <strong>só do treino</strong></td><td>G16</td></tr>
<tr><td>Limiar de decisão</td><td>Fixo em 0,5, aplicado sobre
probabilidades deslocadas pela reamostragem</td>
<td><strong>Escolhido na validação</strong>, maximizando F2, e aplicado ao
teste, em todos os blocos e algoritmos. No <code>stratified-k-fold</code>, em
que a validação é o próprio conjunto avaliado, o limiar de cada dobra vem da
validação das <em>outras</em> dobras</td><td>G11</td></tr>
<tr><td>Pesos de classe</td><td>Não existiam</td>
<td>Dois braços de balanceamento novos, ao lado de <code>none</code>, RUS,
RUS+SMOTE e Tomek Links: <code>sqrt-balanced</code> (√R) e
<code>balanced</code> (R), com R medido no treino de cada dobra. Rodam só sobre
as configurações sem reamostragem, nos três algoritmos que aceitam peso:
regressão logística, CatBoost e XGBoost</td><td>G10, C3.14</td></tr>
<tr><td>Titular na IEEE-CIS</td><td><code>card1</code>, que é informação do cartão</td>
<td><code>card1</code> + <code>addr1</code>, combinação citada tanto no
relatório de revisão quanto nas soluções publicadas da competição no
Kaggle</td><td>G8</td></tr>
<tr><td><code>fold_index</code></td><td>0-based, vazio quando não havia dobras</td>
<td><strong>1-based</strong>, sempre preenchido</td><td>C4.7</td></tr>
<tr><td>MD5 dos arquivos</td><td>Só identidade da configuração</td>
<td>Identidade <strong>+ MD5 de conteúdo</strong> de treino/validação/teste</td><td>N2</td></tr>
<tr><td>Semente</td><td>Única, implícita</td>
<td><code>random_seed</code> na identidade; <code>SEED_LIST</code> repete a grade</td><td>G13</td></tr>
<tr><td>Cabeça dos Transformers</td><td><code>linear</code>, <code>mlp</code>, <code>lstm</code></td>
<td><code>linear</code> removida; ficam <code>mlp</code> e <code>lstm</code></td><td>—</td></tr>
<tr><td>Regressão linear</td><td>Avaliada como classificador</td>
<td><strong>Removida da grade.</strong> Sua saída não é uma probabilidade
calibrada, pode cair fora de [0, 1] e o limiar de 0,5 não tem interpretação
sobre ela</td><td>C3.4</td></tr>
<tr><td>Registros duplicados</td><td>1.081 linhas idênticas espalhadas entre treino e teste</td>
<td>Removidas na carga</td><td>N1</td></tr>
</tbody></table>
</section>

<section id="bases">
<h2>As bases e o que roda em cada uma</h2>

<p>São <strong>7 configurações de 4 fontes</strong>. <code>eqm-ref-21</code>
não aparece em lugar nenhum: era uma republicação da ULB que produzia
resultados idênticos aos de <code>ulb</code> e dava peso dobrado a ela em
toda média agregada.</p>

<p>A novidade é <code>home-credit-somente-aplicacao</code>: a Home Credit lida apenas
de <code>application_data.csv</code>, <strong>sem a junção</strong> com
<code>previous_application.csv</code>. São 307.511 pedidos, um por cliente.
Pareada com <code>home-credit-completa</code>, ela isola o efeito da junção — tanto a
informação que o histórico acrescenta quanto a replicação de cada cliente em
~4,9 linhas, que era a origem do vazamento apontado no item G7. Como aqui cada
cliente aparece uma única vez, a partição aleatória já é segura por construção,
e o contraste entre as duas mostra quanto do desempenho anterior vinha do
histórico e quanto vinha da replicação. Ela também não sofre o viés de seleção
da junção interna, que descartava os 16.454 clientes sem pedidos anteriores
(307.511 menos 291.057).</p>

{build_dataset_table()}

<div class="callout">
<p><strong>A PCA é aplicada apenas para os algoritmos clássicos.</strong> O
treinamento gerado para os Transformers e para o boosting não utiliza essa
técnica. Entre as bases, ela roda em todas menos na ULB: os atributos
<code>V1</code>–<code>V28</code> já são componentes principais publicados, e uma
segunda PCA sobre eles é, na prática, uma rotação seguida do descarte de um ou
dois componentes, que podem carregar informação discriminativa. A PCA deixou de
ser eixo da grade e virou propriedade da base, ajustada no treino de cada
partição.</p>
<p>Optou-se por não aplicar uma PCA adicional nos demais blocos porque ela
eliminaria a semântica individual dos atributos, impediria o tratamento nativo
das variáveis categóricas e alteraria a representação por campo utilizada pelos
Transformers. Além disso, por ser não supervisionada, a PCA preserva variância
dos preditores, não necessariamente informação relevante para a classificação de
fraude.</p>
<p class="small muted">Verificado por teste automatizado
(<code>test_pca_is_declared_only_for_the_classic_block</code>), que lê o código
de preparação e confirma que a chamada de PCA vive apenas no ramo clássico.</p>
</div>

<h3>Atributos removidos antes do ajuste</h3>
<p>Identificadores, dados pessoais, colunas de agrupamento e marcas de tempo
absolutas são usados para juntar, agrupar e ordenar — e então descartados. Eles
nunca chegam ao modelo. A verificação <strong>P3</strong> falha se qualquer um
sobrar nos arquivos preparados.</p>

<div class="callout">
<p><strong>O descarte é o mesmo para todas as famílias de algoritmos.</strong>
Ele acontece no particionamento, antes de qualquer ajuste, então clássicos,
boosting e FT-Transformer recebem exatamente o mesmo conjunto de colunas. A
única diferença é o <strong>TabBERT</strong>: no modo sequencial, o titular e a
coluna de ordem <em>sobrevivem no arquivo</em>, porque o índice de janelas
causais as referencia — mas são excluídos do vocabulário, de modo que nunca
viram campo do modelo. A verificação <strong>P8</strong> confirma isso lendo o
vocabulário gravado.</p>
<p>Na <strong>Sparkov</strong>, concretamente: <code>cc_num</code>,
<code>trans_num</code>, o índice sem nome, <code>first</code>,
<code>last</code>, <code>street</code>, <code>dob</code>,
<code>unix_time</code> e a data bruta saem dos atributos em <em>todos</em> os
blocos. <code>cc_num</code> e <code>unix_time</code> permanecem no arquivo no
modo TabBERT, como titular e ordem, sem entrar no vocabulário. As quatro
coordenadas (<code>lat</code>, <code>long</code>, <code>merch_lat</code>,
<code>merch_long</code>) <strong>permanecem como atributos em todos os
blocos</strong>: a localização de uma transação é informação legítima para
identificar fraude. A distância cliente–estabelecimento entra como atributo
derivado, ao lado delas.</p>
</div>

<div class="callout warn">
<p><strong>Os arquivos <code>test_transaction.csv</code> e
<code>test_identity.csv</code> da IEEE-CIS nunca são lidos</strong>, em nenhum
bloco e em nenhum protocolo, porque <strong>não são rotulados</strong>: são o
conjunto de submissão da competição, sem a coluna <code>isFraud</code>. Todas as
três visões da IEEE-CIS usam exclusivamente <code>train_transaction.csv</code> e
<code>train_identity.csv</code>.</p>
</div>

{build_dropped_columns_list()}

<h3>Tratamento específico</h3>
<h4>ULB (<code>ulb</code>)</h4>
<p><code>Time</code> sai dos atributos e vira <code>hour_of_day</code>. Não tem
atributo categórico nem valor faltante, então os eixos de codificação e
imputação não variam nela — antes, 2 codificações × 4 imputações geravam 8
cópias idênticas de cada experimento. Fica fora do TabBERT: suas linhas são
transações anônimas, sem portador, e não há sequência a modelar.</p>

<h4>Home Credit (<code>home-credit-completa</code>)</h4>
<p>As sete colunas de <code>previous_application</code> homônimas das de
<code>application_data</code> deixaram de ser descartadas como "redundantes" e
entram com prefixo <code>PREV_</code>: descrevem o pedido <em>anterior</em>, e
não o atual. Toda partição é por cliente, porque a junção replica cada um em
~4,9 linhas com o mesmo <code>TARGET</code>. Não tem protocolo temporal: a
unidade é o cliente, não a transação.</p>

<h4>Sparkov (<code>sparkov</code>)</h4>
<p>Entram hora, dia da semana, fim de semana, mês, dia, idade na data da
transação e distância cliente–estabelecimento. O <code>trans_num</code>, único
por transação, é a explicação mais provável para o XGBoost sinalizar ~80% das
transações nessa base na versão anterior. O protocolo temporal usa a separação
oficial <code>fraudTrain</code>/<code>fraudTest</code>.</p>

<h4>IEEE-CIS (<code>ieee-cis-completa*</code>)</h4>
<p>O agrupamento usa um <strong>identificador aproximado de titular</strong>,
combinando <code>card1</code> + <code>addr1</code>. Agrupar só por
<code>card1</code> misturava clientes diferentes, porque
<code>card1</code>–<code>card6</code> são informações do cartão — tipo,
categoria, banco emissor —, e não do portador. Os arquivos
<code>test_transaction.csv</code> e <code>test_identity.csv</code>, sem rótulo,
nunca são lidos.</p>
</section>

<section id="protocolos">
<h2>Protocolos de avaliação</h2>

<h3>Particionamento por família de algoritmos</h3>

<p>O agrupamento é decidido pela <em>base</em>, não pelo algoritmo, então as
três famílias recebem a mesma partição — é isso que torna as comparações
pareadas possíveis. O que muda é o que cada uma faz com ela:</p>

<table>
<thead><tr><th>Família</th><th>Partição</th><th>Colunas usadas para particionar</th><th>O que mudou</th></tr></thead>
<tbody>
<tr><td><strong>Clássicos</strong><br><span class="muted">regressão logística, Naive Bayes, KNN</span></td>
<td>Os três protocolos, com <code>GroupShuffleSplit</code> e
<code>StratifiedGroupKFold</code></td>
<td><code>SK_ID_CURR</code> (Home Credit com junção), <code>cc_num</code>
(Sparkov), <code>card1</code>+<code>addr1</code> (IEEE-CIS). A ULB e a Home
Credit sem junção não precisam: a primeira não tem portador, a segunda tem uma
linha por cliente</td>
<td>Antes a partição era por linha. O KNN era o caso extremo: o vizinho mais
próximo de uma linha de teste tendia a ser outra linha do mesmo cliente, com o
mesmo rótulo</td></tr>
<tr><td><strong>Boosting nativo</strong><br><span class="muted">CatBoost, XGBoost</span></td>
<td>Os três protocolos, <strong>as mesmas partições dos clássicos</strong></td>
<td>Idênticas às dos clássicos</td>
<td>Antes as partições das duas famílias podiam divergir, porque a opção
<code>none</code> removia linhas em um bloco e as mantinha no outro. Agora os
índices são os mesmos, e a comparação entre famílias é pareada de fato</td></tr>
<tr><td><strong>FT-Transformer</strong></td>
<td>Os dois protocolos de holdout, <strong>as mesmas partições dos demais</strong></td>
<td>Idênticas às dos clássicos</td>
<td>Antes usava um arranjo próprio (60/20/20) que não coincidia com nenhum
outro bloco, o que impedia o pareamento. Agora suas métricas são comparáveis
dobra a dobra com clássicos e boosting</td></tr>
<tr><td><strong>TabBERT</strong></td>
<td><code>temporal-holdout</code> na Sparkov e na IEEE-CIS;
<code>stratified-holdout</code> agrupado por cliente na Home Credit, que não tem
ordem temporal. <strong>As mesmas partições dos demais</strong></td>
<td>Idênticas às dos clássicos. Titular e ordem ficam nos arquivos para montar
as janelas (na IEEE-CIS, <code>card1</code>+<code>addr1</code> e
<code>TransactionDT</code>), mas nunca viram atributo</td>
<td>Antes usava um corte próprio dentro do histórico de cada titular, que não
coincidia com partição alguma dos outros blocos, e agrupava a IEEE-CIS por
<code>card1</code> sozinho. Agora é pareado com os demais nos holdouts. Só os
protocolos que preservam as janelas causais servem: o embaralhamento por linha
espalharia o histórico de um titular entre os splits</td></tr>
</tbody></table>

<h3>Os três protocolos</h3>

<p>Três, todos sem parâmetro de tamanho:</p>

<ul class="spaced">
<li><strong><code>stratified-holdout</code></strong> — partição única 70/15/15,
estratificada, respeitando o agrupamento. 1 dobra.</li>
<li><strong><code>stratified-k-fold</code></strong> — 5 dobras estratificadas
por grupo; cada dobra dá treino e validação. É daqui que sai a estimativa de
dispersão entre partições que faltava. 5 dobras.</li>
<li><strong><code>temporal-holdout</code></strong> — corte cronológico
70/15/15; na Sparkov, respeita os arquivos oficiais. É o único que estima o
desempenho esperado em operação. 1 dobra.</li>
</ul>

<p>O TabBERT usa os <strong>mesmos protocolos</strong> dos demais blocos, com a
mesma semente: <code>temporal-holdout</code> onde a base tem ordem temporal, e
<code>stratified-holdout</code> agrupado pelo titular na Home Credit. Suas
janelas causais usam como histórico os registros anteriores do titular nos
splits anteriores — no corte temporal, o treino vem antes da validação e do
teste para todo titular; no agrupado, cada titular cai inteiro num único split.
O arranjo próprio anterior (<code>sequence-chronological</code>, corte dentro
do histórico de cada titular) saiu da grade por não coincidir com partição
alguma dos outros blocos.</p>

<p><strong>Saíram:</strong> <code>random-split</code> (nome enganoso — era
estratificado em quase todas as bases e temporal por acidente na Sparkov) e
<code>stratified-train-test-split</code> (que descartava de 20% a 40% dos dados
sem justificativa e, na Sparkov, jogava fora o <code>fraudTest.csv</code>
inteiro: 30,0% das transações e 22,2% das fraudes).</p>
</section>

<section id="grade">
<h2>Grade experimental</h2>

<p>A grade é gerada por <code>experiment_grid.py</code>, que consulta o catálogo
de bases e <strong>só varia um eixo onde o fenômeno existe</strong>. É isso que
elimina as configurações idênticas da versão anterior.</p>

{build_grid_table()}

<h3>Configurações por base</h3>
{build_grid_per_dataset_table()}

<p>A ULB tem poucas configurações porque os eixos de codificação e imputação não
se aplicam a ela, e nenhuma no TabBERT porque não tem portador. O TabBERT tem
poucas porque não varia reamostragem — RUS e SMOTE quebrariam as janelas
cronológicas, retirando ou inventando posições no meio do histórico de um
titular.</p>

<h3>Eixos do bloco clássico</h3>
<p>Codificação categórica, agora um <strong>fatorial 2×2</strong> —
<code>smoothed-mean:k-fold</code>, <code>smoothed-mean:permutation-prefix</code>,
<code>log-odds:k-fold</code>, <code>log-odds:permutation-prefix</code>, ou
<code>none</code> onde não há categóricas) × imputação (<code>mean</code>,
<code>median</code>, <code>mode</code>, <code>drop-incomplete-rows</code>, ou
<code>none</code> onde não há faltantes) × balanceamento (<code>none</code>,
<code>random-undersampling</code>, <code>RUS+SMOTE</code>,
<code>tomek-links</code>) × protocolo × semente. Total:
{len(classic_entries)} configurações. Na regressão logística, as configurações
com balanceamento <code>none</code> rodam também nos dois braços de pesos de
classe, descritos a seguir.</p>

<h3>Correção do desbalanceamento</h3>

<p>A versão anterior treinava com classes reequilibradas e aplicava um limiar
fixo de 0,5. Depois de reamostrar o treino para 1:1, as probabilidades deixam
de corresponder à prevalência real — que vai de 0,17% a 8,7% — e esse limiar
vira uma escolha arbitrária que produz pontos de operação extremos. O
<em>pipeline</em> trata agora as duas coisas separadamente: <strong>como
balancear</strong> é um eixo da grade, com braços que incluem pesos de classe
(G10), e <strong>onde cortar</strong> é decidido na validação, em todos os
braços (G11).</p>

<h4>Pesos de classe: dois braços do balanceamento</h4>

<p>A revisão pede braços “sem reamostragem” e “pesos de classe” como
alternativas à reamostragem. Os pesos entram, portanto, como braços do eixo de
balanceamento, ao lado de <code>none</code>, <code>random-undersampling</code>,
<code>RUS+SMOTE</code> e <code>tomek-links</code>, e não como um eixo cruzado
com eles. Defina, <strong>exclusivamente com o conjunto de treinamento de cada
dobra</strong>:</p>

<pre><code>R = N_negativos / N_positivos</code></pre>

<table>
<thead><tr><th>Intensidade</th><th>Regressão logística<br><code>class_weight</code></th>
<th>XGBoost<br><code>scale_pos_weight</code></th>
<th>CatBoost<br><code>auto_class_weights</code></th></tr></thead>
<tbody>
<tr><td><strong>Sem ponderação</strong><br><span class="muted">braços <code>none</code> e reamostrados</span></td>
<td><code>None</code></td><td class="num"><code>1.0</code></td><td><code>None</code></td></tr>
<tr><td><strong>Moderada</strong><br><span class="muted">braço <code>sqrt-balanced</code></span></td>
<td><code>{{0: 1, 1: sqrt(R)}}</code></td><td class="num"><code>sqrt(R)</code></td><td><code>"SqrtBalanced"</code></td></tr>
<tr><td><strong>Completa</strong><br><span class="muted">braço <code>balanced</code></span></td>
<td><code>"balanced"</code></td><td class="num"><code>R</code></td><td><code>"Balanced"</code></td></tr>
</tbody></table>

<p>São os <strong>mesmos níveis relativos</strong> nas duas famílias, o que
torna a comparação entre elas interpretável — antes não havia estratégia comum
alguma. A ponderação moderada aumenta a importância das fraudes sem fazer cada
fraude compensar integralmente todas as observações legítimas excedentes, o que
é particularmente interessante quando o balanceamento completo eleva demais os
falsos positivos. O CatBoost implementa <code>SqrtBalanced</code> nativamente.</p>

<h4>Medição real: ULB, holdout, sem reamostragem</h4>
<p>R = 599 nesta configuração. A medição foi feita ainda com o limiar fixo em
0,5, antes da escolha na validação, e mostra por que a opção intermediária
precisa existir:</p>

<table>
<thead><tr><th>Intensidade</th><th class="num">Precisão</th><th class="num">Revocação</th>
<th class="num">F1</th><th class="num">F2</th><th class="num">FPR</th><th class="num">AUC-ROC</th></tr></thead>
<tbody>
<tr><td><code>none</code></td><td class="num">0,824</td><td class="num">0,592</td>
<td class="num">0,689</td><td class="num">0,627</td><td class="num">0,0002</td><td class="num">0,961</td></tr>
<tr><td><code>sqrt-balanced</code></td><td class="num">0,663</td><td class="num">0,803</td>
<td class="num"><strong>0,726</strong></td><td class="num"><strong>0,770</strong></td><td class="num">0,0007</td><td class="num">0,971</td></tr>
<tr><td><code>balanced</code></td><td class="num">0,053</td><td class="num">0,859</td>
<td class="num">0,100</td><td class="num">0,213</td><td class="num">0,0256</td><td class="num">0,970</td></tr>
</tbody></table>

<p>A ponderação completa leva a revocação de 0,592 a 0,859, mas derruba a
precisão de 0,824 para 0,053 e multiplica a FPR por 128. A moderada fica com o
melhor F1 <em>e</em> o melhor F2. É exatamente o argumento de Cui et al. contra
o inverso direto da frequência: ele produz compensação excessiva.</p>

<div class="callout">
<p><strong>Os pesos só rodam sem reamostragem.</strong> R vem do treino da
dobra, e só dele — nunca do teste, nunca de uma razão sobre a base completa. Nas
configurações sem reamostragem esse treino tem a prevalência natural, e R é a
razão que a ponderação deve compensar. Num treino já reamostrado para 1:1,
R ≈ 1 e os pesos só repetiriam o braço <code>none</code>; com Tomek Links, que
mal altera a razão, a ponderação corrigiria o desbalanceamento duas vezes. Por
isso os executores só pedem <code>sqrt-balanced</code> e <code>balanced</code>
nas configurações sem reamostragem, e o código recusa a combinação.</p>
<p>A intensidade é parâmetro de <em>execução</em>, não de preparação: não entra
no <code>configuration_md5</code>, mas entra na linha de resultado, junto com o
R efetivamente medido, e na chave que decide se uma execução já foi registrada.
Naive Bayes e KNN ficam de fora: o primeiro estima densidades condicionais, o
segundo decide por votação entre vizinhos, e nenhum tem noção de peso por
classe.</p>
</div>

<h4>Fundamentação</h4>
<p>A literatura fundamenta o aprendizado sensível a custos e a ponderação por
frequência, mas <strong>não demonstra que R ou √R sejam universalmente
ótimos</strong>. Por isso as duas intensidades entram como braços, ao lado da
ausência de ponderação, e a escolha entre braços é feita na validação,
mantendo o teste reservado para a avaliação final.</p>
<ul class="spaced">
<li>King e Zeng (2001), <em>Logistic Regression in Rare Events Data</em>,
<em>Political Analysis</em> — eventos raros criam dificuldades específicas para
a regressão logística, incluindo viés nas probabilidades estimadas.</li>
<li>Elkan (2001), <em>The Foundations of Cost-Sensitive Learning</em>, IJCAI —
fundamentação decisória, interpretando pesos como diferenças no custo dos
erros.</li>
<li>He e Garcia (2009), <em>Learning from Imbalanced Data</em>, IEEE TKDE —
revisão de reamostragem, aprendizado sensível a custos e alteração dos pesos.</li>
<li>Cui et al. (2019), <em>Class-Balanced Loss Based on Effective Number of
Samples</em>, CVPR — o inverso direto da frequência pode produzir compensação
excessiva; justifica a alternativa intermediária.</li>
</ul>

<h4>Limiar de decisão: escolhido na validação</h4>

<p>Os pesos não resolvem o problema do limiar: assim como a reamostragem, eles
deslocam a probabilidade <em>a priori</em>, e 0,5 continua arbitrário sobre
essas probabilidades. Por isso, em <strong>todos</strong> os braços, blocos e
algoritmos — clássicos, boosting, FT-Transformer e TabBERT —, o limiar deixou
de ser fixo. Ele é o valor que <strong>maximiza F2 sobre as previsões de
validação</strong> (correção (i) do item G11), e é esse valor que se aplica ao
teste. F2 pesa a revocação quatro vezes mais que a precisão, o que corresponde
ao custo assimétrico da fraude.</p>

<table>
<thead><tr><th>Protocolo</th><th>De onde vem o limiar</th><th><code>threshold_strategy</code></th></tr></thead>
<tbody>
<tr><td><code>stratified-holdout</code>, <code>temporal-holdout</code></td>
<td>Validação da própria dobra, aplicado ao teste</td><td><code>validation-f2</code></td></tr>
<tr><td><code>stratified-k-fold</code></td>
<td>Validação das <strong>outras</strong> dobras, reunidas. Aqui a validação é
o próprio conjunto avaliado, e escolher o limiar nela inflaria as
métricas</td><td><code>cross-fold-validation-f2</code></td></tr>
<tr><td>Validação sem as duas classes</td>
<td>Reserva de 0,5, porque o F2 não se define</td><td><code>fixed</code></td></tr>
</tbody></table>

<p>O limiar escolhido e a estratégia são gravados em cada linha de resultado.
Nos <em>holdouts</em>, as métricas da linha <code>split=validation</code> usam o
limiar escolhido nessa mesma validação e são, portanto, otimistas quanto ao
ponto de corte; o relato final vem da linha <code>split=test</code>.
<code>THRESHOLD_STRATEGY=fixed</code> volta ao limiar fixo, para reproduzir
resultados antigos.</p>

<h3>TabBERT</h3>
<p>{len(tabbert_entries)} configurações, nas bases que têm titular e ordem. Cada
uma roda com duas cabeças de classificação (<code>mlp</code>,
<code>lstm</code>) e <strong>três regimes de treino</strong>, todos com o mesmo
total de 10 épocas supervisionadas (item C3.19 da revisão):</p>

<table>
<thead><tr><th>Regime</th><th>Sequência de etapas</th><th>Épocas</th><th>O que mede</th></tr></thead>
<tbody>
<tr><td><code>supervised-only</code> (S)</td><td>ajuste supervisionado</td><td class="num">10</td><td>a referência sem pré-treino</td></tr>
<tr><td><code>mlm-before-finetune</code> (PM→S)</td><td>MLM, ajuste supervisionado</td><td class="num">3 + 10</td><td>a contribuição do pré-treino</td></tr>
<tr><td><code>warmup-mlm-finetune</code> (S→PM→S)</td><td>aquecimento, MLM, ajuste final</td><td class="num">3 + 3 + 7</td><td>se orientar a representação para a classe minoritária antes do pré-treino ajuda</td></tr>
</tbody></table>

<p>O antigo <code>mlm-after-warmup</code> somava o aquecimento a um ajuste completo, com
um terço a mais de épocas supervisionadas, e é recusado. O MLM mascara campos da
sequência e treina o modelo a reconstruí-los, sem usar o rótulo de fraude. Os
regimes leem os mesmos arquivos preparados: a comparação é pareada dobra a
dobra, e o regime não entra no <code>configuration_md5</code>.</p>
</section>

<section id="validacao">
<h2>Validação contra vazamento</h2>

<p>Dois scripts, dois ângulos. Nenhum confia no código: ambos olham para os
dados.</p>

<pre><code>python3 validate_prepared_datasets.py --mode prepared
python3 validate_prepared_datasets.py --mode partition --dataset sparkov
python3 validate_prepared_datasets.py --mode partition   # todas as bases</code></pre>

<h3>Modo <code>prepared</code> — auditoria dos arquivos gravados</h3>
<table>
<thead><tr><th>ID</th><th>Verificação</th><th>Protege contra</th></tr></thead>
<tbody>
<tr><td><strong>P1</strong></td><td>MD5 do conteúdo de cada arquivo confere com o do manifesto</td><td>métrica atribuída a dados que não são os daquela configuração</td></tr>
<tr><td><strong>P2</strong></td><td>Nenhuma linha aparece em duas partições da mesma dobra</td><td>reuso de linha entre treino e teste</td></tr>
<tr><td><strong>P3</strong></td><td>Nenhum identificador, agrupamento ou marca de tempo sobrou nos atributos</td><td>memorização via <code>trans_num</code>, <code>cc_num</code>, <code>TransactionID</code></td></tr>
<tr><td><strong>P4</strong></td><td>Validação e teste mantêm a prevalência da base</td><td>reamostragem vazando para fora do treino</td></tr>
<tr><td><strong>P5</strong></td><td><code>fold_index</code> começa em 1 e é contíguo</td><td>dobras perdidas ou renumeradas</td></tr>
<tr><td><strong>P6</strong></td><td>Duas configurações distintas nunca produzem o mesmo conjunto de arquivos</td><td>as duplicatas de <code>eqm-ref-21</code> e as 8 cópias por base sem categóricas</td></tr>
<tr><td><strong>P7</strong></td><td><code>eqm-ref-21</code>/<code>ulb-pca</code> não aparecem no manifesto</td><td>reintrodução acidental das bases banidas</td></tr>
<tr><td><strong>P8</strong></td><td>Nos modos de Transformer, o vocabulário da dobra não lista nenhum identificador como campo do modelo</td><td>identificador virando atributo no TabBERT, onde a coluna precisa sobreviver no arquivo</td></tr>
<tr><td><strong>P9</strong></td><td>Mesma base, protocolo, semente, dobra e split têm o mesmo <code>partition_md5</code> em todos os blocos</td><td>blocos preparados em momentos diferentes que deixaram de avaliar as mesmas linhas (carga, código ou biblioteca mudou)</td></tr>
</tbody></table>

<div class="callout">
<p><strong>Por que P8 existe?</strong> No TabBERT as colunas de titular e de
ordem <em>precisam</em> sobreviver no arquivo preparado: o índice de janelas
causais as referencia. P3, sozinha, não conseguiria distinguir isso de um
descuido. P8 lê o vocabulário gravado para cada dobra, que lista exatamente os
campos que o modelo recebe — é a leitura mais direta possível da pergunta "o
identificador virou atributo?".</p>
<p>Essa verificação encontrou um vazamento real durante o desenvolvimento: no
modo sequencial, <code>TransactionID</code>, <code>TransactionDT</code>,
<code>approximate_holder_id</code> e, na Sparkov, <code>cc_num</code> e
<code>trans_num</code> entrariam como campos do modelo. A lista de exclusão
passou a ser derivada do catálogo de bases, com uma única fonte de verdade.</p>
</div>

<h3>Modo <code>partition</code> — reconstrução a partir da base bruta</h3>
<p>Verifica o que os arquivos preparados já não conseguem mostrar, porque as
colunas foram removidas. Refaz a partição com a mesma semente, <em>sem</em> o
descarte, e inspeciona.</p>
<table>
<thead><tr><th>ID</th><th>Verificação</th><th>Protege contra</th></tr></thead>
<tbody>
<tr><td><strong>D1</strong></td><td>Nenhum cliente/cartão/titular em duas partições</td><td>vazamento por titular (Home Credit, Sparkov, IEEE-CIS)</td></tr>
<tr><td><strong>D2</strong></td><td>No protocolo temporal, todo treino é anterior a validação e teste</td><td>vazamento temporal</td></tr>
<tr><td><strong>D3</strong></td><td>O treino tem as duas classes</td><td>partição degenerada</td></tr>
<tr><td><strong>D4</strong></td><td>Os índices de origem das partições são disjuntos</td><td>reuso de linha, prova independente dos atributos</td></tr>
</tbody></table>

<div class="callout warn">
<p><strong>D1 sob o protocolo temporal é informativo, não falha.</strong> Na
Sparkov, 908 dos 999 cartões aparecem dos dois lados do corte temporal — e devem
aparecer: o que separa as partições é o <em>período</em>, não o titular, e esse
é o cenário de operação, em que o sistema conhece o histórico do cartão e decide
sobre a próxima transação dele. Exigir grupos disjuntos ali mediria outra coisa:
generalização para titulares nunca vistos. A garantia desse protocolo é D2.</p>
</div>

<h3>Validações já executadas</h3>
<table>
<thead><tr><th>Base</th><th>Modo</th><th class="num">Verificações</th><th class="num">Falhas</th></tr></thead>
<tbody>
<tr><td><code>ulb</code></td><td>partition</td><td class="num">15</td><td class="num">0</td></tr>
<tr><td><code>ulb</code></td><td>prepared (4 configurações)</td><td class="num">34</td><td class="num">0</td></tr>
<tr><td><code>sparkov</code></td><td>partition</td><td class="num">22</td><td class="num">0</td></tr>
<tr><td><code>ieee-cis-transacional</code></td><td>partition</td><td class="num">22</td><td class="num">0</td></tr>
<tr><td><code>ieee-cis-completa</code> (TabBERT, modo sequencial)</td><td>prepared</td><td class="num">8</td><td class="num">0</td></tr>
</tbody></table>

<p>Exemplos do que foi provado com dados:</p>
<pre><code>[OK] D4 sparkov/stratified-holdout/dobra-1: 1852394 linhas distribuidas em
     particoes disjuntas por indice de origem (train=1280919, validation=301482,
     test=269993)
[OK] D1 sparkov/stratified-holdout/dobra-1: 999 grupos de 'cc_num'
     totalmente separados entre as particoes
[OK] D2 sparkov/temporal-holdout/dobra-1: treino inteiramente anterior a
     validacao e ao teste: train=[2019-01-01 00:00:18, 2020-03-19 07:30:44],
     validation=[2020-03-19 07:30:50, 2020-06-21 12:13:37],
     test=[2020-06-21 12:14:25, 2020-12-31 23:59:34]
[OK] D1 ieee-cis-transacional/stratified-holdout/dobra-1: 243759 grupos de
     'approximate_holder_id' totalmente separados entre as particoes
[OK] P4 .../dobra-1: prevalencia validacao/teste ['0.0017','0.0017']
     (treino 0.5)
[OK] P8 ieee-cis-completa/ae4a7b9a: 1 vocabulario(s) sem identificador, agrupamento
     ou marca de tempo entre os campos do modelo</code></pre>

<p>A última linha é a demonstração direta de que a reamostragem toca
<strong>só</strong> o treino: ele vai a 1:1 (0,5) enquanto validação e teste
permanecem em 0,17%.</p>

<h3>Rastreabilidade</h3>

<p>O contrato é de <strong>uma linha por (dobra, split)</strong>, e não uma
linha por dobra. A versão anterior gravava um único split, escolhido pela
prioridade <code>test &gt; validation &gt; train</code>: como os três protocolos
sempre produzem teste, nenhuma linha <code>split=validation</code> chegava ao
CSV e as métricas de validação eram calculadas e descartadas na escrita. Sem
elas, a seleção da melhor configuração só podia acontecer sobre a própria
partição em que o resultado é relatado — a queixa do item G12.</p>

<div class="callout ok">
<p><strong>Protocolo de seleção.</strong> A escolha da melhor configuração usa
as linhas <code>split=validation</code>; <code>split=test</code> fica reservado
para o relato final, e <code>split=train</code> serve só para diagnosticar
sobreajuste. A seleção e a busca de hiperparâmetros permanecem na fase de
análise, fora do pipeline: o que mudou é que agora existe onde fazê-las sem
olhar o teste.</p>
<p>Nos blocos clássico e de boosting isso não custou cálculo algum — as métricas
dos três splits já eram computadas e apenas não eram escritas. Nos Transformers
a validação passou a ser efetivamente avaliada; o treino fica de fora só nessa
família, porque ali exigiria uma passada de inferência inteira sobre o maior
split para produzir uma medida que não entra em seleção nem em relato.</p>
<p>No dashboard, <code>split=test</code> é pré-selecionado ao carregar o
arquivo, de modo que a leitura padrão não some os três splits na mesma
média.</p>
</div>

<p>Cada linha de resultado carrega, além das métricas:</p>

<table>
<thead><tr><th>Campo</th><th>O que permite</th></tr></thead>
<tbody>
<tr><td><code>fold_index</code></td><td>Dobra que produziu a métrica, 1-based e
sempre preenchida — 1 quando o protocolo não tem dobras</td></tr>
<tr><td><code>split</code></td><td>Partição que produziu <em>esta</em>
métrica: <code>train</code>, <code>validation</code> ou <code>test</code>. Com
<code>fold_index</code>, forma a chave que distingue as linhas de uma mesma
dobra</td></tr>
<tr><td><code>train_dataset_md5</code><br><code>validation_dataset_md5</code><br><code>test_dataset_md5</code></td>
<td>MD5 do <strong>conteúdo</strong> de cada arquivo efetivamente lido. Dada uma
métrica, volta-se aos bytes exatos que a produziram</td></tr>
<tr><td><code>random_seed</code></td><td>Semente da partição; parte da
identidade da configuração, de modo que repetições sejam distinguíveis</td></tr>
<tr><td><code>class_weight_intensity</code></td><td>Intensidade da ponderação
aplicada: <code>none</code>, <code>sqrt-balanced</code> ou
<code>balanced</code></td></tr>
<tr><td><code>class_weight_ratio</code></td><td>O <strong>R efetivamente
medido</strong> no treino daquela dobra. Guardar R, e não só o rótulo da
intensidade, permite reconstruir o peso exato aplicado</td></tr>
<tr><td><code>classification_threshold</code></td><td>Ponto de corte
efetivamente aplicado naquela dobra, escolhido na validação</td></tr>
<tr><td><code>threshold_strategy</code></td><td>Como o limiar foi obtido:
<code>validation-f2</code>, <code>cross-fold-validation-f2</code> ou
<code>fixed</code>. Faz parte da chave que decide se uma execução já foi
registrada: um resultado antigo com limiar fixo não impede a nova
execução</td></tr>
<tr><td><code>roc_auc</code> e <code>auc_pr</code></td><td>As duas áreas sob
curva. A AUC-ROC é independente da prevalência e serve para comparar entre
bases; a AUC-PR é mais informativa sob forte desbalanceamento, mas sua linha de
base é a própria prevalência, então compara algoritmos <strong>dentro</strong>
de uma base</td></tr>
<tr><td><code>regime</code></td><td>Regime de pré-treino do TabBERT; vazio para
os demais algoritmos</td></tr>
<tr><td><code>execution_status</code></td><td><code>success</code> ou
<code>failure</code>: falhas entram como linhas, não como ausências</td></tr>
<tr><td><code>failure_category</code> e <code>failure_message</code></td><td>Motivo
da falha: <code>memoria</code>, <code>convergencia</code>,
<code>particao-vazia</code>, <code>armazenamento</code> ou <code>outra</code>, e a
primeira linha da mensagem do erro; vazios nas linhas de sucesso. Um processo
morto pelo sistema não grava linha: o código de saída fica em
<code>registro_execucoes.csv</code>, gravado pelos executores (item G17)</td></tr>
</tbody></table>

<p>A carga recusa a execução se o arquivo no disco divergir do manifesto, de
modo que nenhuma métrica possa ser atribuída a dados que não são os daquela
configuração.</p>
</section>

<section id="comparacoes">
<h2>Comparações possíveis</h2>

<p>Toda comparação é pareada pela chave <code>configuration_md5 +
database_type + algorithm + fold_index + split + regime</code>, e agora também
por <code>random_seed</code> e <code>classification_threshold</code>.</p>

<h3>1. Entre algoritmos, sobre a mesma partição</h3>
<p>Os quatro blocos partem do <strong>mesmo catálogo, dos mesmos protocolos e da
mesma semente</strong>. Para um dado protocolo e dobra, os conjuntos de
treino/validação/teste são os mesmos:</p>
<ul class="spaced">
<li><strong>dentro da família clássica</strong> — regressão linear × logística ×
Naive Bayes × KNN;</li>
<li><strong>dentro do boosting</strong> — CatBoost × XGBoost;</li>
<li><strong>entre famílias</strong> — clássicos × boosting, agora com uma
estratégia de balanceamento comum (<code>random-undersampling</code> 1:1 nos
dois blocos). Antes isso era impossível: o bloco clássico só tinha RUS+SMOTE e
Tomek Links, o nativo só RUS, e a opção <code>none</code> removia linhas em um e
as mantinha no outro, de modo que as partições sequer coincidiam;</li>
<li><strong>clássicos/boosting × FT-Transformer</strong> — nos dois protocolos
de holdout;</li>
<li><strong>FT-Transformer × TabBERT</strong> — com a ressalva de que o TabBERT
usa partição cronológica por titular, e não os protocolos das demais famílias;
a comparação é entre arquiteturas, sobre a mesma base, não sobre a mesma
partição.</li>
</ul>

<h3>2. Entre as bases</h3>

<p>Esta é a comparação mais delicada, porque a prevalência varia por um fator de
50 entre as bases:</p>

{build_prevalence_table()}

<div class="callout warn">
<p><strong>F1, F2, precisão e AUC-PR não são comparáveis entre bases.</strong>
As quatro dependem da prevalência do conjunto avaliado. Um F1 de 0,30 na ULB
(0,173%) e um F1 de 0,30 na IEEE-CIS (7,85%) descrevem desempenhos muito
diferentes; o mesmo vale para a AUC-PR, cuja linha de base é a própria
prevalência, e não 0,5. Para comparar <em>entre</em> bases, use
<strong>AUC-ROC</strong>, que é independente da prevalência, ou o
<strong>ranking médio</strong> dos algoritmos dentro de cada base.</p>

<div class="callout">
<p><strong>As duas áreas sob curva servem a perguntas diferentes.</strong> A
AUC-ROC usa a taxa de falsos positivos, cujo denominador é o número de
negativos — enorme nestas bases. Um aumento grande em termos absolutos de
falsos alarmes mal desloca a curva, então ela é otimista sob forte
desbalanceamento. A AUC-PR mede o que custa em operação: quantos dos alarmes
são falsos.</p>
<p>A primeira medição do <em>pipeline</em> já mostra a tensão. Na ULB, a
AUC-ROC diz que a ponderação moderada é a melhor (0,971 contra 0,961 sem
ponderação); a AUC-PR diz o contrário (0,675 contra 0,686). <strong>Dentro de
uma base, prefira a AUC-PR</strong> para decidir; a AUC-ROC fica para a
comparação entre bases.</p>
</div>
</div>

<h4>O que se pretende comparar</h4>
<table>
<thead><tr><th>Comparação</th><th>O que mede</th><th>Métrica adequada</th></tr></thead>
<tbody>
<tr><td>ULB × Sparkov × IEEE-CIS</td>
<td>Consistência do desempenho relativo entre três fontes de fraude em cartão,
com características muito diferentes: componentes principais anonimizados,
dados sintéticos e dados reais com centenas de atributos</td>
<td>AUC-ROC; ranking médio por base</td></tr>
<tr><td>Dimensionalidade: Sparkov (23 colunas) × IEEE-CIS (394–434)</td>
<td>Se a largura da base explica o desempenho. A versão anterior afirmava que
sim, mas invertia os números: a Sparkov é a base com <em>menos</em> colunas</td>
<td>AUC-ROC pareada por algoritmo</td></tr>
<tr><td>Dado real × sintético: ULB/IEEE-CIS × Sparkov</td>
<td>Se conclusões obtidas em base sintética se transferem. A Sparkov é gerada
pela ferramenta <em>Sparkov Data Generation</em>; isso limita a transferência
dos seus resultados</td>
<td>AUC-ROC; ordenação dos algoritmos</td></tr>
<tr><td>Prevalência: ULB (0,17%) × IEEE-CIS (7,85%)</td>
<td>Como o desbalanceamento afeta o ganho de cada braço de balanceamento e o
limiar escolhido na validação. É aqui que os braços <code>none</code> e de
pesos de classe se tornam informativos</td>
<td>Diferença de F2 entre braços de balanceamento, dentro de cada base</td></tr>
<tr><td>Fraude × inadimplência: bases de cartão × Home Credit</td>
<td>Controle fora do domínio. A Home Credit tem alvo de inadimplência e unidade
de cliente, não de transação</td>
<td>Relatar <strong>separadamente</strong>, nunca na média das bases de fraude</td></tr>
<tr><td>Disponibilidade de protocolo temporal</td>
<td>ULB, Sparkov e IEEE-CIS têm ordem temporal; a Home Credit não. A diferença
entre holdout embaralhado e temporal só é mensurável nas três primeiras</td>
<td>Diferença de revocação e AUC-ROC entre protocolos</td></tr>
</tbody></table>

<div class="callout">
<p><strong>Conte fontes, não configurações.</strong> Ao generalizar, são
<strong>4 fontes</strong>, não 6 bases: <code>ieee-cis-completa</code>,
<code>-noidentity</code> e <code>-transaction</code> são três visões da mesma
IEEE-CIS. Frases como "vence nas cinco bases" superestimavam a amplitude da
conclusão.</p>
</div>

<h3>3. Dentro da IEEE-CIS: duas ablações independentes</h3>
<p>As três visões foram construídas para isolar um fator cada:</p>
<table>
<thead><tr><th>Par</th><th>Linhas</th><th>Atributos</th><th>Isola</th></tr></thead>
<tbody>
<tr><td><code>ieee-cis-completa</code> × <code>ieee-cis-sem-identidade</code></td>
<td>144.233 <strong>iguais</strong></td><td>com × sem identidade</td>
<td><strong>Valor das colunas de identidade</strong>, com o conjunto de linhas fixo</td></tr>
<tr><td><code>ieee-cis-sem-identidade</code> × <code>ieee-cis-transacional</code></td>
<td>144.233 × 590.540</td><td><strong>iguais</strong> (só transacionais)</td>
<td><strong>Efeito do volume</strong>, com o conjunto de atributos fixo</td></tr>
</tbody></table>
<p>Com <code>drop-incomplete-rows</code>, o par com/sem identidade deixava de
comparar a mesma população, porque a visão com identidade tem 40 colunas a mais,
muitas quase vazias. Como a remoção agora vale só para o treino, o
<strong>conjunto de teste permanece idêntico</strong> nas duas visões e o
pareamento se sustenta.</p>

<h3>4. Efeito de cada decisão de preparação</h3>
<p>Pares que diferem em <strong>um</strong> eixo, com todo o resto fixo:</p>
<table>
<thead><tr><th>Comparação</th><th>Eixo isolado</th><th>Bases</th></tr></thead>
<tbody>
<tr><td><code>smoothed-mean</code> × <code>log-odds</code>, com o mecanismo fixo</td>
<td><strong>estatística</strong> da codificação categórica</td><td>19, 20, 22, 22-noid, 22-trans</td></tr>
<tr><td><code>k-fold</code> × <code>permutation-prefix</code>, com a estatística fixa</td>
<td><strong>mecanismo de controle de vazamento</strong> da codificação categórica</td>
<td>19, 20, 22, 22-noid, 22-trans</td></tr>
<tr><td><code>mean</code> × <code>median</code> × <code>mode</code> × <code>drop-incomplete-rows</code></td><td>imputação</td><td>19, 22, 22-noid, 22-trans</td></tr>
<tr><td><code>none</code> × <code>random-undersampling</code> × <code>RUS+SMOTE</code> × <code>tomek-links</code> × <code>sqrt-balanced</code> × <code>balanced</code></td>
<td>balanceamento — os dois últimos são pesos de classe (√R e R, com R medido
no treino de cada dobra), aplicados sem reamostragem</td>
<td>todas (pesos só na regressão logística, CatBoost e XGBoost)</td></tr>
<tr><td><code>quantile</code> × <code>standard</code></td><td>normalização numérica (FT-Transformer)</td><td>todas</td></tr>
<tr><td><code>minimum-frequency</code> × <code>hashing</code></td><td>alta cardinalidade (Transformers)</td><td>todas</td></tr>
<tr><td><code>mlp</code> × <code>lstm</code></td><td>cabeça de classificação</td><td>todas</td></tr>
<tr><td><code>supervised-only</code> × <code>mlm-before-finetune</code> × <code>warmup-mlm-finetune</code></td><td>regime de treino (TabBERT)</td><td>bases do TabBERT</td></tr>
</tbody></table>
<p>O braço <code>none</code> do balanceamento é o que torna a pergunta
"balancear ajuda?" respondível pela primeira vez.</p>
{build_encoding_factorial_section()}

<h3>5. Efeito do protocolo de avaliação</h3>
<p><code>stratified-holdout</code> × <code>stratified-k-fold</code> ×
<code>temporal-holdout</code> sobre a mesma configuração de preparação. A
diferença entre os holdouts embaralhado e temporal é a medida direta do quanto
as partições embaralhadas superestimam o desempenho esperado em operação —
exatamente a diferença que, na versão anterior, aparecia como revocação de 0,697
contra 0,996–0,999 e era atribuída a uma "tarefa mais exigente".</p>
<p>O <code>stratified-k-fold</code> fornece a dispersão entre partições (5
dobras), que permite dizer se uma diferença de 0,007 em F1 entre codificações é
sinal ou ruído.</p>

<h3>6. O que continua não comparável</h3>
<ul class="spaced">
<li><strong>F1, F2, precisão e AUC-PR entre bases</strong> com prevalências
diferentes. A AUC-PR tem como linha de base a própria prevalência, e não 0,5:
uma AUC-PR de 0,69 na ULB (prevalência 0,17%) e uma de 0,69 na IEEE-CIS (7,85%)
descrevem desempenhos muito diferentes.</li>
<li><strong>Resultados deste pipeline com os do anterior.</strong> As partições,
os atributos e a população mudaram. A comparação honesta é conceitual, não
numérica.</li>
<li><strong>Números de Itoo et al.</strong>, artigo retratado em 30/07/2026.</li>
<li><strong>TabBERT e FT-Transformer no <code>stratified-k-fold</code>.</strong>
Os dois só rodam nos holdouts, por custo. O pareamento entre as quatro famílias
vale nos holdouts (<code>temporal-holdout</code>, e
<code>stratified-holdout</code> na Home Credit); no k-fold, só clássicos e
boosting são pareados.</li>
</ul>
</section>

<section id="observacoes">
<h2>Como cada observação foi endereçada</h2>

<p>Os itens seguem a numeração do relatório de revisão: <strong>G1–G23</strong>
(problemas estruturais e metodológicos), <strong>F1–F11</strong> (formatação,
LaTeX, figuras, tabelas e referências) e os itens <strong>C*.*</strong> pontuais
que dependem de código. <strong>N1</strong> e <strong>N2</strong> são achados
deste trabalho, não registrados na revisão.</p>

<div class="filters">
  <button data-filter="todos" aria-pressed="true">Todos ({total_items})</button>
  <button data-filter="{STATUS_RESOLVED}" aria-pressed="false">Resolvido ({counts.get(STATUS_RESOLVED, 0)})</button>
  <button data-filter="{STATUS_PARTIAL}" aria-pressed="false">Parcial ({counts.get(STATUS_PARTIAL, 0)})</button>
  <button data-filter="{STATUS_THESIS}" aria-pressed="false">Dissertação ({counts.get(STATUS_THESIS, 0)})</button>
</div>

{build_review_table()}
</section>

<section id="tempo">
<h2>Estimativa de tempo</h2>

<p>A estimativa é construída a partir de medições reais e é auditável: todas as
premissas de extrapolação são constantes nomeadas no topo de
<code>estimate_execution_time.py</code>.</p>

<h3>Base de medição</h3>
<ol class="spaced">
<li><strong>Medições próprias</strong> neste <em>pipeline</em>, na ULB, holdout
sem reamostragem — o cenário mais caro dos clássicos, porque o treino inteiro
entra no ajuste: Naive Bayes 2,1 s; regressão linear 3,6 s; regressão logística
3,8 s; KNN 39,2 s por dobra.</li>
<li><strong>Medianas das execuções históricas</strong> por algoritmo e base,
extraídas dos resultados arquivados — cobrem CatBoost e XGBoost.</li>
<li><strong>Medições de GPU</strong> do FT-Transformer e do TabBERT da execução
anterior (RTX 4050). O TabBERT na Home Credit levou 135 min 32 s com uma época
de MLM mais uma supervisionada, o que dá ~4.065 s por época nessa base.</li>
</ol>

<h3>Premissas</h3>
<p>Regressões e Naive Bayes crescem linearmente com as linhas; o KNN cresce com
<code>n^1.5</code> (entre linear e quadrático, por causa da indexação em
árvore); boosting usa as medianas históricas × 2,5, fator que corrige a
introdução do braço sem reamostragem; o TabBERT escala com as linhas e com o
número de épocas de cada regime (13 e 16). A PCA na Sparkov e o descarte dos
identificadores <strong>reduzem</strong> o custo e não entram na conta: a
estimativa é conservadora nesse ponto.</p>

{build_time_table()}

<h3>O gargalo</h3>
{build_bottleneck_table()}

<div class="callout warn">
<p>O <strong>boosting na Home Credit</strong> e o <strong>TabBERT</strong>
dominam o total. A Home Credit tem 1,41 milhão de linhas, 29 atributos
categóricos de alta cardinalidade e 1.000 iterações no padrão da biblioteca; o
TabBERT precisa de 13 a 16 épocas sobre janelas causais de todo o histórico.</p>
<p><strong>Reduções possíveis, se o prazo apertar</strong> (nenhuma aplicada;
todas mudam o escopo): limitar a Home Credit ao <code>stratified-holdout</code>
no boosting; usar parada antecipada na validação para CatBoost/XGBoost, que a
revisão pede de qualquer forma; reduzir as épocas do TabBERT; rodar os blocos em
paralelo, já que clássicos e boosting usam CPU enquanto os Transformers usam
GPU.</p>
</div>

<p class="muted">Total sequencial estimado:
<strong>{escape(estimator._format_duration(grand_total))}</strong>
(faixa {escape(estimator._format_duration(grand_total * estimator.UNCERTAINTY_LOW))}
– {escape(estimator._format_duration(grand_total * estimator.UNCERTAINTY_HIGH))}).</p>
</section>

<section id="pendencias">
<h2>Pendências</h2>

<h3>Do código</h3>
<ul class="spaced">
<li><strong>Seleção na validação</strong> (G12): o <em>pipeline</em> produz e
grava as métricas de validação, mas a escolha da melhor configuração continua
sendo feita na análise.</li>
<li><strong>Testes estatísticos</strong> (G13): o <em>pipeline</em> fornece a
dispersão entre dobras e o <code>SEED_LIST</code>; Wilcoxon e Friedman são da
fase de análise.</li>
<li><strong>Validação das bases restantes</strong>: Home Credit (as duas
visões) e as duas visões da IEEE-CIS com identidade ainda não passaram pelo
modo <code>partition</code>.</li>
<li><strong>Dashboard e <code>test_data_size</code>.</strong> A AUC-PR já entrou
como métrica opcional no dashboard, de forma retrocompatível. Mas ele ainda
exige <code>test_data_size</code> em <code>FILTER_COLUMNS</code> e
<code>REQUIRED_COLUMNS</code>, coluna que este <em>pipeline</em> não grava mais:
os resultados novos seriam rejeitados na carga. Resolver isso é uma decisão
sobre o dashboard, não sobre o <em>pipeline</em>.</li>
</ul>

<h3>Da dissertação</h3>
<p>Os itens marcados como <em>Dissertação</em> na tabela acima: retratação de
Itoo et al. (G1), enquadramento (G3), Resumo e <em>Abstract</em> (G5), objetivos
e hipóteses (G6), organização dos capítulos (G19), uso de "generalização"
(G20), revisão bibliográfica (G21), terminologia e redação (G22, G23) e toda a
seção de formatação (F1–F11).</p>
</section>

<section id="distribuicao">
<h2>Distribuição da execução</h2>

<p>A distribuição da execução dos algoritmos e das bases dar-se-á conforme a
tabela de <a href="#protocolos">particionamento por família de algoritmos</a>,
que define a estratégia de criação dos subconjuntos de dados de cada família,
e de acordo com a <a href="#grade">grade experimental</a>, que define quais
configurações existem para cada base. Nenhuma combinação é escolhida à mão: a
grade decide as bases de cada bloco, e o bloco decide os algoritmos.</p>

<h3>Do bloco ao algoritmo</h3>

<p>Os quatro blocos gravam no mesmo manifesto
(<code>prepared_dataset_configurations.tsv</code>), e cada linha leva o
<code>categorical_handling_mode</code> do bloco que a preparou. Cada executor lê
só as linhas do seu modo e ignora as demais, de modo que a ordem em que os
blocos são preparados não importa.</p>

{build_execution_routing_table()}

<p>Todos os scripts recebem como primeiro argumento o diretório dos arquivos
preparados (<code>/media/marcospiaia/dayasets</code> ou
<code>/media/marcospiaia/new-datasets</code>); a execução de um bloco usa o
mesmo diretório da sua preparação.</p>

<h3>Combinações de base e bloco</h3>

<p>Cada célula indica <em>configurações preparadas / execuções</em>, em que uma
execução é um par configuração/algoritmo — incluindo cada intensidade de
ponderação de classe e, no TabBERT, cada regime de pré-treino. Os braços de
pesos de classe (<code>[sqrt-balanced]</code> e <code>[balanced]</code>) só
rodam sobre as configurações sem reamostragem. O traço marca as combinações que
não serão executadas.</p>

{build_execution_combinations_table()}

<p>As ausências no TabBERT são deliberadas: o bloco só se aplica a bases com
titular e ordem. A ULB (<code>ulb</code>) não tem portador, e a Home
Credit sem junção (<code>home-credit-somente-aplicacao</code>) tem uma linha por
cliente, sem histórico a modelar.</p>
</section>

<footer>
<p>Documento gerado por <code>build_public_report.py</code> a partir do código do
<em>pipeline</em>: as tabelas de bases, grade e estimativa de tempo são lidas de
<code>experiment_datasets.py</code>, <code>experiment_grid.py</code> e
<code>estimate_execution_time.py</code>, e a tabela de observações, de
<code>review_items.py</code>.</p>
<p>{escape(INSTITUTION)} · {escape(AUTHOR)} · {escape(generation_date)}</p>
</footer>

</main>
</div>
<script>{FILTER_SCRIPT}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        default="report",
        help="Diretorio onde index.html sera gravado. Padrao: report/",
    )
    arguments = parser.parse_args()

    output_directory = Path(arguments.output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / "index.html"
    output_path.write_text(build_html(), encoding="utf-8")

    size_kilobytes = output_path.stat().st_size / 1024
    print(f"Relatorio gravado em {output_path} ({size_kilobytes:.0f} KB)")
    print(f"Itens de revisao: {len(review_items.REVIEW_ITEMS)}")
    for status, count in sorted(review_items.status_counts().items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
