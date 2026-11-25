"""Figuras, tabelas, ablação H4 e roteiro da discussão sobre a grade revisada.

Itens F6 e F7 da Seção 4 do relatório de revisão e pendências dos Capítulos
4 a 6 da dissertação.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
TESTS_DIRECTORY = Path(__file__).resolve().parent
for directory in (PIPELINE_DIRECTORY, TESTS_DIRECTORY):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import analise_ablacao_ieee as ablacao  # noqa: E402
import analise_hipoteses  # noqa: E402
import dados_grade_revisada  # noqa: E402
import estilo_figuras as estilo  # noqa: E402
import figuras_discussao  # noqa: E402
import roteiro_capitulos  # noqa: E402
import tabelas_latex  # noqa: E402


@pytest.fixture(scope="module")
def results(tmp_path_factory) -> Path:
    return dados_grade_revisada.build(tmp_path_factory.mktemp("resultados"))


@pytest.fixture(scope="module")
def outputs(results, tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("saidas")
    figures, hypotheses, ablation, tables = (root / name for name in ("figuras", "hip", "h4", "tabelas"))
    summary = figuras_discussao.run(results, figures)
    (root / "figuras_discussao_resumo.json").write_text(
        json.dumps(analise_hipoteses.json_ready(summary), ensure_ascii=False), encoding="utf-8")
    tables_h, summary_h = analise_hipoteses.run_analysis(analise_hipoteses.load_results(results), bootstrap_samples=200)
    analise_hipoteses.write_outputs(tables_h, summary_h, hypotheses)
    tables_a, summary_a = ablacao.run(results, samples=300)
    ablation.mkdir()
    for name, table in tables_a.items():
        table.to_csv(ablation / f"{name}.csv", index=False)
    (ablation / "resumo_h4.json").write_text(json.dumps(analise_hipoteses.json_ready(summary_a)), encoding="utf-8")
    tabelas_latex.run(results, hypotheses, ablation, tables)
    return {"root": root, "figuras": figures, "hip": hypotheses, "h4": ablation, "tabelas": tables}


# ---------------------------------------------------------------- padrão numérico e rótulos

def test_number_uses_comma_and_no_negative_zero():
    assert estilo.number(0.12345) == "0,123"
    assert estilo.number(-0.0004) == "0,000"
    assert estilo.number(-0.0123) == "$-$0,012"
    assert estilo.number_plain(-0.0123) == "−0,012"


def test_labels_are_portuguese_and_descriptive():
    assert estilo.METRIC_LABELS["roc_auc"] == "AUC-ROC"
    assert estilo.METRIC_LABELS["f_beta_2"] == "F2"
    assert estilo.dataset_label("eqm-ref-18") == "ULB"
    assert estilo.dataset_label("ieee-cis-sem-identidade") == "IEEE-CIS-sem-identidade"
    assert estilo.label("mean") == "média"
    assert "linear-regression" not in figuras_discussao.FAMILIES["classicos"]


# ---------------------------------------------------------------- figuras

def test_revised_figures_cover_every_family(outputs):
    written = {path.name for path in outputs["figuras"].glob("*.png")}
    expected = {f"discussao_desempenho_{family}.png" for family in figuras_discussao.FAMILIES}
    expected |= {"discussao_balanceamento_classicos.png", "discussao_codificacao_classicos.png",
                 "discussao_imputacao.png", "discussao_protocolo.png", "discussao_anomalia_rotulos.png",
                 "discussao_transformers.png"}
    assert expected <= written


def test_tabbert_is_excluded_as_future_work(results, outputs):
    loaded = analise_hipoteses.load_results(results)
    raw = pd.concat(pd.read_csv(path) for path in results.glob("resultados_tabbert*.csv"))
    assert not raw.empty
    assert not loaded["algorithm"].isin(analise_hipoteses.TABBERT_ALGORITHMS).any()
    assert loaded["algorithm"].isin(analise_hipoteses.FT_TRANSFORMER_ALGORITHMS).any()
    text = (outputs["root"] / "figuras_discussao_resumo.json").read_text(encoding="utf-8")
    assert "tabbert" not in text.lower()


def test_balancing_baseline_read_as_empty_string(results, tmp_path):
    executions = figuras_discussao.load(results)
    baseline = executions["undersampling_technique"].eq(figuras_discussao.BASELINE_ARM)
    assert baseline.any()
    executions.loc[baseline, "undersampling_technique"] = ""
    summary: dict = {}
    figuras_discussao.balancing_figure(executions, tmp_path, summary)
    assert (tmp_path / "discussao_balanceamento_classicos.png").exists()
    assert summary["balanceamento_classicos"]


def test_figure_summary_has_no_legacy_aliases(outputs):
    text = (outputs["root"] / "figuras_discussao_resumo.json").read_text(encoding="utf-8")
    assert "eqm-ref" not in text
    summary = json.loads(text)
    assert summary["execucoes"] > 0
    # Os cinco braços comparados com a linha de base sem reamostragem.
    balancing = json.dumps(summary["balanceamento_classicos"])
    for arm in ("random-undersampling", "RUS+SMOTE", "tomek-links", "sqrt-balanced", "balanced"):
        assert f'"{arm}"' in balancing


# ---------------------------------------------------------------- ablação H4

@pytest.mark.parametrize("low, high, point, expected", [
    (-0.004, 0.006, 0.001, "nao-mensuravel"),
    (-0.004, 0.014, 0.005, "baixo-inconclusivo"),
    (0.012, 0.030, 0.020, "relevante"),
    (-0.002, 0.030, 0.014, "inconclusivo"),
])
def test_h4_outcome_rule(low, high, point, expected):
    assert ablacao.outcome(low, high, point, 0.010) == expected


def test_h4_hypothesis_outcome():
    assert ablacao.hypothesis_outcome(["nao-mensuravel"] * 5) == "corroborated"
    assert ablacao.hypothesis_outcome(["nao-mensuravel", "relevante"]) == "refuted"
    assert ablacao.hypothesis_outcome(["nao-mensuravel", "baixo-inconclusivo"]) == "inconclusive"


def test_h4_pairs_and_counts_incomplete(results, tmp_path):
    data = ablacao.test_rows(results)
    paired, counts = ablacao.pairs(data)
    assert counts["pares"] > 0 and len(paired) == counts["pares"]
    trimmed = data.drop(data[data["database_type"].eq(ablacao.WITHOUT_IDENTITY)].index[:3])
    _, counts_trimmed = ablacao.pairs(trimmed)
    assert counts_trimmed["so_com_identidade"] == 3


def test_h4_detects_relevant_identity_effect(results):
    data = ablacao.test_rows(results)
    boosted = data.copy()
    mask = boosted["database_type"].eq(ablacao.WITH_IDENTITY)
    for metric in ablacao.MARGINS:
        boosted.loc[mask, metric] = boosted.loc[mask, metric] + 0.05
    paired, _ = ablacao.pairs(boosted)
    summary = ablacao.summarize(paired, samples=200)
    assert set(summary["desfecho"]) == {"relevante"}


# ---------------------------------------------------------------- tabelas LaTeX

def test_tables_follow_thesis_standard(outputs):
    tables = sorted(outputs["tabelas"].glob("tab_*.tex"))
    assert {path.name for path in tables} >= {"tab_execucoes.tex", "tab_h1.tex", "tab_q1.tex", "tab_h4.tex"}
    for path in tables:
        text = path.read_text(encoding="utf-8")
        assert r"\toprule" in text and r"\bottomrule" in text and r"\hline" not in text
        assert text.index(r"\caption") < text.index(r"\begin{tabular}")
        assert "-0,000" not in text and "$-$0,000" not in text
        # Vírgula decimal: nenhum número com ponto decimal (o ponto só separa milhares).
        assert not re.search(r"\d\.\d{1,2}(?!\d)|\d\.\d{4,}", text)
        assert "eqm-ref" not in text


def test_h4_table_reports_outcome(outputs):
    text = (outputs["tabelas"] / "tab_h4.tex").read_text(encoding="utf-8")
    assert r"\acs{aucroc}" in text and "Desfecho de H4" in text


# ---------------------------------------------------------------- roteiro

def test_roadmap_maps_pending_items(outputs, tmp_path):
    thesis = tmp_path / "tese"
    thesis.mkdir()
    (thesis / "resultados.tex").write_text(
        "\\section{Ablação}\\label{sec:resultados-ablacao-ieee}\n\\todo{G19: reexecução} Diferenças pareadas.\n",
        encoding="utf-8")
    (thesis / "conclusoes.tex").write_text(
        "\\section{Síntese}\\label{sec:conclusoes-sintese}\n\\todo{G19: reexecução} Síntese do trabalho.\n",
        encoding="utf-8")
    items = roteiro_capitulos.pending_items(thesis)
    assert [item["secao"] for item in items] == ["sec:resultados-ablacao-ieee", "sec:conclusoes-sintese"]
    text = roteiro_capitulos.render(items, [outputs["tabelas"], outputs["h4"], outputs["hip"], outputs["root"]])
    assert "tab_h4.tex" in text
    assert "sem saída associada: 0" in text
