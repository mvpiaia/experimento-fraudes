import pandas as pd

from tabela_execucoes import full_table, load, summary_table


def _row(bloco, algorithm, strategy, previstas, sucesso, equivalente=0, incompleta=0):
    return {"bloco": bloco, "dataset_id": "ulb", "algorithm": algorithm, "test_strategy": strategy,
            "unidades_previstas": previstas, "unidades_sucesso": sucesso, "unidades_falha": 0,
            "unidades_incompleta": incompleta, "unidades_ausente": 0, "unidades_equivalente": equivalente,
            "linhas_previstas": previstas, "linhas_sucesso": sucesso, "linhas_falha": 0,
            "linhas_incompleta": incompleta, "linhas_ausente": 0, "linhas_equivalente": equivalente}


def test_exclui_tabbert_soma_incompletas_as_falhas_e_separa_protocolos(tmp_path):
    path = tmp_path / "appendix_run_counts.csv"
    pd.DataFrame([
        _row("classic", "knn", "stratified-holdout", 4, 3, incompleta=1),
        _row("classic", "knn", "temporal-holdout", 2, 2),
        _row("ft-transformer", "ft-transformer-mlp", "stratified-holdout", 2, 1, equivalente=1),
        _row("ft-transformer", "tabbert-mlp", "stratified-holdout", 9, 0),
    ]).to_csv(path, index=False)
    counts = load(path)
    assert "tabbert-mlp" not in set(counts["algorithm"])
    assert counts["unidades_falha"].sum() == 1
    summary = summary_table(counts, "22/11/2026")
    assert r"\acs{knn} & 6 & 5 & 0 & 1 & 0 & 3 & 2 & 0 \\" in summary
    assert r"Total & 8 & 6 & 1 & 1 & 0 & 4 & 2 & 0 \\" in summary
    assert r"\multicolumn{2}{l}{Total} & 8 & 6 & 1 & 1 & 0 & 8 & 6 \\" in full_table(counts, "22/11/2026")
