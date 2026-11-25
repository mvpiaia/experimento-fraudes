import pandas as pd

from analise_reamostragem import report_rows, summarize_sizes


def _row(technique, mode, rows_after, positives_after, protocol="temporal-holdout", fill="median"):
    return {"dataset_id": "base", "mode": mode, "fill_na_strategy": fill, "categorical_encoding_strategy": "x",
            "technique": technique, "test_strategy": protocol, "configuration_md5": technique + mode,
            "fold_index": 1, "rows_before": 1000, "positives_before": 10,
            "rows_after": rows_after, "positives_after": positives_after}


def test_report_rows_le_antes_e_depois_da_reamostragem():
    split = lambda rows, positives: {"rows": rows, "positives": positives}  # noqa: E731
    report = {"splits": {"train": {"after_imputation": split(1000, 10), "output": split(20, 10)}}}
    assert report_rows(report) == {"rows_before": 1000, "positives_before": 10,
                                   "rows_after": 20, "positives_after": 10}


def test_resumo_usa_o_temporal_e_mede_rus_e_tomek():
    sizes = pd.DataFrame([
        _row("none", "encoded", 1000, 10),
        _row("random-undersampling", "encoded", 20, 10),
        _row("random-undersampling", "native", 20, 10),
        _row("tomek-links", "encoded", 997, 10),
        _row("tomek-links", "encoded", 995, 10),
        _row("random-undersampling", "encoded", 500, 10, protocol="stratified-holdout"),
    ])
    summary = summarize_sizes(sizes).iloc[0]
    assert summary["test_strategy"] == "temporal-holdout"
    assert summary["ratio"] == 99
    assert summary["rus_rows_encoded"] == 20
    assert abs(summary["rus_negatives_kept_encoded"] - 10 / 990) < 1e-12
    assert (summary["tomek_removed_min"], summary["tomek_removed_max"]) == (3, 5)
