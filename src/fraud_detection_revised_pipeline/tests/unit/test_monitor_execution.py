"""Painel de execucao: tempo da lane restrito as configuracoes do plano."""

import monitor_execution as monitor

HEADER = "timestamp_inicio,timestamp_fim,bloco,configuration_md5,algoritmo,class_weight_intensity,label_fraction,regime,tentativa,codigo_saida,categoria_saida\n"


def _write_log(directory, rows):
    (directory / "registro_execucoes.csv").write_text(HEADER + "".join(rows), encoding="utf-8")


def test_lane_times_ignores_rows_from_another_phase(tmp_path):
    _write_log(tmp_path, [
        "2026-11-09T14:00:00-03:00,2026-11-09T15:00:00-03:00,classic,AAAA,knn,none,,,1,0,sucesso\n",
        "2026-11-13T08:00:00-03:00,2026-11-13T09:00:00-03:00,boosting,bbbb,catboost xgboost,none,,,1,1,falha\n",
    ])
    first, last, nonzero = monitor._lane_times(tmp_path, {"bbbb"})
    assert first == "2026-11-13T08:00:00-03:00"
    assert last == "2026-11-13T09:00:00-03:00"
    assert nonzero == 1


def test_lane_times_without_filter_reads_every_row(tmp_path):
    _write_log(tmp_path, [
        "2026-11-09T14:00:00-03:00,2026-11-09T15:00:00-03:00,classic,aaaa,knn,none,,,1,0,sucesso\n",
        "2026-11-13T08:00:00-03:00,2026-11-13T09:00:00-03:00,boosting,bbbb,catboost xgboost,none,,,1,0,sucesso\n",
    ])
    assert monitor._lane_times(tmp_path)[:2] == ("2026-11-09T14:00:00-03:00", "2026-11-13T09:00:00-03:00")


def test_plan_option_accepts_several_phases():
    arguments = monitor.build_argument_parser().parse_args(["--plan", "fase1", "--plan", "pesada"])
    assert arguments.plan == ["fase1", "pesada"]
