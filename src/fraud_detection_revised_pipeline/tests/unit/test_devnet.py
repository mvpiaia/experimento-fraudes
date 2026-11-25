"""DevNet com fracao de rotulos (item G3 da revisao).

Protege o desenho do experimento de rotulos escassos: o sorteio das fraudes
rotuladas e reprodutivel e independente da particao, a fracao vira
ceil(fracao x fraudes do treino) em cada dobra, o orcamento nao e reduzido em
silencio, a perda e a dos autores, e fracoes e sorteios diferentes sao
execucoes diferentes na retomada.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import anomaly_detection_specific as anomaly  # noqa: E402
import devnet_model  # noqa: E402
from test_anomaly_detection import (  # noqa: E402
    TARGET,
    _test_data_information,
    prepare_native_configuration,
)

# Treino curto: os testes verificam o mecanismo, nao reproduzem o artigo.
FAST_SETTINGS = devnet_model.DevNetSettings(epochs=5, batches_per_epoch=10, batch_size=64)


def test_label_draw_is_reproducible_and_seed_dependent():
    is_positive = np.zeros(1000, dtype=bool)
    is_positive[::10] = True

    first = devnet_model.draw_labeled_anomalies(is_positive, 30, draw_seed=0)
    again = devnet_model.draw_labeled_anomalies(is_positive, 30, draw_seed=0)
    other = devnet_model.draw_labeled_anomalies(is_positive, 30, draw_seed=1)

    np.testing.assert_array_equal(first, again)
    assert not np.array_equal(first, other)
    assert len(first) == 30 and is_positive[first].all()


def test_budget_above_available_positives_fails_explicitly():
    is_positive = np.array([True] * 5 + [False] * 95)
    with pytest.raises(ValueError, match="exceeds the 5 positives"):
        devnet_model.draw_labeled_anomalies(is_positive, 10, draw_seed=0)


def test_deviation_loss_matches_the_published_formula():
    reference = torch.tensor([-1.0, 1.0])  # media 0, desvio-padrao populacional 1
    scores = torch.tensor([0.0, 2.0])
    labels = torch.tensor([0.0, 1.0])

    loss = devnet_model.deviation_loss(scores, labels, reference, margin=5.0)

    # Nao rotulado: |0| = 0. Anomalia: dev = 2, perda = max(0, 5 - 2) = 3.
    assert loss.item() == pytest.approx((0.0 + 3.0) / 2)


def test_trained_network_scores_labeled_kind_of_anomaly_higher():
    rng = np.random.default_rng(0)
    normal = rng.normal(0.0, 1.0, size=(500, 3))
    anomalies = rng.normal(4.0, 0.5, size=(40, 3))
    features = np.vstack([normal, anomalies]).astype(np.float32)
    is_positive = np.r_[np.zeros(500, bool), np.ones(40, bool)]
    labeled = devnet_model.draw_labeled_anomalies(is_positive, 10, draw_seed=0)

    model = devnet_model.train_devnet(features, labeled, FAST_SETTINGS)
    scores = devnet_model.score_devnet(model, features)

    # As 30 fraudes nao rotuladas tambem sobem: e o que o metodo promete.
    unlabeled_anomalies = np.setdiff1d(np.flatnonzero(is_positive), labeled)
    assert scores[unlabeled_anomalies].mean() > scores[:500].mean()


def test_draw_seed_changes_only_the_labeled_subset():
    """Inicializacao e lotes ficam em 42: o mesmo sorteio da o mesmo modelo."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(200, 2)).astype(np.float32)
    labeled = np.arange(5)

    first = devnet_model.score_devnet(devnet_model.train_devnet(features, labeled, FAST_SETTINGS), features)
    again = devnet_model.score_devnet(devnet_model.train_devnet(features, labeled, FAST_SETTINGS), features)
    np.testing.assert_allclose(first, again)


def test_labeled_count_is_the_ceiling_of_the_fraction():
    # Fracoes da revisao sobre as 331 fraudes do treino da ULB.
    assert [devnet_model.labeled_anomaly_count(331, f) for f in (0.01, 0.05, 0.10)] == [4, 17, 34]
    # O erro de ponto flutuante de 0,1 x 60 nao acrescenta uma fraude.
    assert devnet_model.labeled_anomaly_count(60, 0.10) == 6
    # Ao menos uma fraude rotulada, mesmo em treinos pequenos.
    assert devnet_model.labeled_anomaly_count(8, 0.01) == 1
    assert devnet_model.labeled_anomaly_count(8, 1.0) == 8


@pytest.mark.parametrize("fraction", [0.0, -0.1, 1.5])
def test_fraction_outside_the_unit_interval_is_rejected(fraction):
    with pytest.raises(ValueError, match="label_fraction"):
        devnet_model.labeled_anomaly_count(100, fraction)


def test_fold_records_fraction_budget_seed_and_contamination():
    from catboost_xgboost_specific import prepare_fold_data

    rng = np.random.default_rng(0)
    fold_data = prepare_fold_data(_test_data_information(rng)["test_info"][0], TARGET)
    # 0,6 das 8 fraudes do treino: ceil(4,8) = 5.
    fold = anomaly.execute_devnet_fold(fold_data, 0.6, 3, FAST_SETTINGS)

    assert fold["label_fraction"] == 0.6
    assert fold["label_budget"] == 5
    assert fold["execution_seed"] == 3
    assert fold["training_summary"]["labeled_anomalies"] == 5
    # O treino tem 8 fraudes: 3 ficam sem rotulo.
    assert fold["training_summary"]["unlabeled_anomalies"] == 3
    # A base sintetica nao tem ausentes: nenhum indicador entra na rede.
    assert fold["training_summary"]["missing_indicators"] == 0
    assert fold["model_hyperparameters"]["margin"] == 5.0
    assert fold["model_hyperparameters"]["gradient_clip_norm"] == 1.0


def test_run_configuration_records_every_arm_and_resumes_per_arm(tmp_path, monkeypatch):
    original_fold = anomaly.execute_devnet_fold
    monkeypatch.setattr(
        anomaly,
        "execute_devnet_fold",
        lambda fold_data, label_fraction, label_draw_seed=0, settings=None, monetary_columns=(): original_fold(
            fold_data, label_fraction, label_draw_seed, FAST_SETTINGS, monetary_columns
        ),
    )
    _, command_line = prepare_native_configuration(tmp_path, np.random.default_rng(7))
    command_line[command_line.index("isolation-forest")] = "devnet"
    parser = anomaly.build_argument_parser()

    # 8 fraudes no treino: 0,25 rotula 2 e 0,5 rotula 4.
    arms = ["--label-fractions", "0.25", "0.5", "--label-draw-seeds", "0", "1"]
    assert anomaly.run_configuration(parser.parse_args(command_line + arms)) == 0

    results_path = tmp_path / "results" / "resultados_devnet.csv"
    recorded = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    test_rows = recorded[recorded["split"] == "test"]
    assert sorted(
        zip(test_rows["label_fraction"], test_rows["label_budget"], test_rows["execution_seed"])
    ) == [
        ("0.25", "2", "0"), ("0.25", "2", "1"), ("0.5", "4", "0"), ("0.5", "4", "1"),
    ]
    assert json.loads(test_rows["model_hyperparameters"].iloc[0])["hidden_units"] == 20

    # Um braco novo roda; os ja gravados sao pulados.
    more_arms = ["--label-fractions", "0.25", "0.75", "--label-draw-seeds", "0"]
    assert anomaly.run_configuration(parser.parse_args(command_line + more_arms)) == 0
    recorded = pd.read_csv(results_path, dtype=str, keep_default_na=False)
    test_rows = recorded[recorded["split"] == "test"]
    assert len(test_rows) == 5
    assert ("0.75", "6", "0") in set(
        zip(test_rows["label_fraction"], test_rows["label_budget"], test_rows["execution_seed"])
    )


def test_fraction_above_one_is_rejected_before_running(tmp_path):
    _, command_line = prepare_native_configuration(tmp_path, np.random.default_rng(8))
    command_line[command_line.index("isolation-forest")] = "devnet"
    arms = ["--label-fractions", "1.5", "--label-draw-seeds", "0"]
    with pytest.raises(ValueError, match="label fractions must be in"):
        anomaly.run_configuration(
            anomaly.build_argument_parser().parse_args(command_line + arms)
        )
    assert not (tmp_path / "results" / "resultados_devnet.csv").exists()


def test_devnet_without_fractions_is_rejected(tmp_path):
    _, command_line = prepare_native_configuration(tmp_path, np.random.default_rng(9))
    command_line[command_line.index("isolation-forest")] = "devnet"
    with pytest.raises(ValueError, match="requires --label-fractions"):
        anomaly.run_configuration(anomaly.build_argument_parser().parse_args(command_line))


def test_initialization_follows_the_keras_dense_defaults():
    network = devnet_model.DeviationNetwork(input_size=10, hidden_units=20)
    assert torch.count_nonzero(network.hidden.bias) == 0
    # Glorot uniforme: |w| <= sqrt(6 / (fan_in + fan_out)).
    assert network.hidden.weight.abs().max() <= (6 / (10 + 20)) ** 0.5
