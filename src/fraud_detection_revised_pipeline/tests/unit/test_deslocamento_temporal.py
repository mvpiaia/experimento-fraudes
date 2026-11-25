import numpy as np
import pandas as pd

from analise_deslocamento_temporal import (
    adversarial_auc,
    adversarial_frame,
    temporal_split_labels,
)
from data_partitioning import AUXILIARY_SOURCE_COLUMN


def _frame(rng, n, shift=0.0):
    return pd.DataFrame({
        "valor": rng.normal(shift, 1.0, n),
        "ruido": rng.normal(0.0, 1.0, n),
        "categoria": rng.choice(["a", "b", "c"], n),
        "rotulo": rng.integers(0, 2, n),
    })


def test_adversarial_sem_deslocamento_fica_perto_do_acaso():
    rng = np.random.default_rng(0)
    features, origin = adversarial_frame(_frame(rng, 2000), _frame(rng, 2000), "rotulo")
    assert "rotulo" not in features.columns
    assert origin.sum() == 2000
    assert adversarial_auc(features, origin)["auc_mean"] < 0.6


def test_adversarial_detecta_o_atributo_deslocado():
    rng = np.random.default_rng(1)
    features, origin = adversarial_frame(_frame(rng, 2000), _frame(rng, 2000, shift=2.0), "rotulo")
    result = adversarial_auc(features, origin)
    # Deslocamento de 2 desvios: AUC teórica de Phi(2 / sqrt(2)) = 0,92.
    assert result["auc_mean"] > 0.85
    assert result["gain"].index[0] == "valor"


def test_corte_temporal_usa_o_arquivo_auxiliar_como_teste():
    times = pd.date_range("2019-01-01", periods=100, freq="D")
    data = pd.DataFrame({
        "trans_date_trans_time": times,
        "label": 0,
        AUXILIARY_SOURCE_COLUMN: [False] * 80 + [True] * 20,
    }).sample(frac=1.0, random_state=3).reset_index(drop=True)
    split = temporal_split_labels(data, "sparkov")
    assert (split[data[AUXILIARY_SOURCE_COLUMN]] == "teste").all()
    main = data[~data[AUXILIARY_SOURCE_COLUMN]]
    train_times = main.loc[split[main.index] == "treino", "trans_date_trans_time"]
    validation_times = main.loc[split[main.index] == "validação", "trans_date_trans_time"]
    assert train_times.max() < validation_times.min()
    assert len(train_times) == round(80 * 0.70 / 0.85)
