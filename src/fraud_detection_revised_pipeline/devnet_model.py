"""Deviation Network (DevNet; Pang, Shen e van den Hengel, KDD 2019).

Detector fracamente supervisionado: aprende um escore de anomalia a partir de
poucas anomalias rotuladas e de um conjunto grande de dados nao rotulados, que
pode conter anomalias nao identificadas. A perda empurra o escore dos nao
rotulados para a media de uma distribuicao de referencia N(0, 1) e o das
anomalias rotuladas para pelo menos `margin` desvios-padrao acima dela
(equacoes 2.12 e 2.13 da dissertacao).

Reimplementacao em PyTorch do codigo de referencia dos autores
(github.com/GuansongPang/deviation-network, devnet.py), escrito para Keras
2.2.4. Os hiperparametros sao os dos autores, fixos: o objetivo e medir o
metodo como publicado sob diferentes orcamentos de rotulos, e nao ajusta-lo a
cada base. A origem de cada valor esta anotada em DevNetSettings: parte vem do
artigo (Secoes 4.2 e 5.3), parte so existe no codigo de referencia, e parte e
padrao do Keras 2.2.4 que o codigo herda sem declarar.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class DevNetSettings:
    # Artigo, Secao 5.3: uma camada oculta de 20 unidades ReLU, a arquitetura
    # que os autores relatam como a melhor em todas as bases. No codigo e
    # dev_network_s, o padrao (--network_depth=2).
    hidden_units: int = 20
    # Artigo, Secao 5.3: "an l2-norm regularizer is applied to every hidden
    # layer". O valor 0.01 so aparece no codigo: kernel_regularizer=l2(0.01).
    l2_penalty: float = 0.01
    # Artigo, Secao 4.2: a = 5 da nivel de significancia 5.73e-07 a toda
    # anomalia rotulada; l = 5000 escores de referencia de N(0, 1).
    margin: float = 5.0
    reference_size: int = 5000
    # Artigo, Secao 5.3: RMSprop. O codigo instancia RMSprop(clipnorm=1.), e
    # o resto vem dos padroes do Keras 2.2.4: lr=0.001, rho=0.9 e
    # epsilon=K.epsilon()=1e-7. No Keras 2.2.4 o clipnorm limita a norma
    # GLOBAL do gradiente (Optimizer.get_gradients), nao a de cada tensor.
    learning_rate: float = 1e-3
    rmsprop_alpha: float = 0.9
    rmsprop_epsilon: float = 1e-7
    gradient_clip_norm: float = 1.0
    # Artigo, Secao 5.3: 50 epocas de 20 mini-lotes. O tamanho 512 so aparece
    # no codigo (--batch_size=512).
    batch_size: int = 512
    epochs: int = 50
    batches_per_epoch: int = 20
    # Inicializacao, amostragem dos lotes e escores de referencia. Fixa em 42
    # de proposito: so o sorteio das anomalias rotuladas varia entre
    # repeticoes, para que a dispersao reflita apenas "quais fraudes foram
    # rotuladas".
    seed: int = 42

    def as_dict(self) -> dict:
        return asdict(self)


DEFAULT_DEVNET_SETTINGS = DevNetSettings()


class DeviationNetwork(nn.Module):
    def __init__(self, input_size: int, hidden_units: int = 20) -> None:
        super().__init__()
        self.hidden = nn.Linear(input_size, hidden_units)
        self.output = nn.Linear(hidden_units, 1)
        # Inicializacao do Dense do Keras: Glorot uniforme e vies zero. A do
        # PyTorch (Kaiming uniforme) mudaria o ponto de partida do treino.
        for layer in (self.hidden, self.output):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(torch.relu(self.hidden(features))).squeeze(-1)


def deviation_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    reference_scores: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """(1 - y)|dev| + y max(0, a - dev), com dev = (s - mu_R) / sigma_R.

    sigma_R e o desvio-padrao populacional, como o K.std do Keras.
    """
    deviation = (scores - reference_scores.mean()) / reference_scores.std(unbiased=False)
    inlier_loss = deviation.abs()
    outlier_loss = torch.clamp(margin - deviation, min=0.0)
    return torch.mean((1.0 - labels) * inlier_loss + labels * outlier_loss)


def labeled_anomaly_count(positive_count: int, label_fraction: float) -> int:
    """Fraudes rotuladas na dobra: ceil(fracao x fraudes do treino).

    A fracao, e nao uma contagem fixa, e o parametro do braco: com contagens
    absolutas, 10 fraudes seriam 3% do treino da ULB e menos de 0,2% da
    Sparkov, e o eixo mediria o tamanho da base. O teto garante ao menos uma
    fraude rotulada. O arredondamento antes do teto evita que o erro de ponto
    flutuante (0,1 x 60 = 6,000000000000001) acrescente uma fraude.
    """
    if not 0.0 < label_fraction <= 1.0:
        raise ValueError(f"label_fraction must be in (0, 1], got {label_fraction}")
    if positive_count < 1:
        raise ValueError("the training split has no positives to label")
    return math.ceil(round(label_fraction * positive_count, 9))


def draw_labeled_anomalies(
    is_positive: np.ndarray, label_budget: int, draw_seed: int
) -> np.ndarray:
    """Indices das `label_budget` fraudes que ficam rotuladas.

    Sorteio sem reposicao entre as fraudes do treino. As demais fraudes
    continuam no treino como nao rotuladas, contaminando o conjunto tratado
    como normal, exatamente o cenario que o DevNet foi desenhado para tolerar.
    """
    positive_indices = np.flatnonzero(np.asarray(is_positive, dtype=bool))
    if label_budget < 1:
        raise ValueError(f"label_budget must be at least 1, got {label_budget}")
    if label_budget > len(positive_indices):
        raise ValueError(
            f"label_budget={label_budget} exceeds the {len(positive_indices)} "
            "positives available in the training split"
        )
    generator = np.random.default_rng(draw_seed)
    return np.sort(generator.choice(positive_indices, size=label_budget, replace=False))


def train_devnet(
    features: np.ndarray,
    labeled_anomaly_indices: np.ndarray,
    settings: DevNetSettings = DEFAULT_DEVNET_SETTINGS,
) -> DeviationNetwork:
    """Treina com lotes balanceados: metade anomalias rotuladas, metade nao rotulados."""
    features = np.asarray(features, dtype=np.float32)
    labeled = np.asarray(labeled_anomaly_indices)
    unlabeled = np.setdiff1d(np.arange(len(features)), labeled)
    if len(labeled) == 0 or len(unlabeled) == 0:
        raise ValueError("DevNet needs labeled anomalies and unlabeled rows")

    torch.manual_seed(settings.seed)
    model = DeviationNetwork(features.shape[1], settings.hidden_units)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=settings.learning_rate,
        alpha=settings.rmsprop_alpha,
        eps=settings.rmsprop_epsilon,
    )
    generator = torch.Generator().manual_seed(settings.seed)
    # Como no original, a referencia e sorteada uma vez e fica fixa.
    reference_scores = torch.randn(settings.reference_size, generator=generator)
    batch_rng = np.random.default_rng(settings.seed)
    half_batch = settings.batch_size // 2
    labels = torch.cat([torch.zeros(half_batch), torch.ones(half_batch)])
    feature_tensor = torch.from_numpy(features)

    model.train()
    for _ in range(settings.epochs):
        for _ in range(settings.batches_per_epoch):
            batch_indices = np.concatenate(
                [
                    batch_rng.choice(unlabeled, size=half_batch, replace=True),
                    batch_rng.choice(labeled, size=half_batch, replace=True),
                ]
            )
            optimizer.zero_grad(set_to_none=True)
            scores = model(feature_tensor[batch_indices])
            loss = deviation_loss(scores, labels, reference_scores, settings.margin)
            loss = loss + settings.l2_penalty * model.hidden.weight.pow(2).sum()
            loss.backward()
            # Norma global, como o clipnorm do Keras 2.2.4.
            nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_clip_norm)
            optimizer.step()
    return model


# Linhas por bloco na inferencia. A IEEE-CIS tem 590 mil linhas e ~400
# colunas: pontuar tudo de uma vez criaria um tensor de cerca de 1 GB.
SCORING_CHUNK_SIZE = 65_536


def score_devnet(model: DeviationNetwork, features: np.ndarray) -> np.ndarray:
    """Escore de anomalia: maior para pontos mais anomalos."""
    features = np.asarray(features, dtype=np.float32)
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [
                model(torch.from_numpy(features[start : start + SCORING_CHUNK_SIZE])).numpy()
                for start in range(0, len(features), SCORING_CHUNK_SIZE)
            ]
            or [np.empty(0, dtype=np.float32)]
        )
