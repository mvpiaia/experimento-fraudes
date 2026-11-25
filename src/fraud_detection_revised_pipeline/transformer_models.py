"""PyTorch implementations used by the prepared Transformer experiments."""

from __future__ import annotations

import torch
from torch import nn


# Tipos de cabecalho de classificacao aceitos por FTTransformer e TabBERT.
# "linear" preserva o cabecalho original de cada modelo, usado por todo
# resultado historico registrado como "ft-transformer"/"tabbert".
CLASSIFIER_HEAD_TYPES = ("linear", "mlp", "lstm")


def _validate_classifier_head(classifier_head: str) -> None:
    if classifier_head not in CLASSIFIER_HEAD_TYPES:
        raise ValueError(
            f"classifier_head must be one of {CLASSIFIER_HEAD_TYPES}, got {classifier_head!r}"
        )


def _build_mlp_head(dimension: int, dropout: float) -> nn.Module:
    """Cabecalho MLP compartilhado: uma camada oculta com a mesma dimensao."""
    return nn.Sequential(
        nn.LayerNorm(dimension),
        nn.Linear(dimension, dimension),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(dimension, 1),
    )


class FTTransformer(nn.Module):
    def __init__(
        self,
        numerical_feature_count: int,
        categorical_cardinalities: list[int],
        dimension: int = 64,
        heads: int = 8,
        layers: int = 3,
        dropout: float = 0.1,
        classifier_head: str = "linear",
    ) -> None:
        super().__init__()
        _validate_classifier_head(classifier_head)
        self.classifier_head = classifier_head
        self.numerical_weights = nn.Parameter(
            torch.empty(numerical_feature_count, dimension)
        )
        self.numerical_bias = nn.Parameter(
            torch.zeros(numerical_feature_count, dimension)
        )
        nn.init.xavier_uniform_(self.numerical_weights)
        self.categorical_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dimension, padding_idx=0)
            for cardinality in categorical_cardinalities
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dimension))
        encoder_layer = nn.TransformerEncoderLayer(
            dimension, heads, dimension * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, layers)
        if classifier_head == "mlp":
            self.head = _build_mlp_head(dimension, dropout)
        elif classifier_head == "lstm":
            # Sem dimensao temporal por amostra, a LSTM roda sobre a sequencia
            # de tokens de campo (ja contextualizados pelo encoder), tratando
            # cada campo como um passo; o estado final agrega a linha inteira,
            # papel que o token CLS cumpre no cabecalho linear/MLP.
            self.sequence_classifier = nn.LSTM(dimension, dimension, batch_first=True)
            self.head = nn.Linear(dimension, 1)
        else:
            self.head = nn.Sequential(nn.LayerNorm(dimension), nn.ReLU(), nn.Linear(dimension, 1))

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        tokens = []
        if numerical.shape[1]:
            tokens.append(
                numerical.unsqueeze(-1) * self.numerical_weights.unsqueeze(0)
                + self.numerical_bias.unsqueeze(0)
            )
        if categorical.shape[1]:
            tokens.append(
                torch.stack(
                    [embedding(categorical[:, index]) for index, embedding in enumerate(self.categorical_embeddings)],
                    dim=1,
                )
            )
        batch_size = numerical.shape[0] if numerical.ndim == 2 else categorical.shape[0]
        feature_tokens = torch.cat(tokens, dim=1)
        cls = self.cls_token.expand(batch_size, -1, -1)
        encoded = self.encoder(torch.cat([cls, feature_tokens], dim=1))
        if self.classifier_head == "lstm":
            _output, (hidden, _cell) = self.sequence_classifier(encoded[:, 1:])
            return self.head(hidden[-1]).squeeze(-1)
        return self.head(encoded[:, 0]).squeeze(-1)


class TabBERT(nn.Module):
    def __init__(
        self,
        field_cardinalities: list[int],
        dimension: int = 64,
        heads: int = 8,
        field_layers: int = 2,
        sequence_layers: int = 3,
        dropout: float = 0.1,
        classifier_head: str = "linear",
    ) -> None:
        super().__init__()
        _validate_classifier_head(classifier_head)
        self.classifier_head = classifier_head
        self.field_embeddings = nn.ModuleList(
            nn.Embedding(cardinality, dimension, padding_idx=0)
            for cardinality in field_cardinalities
        )
        self.field_identity = nn.Parameter(torch.zeros(1, 1, len(field_cardinalities), dimension))
        field_layer = nn.TransformerEncoderLayer(
            dimension, heads, dimension * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        sequence_layer = nn.TransformerEncoderLayer(
            dimension, heads, dimension * 4, dropout, batch_first=True,
            norm_first=True, activation="gelu",
        )
        self.field_encoder = nn.TransformerEncoder(field_layer, field_layers)
        self.sequence_encoder = nn.TransformerEncoder(sequence_layer, sequence_layers)
        if classifier_head == "mlp":
            self.classifier = _build_mlp_head(dimension, dropout)
        elif classifier_head == "lstm":
            # Unidirecional: preserva a ausencia de vazamento de futuro que o
            # sequence_encoder causal ja garante para cada posicao da janela.
            self.sequence_classifier = nn.LSTM(dimension, dimension, batch_first=True)
            self.classifier = nn.Linear(dimension, 1)
        else:
            self.classifier = nn.Sequential(nn.LayerNorm(dimension), nn.Linear(dimension, 1))
        self.mlm_heads = nn.ModuleList(
            nn.Linear(dimension, cardinality) for cardinality in field_cardinalities
        )

    def encode(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, fields = tokens.shape
        embedded = torch.stack(
            [embedding(tokens[:, :, index]) for index, embedding in enumerate(self.field_embeddings)],
            dim=2,
        ) + self.field_identity
        field_context = self.field_encoder(embedded.reshape(batch * sequence, fields, -1))
        field_context = field_context.reshape(batch, sequence, fields, -1)
        row_context = field_context.mean(dim=2)
        causal_mask = torch.triu(
            torch.ones(sequence, sequence, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        effective_padding_mask = padding_mask.clone()
        # A fully padded causal prefix would leave its first query without any
        # attendable key and produce NaNs. Keep one zero-valued anchor visible.
        effective_padding_mask[:, 0] = False
        sequence_context = self.sequence_encoder(
            row_context,
            mask=causal_mask,
            src_key_padding_mask=effective_padding_mask,
        )
        sequence_context = sequence_context.masked_fill(padding_mask.unsqueeze(-1), 0)
        return field_context, sequence_context

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        _, sequence_context = self.encode(tokens, padding_mask)
        if self.classifier_head == "lstm":
            sequence_context, _ = self.sequence_classifier(sequence_context)
        return self.classifier(sequence_context).squeeze(-1)

    def masked_field_logits(
        self,
        tokens: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> list[torch.Tensor]:
        field_context, _ = self.encode(tokens, padding_mask)
        return [head(field_context[:, :, index]) for index, head in enumerate(self.mlm_heads)]


# Mesmos valores de experiment_grid.TABBERT_REGIMES (conferido nos testes).
TABBERT_TRAINING_REGIMES = ("supervised-only", "mlm-before-finetune", "warmup-mlm-finetune")


def tabbert_training_stages(regime: str) -> tuple[str, ...]:
    """Etapas de cada regime (item C3.19): todos tem o mesmo total de epocas
    supervisionadas.

    ``warmup-mlm-finetune`` (S->PM->S) divide esse total entre o aquecimento
    (um terco) e o ajuste final (o restante), como sugerido pelo revisor. O
    antigo ``mlm-after-warmup`` somava o aquecimento a um ajuste final
    completo, com um terco a mais de epocas supervisionadas, e e recusado.
    """
    if regime == "supervised-only":
        return ("supervised",)
    if regime == "mlm-before-finetune":
        return ("mlm", "supervised")
    if regime == "warmup-mlm-finetune":
        return ("supervised-warmup", "mlm", "supervised")
    raise ValueError(
        f"unknown TabBERT training regime {regime!r}; "
        "expected 'supervised-only', 'mlm-before-finetune' or 'warmup-mlm-finetune'"
    )


def tabbert_supervised_epochs(stages: tuple[str, ...], total_epochs: int) -> dict[str, int]:
    """Epocas de cada etapa supervisionada: o aquecimento leva um terco do total
    e o ajuste final, o restante, de modo que a soma e sempre ``total_epochs``."""
    if "supervised-warmup" not in stages:
        return {"supervised": total_epochs}
    warmup = max(1, total_epochs // 3)
    return {"supervised-warmup": warmup, "supervised": total_epochs - warmup}
