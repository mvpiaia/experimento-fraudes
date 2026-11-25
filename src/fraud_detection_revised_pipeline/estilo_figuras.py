"""Padrão visual e de números das figuras e tabelas da dissertação.

Atende aos itens F6 e F7 da revisão:

* rótulos em português, nomes descritivos das bases (``dataset_name`` de
  ``experiment_datasets``) e as siglas da dissertação (AUC-ROC, AUC-PR, F2);
* vírgula decimal (locale ``pt_BR``), três casas e sem zero negativo;
* legenda fora da área dos painéis, fonte de pelo menos 8 pt no tamanho
  impresso (figuras desenhadas com a largura útil da página, 16 cm);
* séries distinguíveis sem cor (marcadores e hachuras) e paleta segura para
  daltonismo.

O módulo só lê constantes da grade; não altera nada da preparação nem da
execução.
"""

from __future__ import annotations

import locale
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

import experiment_datasets  # noqa: E402

# Largura útil da página A4 com margens de 2,5 cm.
PAGE_WIDTH_IN = 16.0 / 2.54
FONT_SIZE = 8
TITLE_SIZE = 9

METRIC_LABELS = {
    "f1": "F1",
    "f_beta_2": "F2",
    "precision": "Precisão",
    "recall": "Revocação",
    "roc_auc": "AUC-ROC",
    "auc_pr": "AUC-PR",
    "accuracy": "Acurácia",
    "false_positive_rate": "FPR",
}
# Siglas como \acs{} nas tabelas LaTeX (pacote acro da dissertação).
METRIC_LATEX = {
    "f1": "F1",
    "f_beta_2": "F2",
    "precision": "Precisão",
    "recall": "Revocação",
    "roc_auc": r"\acs{aucroc}",
    "auc_pr": r"\acs{aucpr}",
    "accuracy": "Acurácia",
    "false_positive_rate": r"\acs{fpr}",
}
ALGORITHM_LABELS = {
    "logistic-regression": "Regressão logística",
    "naive-bayes": "Naive Bayes",
    "knn": "KNN",
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "isolation-forest": "Isolation Forest",
    "devnet": "DevNet",
    "ft-transformer-mlp": "FT-Transformer (MLP)",
    "ft-transformer-lstm": "FT-Transformer (LSTM)",
    "tabbert-mlp": "TabBERT (MLP)",
    "tabbert-lstm": "TabBERT (LSTM)",
    "linear-regression": "Regressão linear",
}
ALGORITHM_LATEX = {**ALGORITHM_LABELS, "knn": r"\acs{knn}",
                   "ft-transformer-mlp": r"FT-Transformer (\acs{mlp})",
                   "ft-transformer-lstm": r"FT-Transformer (\acs{lstm})",
                   "tabbert-mlp": r"TabBERT (\acs{mlp})",
                   "tabbert-lstm": r"TabBERT (\acs{lstm})"}
STRATEGY_LABELS = {
    # balanceamento e ponderação
    "none": "sem reamostragem",
    "random-undersampling": "RUS",
    "RUS+SMOTE": "RUS+SMOTE",
    "tomek-links": "Tomek Links",
    "sqrt-balanced": "peso √R",
    "balanced": "peso R",
    # faltantes
    "mean": "média",
    "median": "mediana",
    "mode": "moda",
    "drop-incomplete-rows": "remoção de incompletos",
    "native": "nativo",
    # codificação (fatorial 2x2)
    "smoothed-mean:k-fold": "média suavizada | k-fold",
    "smoothed-mean:permutation-prefix": "média suavizada | prefixo",
    "log-odds:k-fold": "log-odds | k-fold",
    "log-odds:permutation-prefix": "log-odds | prefixo",
    "ordered-target": "OTE",
    "woe": "WoE",
    # protocolos
    "stratified-holdout": "holdout estratificado",
    "stratified-k-fold": "validação cruzada estratificada",
    "temporal-holdout": "holdout temporal",
    # regimes do TabBERT e normalizações
    "supervised-only": "S",
    "mlm-before-finetune": "PM→S",
    "warmup-mlm-finetune": "S→PM→S",
    "quantile": "quantis",
    "standard": "padronização",
}
OUTCOME_LABELS = {
    "corroborated": "corroborada",
    "refuted": "refutada",
    "inconclusive": "inconclusiva",
    "impaired": "prejudicada",
    "non-inferior": "não inferior",
    "inferior": "inferior",
}
MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*", "<", ">")
HATCHES = ("", "//", "\\\\", "xx", "..", "--", "++", "oo")


def _dataset_names() -> dict[str, str]:
    """Nome descritivo de cada base, pelo id canônico e pelos apelidos eqm-ref-NN."""
    definitions = list(getattr(experiment_datasets, "DATASET_DEFINITIONS", {}).values())
    # Variantes de ablação definidas fora do dicionário (Sparkov-sem-titular etc.).
    definitions += [
        value for value in vars(experiment_datasets).values()
        if hasattr(value, "dataset_id") and hasattr(value, "dataset_name")
    ]
    names = {definition.dataset_id: definition.dataset_name for definition in definitions}
    for alias, dataset_id in getattr(experiment_datasets, "DATASET_ID_BY_ALIAS", {}).items():
        names[alias] = names.get(dataset_id, dataset_id)
    return names


def dataset_label(value: str) -> str:
    """Nome descritivo da base (ULB, Sparkov, ...), a partir do id ou do apelido."""
    names = DATASET_NAMES
    dataset_id = getattr(experiment_datasets, "DATASET_ID_BY_ALIAS", {}).get(value, value)
    return names.get(dataset_id, names.get(value, str(value)))


def label(value) -> str:
    return STRATEGY_LABELS.get(value, ALGORITHM_LABELS.get(value, METRIC_LABELS.get(value, str(value))))


def number(value, digits: int = 3) -> str:
    """Número com vírgula decimal, ``digits`` casas e sem zero negativo."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    rounded = round(float(value), digits)
    if rounded == 0:
        rounded = 0.0
    text = f"{rounded:.{digits}f}".replace(".", ",")
    return text.replace("-", "$-$") if rounded < 0 else text


def number_plain(value, digits: int = 3) -> str:
    """Como ``number``, para figuras (sinal de menos tipográfico, sem LaTeX)."""
    text = number(value, digits)
    return text.replace("$-$", "−")


def configure() -> None:
    """Tema, paleta, fontes e locale das figuras."""
    try:
        locale.setlocale(locale.LC_NUMERIC, "pt_BR.UTF-8")
        use_locale = True
    except locale.Error:
        use_locale = False
    sns.set_theme(style="whitegrid", context="paper", palette="colorblind")
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": FONT_SIZE,
        "axes.titlesize": TITLE_SIZE,
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE,
        "axes.formatter.use_locale": use_locale,
        "axes.unicode_minus": True,
        "figure.dpi": 150,
        "savefig.dpi": 300,
    })


def legend_below(fig: plt.Figure, handles, labels, ncol: int | None = None) -> None:
    """Legenda única abaixo dos painéis, sem sobrepor títulos."""
    ncol = ncol or min(len(labels), 4)
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=ncol, frameon=False)
    rows = math.ceil(len(labels) / ncol)
    fig.tight_layout(rect=(0, 0.05 + 0.035 * rows, 1, 1))


def hatch_by_color(axis, n_hue: int, legend_handles=()) -> None:
    """Hachura de cada caixa ou barra pela cor da série, e não pela posição."""
    palette = [tuple(round(c, 3) for c in color) for color in sns.color_palette(n_colors=n_hue)]

    def index(color) -> int:
        rgb = tuple(round(c, 3) for c in color[:3])
        return min(range(n_hue), key=lambda i: sum((a - b) ** 2 for a, b in zip(rgb, palette[i])))

    for patch in [*axis.patches, *legend_handles]:
        try:
            patch.set_hatch(HATCHES[index(patch.get_facecolor())])
        except (AttributeError, TypeError):
            continue


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


DATASET_NAMES = _dataset_names()
