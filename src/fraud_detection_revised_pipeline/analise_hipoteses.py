"""Evidencia das hipoteses H1-H3 e das questoes Q1-Q3 da dissertacao (item G20).

As regras de decisao estao no Capitulo 3 da dissertacao ("Revisao das
Hipoteses") e foram fixadas antes da reexecucao da grade:

* a execucao representativa de cada algoritmo em cada base e a de maior F2 na
  VALIDACAO; as metricas sao lidas no teste (item G12);
* as fontes de fraude sao ULB, Sparkov e IEEE-CIS-completa, no
  temporal-holdout; a Home Credit e ordenada a parte (item G7-i);
* H1: melhor estrategia sem dados sinteticos x melhor RUS+SMOTE, nao
  inferioridade com margem de 0,010 em F2 (a mesma da ablacao IEEE-CIS),
  com sensibilidade a 0,005 e 0,020;
* H2: um mesmo algoritmo e o primeiro em F2 nas tres fontes, com o IC da
  diferenca para o segundo acima de zero em cada uma ("robustez entre
  fontes", e nao generalizacao); ranking medio, Friedman e Nemenyi
  [Demsar, 2006] entram como medida descritiva;
* H3: H2 sem RUS+SMOTE; fica prejudicada se H2 nao for corroborada;
* Q1-Q3 sao exploratorias e nao recebem desfecho.

O modulo so le os resultados: nao importa nada da preparacao alem das listas
de algoritmos e dos aliases de base, e nao altera manifestos nem executores.

Unidade de incerteza da diferenca de F2 entre duas execucoes, em ordem de
preferencia:

``transaction``
    bootstrap pareado e estratificado por classe sobre as linhas do teste, a
    partir dos escores gravados por linha (``resultados_*_test_scores.jsonl``
    dos blocos classico, de boosting e de anomalia; ``transaction_predictions``
    do ``_details.jsonl`` dos Transformers). As duas execucoes precisam ter
    avaliado as mesmas linhas: mesmas sementes e dobras, mesmo tamanho, mesma
    sequencia de rotulos e, quando gravado, o mesmo ``test_partition_md5``. O
    sorteio e feito na tabela conjunta (rotulo, predicao de A, predicao de B),
    multinomial dentro de cada classe, o que equivale a reamostrar as linhas.
``seed``
    duas ou mais sementes em comum; intervalo t da media das diferencas
    pareadas por semente.
``confusion``
    a matriz de confusao de cada execucao e reconstruida a partir de
    precisao, revocacao, FPR e tamanho do teste, e cada lado e reamostrado
    por bootstrap parametrico. Os dois lados sao sorteados de forma
    independente: o intervalo ignora a correlacao positiva entre execucoes
    avaliadas no mesmo teste e e, portanto, CONSERVADOR (mais largo).
``none``
    nenhuma das duas e possivel; o intervalo fica vazio e a decisao que
    depende dele fica inconclusiva.

Uso::

    python3 analise_hipoteses.py --results-directory resultados \
        --output-directory analise_hipoteses
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import experiment_datasets
import experiment_grid


FRAUD_SOURCES = ("ulb", "sparkov", "ieee-cis-completa")
IEEE_SENSITIVITY_VARIANTS = ("ieee-cis-sem-identidade", "ieee-cis-transacional")
HOME_CREDIT = "home-credit-completa"
HIGH_CARDINALITY_ABLATION = ("ieee-cis-completa", "ieee-cis-completa-sem-alta-cardinalidade")

TEMPORAL = "temporal-holdout"
STRATIFIED = "stratified-holdout"
SYNTHETIC_TECHNIQUE = "RUS+SMOTE"
H1_MARGIN = 0.010
# Analise de sensibilidade da margem de H1: o desfecho principal usa
# H1_MARGIN; as demais margens so mostram se a conclusao depende dela.
H1_SENSITIVITY_MARGINS = (0.005, H1_MARGIN, 0.020)
CONFIDENCE = 0.95

SELECTION_METRIC = "f_beta_2"
METRICS = ("f_beta_2", "auc_pr", "roc_auc", "precision", "recall", "false_positive_rate")
COST_COLUMNS = ("fit_time", "scoring_time")

CLASSIC_ALGORITHMS = tuple(experiment_grid.CLASSIC_ALGORITHMS)
BOOSTING_ALGORITHMS = tuple(experiment_grid.BOOSTING_ALGORITHMS)
FT_TRANSFORMER_ALGORITHMS = tuple(f"ft-transformer-{head}" for head in experiment_grid.TRANSFORMER_HEADS)
TABBERT_ALGORITHMS = tuple(experiment_grid.TABBERT_ALGORITHMS)
SUPERVISED_ALGORITHMS = (
    CLASSIC_ALGORITHMS + BOOSTING_ALGORITHMS + FT_TRANSFORMER_ALGORITHMS + TABBERT_ALGORITHMS
)
# O TabBERT fica como trabalho futuro na dissertacao: sua execucao nao foi
# concluida no prazo, e nem as execucoes concluidas sao relatadas, para que o
# conjunto comparado nao dependa de quais bracos terminaram primeiro. A
# exclusao e feita na leitura, e vale para todas as analises e figuras.
EXCLUDED_ALGORITHMS = TABBERT_ALGORITHMS

# Eixos que, com o configuration_md5, identificam uma execucao. Os bracos de
# ponderacao de classe reusam o md5 da configuracao sem reamostragem, e os
# regimes do TabBERT reusam o da preparacao; por isso entram na chave.
KEY_COLUMNS = (
    "configuration_md5",
    "database_type",
    "algorithm",
    "train_validation_test_strategy",
    "undersampling_technique",
    "class_weight_intensity",
    "threshold_strategy",
    "regime",
    "label_fraction",
    "categorical_encoding_strategy",
    "fill_na_strategy",
)
ARM_COLUMNS = KEY_COLUMNS[4:]

# Valores criticos q_0,05 do teste de Nemenyi (Demsar, 2006, Tabela 5a).
NEMENYI_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}

OUTCOME_CORROBORATED = "corroborated"
OUTCOME_REFUTED = "refuted"
OUTCOME_INCONCLUSIVE = "inconclusive"
OUTCOME_IMPAIRED = "impaired"


# --------------------------------------------------------------- carregamento


def load_results(results_directory: str | Path) -> pd.DataFrame:
    """Le todos os ``resultados_<algoritmo>.csv`` do diretorio, sem ``EXCLUDED_ALGORITHMS``."""
    paths = sorted(Path(results_directory).glob("resultados_*.csv"))
    frames = [pd.read_csv(path, low_memory=False) for path in paths]
    if not frames:
        return normalize_results(pd.DataFrame())
    data = normalize_results(pd.concat(frames, ignore_index=True, sort=False))
    return data[~data["algorithm"].isin(EXCLUDED_ALGORITHMS)].reset_index(drop=True)


def normalize_results(data: pd.DataFrame) -> pd.DataFrame:
    """Linhas de sucesso avaliadas por registro, com nomes uniformes.

    * ``split`` perde o sufixo ``_data`` dos blocos classico e de boosting,
      para coincidir com o ``validation``/``test`` dos Transformers;
    * ``database_type`` passa ao ``dataset_id`` canonico (aliases eqm-ref-NN);
    * metricas negativas (marca antiga de falha) viram ausentes.
    """
    data = data.copy()
    for column in (*KEY_COLUMNS, "split", "random_seed", "test_df_size", *METRICS, *COST_COLUMNS):
        if column not in data.columns:
            data[column] = np.nan
    identity = data["dataset_id"] if "dataset_id" in data.columns else data["database_type"]
    identity = identity.where(identity.notna(), data["database_type"])
    data["database_type"] = identity.map(
        lambda value: experiment_datasets.DATASET_ID_BY_ALIAS.get(value, value)
    )
    data["split"] = (
        data["split"].fillna("test").astype(str).str.replace(r"_data$", "", regex=True)
    )
    if "evaluation_unit" in data.columns:
        data = data[data["evaluation_unit"].fillna("row").eq("row")]
    if "execution_status" in data.columns:
        data = data[data["execution_status"].fillna("success").eq("success")]
    for column in (*METRICS, *COST_COLUMNS, "test_df_size", "random_seed"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    for metric in METRICS:
        data.loc[data[metric] < 0, metric] = np.nan
    data = data[data[SELECTION_METRIC].notna()].copy()
    data["random_seed"] = data["random_seed"].fillna(0).astype(int)
    for column in KEY_COLUMNS:
        data[column] = data[column].fillna("").astype(str)
    return data.reset_index(drop=True)


# ---------------------------------------------------------------- execucoes


def per_seed_metrics(data: pd.DataFrame, split: str) -> pd.DataFrame:
    """Media das dobras por execucao e semente, em um split."""
    rows = data[data["split"].eq(split)]
    columns = [*METRICS, *COST_COLUMNS, "test_df_size"]
    return rows.groupby([*KEY_COLUMNS, "random_seed"], as_index=False)[columns].mean()


def build_executions(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Uma linha por execucao com validacao, e as metricas de teste por semente.

    ``validation_f_beta_2`` e a media das sementes na validacao; as metricas de
    teste sao a media das sementes. Execucoes sem validacao nao sao elegiveis.
    """
    validation = per_seed_metrics(data, "validation")
    test = per_seed_metrics(data, "test")
    if validation.empty or test.empty:
        empty = pd.DataFrame(columns=[*KEY_COLUMNS, "validation_f_beta_2", *METRICS, "seeds"])
        return empty, test
    selection = (
        validation.groupby(list(KEY_COLUMNS))[SELECTION_METRIC].mean().rename("validation_f_beta_2")
    )
    test_means = test.groupby(list(KEY_COLUMNS))[[*METRICS, *COST_COLUMNS]].mean()
    seeds = test.groupby(list(KEY_COLUMNS))["random_seed"].nunique().rename("seeds")
    executions = test_means.join(selection, how="inner").join(seeds).reset_index()
    return executions, test


def select_representatives(executions: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Execucao de maior F2 na validacao em cada grupo; desempate pela chave."""
    if executions.empty:
        return executions.copy()
    ordered = executions.sort_values(
        ["validation_f_beta_2", *KEY_COLUMNS],
        ascending=[False] + [True] * len(KEY_COLUMNS),
        kind="mergesort",
    )
    return ordered.groupby(by, sort=True).head(1).reset_index(drop=True)


def arm_label(row: pd.Series) -> str:
    return "|".join(str(row[column]) for column in ARM_COLUMNS)


def execution_key(row: pd.Series) -> tuple:
    return tuple(row[column] for column in KEY_COLUMNS)


# ------------------------------------------------ escores do teste por linha

TEST_SCORES_PATTERN = "resultados_*_test_scores.jsonl"
DETAILS_PATTERN = "resultados_*_details.jsonl"
# Mesmo formato de lr_nb_knn_specific.encode_test_scores.
TEST_SCORES_ENCODING = "scores=float32-le-base64;targets=packbits-base64"
# Atributos que, se gravados no registro, precisam coincidir com a execucao.
SCORE_MATCH_ATTRIBUTES = ("class_weight_intensity", "threshold_strategy", "undersampling_technique")


def decode_test_scores(record: dict) -> tuple[np.ndarray, np.ndarray]:
    """(rotulos 0/1, escores) de um registro ``test_scores``."""
    if record.get("encoding") != TEST_SCORES_ENCODING:
        raise ValueError(f"unknown test score encoding: {record.get('encoding')}")
    rows = int(record["rows"])
    scores = np.frombuffer(base64.b64decode(record["scores"]), dtype="<f4")[:rows]
    targets = np.unpackbits(np.frombuffer(base64.b64decode(record["targets"]), dtype=np.uint8))[:rows]
    return targets.astype(np.uint8), scores.astype(np.float64)


def _decode_details(record: dict) -> tuple[np.ndarray, np.ndarray]:
    predictions = record["transaction_predictions"]
    targets = np.array([int(round(float(item["target"]))) for item in predictions], dtype=np.uint8)
    scores = np.array([float(item["probability"]) for item in predictions], dtype=np.float64)
    return targets, scores


def _same_value(first, second) -> bool:
    if first is None or second is None or "" in (first, second):
        return True
    try:
        return math.isclose(float(first), float(second))
    except (TypeError, ValueError):
        return str(first) == str(second)


def _score_lookup_key(configuration_md5, algorithm, regime, seed, fold) -> tuple:
    regime = "" if regime is None or (isinstance(regime, float) and math.isnan(regime)) else str(regime)
    return (str(configuration_md5), str(algorithm), regime, int(seed), int(fold))


class TestScoreStore:
    """Indice dos escores do teste por linha, lidos sob demanda.

    A varredura guarda so os metadados e a posicao de cada registro; os
    escores sao decodificados quando um par de execucoes os pede.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple, list[dict]] = {}
        self._cache: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def from_directory(cls, results_directory: str | Path) -> "TestScoreStore":
        store = cls()
        directory = Path(results_directory)
        for pattern, kind in ((TEST_SCORES_PATTERN, "test_scores"), (DETAILS_PATTERN, "details")):
            for path in sorted(directory.glob(pattern)):
                store._scan(path, kind)
        return store

    def _scan(self, path: Path, kind: str) -> None:
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                record = json.loads(line)
                self.add(record, kind, location=(str(path), offset))

    def add(self, record: dict, kind: str, location: tuple | None = None) -> None:
        if str(record.get("split", "")).removesuffix("_data") != "test":
            return
        if kind == "details" and not record.get("transaction_predictions"):
            return
        if kind == "test_scores" and record.get("evaluation_unit", "row") != "row":
            return
        try:
            key = _score_lookup_key(record["configuration_md5"], record["algorithm"], record.get("regime"),
                                    record.get("random_seed") or 0, record["fold_index"])
        except (KeyError, TypeError, ValueError):
            return
        entry = {
            "kind": kind,
            "location": location,
            "record": None if location else record,
            "attributes": {name: record.get(name) for name in SCORE_MATCH_ATTRIBUTES},
            "test_partition_md5": record.get("test_partition_md5") or "",
        }
        # Uma reexecucao forcada grava de novo: vale o ultimo registro.
        candidates = self.entries.setdefault(key, [])
        candidates[:] = [item for item in candidates if item["attributes"] != entry["attributes"]]
        candidates.append(entry)

    def __len__(self) -> int:
        return len(self.entries)

    def lookup(self, execution: pd.Series, seed: int, fold: int) -> dict | None:
        key = _score_lookup_key(execution["configuration_md5"], execution["algorithm"], execution["regime"], seed, fold)
        matches = [
            entry for entry in self.entries.get(key, [])
            if all(_same_value(entry["attributes"][name], execution.get(name)) for name in SCORE_MATCH_ATTRIBUTES
                   if entry["attributes"][name] not in (None, ""))
        ]
        return matches[0] if len(matches) == 1 else None

    def arrays(self, entry: dict) -> tuple[np.ndarray, np.ndarray]:
        cache_key = (entry["location"] or id(entry["record"]),)
        if cache_key not in self._cache:
            record = entry["record"]
            if record is None:
                path, offset = entry["location"]
                with open(path, "rb") as handle:
                    handle.seek(offset)
                    record = json.loads(handle.readline())
            decoded = decode_test_scores(record) if entry["kind"] == "test_scores" else _decode_details(record)
            if len(self._cache) >= 64:
                self._cache.pop(next(iter(self._cache)))
            self._cache[cache_key] = decoded
        return self._cache[cache_key]


def fold_thresholds(data: pd.DataFrame) -> dict[tuple, dict[int, float]]:
    """Limiar de cada dobra do teste, por (execucao, semente)."""
    if "fold_index" not in data.columns or "classification_threshold" not in data.columns:
        return {}
    rows = data[data["split"].eq("test")].copy()
    rows["fold_index"] = pd.to_numeric(rows["fold_index"], errors="coerce")
    rows["classification_threshold"] = pd.to_numeric(rows["classification_threshold"], errors="coerce")
    rows = rows.dropna(subset=["fold_index", "classification_threshold"])
    thresholds: dict[tuple, dict[int, float]] = {}
    for row in rows.itertuples(index=False):
        key = (*(getattr(row, column) for column in KEY_COLUMNS), int(row.random_seed))
        thresholds.setdefault(key, {})[int(row.fold_index)] = float(row.classification_threshold)
    return thresholds


def _f2_from_counts(true_positive, positives, false_positive):
    denominator = 5 * true_positive + 4 * (positives - true_positive) + false_positive
    return np.divide(5 * true_positive, denominator, out=np.zeros(np.shape(denominator), dtype=float),
                     where=denominator > 0)


def paired_transaction_bootstrap(
    units: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    samples: int,
) -> tuple[float, np.ndarray]:
    """Diferenca de F2 (A - B) e suas reamostras, media das unidades.

    Cada unidade (semente, dobra) traz rotulos e as predicoes binarias de A e
    de B nas mesmas linhas. Dentro de cada classe, as linhas caem em quatro
    celulas (predicao de A, predicao de B); reamostrar as linhas com reposicao,
    estratificado por classe, e sortear essas contagens numa multinomial.
    """
    points, draws = [], []
    for targets, first, second in units:
        positive = targets == 1
        cells = {}
        for label, mask in ((1, positive), (0, ~positive)):
            a, b = first[mask], second[mask]
            counts = np.array([np.sum(a & b), np.sum(a & ~b), np.sum(~a & b), np.sum(~a & ~b)])
            total = int(counts.sum())
            sampled = rng.multinomial(total, counts / total, size=samples) if total else np.zeros((samples, 4), int)
            cells[label] = (counts, total, sampled)
        pos_counts, positives, pos_draw = cells[1]
        neg_counts, _, neg_draw = cells[0]
        point = (_f2_from_counts(pos_counts[0] + pos_counts[1], positives, neg_counts[0] + neg_counts[1])
                 - _f2_from_counts(pos_counts[0] + pos_counts[2], positives, neg_counts[0] + neg_counts[2]))
        draw = (_f2_from_counts(pos_draw[:, 0] + pos_draw[:, 1], positives, neg_draw[:, 0] + neg_draw[:, 1])
                - _f2_from_counts(pos_draw[:, 0] + pos_draw[:, 2], positives, neg_draw[:, 0] + neg_draw[:, 2]))
        points.append(float(point))
        draws.append(draw)
    return float(np.mean(points)), np.mean(draws, axis=0)


# ------------------------------------------------------ unidade de incerteza


def estimate_positives(per_seed_test: pd.DataFrame) -> dict[tuple, int]:
    """Positivos do teste por (base, protocolo, semente, tamanho do teste).

    Com precisao p, revocacao r, FPR f e tamanho S: P = p f S / (r + p f - p r).
    Todas as execucoes de uma mesma chave avaliam o mesmo teste; a mediana das
    estimativas validas cobre as execucoes em que a formula nao se aplica
    (revocacao ou FPR nulos).
    """
    rows = per_seed_test.dropna(subset=["precision", "recall", "false_positive_rate", "test_df_size"])
    p, r, f, size = (rows[column] for column in ("precision", "recall", "false_positive_rate", "test_df_size"))
    denominator = r + p * f - p * r
    valid = (p > 0) & (r > 0) & (f > 0) & (denominator > 0)
    estimates = (p * f * size / denominator).where(valid)
    keyed = rows.assign(positives=estimates).dropna(subset=["positives"])
    keys = ["database_type", "train_validation_test_strategy", "random_seed", "test_df_size"]
    medians = keyed.groupby(keys)["positives"].median()
    return {
        key: int(round(value))
        for key, value in medians.items()
        if 0 < round(value) < key[-1]
    }


def _positives_key(row: pd.Series) -> tuple:
    return (row["database_type"], row["train_validation_test_strategy"], int(row["random_seed"]), row["test_df_size"])


def confusion_bootstrap(
    seed_rows: pd.DataFrame,
    positives: dict[tuple, int],
    rng: np.random.Generator,
    samples: int,
) -> np.ndarray | None:
    """F2 reamostrado da matriz de confusao reconstruida (media das sementes)."""
    draws = []
    for _, row in seed_rows.iterrows():
        key = _positives_key(row)
        if key not in positives or pd.isna(row["recall"]) or pd.isna(row["false_positive_rate"]):
            return None
        total_positive = positives[key]
        total_negative = int(row["test_df_size"]) - total_positive
        true_positive = int(round(row["recall"] * total_positive))
        false_positive = int(round(row["false_positive_rate"] * total_negative))
        tp = rng.binomial(total_positive, true_positive / total_positive, samples)
        fp = rng.binomial(total_negative, false_positive / total_negative, samples) if total_negative else 0
        denominator = 5 * tp + 4 * (total_positive - tp) + fp
        draws.append(np.divide(5 * tp, denominator, out=np.zeros(samples), where=denominator > 0))
    return np.mean(draws, axis=0) if draws else None


class DifferenceEstimator:
    """IC da diferenca de F2 de teste entre duas execucoes (A - B)."""

    def __init__(
        self,
        per_seed_test: pd.DataFrame,
        samples: int = 2000,
        seed: int = 20,
        scores: TestScoreStore | None = None,
        thresholds: dict[tuple, dict[int, float]] | None = None,
    ):
        self.groups = {
            key: frame.set_index("random_seed")
            for key, frame in per_seed_test.groupby(list(KEY_COLUMNS))
        }
        self.positives = estimate_positives(per_seed_test)
        self.samples = samples
        self.seed = seed
        self.scores = scores or TestScoreStore()
        self.thresholds = thresholds or {}

    def _transaction_units(self, first: pd.Series, second: pd.Series, seeds) -> list | None:
        """Unidades (semente, dobra) em que A e B avaliaram as mesmas linhas."""
        units = []
        for seed in sorted(seeds):
            folds_a = self.thresholds.get((*execution_key(first), int(seed)), {})
            folds_b = self.thresholds.get((*execution_key(second), int(seed)), {})
            if not folds_a or set(folds_a) != set(folds_b):
                return None
            for fold in sorted(folds_a):
                entry_a = self.scores.lookup(first, seed, fold)
                entry_b = self.scores.lookup(second, seed, fold)
                if entry_a is None or entry_b is None:
                    return None
                if entry_a["test_partition_md5"] and entry_b["test_partition_md5"] and (
                    entry_a["test_partition_md5"] != entry_b["test_partition_md5"]
                ):
                    return None
                targets_a, scores_a = self.scores.arrays(entry_a)
                targets_b, scores_b = self.scores.arrays(entry_b)
                if len(targets_a) != len(targets_b) or not np.array_equal(targets_a, targets_b):
                    return None
                units.append((targets_a, scores_a >= folds_a[fold], scores_b >= folds_b[fold]))
        return units or None

    def interval(self, first: pd.Series, second: pd.Series) -> dict:
        a = self.groups.get(execution_key(first))
        b = self.groups.get(execution_key(second))
        point = float(first[SELECTION_METRIC] - second[SELECTION_METRIC])
        result = {"difference": point, "ci_low": np.nan, "ci_high": np.nan, "uncertainty_unit": "none", "units": 0}
        if a is None or b is None:
            return result
        common = a.index.intersection(b.index)
        if len(self.scores) and len(common):
            units = self._transaction_units(first, second, common)
            if units:
                rng = np.random.default_rng(self.seed)
                mean, draws = paired_transaction_bootstrap(units, rng, self.samples)
                tail = (1 - CONFIDENCE) / 2 * 100
                low, high = np.percentile(draws, [tail, 100 - tail])
                return {**result, "difference": mean, "ci_low": float(low), "ci_high": float(high),
                        "uncertainty_unit": "transaction", "units": int(sum(len(unit[0]) for unit in units))}
        if len(common) >= 2:
            differences = a.loc[common, SELECTION_METRIC] - b.loc[common, SELECTION_METRIC]
            n = len(differences)
            half = stats.t.ppf(0.5 + CONFIDENCE / 2, n - 1) * differences.std(ddof=1) / math.sqrt(n)
            mean = float(differences.mean())
            return {**result, "difference": mean, "ci_low": mean - half, "ci_high": mean + half,
                    "uncertainty_unit": "seed", "units": n}
        rng = np.random.default_rng(self.seed)
        first_draws = confusion_bootstrap(a.reset_index(), self.positives, rng, self.samples)
        second_draws = confusion_bootstrap(b.reset_index(), self.positives, rng, self.samples)
        if first_draws is None or second_draws is None:
            return result
        differences = first_draws - second_draws
        tail = (1 - CONFIDENCE) / 2 * 100
        low, high = np.percentile(differences, [tail, 100 - tail])
        return {**result, "ci_low": float(low), "ci_high": float(high),
                "uncertainty_unit": "confusion", "units": self.samples}


# ------------------------------------------------------------------------ H1


def analyse_h1(
    executions: pd.DataFrame, estimator: DifferenceEstimator
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    rows = []
    scope = executions[
        executions["algorithm"].isin(CLASSIC_ALGORITHMS)
        & executions["database_type"].isin(FRAUD_SOURCES)
        & executions["train_validation_test_strategy"].eq(TEMPORAL)
    ]
    for algorithm in CLASSIC_ALGORITHMS:
        for source in FRAUD_SOURCES:
            group = scope[scope["algorithm"].eq(algorithm) & scope["database_type"].eq(source)]
            synthetic = group[group["undersampling_technique"].eq(SYNTHETIC_TECHNIQUE)]
            without = group[~group["undersampling_technique"].eq(SYNTHETIC_TECHNIQUE)]
            record = {"algorithm": algorithm, "dataset_id": source}
            if synthetic.empty or without.empty:
                rows.append({**record, "status": "missing"})
                continue
            best_without = select_representatives(without, ["algorithm"]).iloc[0]
            best_synthetic = select_representatives(synthetic, ["algorithm"]).iloc[0]
            rows.append({
                **record,
                "status": "compared",
                "without_synthetic_arm": arm_label(best_without),
                "without_synthetic_md5": best_without["configuration_md5"],
                "without_synthetic_f_beta_2": best_without[SELECTION_METRIC],
                "rus_smote_arm": arm_label(best_synthetic),
                "rus_smote_md5": best_synthetic["configuration_md5"],
                "rus_smote_f_beta_2": best_synthetic[SELECTION_METRIC],
                **estimator.interval(best_without, best_synthetic),
            })
    table = pd.DataFrame(rows)
    if not table.empty:
        table["pair_outcome"] = table.apply(h1_pair_outcome, axis=1)
    sensitivity = h1_margin_sensitivity(table)
    summary = {
        "outcome": h1_outcome(table),
        "margin": H1_MARGIN,
        **_pair_counts(table),
        "margin_sensitivity": {
            f"{margin:.3f}": h1_outcome(table, margin) for margin in H1_SENSITIVITY_MARGINS
        },
    }
    summary["margin_sensitivity_stable"] = len(set(summary["margin_sensitivity"].values())) == 1
    return table, summary, sensitivity


def h1_pair_outcome(row: pd.Series, margin: float = H1_MARGIN) -> str:
    """Desfecho de um par (algoritmo, base) para uma margem."""
    if row.get("status") != "compared" or pd.isna(row.get("ci_low")) or pd.isna(row.get("ci_high")):
        return OUTCOME_INCONCLUSIVE
    if row["ci_high"] < -margin:
        return OUTCOME_REFUTED
    if row["ci_low"] > -margin:
        return OUTCOME_CORROBORATED
    return OUTCOME_INCONCLUSIVE


def h1_outcome(table: pd.DataFrame, margin: float = H1_MARGIN) -> str:
    if table.empty or "ci_high" not in table.columns:
        return OUTCOME_INCONCLUSIVE
    if (table["ci_high"] < -margin).any():
        return OUTCOME_REFUTED
    complete = table["status"].eq("compared").all() and table["ci_low"].notna().all()
    if complete and (table["ci_low"] > -margin).all():
        return OUTCOME_CORROBORATED
    return OUTCOME_INCONCLUSIVE


def h1_margin_sensitivity(table: pd.DataFrame) -> pd.DataFrame:
    """Desfecho de cada par e o global para cada margem de H1_SENSITIVITY_MARGINS."""
    columns = ["margin", "primary", "algorithm", "dataset_id", "pair_outcome", "global_outcome"]
    if table.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for margin in H1_SENSITIVITY_MARGINS:
        global_outcome = h1_outcome(table, margin)
        for _, row in table.iterrows():
            rows.append({
                "margin": margin,
                "primary": math.isclose(margin, H1_MARGIN),
                "algorithm": row["algorithm"],
                "dataset_id": row["dataset_id"],
                "pair_outcome": h1_pair_outcome(row, margin),
                "global_outcome": global_outcome,
            })
    return pd.DataFrame(rows, columns=columns)


def _pair_counts(table: pd.DataFrame) -> dict:
    if table.empty:
        return {"expected_pairs": len(CLASSIC_ALGORITHMS) * len(FRAUD_SOURCES), "compared_pairs": 0, "pairs_with_ci": 0}
    compared = table["status"].eq("compared")
    with_ci = table["ci_low"].notna() if "ci_low" in table.columns else compared & False
    return {
        "expected_pairs": len(CLASSIC_ALGORITHMS) * len(FRAUD_SOURCES),
        "compared_pairs": int(compared.sum()),
        "pairs_with_ci": int(with_ci.sum()),
    }


# --------------------------------------------------------------------- H2/H3


def representatives_by_source(
    executions: pd.DataFrame,
    sources: tuple[str, ...],
    protocol: str,
    exclude_synthetic: bool = False,
) -> pd.DataFrame:
    scope = executions[
        executions["algorithm"].isin(SUPERVISED_ALGORITHMS)
        & executions["database_type"].isin(sources)
        & executions["train_validation_test_strategy"].eq(protocol)
    ]
    if exclude_synthetic:
        scope = scope[~scope["undersampling_technique"].eq(SYNTHETIC_TECHNIQUE)]
    return select_representatives(scope, ["database_type", "algorithm"])


def rank_algorithms(representatives: pd.DataFrame, sources: tuple[str, ...]) -> tuple[pd.DataFrame, list[str]]:
    """Postos por fonte, so dos algoritmos presentes em todas as fontes."""
    present = representatives.groupby("algorithm")["database_type"].nunique()
    complete = sorted(present[present == len(sources)].index)
    excluded = sorted(set(representatives["algorithm"]) - set(complete))
    table = representatives[representatives["algorithm"].isin(complete)].copy()
    for metric in ("f_beta_2", "auc_pr"):
        table[f"rank_{metric}"] = table.groupby("database_type")[metric].rank(ascending=False, method="average")
    table["arm"] = table.apply(arm_label, axis=1) if not table.empty else []
    columns = ["database_type", "algorithm", "f_beta_2", "auc_pr", "rank_f_beta_2", "rank_auc_pr",
               "validation_f_beta_2", "configuration_md5", "arm"]
    table = table.reindex(columns=columns).rename(columns={"database_type": "dataset_id"})
    return table.sort_values(["dataset_id", "rank_f_beta_2"]).reset_index(drop=True), excluded


def mean_ranks(ranks: pd.DataFrame) -> pd.DataFrame:
    if ranks.empty:
        return pd.DataFrame(columns=["algorithm", "mean_rank_f_beta_2", "mean_rank_auc_pr"])
    table = ranks.groupby("algorithm", as_index=False).agg(
        mean_rank_f_beta_2=("rank_f_beta_2", "mean"), mean_rank_auc_pr=("rank_auc_pr", "mean")
    )
    return table.sort_values(["mean_rank_f_beta_2", "algorithm"]).reset_index(drop=True)


def friedman_summary(ranks: pd.DataFrame, metric: str) -> dict:
    """Friedman e distancia critica de Nemenyi (alfa = 0,05) sobre um metrica."""
    if ranks.empty:
        return {"n_sources": 0, "k_algorithms": 0}
    pivot = ranks.pivot(index="dataset_id", columns="algorithm", values=metric).dropna(axis=1)
    n, k = pivot.shape
    summary = {"n_sources": int(n), "k_algorithms": int(k), "statistic": None, "p_value": None,
               "nemenyi_critical_distance": None}
    if k >= 3 and n >= 2:
        # Com todos os postos empatados a estatistica e indefinida (NaN), e o
        # resumo grava null em vez de um aviso do scipy.
        with np.errstate(invalid="ignore", divide="ignore"):
            result = stats.friedmanchisquare(*[pivot[column].to_numpy() for column in pivot.columns])
        summary["statistic"] = float(result.statistic)
        summary["p_value"] = float(result.pvalue)
    if k in NEMENYI_Q05 and n >= 1:
        summary["nemenyi_critical_distance"] = NEMENYI_Q05[k] * math.sqrt(k * (k + 1) / (6 * n))
    return summary


def leadership_outcome(
    ranks: pd.DataFrame,
    representatives: pd.DataFrame,
    sources: tuple[str, ...],
    estimator: DifferenceEstimator,
) -> tuple[str, list[dict]]:
    """Desfecho de H2 (ou H3) e a diferenca entre o primeiro e o segundo por fonte."""
    if ranks.empty or set(ranks["dataset_id"]) != set(sources):
        return OUTCOME_INCONCLUSIVE, []
    details, leaders = [], set()
    for source in sources:
        ordered = ranks[ranks["dataset_id"].eq(source)].sort_values(["rank_f_beta_2", "algorithm"])
        first = ordered.iloc[0]
        tied = ordered["rank_f_beta_2"].eq(first["rank_f_beta_2"]).sum() > 1
        leaders.add(first["algorithm"] if not tied else "__tie__")
        record = {"dataset_id": source, "leader": first["algorithm"], "tie_at_first": bool(tied)}
        if len(ordered) >= 2:
            second = ordered.iloc[1]
            pick = lambda name: representatives[  # noqa: E731
                representatives["database_type"].eq(source) & representatives["algorithm"].eq(name)
            ].iloc[0]
            record.update({"runner_up": second["algorithm"],
                           **estimator.interval(pick(first["algorithm"]), pick(second["algorithm"]))})
        details.append(record)
    if "__tie__" in leaders:
        return OUTCOME_INCONCLUSIVE, details
    if len(leaders) > 1:
        return OUTCOME_REFUTED, details
    lows = [record.get("ci_low", np.nan) for record in details]
    if all(pd.notna(low) and low > 0 for low in lows):
        return OUTCOME_CORROBORATED, details
    return OUTCOME_INCONCLUSIVE, details


def analyse_leadership(
    executions: pd.DataFrame,
    estimator: DifferenceEstimator,
    exclude_synthetic: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    representatives = representatives_by_source(executions, FRAUD_SOURCES, TEMPORAL, exclude_synthetic)
    ranks, excluded = rank_algorithms(representatives, FRAUD_SOURCES)
    outcome, details = leadership_outcome(ranks, representatives, FRAUD_SOURCES, estimator)
    summary = {
        "outcome": outcome,
        "sources": list(FRAUD_SOURCES),
        "protocol": TEMPORAL,
        "algorithms_ranked": sorted(set(ranks["algorithm"])),
        "algorithms_excluded_missing_sources": excluded,
        "leader_vs_runner_up": details,
        "friedman_f_beta_2": friedman_summary(ranks, "f_beta_2"),
        "friedman_auc_pr": friedman_summary(ranks, "auc_pr"),
    }
    return ranks, mean_ranks(ranks), summary


def ieee_sensitivity(executions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in ("ieee-cis-completa", *IEEE_SENSITIVITY_VARIANTS):
        sources = ("ulb", "sparkov", variant)
        ranks, _ = rank_algorithms(representatives_by_source(executions, sources, TEMPORAL), sources)
        for source in sources:
            ordered = ranks[ranks["dataset_id"].eq(source)].sort_values(["rank_f_beta_2", "algorithm"])
            rows.append({"ieee_variant": variant, "dataset_id": source,
                         "leader": ordered.iloc[0]["algorithm"] if not ordered.empty else None})
    return pd.DataFrame(rows, columns=["ieee_variant", "dataset_id", "leader"])


def home_credit_ranks(executions: pd.DataFrame) -> pd.DataFrame:
    sources = (HOME_CREDIT,)
    ranks, _ = rank_algorithms(representatives_by_source(executions, sources, STRATIFIED), sources)
    return ranks


# ---------------------------------------------------------------- Q1 e Q3


def analyse_q1(executions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    by_protocol = {
        protocol: representatives_by_source(executions, FRAUD_SOURCES, protocol)
        for protocol in (STRATIFIED, TEMPORAL)
    }
    merged = by_protocol[STRATIFIED].merge(
        by_protocol[TEMPORAL], on=["database_type", "algorithm"], suffixes=("_stratified", "_temporal")
    )
    rows = []
    for _, row in merged.iterrows():
        record = {"dataset_id": row["database_type"], "algorithm": row["algorithm"]}
        for metric in ("f_beta_2", "auc_pr", "roc_auc"):
            record[f"{metric}_stratified"] = row[f"{metric}_stratified"]
            record[f"{metric}_temporal"] = row[f"{metric}_temporal"]
            record[f"{metric}_temporal_minus_stratified"] = row[f"{metric}_temporal"] - row[f"{metric}_stratified"]
        rows.append(record)
    table = pd.DataFrame(rows)
    leaders = {}
    for source in FRAUD_SOURCES:
        subset = table[table["dataset_id"].eq(source)] if not table.empty else table
        if subset.empty:
            continue
        leaders[source] = {
            protocol: subset.sort_values([f"f_beta_2_{protocol}", "algorithm"], ascending=[False, True]).iloc[0]["algorithm"]
            for protocol in ("stratified", "temporal")
        }
    return table, {"kind": "exploratory", "leader_by_protocol": leaders}


def _preferred_protocol(executions: pd.DataFrame, dataset: str) -> str:
    protocols = set(executions.loc[executions["database_type"].eq(dataset), "train_validation_test_strategy"])
    return TEMPORAL if TEMPORAL in protocols else STRATIFIED


def analyse_q3(executions: pd.DataFrame, data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    families = {"classic": CLASSIC_ALGORITHMS, "boosting": BOOSTING_ALGORITHMS}
    leader_rows = []
    for dataset in sorted(set(executions["database_type"])):
        protocol = _preferred_protocol(executions, dataset)
        scope = executions[executions["database_type"].eq(dataset)
                           & executions["train_validation_test_strategy"].eq(protocol)]
        record = {"dataset_id": dataset, "protocol": protocol}
        for family, algorithms in families.items():
            subset = scope[scope["algorithm"].isin(algorithms)]
            if subset.empty:
                continue
            best = select_representatives(subset, ["database_type"]).iloc[0]
            record.update({f"{family}_leader": best["algorithm"], f"{family}_f_beta_2": best["f_beta_2"],
                           f"{family}_auc_pr": best["auc_pr"]})
        if "classic_f_beta_2" in record and "boosting_f_beta_2" in record:
            record["boosting_minus_classic_f_beta_2"] = record["boosting_f_beta_2"] - record["classic_f_beta_2"]
        leader_rows.append(record)

    ablation_rows = []
    base, ablated = HIGH_CARDINALITY_ABLATION
    protocol = _preferred_protocol(executions, base)
    for algorithm in BOOSTING_ALGORITHMS:
        sides = {}
        for dataset in (base, ablated):
            subset = executions[executions["algorithm"].eq(algorithm) & executions["database_type"].eq(dataset)
                                & executions["train_validation_test_strategy"].eq(protocol)]
            if not subset.empty:
                sides[dataset] = select_representatives(subset, ["algorithm"]).iloc[0]
        if len(sides) == 2:
            ablation_rows.append({
                "algorithm": algorithm, "protocol": protocol,
                "f_beta_2_complete": sides[base]["f_beta_2"], "f_beta_2_without_high_cardinality": sides[ablated]["f_beta_2"],
                "complete_minus_ablated_f_beta_2": sides[base]["f_beta_2"] - sides[ablated]["f_beta_2"],
                "complete_minus_ablated_auc_pr": sides[base]["auc_pr"] - sides[ablated]["auc_pr"],
            })

    test_rows = data[data["split"].eq("test")]
    cost = (
        test_rows.groupby(["database_type", "algorithm"], as_index=False)[list(COST_COLUMNS)].median()
        .rename(columns={"database_type": "dataset_id", "fit_time": "median_fit_time",
                         "scoring_time": "median_scoring_time"})
    )
    return pd.DataFrame(leader_rows), pd.DataFrame(ablation_rows), cost, {"kind": "exploratory"}


# ------------------------------------------------------------------- execucao


def run_analysis(
    data: pd.DataFrame,
    bootstrap_samples: int = 2000,
    scores: TestScoreStore | None = None,
) -> tuple[dict[str, pd.DataFrame], dict]:
    executions, per_seed_test = build_executions(data)
    estimator = DifferenceEstimator(
        per_seed_test, samples=bootstrap_samples, scores=scores, thresholds=fold_thresholds(data)
    )

    h1_table, h1_summary, h1_sensitivity = analyse_h1(executions, estimator)
    h2_ranks, h2_mean, h2_summary = analyse_leadership(executions, estimator, exclude_synthetic=False)
    h3_ranks, h3_mean, h3_summary = analyse_leadership(executions, estimator, exclude_synthetic=True)
    h3_summary["computed_outcome"] = h3_summary["outcome"]
    if h2_summary["outcome"] != OUTCOME_CORROBORATED:
        h3_summary["outcome"] = OUTCOME_IMPAIRED
    q1_table, q1_summary = analyse_q1(executions)
    q3_leaders, q3_ablation, q3_cost, q3_summary = analyse_q3(executions, data)

    tables = {
        "h1_pares": h1_table,
        "h1_sensibilidade_margem": h1_sensitivity,
        "h2_postos": h2_ranks,
        "h2_ranking_medio": h2_mean,
        "h2_sensibilidade_ieee": ieee_sensitivity(executions),
        "h2_home_credit": home_credit_ranks(executions),
        "h3_postos": h3_ranks,
        "h3_ranking_medio": h3_mean,
        "q1_temporal": q1_table,
        "q3_familias": q3_leaders,
        "q3_ablacao_alta_cardinalidade": q3_ablation,
        "q3_custo": q3_cost,
    }
    units = set()
    for table in (h1_table,):
        if "uncertainty_unit" in table.columns:
            units |= set(table["uncertainty_unit"].dropna())
    for summary in (h2_summary, h3_summary):
        units |= {detail.get("uncertainty_unit") for detail in summary["leader_vs_runner_up"]
                  if detail.get("uncertainty_unit")}
    summary = {
        "selection": "maior f_beta_2 medio na validacao; metricas lidas no teste",
        "uncertainty_units": sorted(units),
        "uncertainty_note": (
            "transaction = bootstrap pareado e estratificado sobre as linhas do teste; "
            "seed = intervalo t pareado por semente; confusion = bootstrap parametrico nao "
            "pareado da matriz de confusao reconstruida (conservador); none = sem intervalo"
        ),
        "executions_eligible": int(len(executions)),
        "test_score_records": int(sum(len(items) for items in estimator.scores.entries.values())),
        "H1": h1_summary,
        "H2": h2_summary,
        "H3": h3_summary,
        "H4": {"note": "avaliada pela analise da ablacao IEEE-CIS (Capitulo 3), fora deste script"},
        "Q1": q1_summary,
        "Q2": {"kind": "exploratory", "source": "history_strata_analysis.py"},
        "Q3": q3_summary,
        "empty_tables": sorted(name for name, table in tables.items() if table.empty),
    }
    return tables, summary


def write_outputs(tables: dict[str, pd.DataFrame], summary: dict, output_directory: str | Path) -> None:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / f"{name}.csv", index=False)
    with open(output / "resumo_hipoteses.json", "w", encoding="utf-8") as handle:
        json.dump(json_ready(summary), handle, ensure_ascii=False, indent=2, allow_nan=False)


def json_ready(value):
    """Converte tipos numpy e NaN (que nao e JSON valido) para tipos nativos."""
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_ready(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if math.isnan(value) else float(value)
    return value


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results-directory", default=".", help="Diretorio com os resultados_<algoritmo>.csv.")
    parser.add_argument("--output-directory", default="analise_hipoteses", help="Diretorio das tabelas e do resumo.")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    data = load_results(arguments.results_directory)
    scores = TestScoreStore.from_directory(arguments.results_directory)
    tables, summary = run_analysis(data, arguments.bootstrap_samples, scores)
    write_outputs(tables, summary, arguments.output_directory)
    for hypothesis in ("H1", "H2", "H3"):
        print(f"{hypothesis}: {summary[hypothesis]['outcome']}")
    print(f"unidades de incerteza: {', '.join(summary['uncertainty_units']) or 'nenhuma'}")
    print(f"saidas em {arguments.output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
