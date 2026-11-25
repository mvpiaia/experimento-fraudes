"""DevNet na ULB no protocolo do artigo original: o que explica a diferença?

Experimento controlado (fatorial 2 x 2 x 2) com a implementação da DevNet
desta grade (``devnet_model``), na mesma divisão usada pelos autores
(estratificada 80/20, ``random_state=42``) e com 30 fraudes rotuladas, como
o padrão do código público (github.com/GuansongPang/deviation-network):

* preparação: ``autores`` (escala min-max [0, 1] na base inteira, sem
  ``Time``, ``Amount`` sem log, como o arquivo ``creditcardfraud_normalised``
  do repositório) ou ``grade`` (``hour_of_day`` no lugar de ``Time``,
  log(1+x) em ``Amount`` e padronização ajustada no treino, como nesta grade);
  com ``--ablation``, cada um dos três componentes da preparação da grade é
  acrescentado isoladamente à dos autores;
* duplicatas: ``mantidas`` (284.807 linhas, como no arquivo dos autores) ou
  ``removidas`` (283.726, como nesta grade);
* rotulagem: ``autores`` (as fraudes do treino além das 30 rotuladas são
  removidas e 2% de ruído sintético, rotulado como normal, é injetado a
  partir das fraudes da base, como ``inject_noise`` no código dos autores) ou
  ``grade`` (as demais fraudes do treino ficam no conjunto não rotulado).

Cada braço é repetido com 10 sementes (sorteio dos rotulados e do ruído); a
inicialização da rede segue a semente fixa da grade. Métricas no teste:
AUC-ROC e AUC-PR (precisão média), como no artigo.

Uso::

    python3 devnet_protocolo_artigo.py --creditcard creditcard.csv \\
        --output-directory saidas [--runs 10]
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from devnet_model import DEFAULT_DEVNET_SETTINGS, score_devnet, train_devnet

KNOWN_OUTLIERS = 30
CONTAMINATION = 0.02
PCA_COLUMNS = [f"V{i}" for i in range(1, 29)]


PREPARATIONS = {
    # nome: (hora do dia, log em Amount, padronização no treino em vez de min-max na base)
    "autores": (False, False, False),
    "grade": (True, True, True),
    "autores+hora": (True, False, False),
    "autores+log": (False, True, False),
    "padronizada": (False, False, True),
}


def features(raw: pd.DataFrame, preparation: str) -> pd.DataFrame:
    hour, log_amount, standardize = PREPARATIONS[preparation]
    x = raw[PCA_COLUMNS].astype(float).copy()
    x["Amount"] = np.log1p(raw["Amount"].astype(float)) if log_amount else raw["Amount"].astype(float)
    if hour:
        x["hour_of_day"] = (raw["Time"] / 3600.0) % 24
    if standardize:
        return x  # padronizada depois, só com o treino
    return (x - x.min()) / (x.max() - x.min())


def inject_noise(outliers: np.ndarray, count: int, rng: np.random.RandomState) -> np.ndarray:
    """Como no código dos autores: copia uma fraude e troca 5% dos atributos por outra."""
    dim = outliers.shape[1]
    swap = max(1, int(0.05 * dim))
    noise = np.empty((count, dim))
    for i in range(count):
        first, second = rng.choice(len(outliers), 2, replace=False)
        noise[i] = outliers[first]
        cols = rng.choice(dim, swap, replace=False)
        noise[i, cols] = outliers[second, cols]
    return noise


def run_arm(raw: pd.DataFrame, preparation: str, labelling: str, seed: int) -> dict:
    x = features(raw, preparation).to_numpy()
    y = raw["Class"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    if PREPARATIONS[preparation][2]:
        mean, std = x_train.mean(axis=0), x_train.std(axis=0)
        std[std == 0] = 1.0
        x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        all_outliers = (x[y == 1] - mean) / std
    else:
        all_outliers = x[y == 1]
    rng = np.random.RandomState(seed)
    outlier_idx = np.where(y_train == 1)[0]
    labeled = rng.choice(outlier_idx, KNOWN_OUTLIERS, replace=False)
    if labelling == "autores":
        keep = np.setdiff1d(np.arange(len(x_train)), np.setdiff1d(outlier_idx, labeled))
        x_train = x_train[keep]
        labeled = np.searchsorted(keep, labeled)
        inliers = len(x_train) - KNOWN_OUTLIERS
        noise = inject_noise(all_outliers, int(inliers * CONTAMINATION / (1 - CONTAMINATION)), rng)
        x_train = np.vstack([x_train, noise])
    started = time.perf_counter()
    model = train_devnet(x_train, labeled, DEFAULT_DEVNET_SETTINGS)
    scores = score_devnet(model, x_test)
    return {"roc_auc": roc_auc_score(y_test, scores), "auc_pr": average_precision_score(y_test, scores),
            "train_rows": len(x_train), "test_rows": len(x_test), "test_frauds": int(y_test.sum()),
            "seconds": round(time.perf_counter() - started, 2)}


def main() -> None:
    parser = argparse.ArgumentParser(description="DevNet na ULB no protocolo do artigo.")
    parser.add_argument("--creditcard", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--ablation", action="store_true",
                        help="Só duplicatas removidas e rotulagem da grade, variando cada componente da preparação.")
    args = parser.parse_args()
    raw_all = pd.read_csv(args.creditcard)
    datasets = {"mantidas": raw_all, "removidas": raw_all.drop_duplicates().reset_index(drop=True)}
    rows = []
    arms = (itertools.product(["removidas"], PREPARATIONS, ["grade"]) if args.ablation
            else itertools.product(datasets, ("autores", "grade"), ("autores", "grade")))
    for duplicates, preparation, labelling in arms:
        for seed in range(args.runs):
            result = run_arm(datasets[duplicates], preparation, labelling, seed)
            rows.append({"duplicatas": duplicates, "preparacao": preparation, "rotulagem": labelling,
                         "seed": seed, **result})
            print(rows[-1], flush=True)
    out = Path(args.output_directory)
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    name = "devnet_protocolo_artigo_ablacao" if args.ablation else "devnet_protocolo_artigo"
    table.to_csv(out / f"{name}.csv", index=False)
    summary = (table.groupby(["duplicatas", "preparacao", "rotulagem"])[["roc_auc", "auc_pr"]]
               .agg(["mean", "std"]).round(3))
    summary.to_csv(out / f"{name}_resumo.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
