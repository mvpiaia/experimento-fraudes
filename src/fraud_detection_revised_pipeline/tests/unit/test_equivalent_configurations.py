"""Configuracoes gemeas por eixo de alta cardinalidade inerte.

Numa base sem colunas de alta cardinalidade, ``minimum-frequency`` e
``hashing`` geram os mesmos splits e vocabularios. A validacao aceita as
gemeas (P6), o plano executa so o representante e a R4 conta a gemea como
``equivalente`` quando o representante teve sucesso.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import equivalent_configurations as equivalent  # noqa: E402
import grid_plan  # noqa: E402
import preparation_sanity_checks as sanity  # noqa: E402

from test_g18_grid_distribution import _calibration  # noqa: E402


def _rows(md5, *, dataset="ulb", mode="transformer-indexed", strategy="minimum-frequency",
          parameter="10", content="same", vocabulary="v1", undersampling="none"):
    return [
        {
            "database_type": dataset, "dataset_id": dataset, "configuration_md5": md5,
            "dataset_md5": f"{md5}-{split}", "content_md5": f"{content}-{split}",
            "fold_index": "1", "split": split, "categorical_handling_mode": mode,
            "categorical_encoding_strategy": "none", "fill_na_strategy": "model-managed",
            "undersampling_technique": undersampling, "undersampling_parameter_values": "none",
            "test_strategy": "stratified-holdout", "random_seed": "42",
            "numeric_normalization": "quantile", "high_cardinality_strategy": strategy,
            "high_cardinality_parameter": parameter, "vocabulary_md5": vocabulary,
        }
        for split in ("train", "validation", "test")
    ]


def _manifest(*configurations) -> pd.DataFrame:
    return pd.DataFrame([row for rows in configurations for row in rows])


def test_only_the_inert_axis_makes_twins():
    manifest = _manifest(
        _rows("b-min"),
        _rows("a-hash", strategy="hashing", parameter="1024"),
        # Mesmo eixo, mas vocabulario diferente: o eixo atuou.
        _rows("c-min", content="other", vocabulary="v2"),
        _rows("d-hash", content="other", vocabulary="v3", strategy="hashing", parameter="1024"),
        # Mesmos arquivos, mas outro eixo difere: nao e equivalencia.
        _rows("e-min", content="third", undersampling="random-undersampling"),
        _rows("f-hash", content="third", strategy="hashing", parameter="1024"),
        # Fora dos Transformers o eixo nao existe.
        _rows("g-nat", mode="native", content="fourth"),
        _rows("h-nat", mode="native", content="fourth", strategy="hashing", parameter="1024"),
    )
    groups = equivalent.inert_high_cardinality_groups(manifest)
    # O representante e a estrategia padrao, mesmo com md5 maior.
    assert [(group.representative, group.twins) for group in groups] == [("b-min", ("a-hash",))]
    assert equivalent.twin_map(groups) == {"a-hash": "b-min"}
    table = equivalent.groups_table(groups)
    assert table.to_dict("records")[0]["twin_configuration_md5"] == "a-hash"


def _expected(*md5s) -> pd.DataFrame:
    # Mesma normalizacao da chave que o validador aplica aos resultados.
    keys = sanity._with_run_key(pd.concat([_results(md5, "success") for md5 in md5s]))
    expected = keys[list(sanity.RUN_KEY)].reset_index(drop=True)
    return expected.assign(
        bloco="ft-transformer", dataset_id="ulb", undersampling_technique="none",
        test_strategy="stratified-holdout", random_seed="42", linhas_teste_previstas=1,
    )


def _results(md5, status) -> pd.DataFrame:
    return pd.DataFrame(
        [{"configuration_md5": md5, "algorithm": "ft-transformer", "class_weight_intensity": "",
          "regime": "", "label_fraction": "", "execution_seed": "", "fold_index": 1,
          "split": "test", "execution_status": status, "failure_category": "",
          "failure_message": ""}]
    )


def test_twin_counts_as_equivalent_only_when_the_representative_succeeded():
    empty_log = pd.DataFrame(columns=["configuration_md5", "algoritmo", "categoria_saida"])
    runs = sanity.classify_runs(
        _results("b-min", "success"), _expected("a-hash", "b-min"), empty_log, {"a-hash": "b-min"}
    )
    assert dict(zip(runs["configuration_md5"], runs["estado"])) == {
        "a-hash": "equivalente", "b-min": "sucesso"
    }
    runs = sanity.classify_runs(
        _results("b-min", "failure"), _expected("a-hash", "b-min"), empty_log, {"a-hash": "b-min"}
    )
    assert dict(zip(runs["configuration_md5"], runs["estado"]))["a-hash"] == "ausente"


def test_plan_runs_only_the_representative(tmp_path):
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    configurations = (
        _rows("b-min"),
        _rows("a-hash", strategy="hashing", parameter="1024"),
        _rows("c-min", content="other", vocabulary="v2"),
    )
    rows = []
    for configuration in configurations:
        for row in configuration:
            text = f"{row['content_md5']}\n" * 50
            (prepared / f"{row['dataset_md5']}.csv").write_text(text)
            rows.append({**row, "content_md5": hashlib.md5(text.encode()).hexdigest()})
    manifest = tmp_path / "manifest.tsv"
    pd.DataFrame(rows).drop(columns=["dataset_id"]).to_csv(manifest, sep="\t", index=False)
    calibrations = tmp_path / "calibration"
    calibrations.mkdir()
    calibration = _calibration("acer", "NVIDIA GeForce RTX 4050 Laptop GPU", 1.0, memory_gb=30)
    (calibrations / "calibration_acer.json").write_text(json.dumps(calibration))
    args = grid_plan.build_argument_parser().parse_args(
        ["allocate", "--manifest", str(manifest), "--prepared-data-directory", str(prepared),
         "--calibration-directory", str(calibrations), "--reference-host", "acer",
         "--plan-directory", str(tmp_path / "plan")]
    )
    grid_plan.run_allocate(args)
    plan = grid_plan.read_plan(tmp_path / "plan")
    planned = {
        pair["configurations"][0]
        for machine in plan["machines"].values()
        for pair in machine["pairs"]
    }
    assert planned == {"b-min", "c-min"}
    assert plan["equivalent_configurations"] == {"a-hash": "b-min"}
