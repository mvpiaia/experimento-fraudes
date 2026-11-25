"""Plano, publicacao e merge da preparacao distribuida."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import distributed_preparation_plan as planning  # noqa: E402
import merge_distributed_preparation as merging  # noqa: E402
import publish_prepared_configuration as publication  # noqa: E402


def _create_args(tmp_path, **overrides):
    values = {
        "run_id": "run-1",
        "mode": "retry",
        "output_directory": str(tmp_path / "plan"),
        "acer_host": "acer",
        "dell_host": "dell",
        "lenovo_host": "lenovo",
        "black_root": str(tmp_path / "black"),
        "transparent_root": str(tmp_path / "transparent"),
        "reserve_bytes": 0,
        "micro_shard_bytes": 0,
        "limit": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_plan_is_complete_disjoint_deterministic_and_keeps_cohorts_together(tmp_path, monkeypatch):
    monkeypatch.setattr(
        planning.execution_provenance,
        "collect_provenance",
        lambda: {
            "code_commit": "abc",
            "code_tree_dirty": False,
            "environment_md5": "env",
            "environment_locked": True,
        },
    )
    monkeypatch.setattr(planning, "source_checksums", lambda: {"raw.csv": "1" * 32})
    monkeypatch.setattr(planning.verify_datasets, "verify_datasets", lambda: [])
    first = planning.create_plan(_create_args(tmp_path))
    plan = json.loads(first.read_text())
    assignment_file = first.parent / "preparation_assignments_acer.tsv"
    assert b"\r\n" not in assignment_file.read_bytes()
    identities = [item["configuration_md5"] for item in plan["assignments"]]
    assert len(identities) == len(set(identities))
    assert set(identities) == {
        planning.configuration_md5(entry)
        for block in planning.BLOCKS
        for entry in planning.experiment_grid.GRID_BUILDERS[block]()
    }
    for cohort in ("home-credit", "ieee-cis", "sparkov"):
        assert len({item["host"] for item in plan["assignments"] if item["cohort"] == cohort}) == 1
    second_dir = tmp_path / "plan-2"
    second = planning.create_plan(_create_args(tmp_path, output_directory=str(second_dir)))
    assert json.loads(second.read_text())["assignments"] == plan["assignments"]


def test_worker_preflight_ignores_provenance_and_force_reuses_no_run_id(tmp_path, monkeypatch):
    provenance = {
        "code_commit": "abc",
        "code_tree_dirty": False,
        "environment_md5": "env",
        "environment_locked": True,
    }
    monkeypatch.setattr(planning.execution_provenance, "collect_provenance", lambda: provenance)
    monkeypatch.setattr(planning, "source_checksums", lambda: {"raw.csv": "1" * 32})
    monkeypatch.setattr(planning.verify_datasets, "verify_datasets", lambda: [])
    root = tmp_path / "black" / "preparation-runs" / "run-1" / "acer"
    plan = {
        **provenance,
        "source_checksums": planning.source_checksums(),
        "grid_environment": {
            name: __import__("os").environ.get(name, "")
            for name in planning.GRID_ENVIRONMENT_VARIABLES
        },
        "reserve_bytes": 0,
        "micro_shard_bytes": 0,
        "hosts": {"acer": {"shard_root": str(root)}},
    }
    assert planning.verify_worker(plan, "acer") == []
    plan["environment_md5"] = "other"
    plan["code_commit"] = "another-commit"
    plan["code_tree_dirty"] = True
    assert planning.verify_worker(plan, "acer") == []

    monkeypatch.setattr(planning, "source_checksums", lambda: {"raw.csv": "2" * 32})
    assert any(
        "source checksum" in problem
        for problem in planning.verify_worker(plan, "acer")
    )
    monkeypatch.setattr(planning, "source_checksums", lambda: plan["source_checksums"])
    plan["grid_environment"][planning.GRID_ENVIRONMENT_VARIABLES[0]] = "different"
    assert any(
        "grid environment" in problem
        for problem in planning.verify_worker(plan, "acer")
    )

    (root / "manifests").mkdir(parents=True)
    with pytest.raises(ValueError, match="force requires a new run-id"):
        planning.create_plan(_create_args(tmp_path, mode="force"))


def _write_configuration(
    directory: Path, md5: str, dataset_md5: str, content: str = "feature,target\n1,0\n", **identities
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{dataset_md5}.csv").write_text(content)
    (directory / f"{dataset_md5}.columns.json").write_text("{}\n")
    (directory / f"{md5}.preparation_timing.json").write_text("{}\n")
    fields = [
        "configuration_md5",
        "dataset_md5",
        "content_md5",
        "fold_index",
        "split",
        "partition_md5",
        "unit_partition_md5",
        "train_population_md5",
        "train_unit_population_md5",
        "vocabulary_path",
        "window_index_path",
    ]
    row = {
        "configuration_md5": md5,
        "dataset_md5": dataset_md5,
        "content_md5": hashlib.md5(content.encode()).hexdigest(),
        "fold_index": "1",
        "split": "test",
        "vocabulary_path": "",
        "window_index_path": "",
        **{field: identities.get(field, f"{field}-value") for field in fields[5:9]},
    }
    manifest = directory / "manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(row)
    return manifest


def test_publication_checks_hash_and_publishes_manifest_last(tmp_path):
    source = tmp_path / "scratch"
    manifest = _write_configuration(source, "a" * 32, "b" * 32)
    shard = tmp_path / "shard"
    published = publication.publish(source, manifest, shard, "a" * 32)
    assert published.is_file()
    # compact-prepared-storage: o split e publicado comprimido.
    assert (shard / "files" / f"{'b' * 32}.csv.zst").is_file()
    assert not (shard / "files" / f"{'b' * 32}.csv").exists()
    assert not (shard / ".incoming" / ("a" * 32)).exists()

    bad_source = tmp_path / "bad"
    bad_manifest = _write_configuration(bad_source, "c" * 32, "d" * 32)
    (bad_source / f"{'d' * 32}.csv").write_text("corrupted")
    with pytest.raises(ValueError, match="content_md5 mismatch"):
        publication.publish(bad_source, bad_manifest, shard, "c" * 32)
    assert not (shard / "manifests" / f"{'c' * 32}.tsv").exists()

    corrupted = tmp_path / "corrupted" / f"{'b' * 32}.csv"
    corrupted.parent.mkdir()
    corrupted.write_text("corrupted")
    packed, _size = publication.compress_prepared_split(corrupted, hashlib.md5(b"corrupted").hexdigest())
    packed.replace(shard / "files" / f"{'b' * 32}.csv.zst")
    with pytest.raises(ValueError, match="content_md5 mismatch"):
        publication.verify_published(shard, "a" * 32)
    published.rename(published.with_suffix(".tsv.invalid"))
    publication.publish(source, manifest, shard, "a" * 32)
    publication.verify_published(shard, "a" * 32)


def test_merge_preserves_pairing_identities_and_rejects_missing_assignment(tmp_path):
    configurations = [("a" * 32, "b" * 32, "acer"), ("c" * 32, "d" * 32, "lenovo")]
    hosts = {}
    assignments = []
    for config, dataset, host in configurations:
        shard = tmp_path / host
        source = tmp_path / f"source-{host}"
        manifest = _write_configuration(
            source,
            config,
            dataset,
            partition_md5="same-partition",
            unit_partition_md5="same-units",
            train_population_md5="same-train",
            train_unit_population_md5="same-train-units",
        )
        publication.publish(source, manifest, shard, config)
        hosts[host] = {"shard_root": str(shard)}
        assignments.append({"configuration_md5": config, "host": host})
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / planning.PLAN_FILENAME).write_text(
        json.dumps({"hosts": hosts, "assignments": assignments, "reserve_bytes": 0})
    )
    canonical = tmp_path / "canonical"
    merged = merging.merge(plan_dir, canonical)
    with merged.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for field in (
        "partition_md5",
        "unit_partition_md5",
        "train_population_md5",
        "train_unit_population_md5",
    ):
        assert {row[field] for row in rows} == {
            {
                "partition_md5": "same-partition",
                "unit_partition_md5": "same-units",
                "train_population_md5": "same-train",
                "train_unit_population_md5": "same-train-units",
            }[field]
        }
    (Path(hosts["lenovo"]["shard_root"]) / "manifests" / f"{'c' * 32}.tsv").unlink()
    with pytest.raises(ValueError, match="missing published configuration"):
        merging.merge(plan_dir, tmp_path / "another")


def test_worker_script_has_retry_force_and_one_configuration_scratch():
    text = (PIPELINE_DIRECTORY / "run_distributed_preparation_worker.sh").read_text()
    assert "retry" in text and "force" in text
    assert '${SCRATCH_ROOT%/}/${configuration_md5}' in text
    assert "CONFIGURATION_MD5_FILTER" in text
    assert "configuration_md5%$'\\r'" in text
    assert 'PREPARATION_SCRATCH_DATA_DIRECTORY="${scratch}"' in text
    assert "STORAGE_ALLOW_ROOT_FILESYSTEM=true" in text
    assert "SCRATCH_RESERVE_BYTES" in text and "scratch_required" in text
    assert text.index("publish_prepared_configuration.py") < text.rindex('rm -rf -- "${scratch}"')
    common = (PIPELINE_DIRECTORY / "prepare_grid_common.sh").read_text()
    assert "CONFIGURATION_MD5_FILTER" in common
    assert common.index("configuration_filter") < common.index("Preparing ${database_type}")


# --- fix-ft-transformer-plan-md5 -------------------------------------------

# md5s do plano prep-20261101-123308, publicados nos shards: a correcao do FT
# nao pode muda-los.
PUBLISHED_PLAN_SAMPLES = {
    "classic": "044d5d522ec7ff01e35a6db44ab03654",
    "boosting": "118d8f41b2dc8e8d6430c208c0d11c0d",
    "tabbert": "6af9191e2a85b1c4aa67f224d7262f73",
}
# Impresso por --print-configuration-md5-only para um FT da Sparkov.
FT_PREPARER_SAMPLE = "eec30bdee1fcc461d9a713e712bcc331"


def _block_identities(block: str, rule=None) -> set[str]:
    rule = rule or planning.configuration_md5
    return {rule(entry) for entry in planning.experiment_grid.GRID_BUILDERS[block]()}


def test_identity_fix_keeps_published_blocks_and_matches_preparer_for_ft():
    for block, md5 in PUBLISHED_PLAN_SAMPLES.items():
        assert md5 in _block_identities(block)
        assert _block_identities(block) == _block_identities(block, planning._legacy_configuration_md5)
    assert FT_PREPARER_SAMPLE in _block_identities("ft-transformer")
    assert FT_PREPARER_SAMPLE not in _block_identities(
        "ft-transformer", planning._legacy_configuration_md5
    )


def _preparer_probe_md5(entry, tmp_path: Path) -> str:
    """Executa o preparador com os argumentos que prepare_grid_common.sh monta."""
    import subprocess

    (
        dataset_id, _name, _filename, prediction, auxiliary, handling, encoding, fill_na,
        technique, parameters, protocol, seed, normalization, cardinality, cardinality_parameter,
    ) = entry.to_tsv().split("\t")
    command = [
        sys.executable, str(PIPELINE_DIRECTORY / "prepare_lr_nb_knn_datasets.py"),
        "--dataset", dataset_id,
        "--prediction-column", prediction,
        "--auxiliary-data-is-test-data", auxiliary,
        "--categorical-handling-mode", handling,
        "--categorical-encoding", encoding,
        "--fill-na-strategy", fill_na,
        "--undersampling-technique", technique,
        "--undersampling-parameters", parameters,
        "--test-strategy", protocol,
        "--random-seed", seed,
        "--output-directory", str(tmp_path),
    ]
    if handling in ("transformer-indexed", "sequence-tokenized"):
        command += [
            "--numeric-normalization", normalization,
            "--high-cardinality-strategy", cardinality,
            "--high-cardinality-parameter", cardinality_parameter,
        ]
    if handling == "sequence-tokenized":
        grid = planning.experiment_grid
        command += [
            "--numeric-bins", str(grid.TABBERT_NUMERIC_BINS),
            "--window-length", str(grid.TABBERT_WINDOW_LENGTH),
            "--history-length", str(grid.TABBERT_HISTORY_LENGTH),
            "--window-stride", str(grid.TABBERT_WINDOW_STRIDE),
        ]
    output = subprocess.run(
        [*command, "--print-configuration-md5-only"],
        capture_output=True, text=True, check=True, cwd=PIPELINE_DIRECTORY,
    ).stdout
    return next(
        line.split("=", 1)[1].strip()
        for line in output.splitlines()
        if line.startswith("configuration_md5=")
    )


@pytest.mark.parametrize("block", planning.BLOCKS)
def test_plan_identity_matches_preparer_probe(block, tmp_path):
    entry = planning.experiment_grid.GRID_BUILDERS[block]()[0]
    assert planning.configuration_md5(entry) == _preparer_probe_md5(entry, tmp_path)


def _legacy_plan(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(
        planning.execution_provenance,
        "collect_provenance",
        lambda: {
            "code_commit": "abc",
            "code_tree_dirty": False,
            "environment_md5": "env",
            "environment_locked": True,
        },
    )
    monkeypatch.setattr(planning, "source_checksums", lambda: {"raw.csv": "1" * 32})
    monkeypatch.setattr(planning.verify_datasets, "verify_datasets", lambda: [])
    current_rule = planning.configuration_md5
    monkeypatch.setattr(planning, "configuration_md5", planning._legacy_configuration_md5)
    path = planning.create_plan(_create_args(tmp_path))
    monkeypatch.setattr(planning, "configuration_md5", current_rule)
    return path.parent


def _plan_files(plan_dir: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in sorted(plan_dir.iterdir())}


def test_amend_renames_only_ft_and_records_the_correction(tmp_path, monkeypatch):
    plan_dir = _legacy_plan(tmp_path, monkeypatch)
    before = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    legacy_ft = _block_identities("ft-transformer", planning._legacy_configuration_md5)

    changes = planning.amend_plan(plan_dir, timestamp="20261105-120000")

    assert {change["old"] for change in changes} == legacy_ft
    assert {change["block"] for change in changes} == {"ft-transformer"}
    after = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    expected = {
        planning.configuration_md5(entry)
        for block in planning.BLOCKS
        for entry in planning.experiment_grid.GRID_BUILDERS[block]()
    }
    assert {item["configuration_md5"] for item in after["assignments"]} == expected
    assert [(item["host"], item["block"], item["dataset_id"]) for item in after["assignments"]] == [
        (item["host"], item["block"], item["dataset_id"]) for item in before["assignments"]
    ]
    assert after["amendments"][0]["backup_suffix"] == ".before-amend-20261105-120000"
    assert len(after["amendments"][0]["changes"]) == len(legacy_ft)
    tsv_md5s = set()
    for path in plan_dir.glob("preparation_assignments_*.tsv"):
        with path.open(newline="") as handle:
            tsv_md5s |= {row["configuration_md5"] for row in csv.DictReader(handle, delimiter="\t")}
        assert (plan_dir / f"{path.name}.before-amend-20261105-120000").is_file()
    assert tsv_md5s == expected
    assert json.loads(
        (plan_dir / f"{planning.PLAN_FILENAME}.before-amend-20261105-120000").read_text()
    ) == before

    snapshot = _plan_files(plan_dir)
    assert planning.amend_plan(plan_dir, timestamp="20261105-130000") == []
    assert _plan_files(plan_dir) == snapshot


def test_amend_refuses_published_or_unknown_identities_without_changes(tmp_path, monkeypatch):
    plan_dir = _legacy_plan(tmp_path, monkeypatch)
    plan = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    ft = next(item for item in plan["assignments"] if item["block"] == "ft-transformer")
    manifests = Path(plan["hosts"][ft["host"]]["shard_root"]) / "manifests"
    manifests.mkdir(parents=True)
    (manifests / f"{ft['configuration_md5']}.tsv").write_text("published\n")
    snapshot = _plan_files(plan_dir)
    with pytest.raises(ValueError, match="already published"):
        planning.amend_plan(plan_dir)
    assert _plan_files(plan_dir) == snapshot

    (manifests / f"{ft['configuration_md5']}.tsv").unlink()
    plan["assignments"][0]["configuration_md5"] = "f" * 32
    (plan_dir / planning.PLAN_FILENAME).write_text(json.dumps(plan))
    snapshot = _plan_files(plan_dir)
    with pytest.raises(ValueError, match="not in the current grid"):
        planning.amend_plan(plan_dir)
    assert _plan_files(plan_dir) == snapshot


def _run_stub_worker(tmp_path: Path, manifest_rows: bool):
    """Executa o worker real com plano, preparador e publicador simulados."""
    import shutil
    import subprocess

    md5 = "a" * 32
    pipeline = tmp_path / "pipeline"
    pipeline.mkdir()
    shutil.copy2(PIPELINE_DIRECTORY / "run_distributed_preparation_worker.sh", pipeline)
    shard = tmp_path / "shard"
    (pipeline / "distributed_preparation_plan.py").write_text(
        "import sys\n"
        f"print({str(shard)!r}) if sys.argv[1] == 'shard-root' else None\n"
    )
    row = f"x\tx\tx\tx\tx\tx\tx\tx\tx\tx\t{md5}\n" if manifest_rows else ""
    (pipeline / "prepare_partition.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"printf 'h1\\th2\\th3\\th4\\th5\\th6\\th7\\th8\\th9\\th10\\tconfiguration_md5\\n{row}' > \"$2\"\n"
    )
    (pipeline / "prepare_partition.sh").chmod(0o755)
    marker = tmp_path / "published"
    (pipeline / "publish_prepared_configuration.py").write_text(
        f"open({str(marker)!r}, 'w').close()\n"
    )
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "preparation_assignments_host.tsv").write_text(
        f"block\tdataset_id\tcohort\tconfiguration_md5\nft-transformer\tsparkov\tsparkov\t{md5}\n"
    )
    scratch = tmp_path / "scratch"
    result = subprocess.run(
        ["bash", str(pipeline / "run_distributed_preparation_worker.sh"), str(plan), "retry"],
        capture_output=True,
        text=True,
        env={
            **__import__("os").environ,
            "PREPARATION_HOST": "host",
            "PYTHON_COMMAND": sys.executable,
            "PREPARATION_SCRATCH_ROOT": str(scratch),
            "PREPARATION_SCRATCH_RESERVE_BYTES": "0",
            "PREPARATION_MICRO_SHARD_BYTES": "0",
        },
    )
    return result, marker, scratch / md5


def test_worker_reports_identity_mismatch_without_publishing(tmp_path):
    result, marker, scratch = _run_stub_worker(tmp_path, manifest_rows=False)
    assert result.returncode == 1
    assert "Identity mismatch: " + "a" * 32 + " (ft-transformer, sparkov)" in result.stderr
    assert not marker.exists()
    assert not scratch.exists()


def test_worker_publishes_when_manifest_has_rows(tmp_path):
    result, marker, scratch = _run_stub_worker(tmp_path, manifest_rows=True)
    assert result.returncode == 0, result.stderr
    assert "Identity mismatch" not in result.stderr
    assert marker.exists()
    assert not scratch.exists()


def _current_plan(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(
        planning.execution_provenance,
        "collect_provenance",
        lambda: {
            "code_commit": "abc",
            "code_tree_dirty": False,
            "environment_md5": "env",
            "environment_locked": True,
        },
    )
    monkeypatch.setattr(planning, "source_checksums", lambda: {"raw.csv": "1" * 32})
    monkeypatch.setattr(planning.verify_datasets, "verify_datasets", lambda: [])
    return planning.create_plan(_create_args(tmp_path)).parent


def _tsv_rows(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def test_reassign_moves_whole_pairs_and_records_the_change(tmp_path, monkeypatch):
    plan_dir = _current_plan(tmp_path, monkeypatch)
    before = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    expected = [
        item
        for item in before["assignments"]
        if item["host"] == "dell" and item["block"] in ("ft-transformer", "tabbert")
    ]
    assert expected

    moved = planning.reassign_plan(
        plan_dir, "dell", "lenovo", ["ft-transformer", "tabbert"], timestamp="20261107-100000"
    )

    assert {item["configuration_md5"] for item in moved} == {
        item["configuration_md5"] for item in expected
    }
    after = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    moved_md5s = {item["configuration_md5"] for item in moved}
    for old, new in zip(before["assignments"], after["assignments"]):
        assert {k: v for k, v in old.items() if k != "host"} == {
            k: v for k, v in new.items() if k != "host"
        }
        assert new["host"] == ("lenovo" if old["configuration_md5"] in moved_md5s else old["host"])
    record = after["reassignments"][0]
    assert record["source_host"] == "dell" and record["target_host"] == "lenovo"
    assert record["backup_suffix"] == ".before-reassign-20261107-100000"
    assert sum(pair["configurations"] for pair in record["pairs"]) == len(moved)
    dell_rows = _tsv_rows(plan_dir / "preparation_assignments_dell.tsv")
    lenovo_rows = _tsv_rows(plan_dir / "preparation_assignments_lenovo.tsv")
    assert not moved_md5s & {row["configuration_md5"] for row in dell_rows}
    assert moved_md5s <= {row["configuration_md5"] for row in lenovo_rows}
    assert {row["configuration_md5"] for row in dell_rows} | {
        row["configuration_md5"] for row in lenovo_rows
    } == {item["configuration_md5"] for item in before["assignments"] if item["host"] in ("dell", "lenovo")}
    for name in ("preparation_assignments_dell.tsv", "preparation_assignments_lenovo.tsv", planning.PLAN_FILENAME):
        assert (plan_dir / f"{name}.before-reassign-20261107-100000").is_file()
    assert b"\r\n" not in (plan_dir / "preparation_assignments_lenovo.tsv").read_bytes()

    moved_item = moved[0]
    only_moved = tmp_path / "only-moved"
    only_moved.mkdir()
    after["assignments"] = [item for item in after["assignments"] if item["configuration_md5"] in moved_md5s]
    (only_moved / planning.PLAN_FILENAME).write_text(json.dumps(after))
    with pytest.raises(ValueError, match=f"missing published configuration: {moved_item['configuration_md5']} on lenovo"):
        merging.merge(only_moved, tmp_path / "canonical")


def test_reassign_refuses_published_pairs_and_unknown_hosts_without_changes(tmp_path, monkeypatch):
    plan_dir = _current_plan(tmp_path, monkeypatch)
    plan = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    ft = next(item for item in plan["assignments"] if item["host"] == "dell" and item["block"] == "ft-transformer")
    manifests = Path(plan["hosts"]["dell"]["shard_root"]) / "manifests"
    manifests.mkdir(parents=True)
    (manifests / f"{ft['configuration_md5']}.tsv").write_text("published\n")
    snapshot = _plan_files(plan_dir)
    with pytest.raises(ValueError, match="published configurations"):
        planning.reassign_plan(plan_dir, "dell", "lenovo", ["ft-transformer"], [ft["dataset_id"]])
    with pytest.raises(ValueError, match="not in the preparation plan"):
        planning.reassign_plan(plan_dir, "dell", "unknown", ["tabbert"])
    with pytest.raises(ValueError, match="no assignments"):
        planning.reassign_plan(plan_dir, "dell", "lenovo", ["tabbert"], ["sparkov"])
    assert _plan_files(plan_dir) == snapshot


def test_merge_rejects_same_split_name_with_different_bytes_across_shards(tmp_path):
    hosts = {}
    assignments = []
    for config, host, content in (
        ("a" * 32, "dell", "feature,target\n1,0\n"),
        ("c" * 32, "lenovo", "feature,target\n2,0\n"),
    ):
        shard = tmp_path / host
        source = tmp_path / f"source-{host}"
        manifest = _write_configuration(source, config, "b" * 32, content=content)
        publication.publish(source, manifest, shard, config)
        hosts[host] = {"shard_root": str(shard)}
        assignments.append({"configuration_md5": config, "host": host})
    plan_dir = tmp_path / "plan"
    plan_dir.mkdir()
    (plan_dir / planning.PLAN_FILENAME).write_text(
        json.dumps({"hosts": hosts, "assignments": assignments, "reserve_bytes": 0})
    )
    with pytest.raises(ValueError, match="canonical file conflict"):
        merging.merge(plan_dir, tmp_path / "canonical")


def test_validator_finds_side_files_by_name_after_the_scratch_is_gone(tmp_path):
    import validate_prepared_datasets as validation

    canonical = tmp_path / "canonical"
    canonical.mkdir()
    vocabulary = canonical / "abc_fold-1_vocabulary.json"
    vocabulary.write_text("{}", encoding="utf-8")
    recorded = "/home/worker/preparation-scratch/abc/abc_fold-1_vocabulary.json"

    assert validation.resolve_prepared_artifact(canonical, recorded) == vocabulary
    # Manifesto de uma maquina so: o caminho gravado ja e o do diretorio.
    assert validation.resolve_prepared_artifact(canonical, str(vocabulary)) == vocabulary
    # Ausente nos dois lugares: continua apontando o caminho gravado.
    missing = validation.resolve_prepared_artifact(tmp_path / "other", recorded)
    assert missing == Path(recorded) and not missing.is_file()


def _pre_revision_plan(tmp_path, monkeypatch) -> Path:
    """Plano criado antes da revisao hc-numeric-rates."""
    current_rule = planning.configuration_md5
    monkeypatch.setattr(
        planning,
        "configuration_md5",
        lambda entry, include_transformer_schema_revision=True: current_rule(entry, False),
    )
    plan_dir = _current_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(planning, "configuration_md5", current_rule)
    return plan_dir


def _home_credit_transformer_md5s(rule) -> set[str]:
    return {
        rule(entry)
        for block in ("ft-transformer", "tabbert")
        for entry in planning.experiment_grid.GRID_BUILDERS[block]()
        if entry.dataset_id == "home-credit-completa"
    }


def test_schema_revision_changes_only_home_credit_completa_transformers():
    for block in planning.BLOCKS:
        for entry in planning.experiment_grid.GRID_BUILDERS[block]():
            changed = planning.configuration_md5(entry) != planning.configuration_md5(entry, False)
            assert changed == (
                entry.dataset_id == "home-credit-completa" and block in ("ft-transformer", "tabbert")
            ), (block, entry.dataset_id)


def test_supersede_replaces_published_home_credit_transformers_and_records_it(tmp_path, monkeypatch):
    plan_dir = _pre_revision_plan(tmp_path, monkeypatch)
    before = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    old = _home_credit_transformer_md5s(lambda entry: planning.configuration_md5(entry, False))
    new = _home_credit_transformer_md5s(planning.configuration_md5)
    published = next(item for item in before["assignments"] if item["configuration_md5"] in old)
    manifests = Path(before["hosts"][published["host"]]["shard_root"]) / "manifests"
    manifests.mkdir(parents=True)
    (manifests / f"{published['configuration_md5']}.tsv").write_text("published\n")

    changes = planning.supersede_plan(plan_dir, timestamp="20261108-120000")

    assert {change["old"] for change in changes} == old
    assert {change["new"] for change in changes} == new
    assert [change["old_published"] for change in changes].count(True) == 1
    after = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    assert [(item["host"], item["block"], item["dataset_id"]) for item in after["assignments"]] == [
        (item["host"], item["block"], item["dataset_id"]) for item in before["assignments"]
    ]
    assert not old & {item["configuration_md5"] for item in after["assignments"]}
    assert after["supersessions"][0]["backup_suffix"] == ".before-supersede-20261108-120000"
    tsv_md5s = {
        row["configuration_md5"]
        for path in plan_dir.glob("preparation_assignments_*.tsv")
        for row in _tsv_rows(path)
    }
    assert new <= tsv_md5s and not old & tsv_md5s
    snapshot = _plan_files(plan_dir)
    assert planning.supersede_plan(plan_dir, timestamp="20261108-130000") == []
    assert _plan_files(plan_dir) == snapshot


def test_supersede_refuses_unknown_identities_without_changes(tmp_path, monkeypatch):
    plan_dir = _pre_revision_plan(tmp_path, monkeypatch)
    plan = json.loads((plan_dir / planning.PLAN_FILENAME).read_text())
    plan["assignments"][0]["configuration_md5"] = "f" * 32
    (plan_dir / planning.PLAN_FILENAME).write_text(json.dumps(plan))
    snapshot = _plan_files(plan_dir)
    with pytest.raises(ValueError, match="not superseded"):
        planning.supersede_plan(plan_dir)
    assert _plan_files(plan_dir) == snapshot
