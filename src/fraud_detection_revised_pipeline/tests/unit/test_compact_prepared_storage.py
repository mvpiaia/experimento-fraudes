"""compact-prepared-storage: splits .csv.zst, deduplicacao, gate de espaco e migracao.

A identidade de um split e o MD5 do CSV descomprimido. Estes testes garantem
que comprimir ou deduplicar nao muda o que validadores, staging e executores
veem, e que nenhuma forma de ausencia vira espera pelo HD (codigo 75).
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PIPELINE_DIRECTORY = Path(__file__).resolve().parents[2]
if str(PIPELINE_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(PIPELINE_DIRECTORY))

import compact_prepared_storage as compact  # noqa: E402
import distributed_preparation_plan as planning  # noqa: E402
import grid_plan  # noqa: E402
import lr_nb_knn_specific as pipeline  # noqa: E402
import prepared_dataset_storage as storage  # noqa: E402
import preparation_sanity_checks as sanity  # noqa: E402
import publish_prepared_configuration as publication  # noqa: E402
import validate_prepared_datasets as validation  # noqa: E402

from test_g14_registry_and_categorical import raw_data_root  # noqa: E402,F401
from test_g15_verifier import CapturingReport, prepared  # noqa: E402,F401

pytestmark = pytest.mark.skipif(shutil.which("zstd") is None, reason="zstd binary not installed")


def _md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _write_split(directory: Path, dataset_md5: str, text: str) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{dataset_md5}.csv").write_text(text, encoding="utf-8")
    (directory / f"{dataset_md5}.columns.json").write_text("{}\n", encoding="utf-8")
    return _md5(text.encode("utf-8"))


# --- 1. Helper ----------------------------------------------------------------


def test_compressed_split_keeps_the_csv_identity(tmp_path):
    text = "a,label\n" + "".join(f"{i}.125,{i % 2}\n" for i in range(2000))
    content = _write_split(tmp_path, "1" * 32, text)
    packed, size = storage.compress_prepared_split(tmp_path / f"{'1' * 32}.csv", content)
    assert packed.name == f"{'1' * 32}.csv.zst" and size == len(text.encode())
    assert storage.split_content_digest(packed) == (content, size)
    (tmp_path / f"{'1' * 32}.csv").unlink()
    assert storage.prepared_split_path(tmp_path, "1" * 32) == packed
    frame = storage.read_prepared_split_csv(packed)
    assert len(frame) == 2000
    assert list(storage.read_prepared_split_csv(packed, nrows=0).columns) == ["a", "label"]
    local = storage.materialize_prepared_split(packed, tmp_path / "local.csv", content)
    assert local.read_text(encoding="utf-8") == text


def test_mixed_directory_prefers_csv_and_accepts_zst(tmp_path):
    _write_split(tmp_path, "1" * 32, "a\n1\n")
    content = _write_split(tmp_path, "2" * 32, "a\n2\n")
    storage.compress_prepared_split(tmp_path / f"{'2' * 32}.csv", content)
    (tmp_path / f"{'2' * 32}.csv").unlink()
    assert storage.prepared_split_path(tmp_path, "1" * 32).name.endswith(".csv")
    assert storage.prepared_split_path(tmp_path, "2" * 32).name.endswith(".csv.zst")
    assert storage.prepared_split_path(tmp_path, "3" * 32) is None


def test_missing_zstd_is_an_explicit_failure_not_a_storage_wait(tmp_path, monkeypatch):
    content = _write_split(tmp_path, "1" * 32, "a\n1\n")
    packed, _ = storage.compress_prepared_split(tmp_path / f"{'1' * 32}.csv", content)
    monkeypatch.setattr(storage.shutil, "which", lambda _name: None)
    with pytest.raises(storage.PreparedSplitDecompressorError, match="apt install zstd") as raised:
        storage.split_content_digest(packed)
    assert not isinstance(raised.value, OSError)
    assert not isinstance(raised.value, pipeline.PreparedDatasetStorageError)


def test_corrupted_split_is_rejected_and_leaves_no_copy(tmp_path):
    content = _write_split(tmp_path, "1" * 32, "a,label\n" + "1,0\n" * 5000)
    packed, _ = storage.compress_prepared_split(tmp_path / f"{'1' * 32}.csv", content)
    data = bytearray(packed.read_bytes())
    data[len(data) // 2] ^= 0xFF
    packed.write_bytes(bytes(data))
    with pytest.raises((OSError, ValueError)):
        storage.materialize_prepared_split(packed, tmp_path / "local.csv", content)
    assert not (tmp_path / "local.csv").exists()
    with pytest.raises(ValueError, match="content_md5 mismatch"):
        storage.materialize_prepared_split(tmp_path / f"{'1' * 32}.csv", tmp_path / "x.csv", "0" * 32)
    assert not (tmp_path / "x.csv").exists()


def test_storage_metadata_records_stored_and_content_bytes(tmp_path):
    text = "a\n" + "7\n" * 10000
    content = _write_split(tmp_path, "1" * 32, text)
    storage.compress_prepared_split(tmp_path / f"{'1' * 32}.csv", content)
    (tmp_path / f"{'1' * 32}.csv").unlink()
    storage.write_storage_metadata(tmp_path, "c" * 32, [{"dataset_md5": "1" * 32, "content_md5": content}])
    split = storage.read_storage_metadata(tmp_path, "c" * 32)["splits"]["1" * 32]
    assert split["codec"] == "zstd" and split["content_bytes"] == len(text)
    assert split["stored_bytes"] == (tmp_path / f"{'1' * 32}.csv.zst").stat().st_size < len(text)


# --- 2. Leitores sobre um diretorio so com .csv.zst ----------------------------


def _canonical_copy(manifest: pd.DataFrame, source: Path, destination: Path) -> Path:
    shutil.copytree(source, destination)
    manifest.to_csv(destination / compact.CANONICAL_MANIFEST_NAME, sep="\t", index=False)
    return destination / compact.CANONICAL_MANIFEST_NAME


def _outcomes(report: CapturingReport) -> list[tuple[str, bool, str]]:
    return sorted((check, passed, detail) for check, _scope, passed, detail in report.outcomes)


def test_readers_see_the_same_data_after_compression(prepared, tmp_path):
    manifest, output, reports = prepared
    plain_manifest = _canonical_copy(manifest, output, tmp_path / "plain")
    packed_manifest = _canonical_copy(manifest, output, tmp_path / "packed")
    plain, packed = plain_manifest.parent, packed_manifest.parent

    summary = compact.run(packed, "compress", log=lambda _line: None)
    assert summary.problems == [] and summary.compressed > 0
    assert not [p for p in packed.glob("*.csv") if re.fullmatch(r"[0-9a-f]{32}\.csv", p.name)]

    # Validador (P1 e demais) e sanity checks com os mesmos resultados.
    for directory, path in ((plain, plain_manifest), (packed, packed_manifest)):
        report = CapturingReport()
        validation.validate_prepared_files(path, directory, report, 0)
        sanity.check_preparation_reports(manifest, directory, reports, report)
        sanity.check_missing_screening(manifest, directory, reports, report)
        if directory == plain:
            expected = _outcomes(report)
        else:
            assert _outcomes(report) == expected
    p1 = [passed for check, passed, _detail in expected if check == "P1"]
    assert p1 and all(p1)

    md5 = manifest["configuration_md5"].iloc[0]
    # Staging entrega os mesmos CSVs locais.
    with pipeline.stage_prepared_dataset_files(plain, plain_manifest, md5, tmp_path / "s1") as one, \
            pipeline.stage_prepared_dataset_files(packed, packed_manifest, md5, tmp_path / "s2") as two:
        first = {p.name: p.read_bytes() for p in one.iterdir()}
        second = {p.name: p.read_bytes() for p in two.iterdir()}
    assert first == second

    # Leitura direta (benchmark) e estimativa de memoria do grid_plan.
    loaded = [
        storage.load_prepared_data_splits(directory, path, md5, manifest["test_strategy"].iloc[0])
        for directory, path in ((plain, plain_manifest), (packed, packed_manifest))
    ]
    for fold_a, fold_b in zip(loaded[0]["test_info"], loaded[1]["test_info"]):
        for split in ("train_data", "validation_data", "test_data"):
            if fold_a.get(split) is not None:
                pd.testing.assert_frame_equal(fold_a[split], fold_b[split])
    raw = pd.read_csv(plain_manifest, sep="\t", dtype=str)
    plain_files = grid_plan._configuration_files(plain, raw)
    packed_files = grid_plan._configuration_files(packed, raw)
    assert grid_plan.configuration_content_bytes(md5, packed_files[md5]) == (
        grid_plan.configuration_content_bytes(md5, plain_files[md5])
    )

    # Rollback devolve os mesmos bytes.
    compact.run(packed, "decompress", log=lambda _line: None)
    for path in plain.glob("*.csv"):
        assert (packed / path.name).read_bytes() == path.read_bytes()


def test_shell_storage_checks_accept_compressed_splits(tmp_path):
    md5, dataset_md5 = "c" * 32, "d" * 32
    manifest = tmp_path / "manifest.tsv"
    columns = [f"c{i}" for i in range(1, 11)] + ["configuration_md5", "dataset_md5"]
    manifest.write_text("\t".join(columns) + "\n" + "\t".join(["x"] * 10 + [md5, dataset_md5]) + "\n")
    content = _write_split(tmp_path, dataset_md5, "a\n1\n")
    storage.compress_prepared_split(tmp_path / f"{dataset_md5}.csv", content)
    (tmp_path / f"{dataset_md5}.csv").unlink()
    script = (
        f"source {PIPELINE_DIRECTORY / 'run_storage_common.sh'}\n"
        f"PREPARED_DATA_DIRECTORY={tmp_path}; runtime_manifest={manifest}\n"
        "STORAGE_CHECK_TIMEOUT_SECONDS=5\n"
        f"configuration_manifest_is_complete {md5} && configuration_files_are_available {md5} && echo ok\n"
        f"rm {tmp_path / (dataset_md5 + '.csv.zst')}\n"
        f"report_unavailable_configuration_files {md5}\n"
    )
    result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
    assert result.stdout.strip() == "ok", result.stderr
    assert "nor .csv.zst" in result.stderr


def test_remote_checksum_script_verifies_compressed_splits(tmp_path):
    text = (PIPELINE_DIRECTORY / "stage_machine_configurations.sh").read_text()
    script = text.split("verify_script='", 1)[1].split("'\n", 1)[0]
    good = _write_split(tmp_path, "1" * 32, "a\n1\n")
    storage.compress_prepared_split(tmp_path / f"{'1' * 32}.csv", good)
    (tmp_path / f"{'1' * 32}.csv").unlink()
    lines = f"{good} {'1' * 32}\n{'0' * 32} {'1' * 32}\n{good} {'2' * 32}\n"
    result = subprocess.run(
        ["bash", "-c", f"cd {tmp_path} && {script}"], input=lines, capture_output=True, text=True
    )
    assert result.stdout.splitlines() == [f"{'1' * 32}.csv.zst", f"{'2' * 32}.csv"]


# Montar o nome de um split no diretorio preparado fora do helper e o que faria
# um runner esperar pelo HD para sempre. Cada excecao diz por que e segura.
SPLIT_NAME_ALLOWED = {
    "grid_plan.py": "lista as duas formas para o staging entre maquinas",
    "validate_prepared_datasets.py": "so na mensagem de split ausente",
    "run_transformer_prepared_experiments.py": "le o diretorio local do staging",
    "prepared_dataset_storage.py": "o proprio helper e a escrita da preparacao",
}


def test_no_split_path_is_built_outside_the_helper():
    pattern = re.compile(r"dataset_md5[^\n]{0,40}\}\.csv\"")
    offenders = [
        f"{path.name}:{number}"
        for path in sorted(PIPELINE_DIRECTORY.glob("*.py"))
        if path.name not in SPLIT_NAME_ALLOWED
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if pattern.search(line) and "csv.zst" not in line
    ]
    assert offenders == []


# --- 3. Publicacao, deduplicacao e worker -------------------------------------


def _configuration(source: Path, md5: str, splits: dict[str, str]) -> Path:
    rows = []
    for dataset_md5, text in splits.items():
        content = _write_split(source, dataset_md5, text)
        rows.append({"configuration_md5": md5, "dataset_md5": dataset_md5, "content_md5": content,
                     "fold_index": "1", "split": "validation"})
    manifest = source / "manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


def test_publication_compresses_and_links_identical_content(tmp_path):
    shard = tmp_path / "shard"
    shared = "a,label\n" + "3.5,1\n" * 3000
    first = _configuration(tmp_path / "s1", "a" * 32, {"1" * 32: shared, "2" * 32: "a,label\n1,0\n"})
    publication.publish(first.parent, first, shard, "a" * 32)
    second = _configuration(tmp_path / "s2", "b" * 32, {"3" * 32: shared})
    publication.publish(second.parent, second, shard, "b" * 32)
    files = shard / "files"
    assert not list(files.glob("*[0-9a-f].csv"))
    assert (files / f"{'1' * 32}.csv.zst").samefile(files / f"{'3' * 32}.csv.zst")
    assert not (files / f"{'2' * 32}.csv.zst").samefile(files / f"{'1' * 32}.csv.zst")
    assert storage.read_storage_metadata(files, "b" * 32)["splits"]["3" * 32]["codec"] == "zstd"
    publication.verify_published(shard, "a" * 32)
    publication.verify_published(shard, "b" * 32)


def test_publication_reports_no_space_with_its_own_code(tmp_path, monkeypatch):
    manifest = _configuration(tmp_path / "s1", "a" * 32, {"1" * 32: "a\n1\n"})

    def full(*_args, **_kwargs):
        raise OSError(publication.errno.ENOSPC, "No space left on device")

    monkeypatch.setattr(publication, "_rsync", full)
    monkeypatch.setattr(sys, "argv", [
        "publish", "--source-directory", str(manifest.parent), "--manifest", str(manifest),
        "--shard-directory", str(tmp_path / "shard"), "--configuration-md5", "a" * 32,
    ])
    assert publication.main() == publication.NO_SPACE_EXIT_CODE


def test_space_requirement_uses_the_largest_published_configuration(tmp_path):
    shard = tmp_path / "shard"
    manifest = _configuration(tmp_path / "s1", "a" * 32, {"1" * 32: "a\n" + "9\n" * 5000})
    publication.publish(manifest.parent, manifest, shard, "a" * 32)
    stored = (shard / "files" / f"{'1' * 32}.csv.zst").stat().st_size
    plan = {
        "reserve_bytes": 100, "micro_shard_bytes": 10**6,
        "hosts": {"h": {"shard_root": str(shard)}},
        "assignments": [
            {"host": "h", "block": "classic", "dataset_id": "d", "configuration_md5": "a" * 32},
            {"host": "h", "block": "classic", "dataset_id": "d", "configuration_md5": "b" * 32},
        ],
    }
    assert planning.configuration_space_requirement(plan, "h", "classic", "d") == 100 + stored
    assert planning.configuration_space_requirement(plan, "h", "boosting", "d") == 100 + 10**6
    assert planning.check_configuration_space(plan, "h", "classic", "d") is None
    plan["reserve_bytes"] = 10**18
    assert "No space for the next configuration" in planning.check_configuration_space(
        plan, "h", "classic", "d"
    )


def _stub_worker(tmp_path: Path, space_code: int = 0, publish_code: int = 0, scratch_ready: bool = False):
    """Worker real com plano, preparador e publicador simulados."""
    md5 = "a" * 32
    pipeline_dir = tmp_path / "pipeline"
    pipeline_dir.mkdir()
    shutil.copy2(PIPELINE_DIRECTORY / "run_distributed_preparation_worker.sh", pipeline_dir)
    calls = tmp_path / "calls"
    (pipeline_dir / "distributed_preparation_plan.py").write_text(
        "import sys\n"
        f"open({str(calls)!r}, 'a').write(sys.argv[1] + '\\n')\n"
        f"print({str(tmp_path / 'shard')!r}) if sys.argv[1] == 'shard-root' else None\n"
        f"sys.exit({space_code} if sys.argv[1] == 'space' else 0)\n"
    )
    (pipeline_dir / "prepare_partition.sh").write_text(
        "#!/usr/bin/env bash\n"
        f"echo prepare >> {calls}\n"
        "printf 'h1\\th2\\th3\\th4\\th5\\th6\\th7\\th8\\th9\\th10\\tconfiguration_md5\\n"
        "x\\tx\\tx\\tx\\tx\\tx\\tx\\tx\\tx\\tx\\t%s\\n' \"$CONFIGURATION_MD5_FILTER\" > \"$2\"\n"
    )
    (pipeline_dir / "prepare_partition.sh").chmod(0o755)
    (pipeline_dir / "publish_prepared_configuration.py").write_text(
        "import sys\n"
        f"open({str(calls)!r}, 'a').write('publish\\n')\n"
        f"sys.exit({publish_code})\n"
    )
    plan = tmp_path / "plan"
    plan.mkdir()
    (plan / "preparation_assignments_host.tsv").write_text(
        "block\tdataset_id\tcohort\tconfiguration_md5\n"
        f"classic\tsparkov\tsparkov\t{md5}\nclassic\tsparkov\tsparkov\t{'b' * 32}\n"
    )
    scratch = tmp_path / "scratch"
    if scratch_ready:
        (scratch / md5).mkdir(parents=True)
        (scratch / md5 / "manifest.tsv").write_text(
            "h1\th2\th3\th4\th5\th6\th7\th8\th9\th10\tconfiguration_md5\n"
            f"x\tx\tx\tx\tx\tx\tx\tx\tx\tx\t{md5}\n"
        )
    result = subprocess.run(
        ["bash", str(pipeline_dir / "run_distributed_preparation_worker.sh"), str(plan), "retry"],
        capture_output=True, text=True,
        env={**os.environ, "PREPARATION_HOST": "host", "PYTHON_COMMAND": sys.executable,
             "PREPARATION_SCRATCH_ROOT": str(scratch), "PREPARATION_SCRATCH_RESERVE_BYTES": "0",
             "PREPARATION_MICRO_SHARD_BYTES": "0"},
    )
    return result, calls.read_text().split()


def test_worker_stops_the_queue_when_the_shard_has_no_space(tmp_path):
    result, calls = _stub_worker(tmp_path, space_code=10)
    assert result.returncode == 10
    assert "Stopping the queue" in result.stderr
    assert "prepare" not in calls and "publish" not in calls


def test_worker_stops_when_publication_runs_out_of_space(tmp_path):
    result, calls = _stub_worker(tmp_path, publish_code=10)
    assert result.returncode == 10
    assert calls.count("prepare") == 1 and calls.count("publish") == 1


def test_retry_publishes_a_prepared_scratch_without_preparing(tmp_path):
    result, calls = _stub_worker(tmp_path, scratch_ready=True)
    assert result.returncode == 0, result.stderr
    assert "without preparing again" in result.stdout
    # A primeira configuracao vem do scratch; so a segunda e preparada.
    assert calls.count("prepare") == 1 and calls.count("publish") == 2
    assert not (tmp_path / "scratch" / ("a" * 32)).exists()


def test_preflight_requires_zstd(tmp_path, monkeypatch):
    monkeypatch.setattr(planning, "source_checksums", lambda: {})
    monkeypatch.setattr(planning.verify_datasets, "verify_datasets", lambda: [])
    plan = {
        "source_checksums": {}, "grid_environment": {name: os.environ.get(name, "")
                                                     for name in planning.GRID_ENVIRONMENT_VARIABLES},
        "reserve_bytes": 0, "micro_shard_bytes": 0, "hosts": {"h": {"shard_root": str(tmp_path)}},
    }
    assert planning.verify_worker(plan, "h") == []
    monkeypatch.setattr(planning.shutil, "which", lambda _name: None)
    assert any("zstd" in problem for problem in planning.verify_worker(plan, "h"))


# --- 4. Migracao --------------------------------------------------------------


def _shard_with_duplicates(tmp_path: Path) -> Path:
    shard = tmp_path / "shard"
    (shard / "manifests").mkdir(parents=True)
    shared = "a,label\n" + "2.25,0\n" * 4000
    for md5, splits in (("a" * 32, {"1" * 32: shared, "2" * 32: "a,label\n5,1\n"}),
                        ("b" * 32, {"3" * 32: shared})):
        rows = []
        for dataset_md5, text in splits.items():
            rows.append({"configuration_md5": md5, "dataset_md5": dataset_md5,
                         "content_md5": _write_split(shard / "files", dataset_md5, text)})
        with (shard / "manifests" / f"{md5}.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    return shard


def test_dedupe_only_links_identical_csvs(tmp_path):
    shard = _shard_with_duplicates(tmp_path)
    files = shard / "files"
    summary = compact.run(shard, "dedupe", log=lambda _line: None)
    assert summary.linked == 1 and summary.problems == []
    assert (files / f"{'1' * 32}.csv").samefile(files / f"{'3' * 32}.csv")
    assert compact.run(shard, "dedupe", log=lambda _line: None).linked == 0


def test_dedupe_refuses_content_that_does_not_match_the_manifest(tmp_path):
    shard = _shard_with_duplicates(tmp_path)
    (shard / "files" / f"{'3' * 32}.csv").write_text("a,label\n9,9\n")
    with pytest.raises(ValueError, match="does not match content_md5"):
        compact.run(shard, "dedupe", log=lambda _line: None)


def test_compression_is_resumable_and_compresses_each_content_once(tmp_path, monkeypatch):
    shard = _shard_with_duplicates(tmp_path)
    files = shard / "files"
    original = {path.name: path.read_bytes() for path in files.glob("*.csv")}
    calls = []
    real = compact.compress_prepared_split

    def interrupted(path, content):
        calls.append(path.name)
        if len(calls) == 2:
            raise KeyboardInterrupt
        return real(path, content)

    monkeypatch.setattr(compact, "compress_prepared_split", interrupted)
    with pytest.raises(KeyboardInterrupt):
        compact.run(shard, "compress", log=lambda _line: None)
    # Cada split continua integro em pelo menos uma forma.
    for name, data in original.items():
        stem = name.removesuffix(".csv")
        path = storage.prepared_split_path(files, stem)
        assert storage.split_content_digest(path)[0] == _md5(data)
    (files / ".leftover.csv.zst.tmp").write_bytes(b"partial")

    monkeypatch.setattr(compact, "compress_prepared_split", real)
    summary = compact.run(shard, "compress", log=lambda _line: None)
    assert summary.problems == []
    assert not list(files.glob("*[0-9a-f].csv")) and not list(files.glob(".*.tmp"))
    assert (files / f"{'1' * 32}.csv.zst").samefile(files / f"{'3' * 32}.csv.zst")
    for md5 in ("a" * 32, "b" * 32):
        assert storage.read_storage_metadata(files, md5) is not None
        publication.verify_published(shard, md5)
    assert compact.run(shard, "compress", log=lambda _line: None).skipped == 2

    compact.run(shard, "decompress", log=lambda _line: None)
    assert {path.name: path.read_bytes() for path in files.glob("*.csv")} == original
