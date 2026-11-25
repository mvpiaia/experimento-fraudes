"""Tempo de cada etapa da preparacao de uma configuracao (item G18).

O revisor pediu os tempos de preparacao por algoritmo e base. Cada etapa da
preparacao e aplicada a todas as dobras de uma vez, entao o tempo e medido por
configuracao e por etapa, e nao por dobra. Ele e gravado como
``<configuration_md5>.preparation_timing.json`` ao lado dos CSVs, em todos os
blocos, inclusive nos Transformers, que nao tem relatorio de preparacao. As
etapas que nao rodaram ficam ausentes, e nao com zero.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import execution_provenance

PREPARATION_TIMING_FILE_SUFFIX = ".preparation_timing.json"

# Ordem de execucao das etapas, a mesma do design do G15.
STEP_STRUCTURAL_LOAD = "structural_load"
STEP_PARTITION = "partition"
STEP_SCREENING = "screening"
STEP_MONETARY_TRANSFORM = "monetary_transform"
STEP_IMPUTATION = "imputation"
STEP_ENCODING = "encoding"
STEP_STANDARDIZATION = "standardization"
STEP_PCA = "pca"
STEP_TRANSFORMER_PREPROCESSING = "transformer_preprocessing"
STEP_RESAMPLING = "resampling"
STEP_WRITING = "writing"


def preparation_timing_path(directory: str | Path, configuration_md5: str) -> Path:
    return Path(directory) / f"{configuration_md5}{PREPARATION_TIMING_FILE_SUFFIX}"


class PreparationTimer:
    """Acumula os segundos de cada etapa, na ordem em que rodaram."""

    def __init__(self) -> None:
        self.step_seconds: dict[str, float] = {}
        self._open_steps: dict[str, float] = {}
        self._started_at = time.perf_counter()

    @contextmanager
    def step(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    # start/stop para etapas que cobrem um bloco longo demais para um `with`.
    def start(self, name: str) -> None:
        self._open_steps[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        elapsed = time.perf_counter() - self._open_steps.pop(name)
        self.step_seconds[name] = round(self.step_seconds.get(name, 0.0) + elapsed, 6)

    def as_dict(self, configuration_md5: str, fold_count: int) -> dict:
        return {
            "configuration_md5": configuration_md5,
            "fold_count": fold_count,
            "step_seconds": dict(self.step_seconds),
            "total_seconds": round(time.perf_counter() - self._started_at, 6),
            # A preparacao roda numa maquina so; a tabela de tempos nunca
            # mistura maquinas.
            **execution_provenance.collect_provenance(),
        }

    def write(self, directory: str | Path, configuration_md5: str, fold_count: int) -> Path:
        path = preparation_timing_path(directory, configuration_md5)
        path.write_text(
            json.dumps(self.as_dict(configuration_md5, fold_count), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return path
