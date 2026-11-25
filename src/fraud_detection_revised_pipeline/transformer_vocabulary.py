"""Target-independent categorical vocabularies for tabular Transformers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


PAD_INDEX = 0
UNK_INDEX = 1
MISSING_INDEX = 2
MASK_INDEX = 3
FIRST_VALUE_INDEX = 4
SPECIAL_TOKENS = {
    "[PAD]": PAD_INDEX,
    "[UNK]": UNK_INDEX,
    "[MISSING]": MISSING_INDEX,
    "[MASK]": MASK_INDEX,
}
HASH_PERSONALIZATION = b"tab-xform-v1"


def canonicalize_category(value: object) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


@dataclass(frozen=True)
class FieldVocabulary:
    column: str
    strategy: str
    values: dict[str, int]
    parameter: int

    @property
    def size(self) -> int:
        if self.strategy == "hashing":
            return FIRST_VALUE_INDEX + self.parameter
        return FIRST_VALUE_INDEX + len(self.values)

    def encode(self, value: object) -> int:
        canonical = canonicalize_category(value)
        if canonical is None:
            return MISSING_INDEX
        if self.strategy == "hashing":
            payload = f"{self.column}\x1f{canonical}".encode("utf-8")
            digest = hashlib.blake2b(
                payload, digest_size=8, person=HASH_PERSONALIZATION
            ).digest()
            return FIRST_VALUE_INDEX + int.from_bytes(digest, "big") % self.parameter
        return self.values.get(canonical, UNK_INDEX)

    def to_metadata(self) -> dict:
        return {
            "strategy": self.strategy,
            "parameter": self.parameter,
            "size": self.size,
            "special_tokens": SPECIAL_TOKENS,
            "values": self.values,
            "hash_algorithm": "blake2b-64" if self.strategy == "hashing" else None,
            "hash_personalization": (
                HASH_PERSONALIZATION.decode("ascii")
                if self.strategy == "hashing" else None
            ),
        }


def fit_field_vocabulary(
    series: pd.Series,
    column: str,
    strategy: str = "minimum-frequency",
    parameter: int = 10,
) -> FieldVocabulary:
    if strategy not in {"minimum-frequency", "hashing"}:
        raise ValueError("high cardinality strategy must be minimum-frequency or hashing")
    if parameter < 1:
        raise ValueError("high cardinality parameter must be positive")
    if strategy == "hashing":
        return FieldVocabulary(column, strategy, {}, int(parameter))
    canonical = series.map(canonicalize_category).dropna()
    counts = canonical.value_counts()
    retained = sorted(str(value) for value in counts[counts >= parameter].index)
    values = {value: FIRST_VALUE_INDEX + index for index, value in enumerate(retained)}
    return FieldVocabulary(column, strategy, values, int(parameter))


def transform_categorical_columns(
    dataframe: pd.DataFrame,
    vocabularies: dict[str, FieldVocabulary],
) -> pd.DataFrame:
    transformed = dataframe.copy()
    for column, vocabulary in vocabularies.items():
        transformed[column] = transformed[column].map(vocabulary.encode).astype("int64")
    return transformed
