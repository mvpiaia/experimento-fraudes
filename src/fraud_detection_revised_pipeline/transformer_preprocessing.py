"""Fold-fitted preprocessing for FT-Transformer and TabBERT."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from transformer_feature_schema import HIGH_CARDINALITY_COLUMNS, resolve_feature_types
from transformer_vocabulary import (
    FIRST_VALUE_INDEX,
    MISSING_INDEX,
    FieldVocabulary,
    fit_field_vocabulary,
    transform_categorical_columns,
)


@dataclass
class FittedTransformerPreprocessor:
    dataset: str
    target_column: str
    mode: str
    categorical_columns: list[str]
    numerical_columns: list[str]
    vocabularies: dict[str, FieldVocabulary]
    categorical_modes: dict[str, object]
    numerical_means: dict[str, float]
    numeric_transformer: object | None
    numeric_bin_edges: dict[str, list[float]]
    numeric_normalization: str

    def transform(self, dataframe: pd.DataFrame | None) -> pd.DataFrame | None:
        if dataframe is None:
            return None
        result = dataframe.copy().replace([np.inf, -np.inf], np.nan)
        if self.target_column in result:
            result = result.dropna(subset=[self.target_column])

        original_missing = result[self.numerical_columns].isna()
        if self.mode == "transformer-indexed":
            numeric = result[self.numerical_columns].fillna(self.numerical_means)
            if self.numerical_columns:
                result[self.numerical_columns] = self.numeric_transformer.transform(numeric)
            for column in self.numerical_columns:
                result[f"{column}__is_missing"] = original_missing[column].astype("int8")
        else:
            for column in self.numerical_columns:
                missing = original_missing[column]
                edges = np.asarray(self.numeric_bin_edges[column], dtype=float)
                values = pd.to_numeric(result[column], errors="coerce")
                tokens = (
                    np.searchsorted(edges[1:-1], values.fillna(0), side="right")
                    + FIRST_VALUE_INDEX
                )
                result[column] = np.where(missing, MISSING_INDEX, tokens).astype("int64")

        return transform_categorical_columns(result, self.vocabularies)

    def metadata(self) -> dict:
        numeric_metadata = {
            "normalization": self.numeric_normalization,
            "means": self.numerical_means,
            "bin_edges": self.numeric_bin_edges,
        }
        if isinstance(self.numeric_transformer, StandardScaler):
            numeric_metadata.update(
                mean=self.numeric_transformer.mean_.tolist(),
                scale=self.numeric_transformer.scale_.tolist(),
            )
        elif isinstance(self.numeric_transformer, QuantileTransformer):
            numeric_metadata.update(
                quantiles=self.numeric_transformer.quantiles_.tolist(),
                references=self.numeric_transformer.references_.tolist(),
                output_distribution=self.numeric_transformer.output_distribution,
            )
        return {
            "dataset": self.dataset,
            "mode": self.mode,
            "target_column": self.target_column,
            "categorical_columns": self.categorical_columns,
            "numerical_columns": self.numerical_columns,
            "categorical_modes": {
                key: None if pd.isna(value) else str(value)
                for key, value in self.categorical_modes.items()
            },
            "vocabularies": {
                column: vocabulary.to_metadata()
                for column, vocabulary in self.vocabularies.items()
            },
            "numeric": numeric_metadata,
        }


def _safe_mean(series: pd.Series) -> float:
    value = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).mean()
    return 0.0 if pd.isna(value) else float(value)


def _mode(series: pd.Series) -> object:
    values = series.dropna().mode()
    return None if values.empty else values.iloc[0]


def fit_transformer_preprocessor(
    train_data: pd.DataFrame,
    dataset: str,
    target_column: str,
    mode: str,
    numeric_normalization: str = "quantile",
    high_cardinality_strategy: str = "minimum-frequency",
    high_cardinality_parameter: int = 10,
    numeric_bins: int = 10,
) -> FittedTransformerPreprocessor:
    if mode not in {"transformer-indexed", "sequence-tokenized"}:
        raise ValueError("unsupported Transformer preparation mode")
    categorical, numerical = resolve_feature_types(dataset, train_data, target_column)
    categorical_modes = {column: _mode(train_data[column]) for column in categorical}
    vocabularies = {}
    for column in categorical:
        is_high_cardinality = (
            column in HIGH_CARDINALITY_COLUMNS
            or train_data[column].nunique(dropna=True) > 1024
        )
        strategy = high_cardinality_strategy if is_high_cardinality else "minimum-frequency"
        parameter = high_cardinality_parameter if is_high_cardinality else 1
        vocabularies[column] = fit_field_vocabulary(
            train_data[column], column, strategy, parameter
        )

    means = {column: _safe_mean(train_data[column]) for column in numerical}
    numeric_transformer = None
    bin_edges: dict[str, list[float]] = {}
    numeric_frame = train_data[numerical].replace([np.inf, -np.inf], np.nan).fillna(means)
    if mode == "transformer-indexed" and numerical:
        if numeric_normalization == "quantile":
            numeric_transformer = QuantileTransformer(
                n_quantiles=min(1000, len(numeric_frame)),
                output_distribution="normal",
                random_state=42,
            )
        elif numeric_normalization == "standard":
            numeric_transformer = StandardScaler()
        else:
            raise ValueError("numeric normalization must be quantile or standard")
        numeric_transformer.fit(numeric_frame)
    elif mode == "sequence-tokenized":
        if numeric_bins < 2:
            raise ValueError("numeric_bins must be at least 2")
        quantiles = np.linspace(0, 1, numeric_bins + 1)
        for column in numerical:
            values = pd.to_numeric(train_data[column], errors="coerce").dropna()
            if values.empty:
                edges = np.array([-np.inf, np.inf])
            else:
                edges = np.unique(values.quantile(quantiles).to_numpy(dtype=float))
                if len(edges) == 1:
                    edges = np.array([edges[0], edges[0]])
                edges[0], edges[-1] = -np.inf, np.inf
            bin_edges[column] = edges.tolist()

    return FittedTransformerPreprocessor(
        dataset=dataset,
        target_column=target_column,
        mode=mode,
        categorical_columns=categorical,
        numerical_columns=numerical,
        vocabularies=vocabularies,
        categorical_modes=categorical_modes,
        numerical_means=means,
        numeric_transformer=numeric_transformer,
        numeric_bin_edges=bin_edges,
        numeric_normalization=(numeric_normalization if mode == "transformer-indexed" else "quantile-bins"),
    )
