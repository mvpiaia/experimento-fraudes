"""Tomek-link and evolutionary under-sampling (TEUS).

Implementation of the method proposed by Gong, Gao and Wang in "A Hybrid
Evolutionary Under-sampling Method for Handling the Class Imbalance Problem
with Overlap in Credit Classification" (2022), DOI 10.1007/s11518-022-5545-5.

The paper does not specify the density estimator used in Equation 7, the
number of crossover points, or behavior when no Tomek link exists. This
implementation uses Gaussian naive-Bayes log likelihoods, two-point crossover,
and the majority sample nearest to the minority class as a boundary fallback.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from sklearn.neighbors import NearestNeighbors


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class _Subgroup:
    """Homogeneous majority subgroup used by the genetic algorithm."""

    sample_indexes: IntArray
    information_contribution: float
    overlap_potential: float

    @property
    def size(self) -> int:
        return len(self.sample_indexes)


class TEUSUnderSampler:
    """Under-sample a binary dataset using the TEUS method.

    Parameters match the paper's experiments by default: ``alpha=5``,
    ``beta=10``, ``gamma=3``, ``quantiles=99``, population size 200,
    mutation rate 0.01, and 300 generations.

    The estimator exposes the familiar ``fit_resample(X, y)`` API. Input
    features must be finite and numeric; categorical data should be encoded
    and all features standardized before use, as in the paper.
    """

    def __init__(
        self,
        *,
        alpha: float = 5.0,
        beta: float = 10.0,
        gamma: float = 3.0,
        quantiles: int = 99,
        population_size: int = 200,
        generations: int = 300,
        crossover_probability: float = 0.8,
        mutation_rate: float = 0.01,
        random_state: int | None = 42,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.quantiles = quantiles
        self.population_size = population_size
        self.generations = generations
        self.crossover_probability = crossover_probability
        self.mutation_rate = mutation_rate
        self.random_state = random_state

    def fit_resample(
        self,
        features: pd.DataFrame | ArrayLike,
        target: pd.Series | ArrayLike,
    ) -> tuple[pd.DataFrame | FloatArray, pd.Series | NDArray]:
        """Fit TEUS and return the selected majority plus all minority rows."""
        feature_values, target_values = self._validate_inputs(features, target)
        self.feature_names_in_ = (
            np.asarray(features.columns, dtype=object)
            if isinstance(features, pd.DataFrame)
            else None
        )
        self.n_features_in_ = feature_values.shape[1]

        class_labels, class_counts = np.unique(target_values, return_counts=True)
        minority_label = class_labels[np.argmin(class_counts)]
        majority_label = class_labels[np.argmax(class_counts)]
        minority_indexes = np.flatnonzero(target_values == minority_label)
        majority_indexes = np.flatnonzero(target_values == majority_label)

        self.classes_ = class_labels
        self.minority_class_ = minority_label
        self.majority_class_ = majority_label

        if len(minority_indexes) == len(majority_indexes):
            self.sample_indices_ = np.arange(len(target_values), dtype=np.int64)
            return self._restore_output_types(features, target, self.sample_indices_)

        boundary_indexes, boundary_radii = self._find_tomek_boundaries(
            feature_values,
            target_values,
            majority_indexes,
            minority_indexes,
        )
        majority_features = feature_values[majority_indexes]
        minority_features = feature_values[minority_indexes]
        information_values = self._calculate_information_contribution(
            majority_features
        )
        overlap_values = self._calculate_overlap_potential(
            majority_features,
            minority_features,
            feature_values[boundary_indexes],
            majority_indexes,
            boundary_indexes,
            boundary_radii,
        )
        subgroups = self._build_subgroups(
            majority_indexes,
            information_values,
            overlap_values,
        )
        rng = np.random.default_rng(self.random_state)
        selected_counts = self._run_genetic_algorithm(
            subgroups,
            minority_size=len(minority_indexes),
            rng=rng,
        )
        selected_majority = self._sample_subgroups(subgroups, selected_counts, rng)

        self.boundary_indices_ = boundary_indexes
        self.information_contribution_ = information_values
        self.overlap_potential_ = overlap_values
        self.selected_subgroup_sizes_ = selected_counts
        self.sample_indices_ = np.concatenate(
            (selected_majority, minority_indexes)
        ).astype(np.int64)
        rng.shuffle(self.sample_indices_)
        return self._restore_output_types(features, target, self.sample_indices_)

    def _validate_inputs(
        self,
        features: pd.DataFrame | ArrayLike,
        target: pd.Series | ArrayLike,
    ) -> tuple[FloatArray, NDArray]:
        if self.alpha <= 0 or self.beta <= 0 or self.gamma <= 0:
            raise ValueError("alpha, beta and gamma must be positive")
        if self.quantiles < 1:
            raise ValueError("quantiles must be at least 1")
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2")
        if self.generations < 1:
            raise ValueError("generations must be at least 1")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("crossover_probability must be between 0 and 1")
        if not 0 <= self.mutation_rate <= 1:
            raise ValueError("mutation_rate must be between 0 and 1")

        try:
            feature_values = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("features must contain only numeric values") from error
        target_values = np.asarray(target)
        if feature_values.ndim != 2 or feature_values.shape[0] == 0:
            raise ValueError("features must be a non-empty two-dimensional dataset")
        if target_values.ndim != 1 or len(target_values) != len(feature_values):
            raise ValueError("target must be one-dimensional and match features")
        if not np.isfinite(feature_values).all():
            raise ValueError("features cannot contain NaN or infinite values")
        class_labels, class_counts = np.unique(target_values, return_counts=True)
        if len(class_labels) != 2:
            raise ValueError("TEUS supports binary classification only")
        if class_counts[0] == class_counts[1]:
            return feature_values, target_values
        if class_counts.min() < 1:
            raise ValueError("both classes must contain at least one sample")
        return feature_values, target_values

    @staticmethod
    def _find_tomek_boundaries(
        features: FloatArray,
        target: NDArray,
        majority_indexes: IntArray,
        minority_indexes: IntArray,
    ) -> tuple[IntArray, dict[int, float]]:
        nearest_model = NearestNeighbors(n_neighbors=2).fit(features)
        distances, neighbors = nearest_model.kneighbors(features)
        nearest_indexes = neighbors[:, 1]
        nearest_distances = distances[:, 1]
        boundary_indexes = []
        boundary_radii = {}

        for majority_index in majority_indexes:
            neighbor_index = nearest_indexes[majority_index]
            is_opposite_class = target[neighbor_index] != target[majority_index]
            is_mutual_neighbor = nearest_indexes[neighbor_index] == majority_index
            if is_opposite_class and is_mutual_neighbor:
                boundary_indexes.append(majority_index)
                boundary_radii[int(majority_index)] = float(
                    nearest_distances[majority_index]
                )

        if boundary_indexes:
            return np.asarray(boundary_indexes, dtype=np.int64), boundary_radii

        minority_model = NearestNeighbors(n_neighbors=1).fit(features[minority_indexes])
        cross_distances, _ = minority_model.kneighbors(features[majority_indexes])
        fallback_position = int(np.argmin(cross_distances[:, 0]))
        fallback_index = int(majority_indexes[fallback_position])
        return (
            np.asarray([fallback_index], dtype=np.int64),
            {fallback_index: float(cross_distances[fallback_position, 0])},
        )

    @staticmethod
    def _calculate_information_contribution(
        majority_features: FloatArray,
    ) -> FloatArray:
        """Calculate Equation 7 using Gaussian conditional densities."""
        means = majority_features.mean(axis=0)
        variances = majority_features.var(axis=0)
        variances = np.maximum(variances, np.finfo(np.float64).eps)
        log_probabilities = -0.5 * (
            np.log(2 * np.pi * variances)
            + np.square(majority_features - means) / variances
        )
        information_values = log_probabilities.sum(axis=1)
        return TEUSUnderSampler._min_max_normalize(information_values)

    @staticmethod
    def _calculate_overlap_potential(
        majority_features: FloatArray,
        minority_features: FloatArray,
        boundary_features: FloatArray,
        majority_indexes: IntArray,
        boundary_indexes: IntArray,
        boundary_radii: dict[int, float],
    ) -> FloatArray:
        """Calculate the local opposite-class density from Equation 9."""
        boundary_model = NearestNeighbors(n_neighbors=1).fit(boundary_features)
        radii, _ = boundary_model.kneighbors(majority_features)
        radii = radii[:, 0]
        majority_positions = {
            int(sample_index): position
            for position, sample_index in enumerate(majority_indexes)
        }
        for boundary_index in boundary_indexes:
            position = majority_positions[int(boundary_index)]
            radii[position] = boundary_radii[int(boundary_index)]

        positive_radii = radii[radii > 0]
        radius_floor = (
            float(positive_radii.min())
            if positive_radii.size
            else np.finfo(np.float64).eps
        )
        radii = np.maximum(radii, radius_floor)
        overlap_values = np.zeros(len(majority_features), dtype=np.float64)
        for sample_position, (sample, radius) in enumerate(
            zip(majority_features, radii)
        ):
            squared_distances = np.square(minority_features - sample).sum(axis=1)
            inside_radius = squared_distances < radius**2
            overlap_values[sample_position] = np.exp(
                -squared_distances[inside_radius] / radius**2
            ).sum()
        return TEUSUnderSampler._min_max_normalize(overlap_values)

    def _build_subgroups(
        self,
        majority_indexes: IntArray,
        information_values: FloatArray,
        overlap_values: FloatArray,
    ) -> list[_Subgroup]:
        quantile_points = np.arange(1, self.quantiles + 1) / (self.quantiles + 1)
        information_limits = np.quantile(information_values, quantile_points)
        overlap_limits = np.quantile(overlap_values, quantile_points)
        information_bins = np.searchsorted(
            information_limits, information_values, side="left"
        )
        overlap_bins = np.searchsorted(overlap_limits, overlap_values, side="left")

        subgroups = []
        for information_bin, overlap_bin in sorted(
            set(zip(information_bins, overlap_bins))
        ):
            mask = (information_bins == information_bin) & (
                overlap_bins == overlap_bin
            )
            subgroup_indexes = majority_indexes[mask]
            subgroups.append(
                _Subgroup(
                    sample_indexes=subgroup_indexes,
                    information_contribution=float(information_values[mask].mean()),
                    overlap_potential=float(overlap_values[mask].mean()),
                )
            )
        return subgroups

    def _run_genetic_algorithm(
        self,
        subgroups: list[_Subgroup],
        minority_size: int,
        rng: np.random.Generator,
    ) -> IntArray:
        upper_bounds = np.asarray([subgroup.size for subgroup in subgroups])
        population = rng.integers(
            low=np.zeros(len(subgroups), dtype=np.int64),
            high=upper_bounds + 1,
            size=(self.population_size, len(subgroups)),
        )
        best_chromosome = population[0].copy()
        best_fitness = -np.inf

        for _ in range(self.generations):
            fitness = self._calculate_fitness(population, subgroups, minority_size)
            generation_best = int(np.argmax(fitness))
            if fitness[generation_best] > best_fitness:
                best_fitness = float(fitness[generation_best])
                best_chromosome = population[generation_best].copy()

            population = self._roulette_selection(population, fitness, rng)
            population = self._crossover_population(population, rng)
            self._mutate_population(population, upper_bounds, rng)
            population[0] = best_chromosome

        final_fitness = self._calculate_fitness(population, subgroups, minority_size)
        final_best = int(np.argmax(final_fitness))
        if final_fitness[final_best] > best_fitness:
            best_chromosome = population[final_best].copy()
        return best_chromosome.astype(np.int64)

    def _calculate_fitness(
        self,
        population: IntArray,
        subgroups: list[_Subgroup],
        minority_size: int,
    ) -> FloatArray:
        information = np.asarray(
            [subgroup.information_contribution for subgroup in subgroups]
        )
        overlap = np.asarray(
            [subgroup.overlap_potential for subgroup in subgroups]
        )
        selected_information = population @ information
        selected_overlap = population @ overlap
        balance_penalty = np.abs(population.sum(axis=1) / minority_size - 1)
        return (
            -self.alpha * selected_overlap
            + self.beta * selected_information
            - self.gamma * balance_penalty
        )

    @staticmethod
    def _roulette_selection(
        population: IntArray,
        fitness: FloatArray,
        rng: np.random.Generator,
    ) -> IntArray:
        shifted_fitness = fitness - fitness.min() + np.finfo(np.float64).eps
        probabilities = shifted_fitness / shifted_fitness.sum()
        selected_indexes = rng.choice(
            len(population),
            size=len(population),
            replace=True,
            p=probabilities,
        )
        return population[selected_indexes].copy()

    def _crossover_population(
        self,
        population: IntArray,
        rng: np.random.Generator,
    ) -> IntArray:
        chromosome_length = population.shape[1]
        if chromosome_length < 2:
            return population
        for first_parent in range(0, len(population) - 1, 2):
            if rng.random() >= self.crossover_probability:
                continue
            second_parent = first_parent + 1
            points = np.sort(
                rng.choice(chromosome_length + 1, size=2, replace=False)
            )
            start, end = int(points[0]), int(points[1])
            exchanged = population[first_parent, start:end].copy()
            population[first_parent, start:end] = population[
                second_parent, start:end
            ]
            population[second_parent, start:end] = exchanged
        return population

    def _mutate_population(
        self,
        population: IntArray,
        upper_bounds: IntArray,
        rng: np.random.Generator,
    ) -> None:
        mutation_mask = rng.random(population.shape) < self.mutation_rate
        for chromosome_index, gene_index in np.argwhere(mutation_mask):
            population[chromosome_index, gene_index] = rng.integers(
                0, upper_bounds[gene_index] + 1
            )

    @staticmethod
    def _sample_subgroups(
        subgroups: list[_Subgroup],
        selected_counts: IntArray,
        rng: np.random.Generator,
    ) -> IntArray:
        selected_parts = []
        for subgroup, selected_count in zip(subgroups, selected_counts):
            if selected_count > 0:
                selected_parts.append(
                    rng.choice(
                        subgroup.sample_indexes,
                        size=int(selected_count),
                        replace=False,
                    )
                )
        if not selected_parts:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(selected_parts).astype(np.int64)

    @staticmethod
    def _min_max_normalize(values: FloatArray) -> FloatArray:
        minimum = float(values.min())
        value_range = float(values.max() - minimum)
        if value_range <= np.finfo(np.float64).eps:
            return np.ones_like(values, dtype=np.float64)
        return (values - minimum) / value_range

    @staticmethod
    def _restore_output_types(
        features: pd.DataFrame | ArrayLike,
        target: pd.Series | ArrayLike,
        selected_indexes: IntArray,
    ) -> tuple[pd.DataFrame | FloatArray, pd.Series | NDArray]:
        if isinstance(features, pd.DataFrame):
            resampled_features = features.iloc[selected_indexes].reset_index(drop=True)
        else:
            resampled_features = np.asarray(features)[selected_indexes]
        if isinstance(target, pd.Series):
            resampled_target = target.iloc[selected_indexes].reset_index(drop=True)
        else:
            resampled_target = np.asarray(target)[selected_indexes]
        return resampled_features, resampled_target


def teus_resample_dataframe(
    dataframe: pd.DataFrame,
    target_column: str,
    **sampler_parameters: object,
) -> pd.DataFrame:
    """Convenience function that applies TEUS directly to a dataframe."""
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError("dataframe must be a non-empty pandas DataFrame")
    if target_column not in dataframe.columns:
        raise ValueError(f"Target column '{target_column}' was not found")

    sampler = TEUSUnderSampler(**sampler_parameters)
    features, target = sampler.fit_resample(
        dataframe.drop(columns=[target_column]),
        dataframe[target_column],
    )
    result = features.copy()
    result[target_column] = np.asarray(target)
    return result
