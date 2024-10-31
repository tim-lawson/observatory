from __future__ import annotations

from enum import Enum
from typing import Callable, List, Optional

import numpy as np
from pydantic import BaseModel
from sklearn import linear_model
from sklearn.metrics import d2_absolute_error_score, r2_score  # type: ignore
from util.types import NDFloatArray

MAX_NORMALIZED_ACTIVATION = 10
VALID_ACTIVATION_TOKENS_ORDERED = list(str(i) for i in range(MAX_NORMALIZED_ACTIVATION + 1))


class ActivationSign(str, Enum):
    POS = "positive"
    NEG = "negative"


class ActivationScale(str, Enum):
    """Which "units" are stored in the expected_activations/distribution_values fields of a
    SequenceSimulation.

    This enum identifies whether the values represent real activations of the neuron or something
    else. Different scales are not necessarily related by a linear transformation.
    """

    NEURON_ACTIVATIONS = "neuron_activations"
    """Values represent real activations of the neuron."""
    SIMULATED_NORMALIZED_ACTIVATIONS = "simulated_normalized_activations"
    """
    Values represent simulated activations of the neuron, normalized to the range [0, 10]. This
    scale is arbitrary and should not be interpreted as a neuron activation.
    """


class SequenceSimulation(BaseModel):
    """The result of a simulation of neuron activations on one text sequence."""

    tokens: list[str]
    """The sequence of tokens that was simulated."""
    expected_activations: list[float]
    """Expected value of the possibly-normalized activation for each token in the sequence."""
    activation_scale: ActivationScale
    """What scale is used for values in the expected_activations field."""
    distribution_probabilities: list[list[float]] | None = None
    """
    For each token in the sequence, the probability of the corresponding value in
    distribution_values.
    """
    distribution_values: Optional[list[list[float]]] = None
    """
    For each token in the sequence, a list of values from the discrete distribution of activations
    produced from simulation, transformed to another unit by calibration.

    When we simulate a neuron, we produce a discrete distribution with values in the arbitrary
    discretized space of the neuron, e.g. 10% chance of 0, 70% chance of 1, 20% chance of 2.
    Which we store as distribution_values = [0, 1, 2], distribution_probabilities = [0.1, 0.7, 0.2].
    When we transform the distribution to the real activation units, we can correspondingly
    transform the values of this distribution to get a distribution in the units of the neuron.
    e.g. if the mapping from the discretized space to the real activation unit of the neuron is
    f(x) = x/2, then the distribution becomes 10% chance of 0, 70% chance of 0.5, 20% chance of 1.
    Which we store as distribution_values = [0, 0.5, 1],
    distribution_probabilities = [0.1, 0.7, 0.2].
    """
    uncalibrated_simulation: Optional[SequenceSimulation] = None
    """The result of the simulation before calibration."""


class ScoredSequenceSimulation(BaseModel):
    """
    SequenceSimulation result with a score (for that sequence only) and ground truth activations.
    """

    simulation: SequenceSimulation
    """The result of a simulation of neuron activations."""
    true_activations: List[float]
    """Ground truth activations on the sequence (not normalized)"""
    ev_correlation_score: float | None
    """
    Correlation coefficient between the expected values of the normalized activations from the
    simulation and the unnormalized true activations of the neuron on the text sequence.
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real))
    """

    def __post_init__(self):
        if self.ev_correlation_score is None:
            self.ev_correlation_score = np.nan


def correlation_score(
    real_activations: List[float] | NDFloatArray,
    predicted_activations: List[float] | NDFloatArray,
) -> float:
    return np.corrcoef(real_activations, predicted_activations)[0, 1]


def rsquared_score_from_sequences(
    real_activations: List[float] | NDFloatArray,
    predicted_activations: List[float] | NDFloatArray,
) -> float:
    return r2_score(real_activations, predicted_activations)  # type: ignore


def absolute_dev_explained_score_from_sequences(
    real_activations: List[float] | NDFloatArray,
    predicted_activations: List[float] | NDFloatArray,
) -> float:
    return d2_absolute_error_score(real_activations, predicted_activations)  # type: ignore


def score_from_simulation(
    real_activations: List[float],
    simulation: SequenceSimulation,
    score_function: Callable[[List[float] | NDFloatArray, List[float] | NDFloatArray], float],
) -> float:
    return score_function(real_activations, simulation.expected_activations)


def apply_calibration(
    values: List[float], regression_model: linear_model.LinearRegression
) -> List[float]:
    return regression_model.predict(np.reshape(np.array(values), (-1, 1))).tolist()  # type: ignore


def calibrate_simulation(
    uncalibrated_simulation: SequenceSimulation, regression_model: linear_model.LinearRegression
) -> SequenceSimulation:
    calibrated_activations = apply_calibration(
        uncalibrated_simulation.expected_activations, regression_model
    )
    calibrated_distribution_values = [
        apply_calibration(dv, regression_model) for dv in np.arange(MAX_NORMALIZED_ACTIVATION + 1)
    ]
    calibrated_simulation = SequenceSimulation(
        tokens=uncalibrated_simulation.tokens,
        expected_activations=calibrated_activations,
        activation_scale=ActivationScale.NEURON_ACTIVATIONS,
        distribution_probabilities=uncalibrated_simulation.distribution_probabilities,
        distribution_values=calibrated_distribution_values,
        uncalibrated_simulation=uncalibrated_simulation,
    )
    return calibrated_simulation


def calibrate_and_score_simulation(
    simulation: SequenceSimulation,
    activations: List[float],
    regression_model: linear_model.LinearRegression,
) -> ScoredSequenceSimulation:
    simulation = calibrate_simulation(simulation, regression_model)

    ev_correlation_score = score_from_simulation(activations, simulation, correlation_score)
    rsquared_score = score_from_simulation(activations, simulation, rsquared_score_from_sequences)
    absolute_dev_explained_score = score_from_simulation(
        activations, simulation, absolute_dev_explained_score_from_sequences
    )
    scored_sequence_simulation = ScoredSequenceSimulation(
        simulation=simulation,
        true_activations=activations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )
    return scored_sequence_simulation
