from __future__ import annotations

# Dataclasses and enums for storing neuron explanations, their scores, and related data. Also,
# related helper functions.
import heapq
from collections import defaultdict
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from activations.activations import ActivationRecord
from activations.exemplars import ExemplarSplit, ExemplarType, NeuronExemplars, NeuronId
from explanations.scoring_utils import (
    ActivationSign,
    ScoredSequenceSimulation,
    SequenceSimulation,
    absolute_dev_explained_score_from_sequences,
    calibrate_and_score_simulation,
    correlation_score,
    rsquared_score_from_sequences,
)
from explanations.simulation_utils import NeuronSimulator
from pydantic import BaseModel
from sklearn import linear_model
from util.types import ChatMessage, NDFloatArray


class ExplanationGenerationMetadata(BaseModel):
    """Simulator parameters and the results of scoring it on multiple sequences"""

    ranks: List[int]
    """Ranks of exemplars used to create the prompt for explanation generation."""
    messages: Optional[List[ChatMessage] | List[List[ChatMessage]]] = None
    """
    messages that led to the full responses. If we sample multiple explanations, we only save
    the first set of messages.
    """
    num_refusals: int
    """Number of refusals from the API."""
    num_format_failures: int
    """Number of failures to parse the explanation from the response."""
    num_iterations: dict[int, int]
    """
    Dictionary with keys being the number of explanations to sample, and values being the
    number of iterations of sampling explanations in order to get the required number.
    """
    exem_indices_for_explanations: Optional[List[List[int]]] = None
    """
    Indices of the exemplars used to generate the explanations.
    """
    responses: Optional[dict[str, List[str]]] = None
    """
    Dictionary with keys "succ" and "fail" indicating whether the response resulted in a
    successful parsing of the explanation.
    """
    exemplars_idx: Optional[List[int]] = None
    """
    Indices of the exemplars used to generate the explanations.
    Only relevant when ExplanationConfig.fix_exemplars_for_exp is True, and is needed to make sure
    we use the same exemplars each time we sample more explanations.
    """
    examples_idx: Optional[List[int]] = None
    """
    Indices of the examples used to generate the explanations.
    Only relevant when ExplanationConfig.fix_examples_for_exp is True, and is needed to make sure
    we use the same examples each time we sample more explanations.
    """

    def update(self, new_metadata: ExplanationGenerationMetadata):
        assert self.ranks == new_metadata.ranks
        assert self.exemplars_idx == new_metadata.exemplars_idx
        assert self.examples_idx == new_metadata.examples_idx
        self.num_refusals += new_metadata.num_refusals
        self.num_format_failures += new_metadata.num_format_failures
        for num_exps, num_iters in new_metadata.num_iterations.items():
            if num_exps in self.num_iterations:
                self.num_iterations[num_exps] += num_iters
            else:
                self.num_iterations[num_exps] = num_iters

        if self.responses is None:
            self.responses = defaultdict(list)

        if new_metadata.responses is not None:
            for response_type, response_list in new_metadata.responses.items():
                self.responses[response_type].extend(response_list)

        if new_metadata.exem_indices_for_explanations is not None:
            if self.exem_indices_for_explanations is None:
                self.exem_indices_for_explanations = new_metadata.exem_indices_for_explanations
            else:
                self.exem_indices_for_explanations.extend(
                    new_metadata.exem_indices_for_explanations
                )


class ExplanationSimulations(BaseModel):
    """Result of scoring a single explanation on multiple sequences."""

    simulation_data: dict[int, ScoredSequenceSimulation]
    """ScoredSequenceSimulation for each sequence"""
    ev_correlation_score: Optional[float] = None
    """
    Correlation coefficient between the expected values of the normalized activations from the
    simulation and the unnormalized true activations on a dataset created from all score_results.
    (Note that this is not equivalent to averaging across sequences.)
    """
    rsquared_score: Optional[float] = None
    """R^2 of the simulated activations."""
    absolute_dev_explained_score: Optional[float] = None
    """
    Score based on absolute difference between real and simulated activations.
    absolute_dev_explained_score = 1 - mean(abs(real-predicted))/ mean(abs(real)).
    """

    def get_preferred_score(self) -> Optional[float]:
        """
        This method may return None in cases where the score is undefined, for example if the
        normalized activations were all zero, yielding a correlation coefficient of NaN.
        """
        return self.ev_correlation_score


class NeuronExplanation(BaseModel):
    """Simulator parameters and the results of scoring it on multiple sequences"""

    explanation: str
    """The explanation used for simulation."""

    simulations: Optional[dict[ExemplarSplit, ExplanationSimulations]] = None
    """Result of scoring the neuron simulator on multiple sequences."""

    def get_preferred_score(self, exemplar_splits: Sequence[ExemplarSplit]) -> Optional[float]:
        """
        This method may return None in cases where the score is undefined, for example if the
        normalized activations were all zero, yielding a correlation coefficient of NaN.
        """
        if self.simulations is None:
            return None
        true_activations: List[List[float]] = []
        flattened_sim_activations: List[float] = []
        seq_sims: List[SequenceSimulation] = []
        for split in exemplar_splits:
            if split not in self.simulations:
                return None
            for _, scored_seq_sim in self.simulations[split].simulation_data.items():
                true_activations.append(scored_seq_sim.true_activations)
                uncalibrated_seq_sim = scored_seq_sim.simulation.uncalibrated_simulation
                assert uncalibrated_seq_sim is not None
                flattened_sim_activations.extend(uncalibrated_seq_sim.expected_activations)
                seq_sims.append(uncalibrated_seq_sim)
        if not seq_sims:
            return None

        flattened_true_activations = np.concatenate(true_activations)
        # Fit a linear model that maps simulated activations to true activations.
        regression_model = linear_model.LinearRegression()
        regression_model.fit(  # type: ignore
            np.array(flattened_sim_activations).reshape(-1, 1), flattened_true_activations
        )

        scored_seq_sims: dict[int, ScoredSequenceSimulation] = {}
        for exemplar_idx, seq_sim in enumerate(seq_sims):
            scored_seq_sims[exemplar_idx] = calibrate_and_score_simulation(
                seq_sim, true_activations[exemplar_idx], regression_model
            )
        expl_sims = aggregate_scored_sequence_simulations(scored_seq_sims)
        return expl_sims.get_preferred_score()

    def parse_simulation_results(
        self,
        exemplar_split: ExemplarSplit,
        calibration_strategy: Optional[Literal["linreg", "norm"]] = "norm",
    ) -> pd.DataFrame | None:
        if self.simulations is None:
            return None
        simulations = self.simulations[exemplar_split]

        results: List[dict[str, Any]] = []
        for i, scored_simulation in simulations.simulation_data.items():
            score = scored_simulation.ev_correlation_score
            rank = i
            simulation = scored_simulation.simulation

            tokens = simulation.tokens
            true_activations = scored_simulation.true_activations

            if calibration_strategy is None:
                assert simulation.uncalibrated_simulation is not None
                simulated_activations = simulation.uncalibrated_simulation.expected_activations
            elif calibration_strategy == "linreg":
                simulated_activations = simulation.expected_activations
            elif calibration_strategy == "norm":
                assert simulation.uncalibrated_simulation is not None
                simulated_activations = np.array(
                    simulation.uncalibrated_simulation.expected_activations
                )
                simulated_activations = simulated_activations / 10 * max(true_activations)

            results.append(
                {
                    "score": score,
                    "rank": rank,
                    "tokens": tokens,
                    "simulated_activations": simulated_activations,
                    "true_activations": true_activations,
                }
            )
        return pd.DataFrame(results)

    def plot_simulation_results(
        self,
        exemplar_split: ExemplarSplit,
        calibration_strategy: Optional[Literal["linreg", "norm"]] = "norm",
    ) -> pd.DataFrame | None:
        results_df = self.parse_simulation_results(exemplar_split, calibration_strategy)
        if results_df is None:
            return None

        fig, axes = plt.subplots(len(results_df), 1, figsize=(15, 3 * len(results_df)))  # type: ignore
        if len(results_df) == 1:
            axes = [axes]

        assert "rank" in results_df.columns
        ranks: List[int] = list(results_df["rank"].values)

        for i, rank in enumerate(sorted(ranks)):
            row: pd.Series = results_df[results_df["rank"] == rank].squeeze()  # type: ignore

            sim_acts: NDFloatArray = np.array(row.simulated_activations, dtype=np.float32)  # type: ignore
            true_acts: NDFloatArray = np.array(row.true_activations, dtype=np.float32)  # type: ignore
            tokens: List[str] = list(row.tokens)  # type: ignore
            score: float = row.score  # type: ignore

            axes[i].plot(sim_acts, label="Predicted", color="tab:blue")
            axes[i].plot(true_acts, label="True", color="tab:orange")

            axes[i].set_xticks(range(len(tokens)))
            axes[i].set_xticklabels(tokens, rotation=90)
            axes[i].set_xlim(0, len(tokens))

            axes[i].set_title(f"Rank {rank} (score {score:.2f})")
            axes[i].set_ylabel("Expected Activation")
            axes[i].legend()
            axes[i].grid(True)

        overall_score = self.get_preferred_score([exemplar_split])
        fig.suptitle(self.explanation + f"(overall score {overall_score:.2f})", y=0.995)  # type: ignore
        fig.tight_layout()
        return results_df


class NeuronExplanations(BaseModel):
    """Simulation results and scores for a neuron."""

    neuron_id: NeuronId
    explanations: dict[ActivationSign, List[NeuronExplanation] | None]
    explanation_generation_metadata: Optional[
        dict[ActivationSign, ExplanationGenerationMetadata | None]
    ] = None

    def get_best_explanations(
        self, exemplar_splits: Sequence[ExemplarSplit]
    ) -> dict[ActivationSign, NeuronExplanation]:
        """
        For each activation sign, returns the ExplanationSimulations object corresponding to the
        explanation that achieves the best measured by get_preferred_score.
        """
        best_explanations: dict[ActivationSign, NeuronExplanation] = {}
        for act_sign, expls_list in self.explanations.items():
            if expls_list is None:
                continue
            scores_and_expls: List[Tuple[float, NeuronExplanation]] = []
            for expl in expls_list:
                score = expl.get_preferred_score(exemplar_splits)
                if score is None:
                    continue
                heapq.heappush(scores_and_expls, (-score, expl))
            if scores_and_expls:
                best_explanations[act_sign] = scores_and_expls[0][1]
        return best_explanations

    def get_all_explanations_and_scores(
        self, exemplar_splits: Sequence[ExemplarSplit]
    ) -> dict[ActivationSign, List[Tuple[str, float | None]]]:
        explanation_and_scores: dict[ActivationSign, List[Tuple[str, float | None]]] = defaultdict(
            list
        )
        for act_sign, ne_list in self.explanations.items():
            if ne_list is None:
                continue
            for ne in ne_list:
                exp_str = ne.explanation
                overall_score: float | None = ne.get_preferred_score(exemplar_splits)
                # if overall_score is None:
                #     continue
                explanation_and_scores[act_sign].append((exp_str, overall_score))
        return explanation_and_scores


class SplitExemplars(BaseModel):
    split: ExemplarSplit
    neuron_exemplars: NeuronExemplars
    exem_idxs: List[int]

    class Config:
        arbitrary_types_allowed = True

    def get_activation_records(
        self, normalize: bool = False, mask_opposite_sign: bool = False, add_ranks: bool = False
    ) -> Dict[ExemplarType, List[ActivationRecord] | Dict[int, ActivationRecord]]:
        if normalize:
            act_recs = self.neuron_exemplars.get_normalized_act_records(
                self.split, mask_opposite_sign
            )
        else:
            act_recs = self.neuron_exemplars.activation_records[self.split]

        if add_ranks:
            return {
                extype: {idx: act_recs[extype][idx] for idx in self.exem_idxs}
                for extype in ExemplarType
            }
        else:
            return {
                extype: [act_recs[extype][idx] for idx in self.exem_idxs] for extype in ExemplarType
            }

    def get_activation_percentiles(self) -> Dict[float, float]:
        return self.neuron_exemplars.activation_percentiles

    def get_ranks(self) -> List[int]:
        return self.exem_idxs


def aggregate_scored_sequence_simulations(
    scored_sequence_simulations: Dict[int, ScoredSequenceSimulation],
) -> ExplanationSimulations:
    """
    Aggregate a list of scored sequence simulations. The logic for doing this is non-trivial for EV
    scores, since we want to calculate the correlation over all activations from all sequences at
    once rather than simply averaging per-sequence correlations.
    """
    all_true_activations: list[float] = []
    all_expected_values: list[float] = []
    for _, scored_sequence_simulation in scored_sequence_simulations.items():
        all_true_activations.extend(scored_sequence_simulation.true_activations or [])
        all_expected_values.extend(scored_sequence_simulation.simulation.expected_activations)
    ev_correlation_score = (
        correlation_score(all_true_activations, all_expected_values)
        if len(all_true_activations) > 0
        else None
    )
    rsquared_score = rsquared_score_from_sequences(all_true_activations, all_expected_values)
    absolute_dev_explained_score = absolute_dev_explained_score_from_sequences(
        all_true_activations, all_expected_values
    )

    return ExplanationSimulations(
        simulation_data=scored_sequence_simulations,
        ev_correlation_score=ev_correlation_score,
        rsquared_score=rsquared_score,
        absolute_dev_explained_score=absolute_dev_explained_score,
    )


def filter_activations(activations: List[float], min_or_max: ExemplarType) -> List[float]:
    if min_or_max == ExemplarType.MAX:
        return np.maximum(activations, 0).tolist()
    else:
        return np.abs(np.minimum(activations, 0)).tolist()


def simulate_and_score(
    split_exemplars: SplitExemplars,
    explanations: List[NeuronExplanation],
    exemplar_type: ExemplarType,
    simulator: NeuronSimulator,
    overwrite: bool = False,
) -> List[NeuronExplanation]:
    exemplars = cast(
        Dict[int, ActivationRecord],
        split_exemplars.get_activation_records(add_ranks=True)[exemplar_type],
    )

    # Go through current data and find which explanations need evaluating on which exemplars.
    explanations_to_eval_per_exemplar: dict[int, List[int]] = defaultdict(
        list
    )  # {exemplar_rank: [expl_idxs]}
    exemplars_to_eval_per_explanation: dict[int, List[int]] = defaultdict(
        list
    )  # {expl_idx: [exemplar_ranks]}
    # Initialize uncalibrated sequence simulations with previous results so that we can
    # use the full set of results for calibration.
    uncalib_seq_sims: dict[int, dict[int, SequenceSimulation]] = defaultdict(
        dict
    )  # {expl_idx: {exemplar_rank: SequenceSimulation}}
    for expl_idx, explanation in enumerate(explanations):
        if explanation.simulations is not None:
            expl_sim_results = explanation.simulations.get(split_exemplars.split, None)
        else:
            expl_sim_results = None

        exemplar_ranks_to_eval = set(exemplars.keys())
        if not overwrite and expl_sim_results is not None:
            already_evaluated_exemplar_ranks = set(expl_sim_results.simulation_data.keys())
            exemplar_ranks_to_eval = exemplar_ranks_to_eval - already_evaluated_exemplar_ranks
            for exemplar_rank, scored_seq_sim in expl_sim_results.simulation_data.items():
                uncalib_sim = scored_seq_sim.simulation.uncalibrated_simulation
                assert uncalib_sim is not None
                uncalib_seq_sims[expl_idx][exemplar_rank] = uncalib_sim

        for exemplar_rank in exemplar_ranks_to_eval:
            explanations_to_eval_per_exemplar[exemplar_rank].append(expl_idx)
            exemplars_to_eval_per_explanation[expl_idx].append(exemplar_rank)
    # TODO(damichoi): Think more carefully about how to deal with the case where there are no
    # exemplars to evaluate on.
    # If there are no more things to evaluate, return.
    if not exemplars_to_eval_per_explanation:
        return explanations

    # Run simulations for explanations and exemplars that we haven't evaluated before.
    for exemplar_rank, expl_idxs in explanations_to_eval_per_exemplar.items():
        explanation_strs = [explanations[idx].explanation for idx in expl_idxs]
        act_rec = exemplars[exemplar_rank]
        sim_results_list = simulator.simulate(
            explanations=explanation_strs, tokens=act_rec.tokens, token_ids=act_rec.token_ids
        )

        for expl_idx, sim_results in zip(expl_idxs, sim_results_list):
            uncalib_seq_sims[expl_idx][exemplar_rank] = sim_results

    # Get true and simulated activations for calibration.
    exemplar_ranks = sorted(explanations_to_eval_per_exemplar.keys())
    true_activations = {
        exemplar_rank: filter_activations(exemplars[exemplar_rank].activations, exemplar_type)
        for exemplar_rank in exemplar_ranks
    }
    flattened_true_activations = np.concatenate(
        [true_activations[exemplar_rank] for exemplar_rank in exemplar_ranks]
    )
    simulated_activations: Dict[int, Dict[int, List[float]]] = defaultdict(
        dict
    )  # {expl_idx: {exemplar_rank: simulated activations}}
    for expl_idx in uncalib_seq_sims:
        for exemplar_rank, uncalib_seq_sim in uncalib_seq_sims[expl_idx].items():
            simulated_activations[expl_idx][exemplar_rank] = uncalib_seq_sim.expected_activations

    results: List[NeuronExplanation] = []
    for expl_idx, explanation in enumerate(explanations):
        if expl_idx not in exemplars_to_eval_per_explanation:
            assert not overwrite and explanation.simulations is not None
            results.append(explanation)
            continue

        flattened_simulated_activations = np.concatenate(
            [simulated_activations[expl_idx][exemplar_rank] for exemplar_rank in exemplar_ranks]
        )
        # Fit a linear model that maps simulated activations to true activations.
        regression_model = linear_model.LinearRegression()
        regression_model.fit(  # type: ignore
            flattened_simulated_activations.reshape(-1, 1), flattened_true_activations
        )

        # Maybe initialize from old simulation_data.
        full_sim_results = (
            explanation.simulations.copy() if explanation.simulations is not None else {}
        )
        scored_sequence_simulations: Dict[int, ScoredSequenceSimulation] = (
            {}
        )  # {exemplar_rank: ScoredSequenceSimulation}
        for exemplar_rank, sim_results in uncalib_seq_sims[expl_idx].items():
            scored_seq_sim = calibrate_and_score_simulation(
                sim_results, true_activations[exemplar_rank], regression_model
            )
            scored_sequence_simulations[exemplar_rank] = scored_seq_sim

        expl_sims = aggregate_scored_sequence_simulations(scored_sequence_simulations)
        full_sim_results[split_exemplars.split] = expl_sims
        results.append(
            NeuronExplanation(
                explanation=explanation.explanation,
                simulations=full_sim_results,
            )
        )
    return results
