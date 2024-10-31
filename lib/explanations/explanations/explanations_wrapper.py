from __future__ import annotations

import json
import os
import random
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from activations.exemplars import ExemplarSplit, ExemplarType, NeuronId
from activations.exemplars_wrapper import ExemplarConfig, ExemplarsWrapper
from explanations.explainer import get_explainer
from explanations.explanation_utils import (
    ExemplarsForExplanationGeneration,
    generate_explanations_for_one_act_sign,
    get_prompt_formatter,
)
from explanations.explanations import (
    ActivationSign,
    ExplanationGenerationMetadata,
    NeuronExplanation,
    NeuronExplanations,
    SplitExemplars,
    simulate_and_score,
)
from explanations.simulation_utils import NeuronSimulator
from pydantic import BaseModel
from tqdm import tqdm
from util.subject import Subject, get_subject_config
from util.types import NDIntArray


class ExplanationConfig(BaseModel):
    exemplar_config: ExemplarConfig

    exem_slice_for_exp: tuple[int, int, int] = (0, 20, 1)
    permute_exemplars_for_exp: bool = True
    num_exem_for_exp: Optional[int] = None  # Legacy
    num_exem_range_for_exp: Optional[tuple[int, int]] = (10, 20)
    fix_exemplars_for_exp: bool = True
    permute_examples_for_exp: bool = True
    num_examples_for_exp: Optional[int] = 1
    fix_examples_for_exp: bool = True
    explainer_model_name: str = "gpt-4o"
    add_special_tokens_for_explainer: bool = True
    explainer_system_prompt_type: str = "no_cot"
    use_puzzle_for_bills: bool = False
    examples_placement: str = "fewshot"
    min_tokens_to_highlight: int = 3
    round_to_int: bool = True
    num_explanation_samples: int = 1
    max_new_tokens_for_explanation_generation: int = 2000
    temperature_for_explanation_generation: float = 1.0
    save_full_explainer_responses: bool = False

    exem_slice_to_score: tuple[int, int, int] = (0, 20, 1)
    num_random_seqs_to_score: int = 5
    simulator_model_name: str = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    simulator_system_prompt_type: str = "unk_base"
    add_special_tokens: bool = False

    seed: int = 54

    def __str__(self):
        return json.dumps(self.model_dump(), indent=4)


class ExplanationsWrapper:
    def __init__(
        self,
        save_path: str,
        config: ExplanationConfig,
        exemplars_data_dir: str,
        subject: Subject,
        overwrite: bool = False,
    ):
        exemplars_wrapper = ExemplarsWrapper(
            data_dir=exemplars_data_dir, config=config.exemplar_config, subject=subject
        )

        os.makedirs(save_path, exist_ok=True)
        for layer in range(subject.L):
            os.makedirs(os.path.join(save_path, "explanations", str(layer)), exist_ok=True)

        # Check whether data already exists in save_path.
        config_file = os.path.join(save_path, "explanation_config.json")
        if not overwrite and os.path.exists(config_file):
            # Check that the configs are the same (excluding some fields that can be different).
            with open(config_file, "r") as f:
                existing_config = ExplanationConfig.model_validate_json(f.read())
            fields_to_exclude = set(["save_full_explainer_responses"])
            existing_config_dict = existing_config.model_dump(exclude=fields_to_exclude)
            config_dict = config.model_dump(exclude=fields_to_exclude)
            for field in existing_config_dict:
                existing_val = existing_config_dict[field]
                curr_val = config_dict[field]
                assert existing_val == curr_val, (
                    f"Value of '{field}' for existing config is '{existing_val}', "
                    f"while the value given in the config is '{curr_val}')"
                )
        else:
            # If there's no data saved yet, save our config.
            with open(config_file, "w") as f:
                f.write(config.model_dump_json())

        # Parse train/valid/test split idxs.
        exem_indices_for_exp = list(range(*config.exem_slice_for_exp))
        exem_indices_to_score = list(range(*config.exem_slice_to_score))

        # Support for legacy config.
        # TODO(damichoi): Remove this at some point.
        if config.num_exem_for_exp is not None:
            config.num_exem_range_for_exp = (config.num_exem_for_exp, config.num_exem_for_exp)

        # Define prompt formatter.
        formatter_args = {
            "examples_placement": config.examples_placement,
            "permute_examples": config.permute_examples_for_exp,
            "num_examples": config.num_examples_for_exp,
            "fix_examples": config.fix_examples_for_exp,
            "min_highlights": config.min_tokens_to_highlight,
            "round_to_int": config.round_to_int,
            "use_puzzle_as_examples": config.use_puzzle_for_bills,
        }
        self.prompt_formatter = get_prompt_formatter(
            config.explainer_system_prompt_type, **formatter_args
        )

        # Don't initialize explainer until we need it, since it might use GPU memory.
        self.explainer = None

        # Keep track of randomness using generator.
        self.rng = random.Random(config.seed)

        self.base_save_path = save_path
        self.config = config
        self.exemplars_wrapper = exemplars_wrapper
        self.exem_indices_for_exp = exem_indices_for_exp
        self.exem_indices_to_score = exem_indices_to_score

    @classmethod
    def from_disk(cls, save_path: str, exemplars_data_dir: str, subject: Optional[Subject] = None):
        config_path = os.path.join(save_path, "explanation_config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"No explanation config found at {config_path}")
        with open(config_path, "r") as f:
            explanation_config = ExplanationConfig.model_validate_json(f.read())
        if subject is None:
            subject = Subject(get_subject_config(explanation_config.exemplar_config.hf_model_id))
        return cls(save_path, explanation_config, exemplars_data_dir, subject)

    def initialize_explainer(self):
        self.explainer = get_explainer(
            model_name=self.config.explainer_model_name,
            max_new_tokens=self.config.max_new_tokens_for_explanation_generation,
            temperature=self.config.temperature_for_explanation_generation,
            add_special_tokens=self.config.add_special_tokens_for_explainer,
        )

    def get_split_neuron_exemplars(
        self,
        to_score: bool,
        split: ExemplarSplit,
        layer: int,
        neuron_idx: int,
    ) -> SplitExemplars:
        neuron_exemplars = self.exemplars_wrapper.get_neuron_exemplars(layer, neuron_idx)
        if split.startswith("random"):
            exem_indices = list(range(self.config.num_random_seqs_to_score))
        else:
            exem_indices = self.exem_indices_for_exp if not to_score else self.exem_indices_to_score
        return SplitExemplars(
            split=split, neuron_exemplars=neuron_exemplars, exem_idxs=exem_indices
        )

    def get_neuron_exemplars_for_explanation_generation(
        self, layer: int, neuron_idx: int, act_sign: ActivationSign
    ) -> ExemplarsForExplanationGeneration:
        neuron_exemplars = self.exemplars_wrapper.get_neuron_exemplars(layer, neuron_idx)
        if (act_sign == ActivationSign.POS and neuron_exemplars.extrema[ExemplarType.MAX] <= 0) or (
            act_sign == ActivationSign.NEG and neuron_exemplars.extrema[ExemplarType.MIN] >= 0
        ):
            raise ValueError(
                f"No exemplars with {act_sign.value} activations for neuron ({layer}, {neuron_idx})"
            )
        exemplars = ExemplarsForExplanationGeneration(
            neuron_exemplars=neuron_exemplars,
            act_sign=act_sign,
            exem_idxs=self.exem_indices_for_exp,
            exemplar_split=ExemplarSplit.TRAIN,
            permute_exemplars=self.config.permute_exemplars_for_exp,
            num_exemplars_range=self.config.num_exem_range_for_exp,
            fix_exemplars=self.config.fix_exemplars_for_exp,
        )
        return exemplars

    def is_neuron_done_explaining(self, layer: int, neuron_idx: int) -> NeuronExplanations | None:
        """
        Checks whether a neuron has all explanations generated.
        """
        neuron_explanations = self.get_neuron_scored_explanations(layer, neuron_idx)
        if neuron_explanations is None:
            return None
        for exps_list in neuron_explanations.explanations.values():
            if exps_list is None:
                continue
            if len(exps_list) < self.config.num_explanation_samples:
                return None
        return neuron_explanations

    def generate_explanations_for_neuron(
        self, layer: int, neuron_idx: int, overwrite: bool = False
    ):
        # Load existing explanations data. If it doesn't exist, neuron_explanations will be None.
        neuron_explanations = self.get_neuron_scored_explanations(layer, neuron_idx)
        prompt_formatter = self.prompt_formatter

        explanation_strs_dict: Dict[ActivationSign, List[str]] = {}
        metadata_dict: Dict[ActivationSign, ExplanationGenerationMetadata] = {}
        for act_sign in ActivationSign:
            metadata = None
            num_exps_left = self.config.num_explanation_samples
            # Compute how many more explanations we need to generate.
            if neuron_explanations is not None and not overwrite:
                if neuron_explanations.explanation_generation_metadata is not None:
                    metadata = neuron_explanations.explanation_generation_metadata[act_sign]
                exps = neuron_explanations.explanations[act_sign]
                num_exps_left = self.config.num_explanation_samples - (
                    len(exps) if exps is not None else 0
                )
                if num_exps_left <= 0:
                    continue

            try:
                exemplars = self.get_neuron_exemplars_for_explanation_generation(
                    layer, neuron_idx, act_sign
                )
            except:
                print(
                    f"Skipping neuron ({layer}, {neuron_idx}, {act_sign.value}) "
                    "due to activations not existing for the sign."
                )
                continue

            # Optionally load exemplars_idx if we're fixing the exemplars.
            if (
                self.config.fix_exemplars_for_exp
                and neuron_explanations is not None
                and metadata is not None
            ):
                exemplars.exemplars_idx = metadata.exemplars_idx
            # Need to reset examples_idx every neuron and activation sign.
            prompt_formatter.examples_idx = None
            # Optionally load examples_idx if we're fixing the examples.
            if (
                self.config.fix_examples_for_exp
                and neuron_explanations is not None
                and metadata is not None
            ):
                prompt_formatter.examples_idx = metadata.examples_idx

            assert self.explainer is not None, "Initialize explainer with initialize_explainer"
            explanation_strs, metadata = generate_explanations_for_one_act_sign(
                exemplars=exemplars,
                prompt_formatter=prompt_formatter,
                num_expl_samples=num_exps_left,
                explainer=self.explainer,
                rng=self.rng,
            )
            explanation_strs_dict[act_sign] = explanation_strs
            metadata_dict[act_sign] = metadata

        if explanation_strs_dict:
            # Saving will combine new explanations with existing ones.
            self.save_neuron_explanations(
                layer, neuron_idx, explanation_strs_dict, metadata_dict, overwrite
            )

    def save_neuron_explanations(
        self,
        layer: int,
        neuron_idx: int,
        explanation_strs_dict: dict[ActivationSign, List[str]],
        metadata_dict: dict[ActivationSign, ExplanationGenerationMetadata],
        overwrite: bool = False,
    ):
        """Saves explanation generation results."""
        save_path = os.path.join(
            self.base_save_path, "explanations", str(layer), f"{neuron_idx}.json"
        )

        # Existing results.
        neuron_explanations = self.get_neuron_scored_explanations(layer, neuron_idx)

        full_explanations_strs_dict: dict[ActivationSign, List[NeuronExplanation] | None] = {}
        full_metadata_dict: dict[ActivationSign, ExplanationGenerationMetadata | None] = {}
        for act_sign in ActivationSign:
            metadata = None
            if (
                not overwrite
                and neuron_explanations is not None
                and act_sign in neuron_explanations.explanations
            ):
                exps = neuron_explanations.explanations[act_sign] or []
                if neuron_explanations.explanation_generation_metadata is not None:
                    metadata = neuron_explanations.explanation_generation_metadata[act_sign]
            else:
                exps: List[NeuronExplanation] = []

            if act_sign in explanation_strs_dict:
                # Add new explanations.
                for exp_str in explanation_strs_dict[act_sign]:
                    exps.append(NeuronExplanation(explanation=exp_str))
                # Update metadata.
                if metadata is not None and act_sign in metadata_dict:
                    metadata.update(metadata_dict[act_sign])
                else:
                    metadata = metadata_dict[act_sign]

            # This happens when there were no positive activations when act_sign == "positive"
            # or no negative activations when act_sign == "negative".
            if not exps:
                full_explanations_strs_dict[act_sign] = None
                full_metadata_dict[act_sign] = None
                continue

            # Check that the number of explanations is consistent with the config.
            assert len(exps) >= self.config.num_explanation_samples, (
                f"Number of explanations to save is less than what we need!: "
                f"{len(exps)} < {self.config.num_explanation_samples}"
            )

            if not self.config.save_full_explainer_responses and metadata is not None:
                metadata.responses = None

            full_explanations_strs_dict[act_sign] = exps
            full_metadata_dict[act_sign] = metadata

        neuron_explanations = NeuronExplanations(
            neuron_id=NeuronId(layer_index=layer, neuron_index=neuron_idx),
            explanations=full_explanations_strs_dict,
            explanation_generation_metadata=full_metadata_dict,
        )

        with open(save_path, "w") as f:
            f.write(neuron_explanations.model_dump_json())

    def is_neuron_explanations_done_scoring(
        self,
        neuron_explanations: NeuronExplanations,
        exem_splits: Sequence[ExemplarSplit],
        idxs_to_check: Optional[dict[ActivationSign, Sequence[int]]] = None,
    ):
        for act_sign, exps_list in neuron_explanations.explanations.items():
            if exps_list is None:
                continue
            if len(exps_list) < self.config.num_explanation_samples:
                return False
            for exp_idx, exp_sims in enumerate(exps_list):
                if idxs_to_check is not None and exp_idx not in idxs_to_check[act_sign]:
                    continue
                for exem_split in exem_splits:
                    # Check that every explanation has been scored.
                    if exp_sims.simulations is None or exem_split not in exp_sims.simulations:
                        return False
                    if exem_split.startswith("random"):
                        if set(range(self.config.num_random_seqs_to_score)) != set(
                            exp_sims.simulations[exem_split].simulation_data.keys()
                        ):
                            return False
                    else:
                        if set(self.exem_indices_to_score) != set(
                            exp_sims.simulations[exem_split].simulation_data.keys()
                        ):
                            return False
        return True

    def save_neuron_scored_explanations(
        self,
        layer: int,
        neuron_idx: int,
        neuron_explanations: NeuronExplanations,
        exem_splits: Sequence[ExemplarSplit],
        idxs_to_check: Optional[dict[ActivationSign, Sequence[int]]] = None,
    ):
        """Saves explanation scoring results."""
        save_path = os.path.join(
            self.base_save_path, "explanations", str(layer), f"{neuron_idx}.json"
        )

        # Check whether we have all required simulation results.
        assert self.is_neuron_explanations_done_scoring(
            neuron_explanations, exem_splits, idxs_to_check
        )

        with open(save_path, "w") as f:
            f.write(neuron_explanations.model_dump_json())

    def is_neuron_done_scoring(
        self,
        layer: int,
        neuron_idx: int,
        exem_splits: Sequence[ExemplarSplit],
    ) -> bool:
        """
        Checks whether a neuron has all explanations generated and scored according to the config.
        """
        neuron_explanations = self.get_neuron_scored_explanations(layer, neuron_idx)
        if neuron_explanations is None:
            return False
        return self.is_neuron_explanations_done_scoring(neuron_explanations, exem_splits)

    def get_neuron_scored_explanations(
        self, layer: int, neuron_idx: int
    ) -> NeuronExplanations | None:
        """Loads NeuronExplanations for given neuron if it exists."""
        save_path = os.path.join(
            self.base_save_path, "explanations", str(layer), f"{neuron_idx}.json"
        )
        if not os.path.exists(save_path):
            # print(f"No explanations found for neuron {neuron_idx} at layer {layer}!")
            return None
        try:
            with open(save_path, "r") as f:
                neuron_explanations = NeuronExplanations.model_validate_json(f.read())
        except:
            raise ValueError(f"Couldn't read file from {save_path}!")

        return neuron_explanations

    def get_explanations_for_neuron(
        self,
        layer: int,
        neuron_idx: int,
        exem_splits: Sequence[ExemplarSplit],
    ) -> dict[ActivationSign, List[Tuple[str, float | None]]] | None:
        """
        Returns just the explanations and their overall scores in a dictionary:
        {ActivationSign: [(explanation1, score1), (explanation2, score2), ...]}
        If scores don't exist, they are set to None.
        """
        neuron_explanations = self.get_neuron_scored_explanations(layer, neuron_idx)
        if neuron_explanations is None:
            return None

        return neuron_explanations.get_all_explanations_and_scores(exem_splits)

    def score_arbitrary_explanation(
        self,
        explanation: str,
        layer: int,
        neuron_idx: int,
        act_sign: ActivationSign,
        simulator: NeuronSimulator,
        exem_splits: Sequence[ExemplarSplit] = (ExemplarSplit.VALID, ExemplarSplit.RANDOM_VALID),
    ) -> NeuronExplanation:
        neuron_exemplars = self.exemplars_wrapper.get_neuron_exemplars(layer, neuron_idx)
        explanations = [NeuronExplanation(explanation=explanation)]
        for split in exem_splits:
            if split.startswith("random"):
                exem_indices = list(range(self.config.num_random_seqs_to_score))
            else:
                exem_indices = self.exem_indices_to_score
            split_exemplars = SplitExemplars(
                split=split, neuron_exemplars=neuron_exemplars, exem_idxs=exem_indices
            )

            extype = ExemplarType.MAX if act_sign == ActivationSign.POS else ExemplarType.MIN
            scored_explanations = simulate_and_score(
                split_exemplars=split_exemplars,
                explanations=explanations,
                exemplar_type=extype,
                simulator=simulator,
                overwrite=True,
            )
            explanations = scored_explanations
        return explanations[0]


def get_neuron_explanations_file_paths_from_exp_path(
    exp_path: str, neurons: Optional[NDIntArray] = None
) -> List[str]:
    """
    Returns a list of paths corresponding to the NeuronExplanations data saved in the
    experiment directory.
    If neurons is not None, only the file paths for the specified neurons are returned.
    """
    exps_save_path = os.path.join(exp_path, "explanations")
    file_paths: List[str] = []
    if neurons is not None:
        for layer, neuron_idx in neurons:
            file_path = os.path.join(exps_save_path, str(layer), f"{neuron_idx}.json")
            if os.path.exists(file_path):
                file_paths.append(file_path)
    else:
        for layer in os.listdir(exps_save_path):
            layer_path = os.path.join(exps_save_path, layer)
            for fname in os.listdir(layer_path):
                neuron_idx, ext = os.path.splitext(fname)
                if ext == ".json":
                    file_paths.append(os.path.join(layer_path, fname))
    return file_paths


def process_results_for_neuron(
    file_path: str,
    exemplar_splits: Sequence[ExemplarSplit],
    max_num_scores: Optional[int] = None,
) -> List[dict[str, Any]] | None:
    try:
        with open(file_path, "rb") as f:
            neuron_explanations = NeuronExplanations.model_validate_json(f.read())
    except:
        print(f"Error reading {file_path}")
        return None
    neuron: NeuronId = neuron_explanations.neuron_id
    layer, neuron_idx = neuron.layer_index, neuron.neuron_index

    metadata = neuron_explanations.explanation_generation_metadata

    results: List[dict[str, Any]] = []
    for act_sign in ActivationSign:
        expls_for_sign = neuron_explanations.explanations[act_sign]
        if expls_for_sign is None:
            continue

        explanations: List[str] = []
        scores: List[float] = []
        for scored_explanation in expls_for_sign:
            score = scored_explanation.get_preferred_score(exemplar_splits)
            if score is not None:
                scores.append(score)
                explanations.append(scored_explanation.explanation)
        if not scores:
            break

        best_idx = np.argmax(scores)
        best_explanation = explanations[best_idx]
        best_score = scores[best_idx]

        exem_idxs_list: List[List[int]] | None = None
        if metadata is not None:
            metadata_for_sign = metadata[act_sign]
            if metadata_for_sign is not None:
                exem_idxs_list = metadata_for_sign.exem_indices_for_explanations

        result = {
            "neuron_idx": neuron_idx,
            "layer": layer,
            "act_sign": act_sign.value,
            "id": (layer, neuron_idx, act_sign.value),
            "explanations": explanations,
            "best_explanation": best_explanation,
            "best_score": best_score,
            "scores": scores,
            "exemplar_idxs_for_best_explanation": (
                exem_idxs_list[best_idx] if exem_idxs_list else None
            ),
        }
        if max_num_scores:
            result["trunc_scores"] = scores[:max_num_scores]
        results.append(result)
    return results


def get_all_scored_explanations_from_experiment_path(
    exp_path: str,
    exemplar_splits: Sequence[ExemplarSplit],
    reload: bool = True,
    max_num_neurons: Optional[int] = None,
    max_num_scores: Optional[int] = None,
):
    suffix = "_".join(sorted(exemplar_splits))
    save_path = os.path.join(exp_path, f"scored_explanations_{suffix}.pkl")
    if not reload and os.path.exists(save_path):
        results_df = pd.read_pickle(save_path)
        # Convert layer and neuron_idx columns to set of tuples.
        done_neurons = set(tuple(x) for x in results_df[["layer", "neuron_idx"]].values)
    else:
        results_df = None
        done_neurons: set[tuple[int, int]] = set()

    exps_save_path = os.path.join(exp_path, "explanations")
    file_paths: List[str] = []
    for layer in os.listdir(exps_save_path):
        layer_path = os.path.join(exps_save_path, layer)
        for fname in os.listdir(layer_path):
            neuron_idx, ext = os.path.splitext(fname)
            if ext == ".json":
                if (int(layer), int(neuron_idx)) not in done_neurons:
                    file_paths.append(os.path.join(layer_path, fname))
    if max_num_neurons is not None:
        np.random.shuffle(file_paths)
        file_paths = file_paths[:max_num_neurons]

    print(f"starting to get results for {len(file_paths)} neurons...")
    with ProcessPoolExecutor() as executor:
        all_results = list(
            tqdm(
                executor.map(
                    partial(
                        process_results_for_neuron,
                        exemplar_splits=exemplar_splits,
                        max_num_scores=max_num_scores,
                    ),
                    file_paths,
                )
            )
        )

    # Flatten results.
    flattened_results: List[dict[str, Any]] = []
    for results in all_results:
        if results is None:
            continue
        for result_for_act_sign in results:
            flattened_results.append(result_for_act_sign)
    df = pd.DataFrame(flattened_results)
    results_df = pd.concat([results_df, df]) if results_df is not None else df

    # Save results.
    results_df.to_pickle(save_path)
    return results_df


def copy_neuron_explanations(file_path: str, dest_exp_path: str):
    try:
        with open(file_path, "rb") as f:
            neuron_explanations = NeuronExplanations.model_validate_json(f.read())
    except:
        raise ValueError(f"Error reading {file_path}")

    neuron_id: NeuronId = neuron_explanations.neuron_id
    layer, neuron_idx = neuron_id.layer_index, neuron_id.neuron_index

    layer_dir = os.path.join(dest_exp_path, "explanations", str(layer))
    dest_path = os.path.join(layer_dir, f"{neuron_idx}.json")

    if os.path.exists(dest_path):
        return
    os.makedirs(layer_dir, exist_ok=True)

    for act_sign in neuron_explanations.explanations:
        explanations = neuron_explanations.explanations[act_sign]
        if explanations is None:
            continue
        for exp in explanations:
            exp.simulations = None

    with open(dest_path, "w") as f:
        f.write(neuron_explanations.model_dump_json())


def copy_explanations_from_one_experiment_to_another(
    source_exp_path: str, dest_exp_path: str, neurons_to_copy: Optional[NDIntArray] = None
):
    """
    Copies just the explanations (not the simulations and scores) from one experiment to another.
    If neurons_to_copy is not None, only the data for the specified neurons is copied.
    """
    file_paths = get_neuron_explanations_file_paths_from_exp_path(source_exp_path, neurons_to_copy)

    print(
        f"starting to copy over {len(file_paths)} neurons "
        f"from '{source_exp_path}' to '{dest_exp_path}'..."
    )
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    partial(copy_neuron_explanations, dest_exp_path=dest_exp_path), file_paths
                )
            )
        )


def copy_scored_neuron_explanations(file_path: str, dest_exp_path: str):
    try:
        with open(file_path, "rb") as f:
            neuron_explanations = NeuronExplanations.model_validate_json(f.read())
    except:
        raise ValueError(f"Error reading {file_path}")

    neuron_id: NeuronId = neuron_explanations.neuron_id
    layer, neuron_idx = neuron_id.layer_index, neuron_id.neuron_index

    layer_dir = os.path.join(dest_exp_path, "explanations", str(layer))
    dest_path = os.path.join(layer_dir, f"{neuron_idx}.json")

    if os.path.exists(dest_path):
        return
    os.makedirs(layer_dir, exist_ok=True)
    shutil.copy(file_path, dest_path)


def copy_scored_explanations_from_one_experiment_to_another(
    source_exp_path: str, dest_exp_path: str, neurons_to_copy: NDIntArray
):
    """
    Copies the explanations and scores from one experiment to another.
    If neurons_to_copy is not None, only the data for the specified neurons is copied.
    """
    file_paths = get_neuron_explanations_file_paths_from_exp_path(source_exp_path, neurons_to_copy)

    print(
        f"starting to copy over {len(file_paths)} neurons "
        f"from '{source_exp_path}' to '{dest_exp_path}'..."
    )
    with ProcessPoolExecutor() as executor:
        list(
            tqdm(
                executor.map(
                    partial(copy_scored_neuron_explanations, dest_exp_path=dest_exp_path),
                    file_paths,
                )
            )
        )


def get_neurons_from_exp_path(exp_path: str) -> NDIntArray:
    exps_save_path = os.path.join(exp_path, "explanations")
    neurons: List[List[int]] = []
    for layer in os.listdir(exps_save_path):
        layer_path = os.path.join(exps_save_path, layer)
        for fname in os.listdir(layer_path):
            neuron_idx, ext = os.path.splitext(fname)
            if ext == ".json":
                neurons.append([int(layer), int(neuron_idx)])
    return np.array(neurons)


def neurons_not_started_yet(exp_path: str, neurons_to_start: NDIntArray) -> NDIntArray:
    neurons_done = get_neurons_from_exp_path(exp_path)
    return np.array(list(set(map(tuple, neurons_to_start)) - set(map(tuple, neurons_done))))
