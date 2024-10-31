import argparse
import concurrent.futures
import math
import os
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import yaml
from activations.exemplars import ExemplarSplit, ExemplarType
from activations.exemplars_wrapper import ExemplarConfig
from explanations.explanations import NeuronExplanations, simulate_and_score
from explanations.explanations_wrapper import ExplanationConfig, ExplanationsWrapper
from explanations.scoring_utils import ActivationSign
from explanations.simulation_utils import FinetunedSimulator, LlamaSimulator
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from util.subject import Subject, get_subject_config


@dataclass
class OpenAIExplanationDataset(Dataset):
    explanations_wrapper: ExplanationsWrapper
    neurons: np.ndarray
    master_process: bool
    torchrun: bool
    return_best_explanation: bool
    splits_to_evaluate_for_best_explanation: Sequence[ExemplarSplit]
    overwrite: bool

    def __len__(self):
        return len(self.neurons)

    def get_best_idxs(self, neuron_explanations: NeuronExplanations) -> dict[ActivationSign, int]:
        # Create a new NeuronExplanations object with only the best explanations
        # (one for each activation sign).
        all_explanations = neuron_explanations.get_all_explanations_and_scores(
            self.splits_to_evaluate_for_best_explanation
        )
        best_idxs = {
            act_sign: np.argmax([score for _, score in expls])
            for act_sign, expls in all_explanations.items()
        }
        return best_idxs

    def __getitem__(self, idx) -> NeuronExplanations:
        layer, neuron_idx = self.neurons[idx]

        neuron_explanations = self.explanations_wrapper.is_neuron_done_explaining(layer, neuron_idx)
        if neuron_explanations is not None:
            best_idxs = (
                self.get_best_idxs(neuron_explanations) if self.return_best_explanation else None
            )
            return neuron_explanations, best_idxs

        if self.explanations_wrapper.explainer is None:
            self.explanations_wrapper.initialize_explainer()

        self.explanations_wrapper.generate_explanations_for_neuron(
            layer, neuron_idx, self.overwrite
        )

        # Load full list of explanations.
        neuron_explanations = self.explanations_wrapper.get_neuron_scored_explanations(
            layer, neuron_idx
        )
        best_idxs = (
            self.get_best_idxs(neuron_explanations) if self.return_best_explanation else None
        )
        return neuron_explanations, best_idxs


def generate_and_score_explanation(
    explanations_wrapper: ExplanationsWrapper,
    neurons: np.ndarray,
    master_process: bool,
    torchrun: bool,
    num_workers: int = 8,
    do_scoring: bool = True,
    sim_gpu_idx: int = 0,
    splits_for_scoring: Sequence[ExemplarSplit] = ("valid", "random_valid"),
    scoring_batch_size: int = 50,
    use_meta_llama: bool = False,
    compile_simulator: bool = False,
    only_evaluate_best_explanation: bool = False,
    splits_to_use_to_get_best_explanation: Sequence[ExemplarSplit] = ("valid", "random_valid"),
    overwrite: bool = False,
):
    # Generate explanations on the fly. This is done by wrapping the explanation generation
    # process using a Pytorch Dataset object. This will take care of parallelizing
    # explanation generation via the dataloader.
    explanation_dataset = OpenAIExplanationDataset(
        explanations_wrapper=explanations_wrapper,
        neurons=neurons,
        master_process=master_process,
        torchrun=torchrun,
        return_best_explanation=only_evaluate_best_explanation,
        splits_to_evaluate_for_best_explanation=splits_to_use_to_get_best_explanation,
        overwrite=overwrite,
    )

    explanation_dataloader = DataLoader(
        explanation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=lambda batch: batch,
        drop_last=False,
    )

    config = explanations_wrapper.config
    # Maybe setup simulators for simulating activations given explanation.
    if do_scoring:
        if "llama" in config.simulator_model_name and use_meta_llama:
            if config.simulator_model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct":
                num_gpus = int(os.environ["WORLD_SIZE"])
                if num_gpus == 2:
                    model_path = (
                        "/home/ubuntu/.llama/checkpoints/Meta-Llama3.1-70B-Instruct/two-nodes"
                    )
                elif num_gpus == 8:
                    model_path = "/home/ubuntu/.llama/checkpoints/Meta-Llama3.1-70B-Instruct"
                else:
                    raise ValueError()
                model_parallel_size = num_gpus
            else:
                raise ValueError()

            simulator = LlamaSimulator.setup(
                model_path=model_path,
                hf_model_id=config.simulator_model_name,
                model_parallel_size=model_parallel_size,
                compile=compile_simulator,
                sys_prompt_type=config.simulator_system_prompt_type,
            )
        else:
            simulator = FinetunedSimulator.setup(
                model_path=config.simulator_model_name,
                add_special_tokens=config.add_special_tokens,
                gpu_idx=sim_gpu_idx,
            )

    if only_evaluate_best_explanation:
        num_scoring_iters = 1
        scoring_batch_size = 1
    else:
        num_scoring_iters = math.ceil(
            explanations_wrapper.config.num_explanation_samples / scoring_batch_size
        )
    # Note: Just doing exemplar generation (setting do_scoring to False) is not efficient
    # since num_workers above can't be too high (due to rate limits for GPT-4o).
    # One way to deal with this is to do explanation generation separately and using the batch API.
    # I thought this way would be fine since we are overlapping compute by
    # doing something else (scoring) while waiting for the explanations to be generated.
    for neuron_explanations in tqdm(explanation_dataloader):
        if not do_scoring:
            continue

        # [0] is because the dataloader adds an extra dimension.
        neuron_explanations, best_idxs = neuron_explanations[0]
        neuron_id = neuron_explanations.neuron_id

        explanations = neuron_explanations.explanations
        for split_for_scoring in splits_for_scoring:
            split_exemplars = explanations_wrapper.get_split_neuron_exemplars(
                True, split_for_scoring, neuron_id.layer_index, neuron_id.neuron_index
            )

            updated_explanations = {act_sign: [] for act_sign in ActivationSign}
            for act_sign, explanations_list in explanations.items():
                # This is a hack to deal with the case where we only have explanations generated for
                # one activation sign (e.g. human labels).
                if explanations_list is None:
                    updated_explanations[act_sign] = None
                    continue
                # Only evaluate the best explanation.
                if only_evaluate_best_explanation and best_idxs is not None:
                    explanations_list = [explanations_list[idx] for idx in best_idxs[act_sign]]

                extype = ExemplarType.MAX if act_sign == ActivationSign.POS else ExemplarType.MIN
                for idx in range(num_scoring_iters):
                    batch_explanations = explanations_list[
                        idx * scoring_batch_size : (idx + 1) * scoring_batch_size
                    ]
                    scored_explanations = simulate_and_score(
                        split_exemplars=split_exemplars,
                        explanations=batch_explanations,
                        exemplar_type=extype,
                        simulator=simulator,
                        overwrite=overwrite,
                    )
                    if only_evaluate_best_explanation:
                        # We need to only update the best explanation.
                        updated_explanations[act_sign] = explanations_list
                        updated_explanations[act_sign][best_idxs[act_sign]] = scored_explanations
                    else:
                        # We can simply extend since we are evaluating all explanations
                        updated_explanations[act_sign].extend(scored_explanations)
            explanations = updated_explanations

        if master_process:
            neuron_explanations.explanations = explanations
            explanations_wrapper.save_neuron_scored_explanations(
                layer=neuron_id.layer_index,
                neuron_idx=neuron_id.neuron_index,
                neuron_explanations=neuron_explanations,
                exem_splits=splits_for_scoring,
                idxs_to_check=best_idxs,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to config yaml path.")
    parser.add_argument(
        "--neurons_file",
        type=str,
        default=None,
        help="Path to npy file that contains (layer, neuron) indices to generate explanations for.",
    )
    parser.add_argument(
        "--layer_index",
        type=int,
        default=None,
        help="Layer from which we pick neurons to generate explanations for.",
    )
    parser.add_argument(
        "--neuron_indices",
        type=int,
        nargs="+",
        default=None,
        help="Neuron indices to generate explanations for. None means all neurons in the layer.",
    )
    parser.add_argument(
        "--neurons_slice",
        type=int,
        nargs="+",
        default=None,
        help="Length-3 list [start, end, step] to slice neuron indices.",
    )
    parser.add_argument(
        "--subject_hf_model_id",
        type=str,
        default="meta-llama/Meta-Llama-3.1-8B-Instruct",
        help="Huggingface model id for model to get activations for.",
    )
    parser.add_argument(
        "--exemplars_data_dir",
        type=str,
        help="Base directory where exemplars are saved.",
    )
    parser.add_argument(
        "--exemplar_config_path",
        type=str,
        help="Path to exemplar config file path.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path of directory where all results will be saved in.",
    )
    parser.add_argument(
        "--overwrite",
        type=eval,
        default=False,
        choices=[True, False],
        help="Whether to re-run explanation generation and scoring for provided neurons.",
    )
    parser.add_argument(
        "--exem_slice_for_exp",
        type=int,
        nargs="+",
        default=[0, 32, 3],
        help="Length-3 list [start, end, step] to slice exemplars for explanation generation.",
    )
    parser.add_argument(
        "--permute_exemplars_for_exp",
        type=eval,
        default=False,
        choices=[True, False],
        help=(
            "Whether to permute exemplar order in explanation generation prompt. "
            "If this is False, we order the exemplars based on their rank."
        ),
    )
    parser.add_argument(
        "--num_exem_range_for_exp",
        type=int,
        nargs="+",
        help=("Range of number of exemplars to use in explanation generation."),
    )
    parser.add_argument(
        "--fix_exemplars_for_exp",
        type=int,
        default=None,
        help=(
            "Given neuron and activation sign, whether to use the same set of exemplars "
            "in the explanation generation prompt. If this is set to False, we sample "
            "the exemplars randomly each time we generate an explanation."
        ),
    )
    parser.add_argument(
        "--permute_examples_for_exp",
        type=eval,
        default=False,
        choices=[True, False],
        help=(
            "Whether to permute few-shot examples' order in explanation generation prompt. "
            "If this is False, we order the exemplars the same way each time."
        ),
    )
    parser.add_argument(
        "--num_examples_for_exp",
        type=int,
        default=None,
        help=(
            "Number of few-shot examples to use in explanation generation. "
            "If this is < total number of few-shot examples, we sample randomly."
        ),
    )
    parser.add_argument(
        "--fix_examples_for_exp",
        type=eval,
        default=False,
        choices=[True, False],
        help=(
            "Given neuron and activation sign, whether to use the same set of few-shot examples "
            "in the explanation generation prompt. If this is set to False, we sample "
            "the examples randomly each time we generate an explanation."
        ),
    )
    parser.add_argument(
        "--permute_exemplars",
        type=eval,
        default=False,
        choices=[True, False],
        help="Number of exemplars to sample for explanation generation.",
    )
    # TODO(damichoi): Add support for using Llama for explanation generation.
    parser.add_argument(
        "--explainer_model_name",
        type=str,
        default="gpt-4o-mini-2024-07-18",
        help="Model to use to get explanations.",
    )
    parser.add_argument(
        "--explainer_system_prompt_type",
        type=str,
        default="default_separate",
        choices=["default_together", "default_separate", "no_cot", "default_activation", "bills"],
        help="Type of system prompt to use for generating explanations.",
    )
    parser.add_argument(
        "--use_puzzle_for_bills",
        type=eval,
        default=False,
        choices=[True, False],
        help="Whether to use puzzles as few-shot examples if explainer_system_prompt_type is 'bills'.",
    )
    parser.add_argument(
        "--examples_placement",
        type=str,
        default="sysprompt",
        choices=["fewshot", "sysprompt", "no_examples"],
        help="Where to put examples in the explanation-generating prompt.",
    )
    parser.add_argument(
        "--min_tokens_to_highlight",
        type=int,
        default=3,
        help="Minimum number of tokens to highlight for the explanation-generating prompt.",
    )
    parser.add_argument(
        "--round_to_int",
        type=eval,
        default=True,
        choices=[True, False],
        help=(
            "Whether to round the activations to integers for the explanation-generating prompt."
            "Relevant only if explainer_system_prompt_type is 'default_activation'."
        ),
    )
    parser.add_argument(
        "--num_explanation_samples",
        type=int,
        default=1,
        help="Number of explanations to sample per neuron.",
    )
    parser.add_argument(
        "--max_new_tokens_for_explanation_generation",
        type=int,
        default=2000,
        help="Maximum number of tokens used to generate an explanation pair.",
    )
    parser.add_argument(
        "--temperature_for_explanation_generation",
        type=int,
        default=1.0,
        help="Temperature for sampling explanations.",
    )
    parser.add_argument(
        "--save_full_explainer_responses",
        type=eval,
        default=False,
        choices=[True, False],
        help="Whether to save full responses from explainer model.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of processes to use for parallelizing explanation generation.",
    )

    parser.add_argument(
        "--do_scoring",
        type=eval,
        default=True,
        choices=[True, False],
        help="Whether to do scoring after generating explanations.",
    )
    parser.add_argument(
        "--sim_gpu_idx",
        type=int,
        default=0,
        help="GPU index to use for simulating activations.",
    )
    parser.add_argument(
        "--only_evaluate_best_explanation",
        type=eval,
        default=False,
        choices=[True, False],
        help="Whether to only score the best explanation.",
    )
    parser.add_argument(
        "--splits_to_use_to_get_best_explanation",
        type=str,
        nargs="+",
        default=["valid", "random_valid"],
        choices=["train", "valid", "test", "random_valid"],
        help=(
            "Which exemplars splits to use to get the best explanation "
            "(only relevant if only_evaluate_best_explanation is True)."
        ),
    )
    parser.add_argument(
        "--splits_for_scoring",
        type=str,
        nargs="+",
        default=["valid", "random_valid"],
        choices=["train", "valid", "test", "random_valid"],
        help="Which exemplars splits to evaluate explanations on.",
    )
    parser.add_argument(
        "--scoring_batch_size",
        type=int,
        default=50,
        help="Explanations batch size when doing scoring.",
    )
    parser.add_argument(
        "--exem_slice_to_score",
        type=int,
        nargs="+",
        default=[1, 32, 3],
        help="Length-3 list [start, end, step] to slice exemplars to score.",
    )
    parser.add_argument(
        "--simulator_model_name",
        type=str,
        default="meta-llama/Meta-Llama-3.1-70B-Instruct",
        choices=[
            "meta-llama/Meta-Llama-3.1-70B-Instruct",
        ],
        help="Model to use to simulate activations.",
    )
    parser.add_argument(
        "--add_special_tokens",
        type=eval,
        default=False,
        choices=[True, False],
    )
    parser.add_argument(
        "--use_meta_llama",
        type=eval,
        default=False,
        choices=[True, False],
        help=(
            "If the simulator is a Llama model, "
            "whether to use meta's version instead of HuggingFace."
        ),
    )
    parser.add_argument(
        "--compile_simulator",
        type=eval,
        default=False,
        choices=[True, False],
        help=(
            "Whether to compile the model when using meta's llama as the simulator. "
            "Set this to true if the number of neurons to evaluate is large."
        ),
    )
    parser.add_argument(
        "--remove_tab",
        type=eval,
        default=False,
        choices=[True, False],
        help="Whether to remove tabs in the simulation prompt.",
    )
    parser.add_argument(
        "--simulator_system_prompt_type",
        type=str,
        default="unk_base",
        help="Type of system prompt to use for simulating activations.",
    )
    parser.add_argument("--seed", type=int, default=64, help="Random seed for reproducibility.")
    args = parser.parse_args()

    torchrun = int(os.environ.get("RANK", -1)) != -1  # is this a torchrun execution?
    master_process = int(os.environ["RANK"]) == 0 if torchrun else True
    if torchrun:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Optionally load YAML config file. Config sets the defaults, which get
    # overrided by the arguments passed through command line.
    if args.config_path:
        with open(args.config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        default_args = argparse.Namespace()
        default_args.__dict__.update(config)
        args = parser.parse_args(namespace=default_args)

    # Define explanations wrapper.
    with open(args.exemplar_config_path, "r") as f:
        exemplar_config = ExemplarConfig.model_validate_json(f.read())

    subject_config = get_subject_config(args.subject_hf_model_id)
    subject = Subject(subject_config)

    explanation_config = ExplanationConfig(
        exemplar_config=exemplar_config,
        exem_slice_for_exp=args.exem_slice_for_exp,
        permute_exemplars_for_exp=args.permute_exemplars_for_exp,
        num_exem_range_for_exp=args.num_exem_range_for_exp,
        fix_exemplars_for_exp=args.fix_exemplars_for_exp,
        permute_examples_for_exp=args.permute_examples_for_exp,
        num_examples_for_exp=args.num_examples_for_exp,
        fix_examples_for_exp=args.fix_examples_for_exp,
        explainer_model_name=args.explainer_model_name,
        explainer_system_prompt_type=args.explainer_system_prompt_type,
        use_puzzle_for_bills=args.use_puzzle_for_bills,
        examples_placement=args.examples_placement,
        min_tokens_to_highlight=args.min_tokens_to_highlight,
        round_to_int=args.round_to_int,
        num_explanation_samples=args.num_explanation_samples,
        max_new_tokens_for_explanation_generation=args.max_new_tokens_for_explanation_generation,
        temperature_for_explanation_generation=args.temperature_for_explanation_generation,
        save_full_explainer_responses=args.save_full_explainer_responses,
        exem_slice_to_score=args.exem_slice_to_score,
        simulator_model_name=args.simulator_model_name,
        add_special_tokens=args.add_special_tokens,
        simulator_system_prompt_type=args.simulator_system_prompt_type,
        seed=args.seed,
    )
    explanations_wrapper = ExplanationsWrapper(
        save_path=args.save_path,
        config=explanation_config,
        exemplars_data_dir=args.exemplars_data_dir,
        subject=subject,
    )

    # Make list of neurons to generate explanations for.
    if args.neurons_file is not None:
        neurons = np.load(args.neurons_file, mmap_mode="r").astype(np.int64)
    else:
        assert args.layer_index is not None
        if args.neuron_indices is not None:
            neuron_indices = args.neuron_indices
        elif args.neurons_slice is not None and len(args.neurons_slice) == 3:
            neuron_indices = range(*args.neurons_slice)
        else:
            neuron_indices = range(subject.I)

        neurons = []
        for idx in neuron_indices:
            neurons.append((args.layer_index, idx))
        neurons = np.stack(neurons)

    def check_neuron(neuron):
        layer, neuron_idx = neuron
        if not explanations_wrapper.is_neuron_done_scoring(
            layer, neuron_idx, args.splits_for_scoring
        ):
            return (layer, neuron_idx)
        return None

    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(check_neuron, neurons)))
    neurons_left = [result for result in results if result is not None]
    neurons_left = np.array(neurons_left)

    generate_and_score_explanation(
        explanations_wrapper=explanations_wrapper,
        neurons=neurons_left,
        master_process=master_process,
        torchrun=torchrun,
        num_workers=args.num_workers,
        do_scoring=args.do_scoring,
        sim_gpu_idx=args.sim_gpu_idx,
        splits_for_scoring=args.splits_for_scoring,
        scoring_batch_size=args.scoring_batch_size,
        use_meta_llama=args.use_meta_llama,
        compile_simulator=args.compile_simulator,
        only_evaluate_best_explanation=args.only_evaluate_best_explanation,
        splits_to_use_to_get_best_explanation=args.splits_to_use_to_get_best_explanation,
        overwrite=args.overwrite,
    )
