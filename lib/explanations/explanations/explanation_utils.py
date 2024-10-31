import copy
import json
import math
import os
import random
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, cast

from activations.activations import ActivationRecord, calculate_max_activation
from activations.exemplars import ExemplarSplit, ExemplarType, NeuronExemplars
from activations.exemplars_wrapper import QUANTILE_KEYS
from explanations.explainer import Explainer
from explanations.explainer_prompts import (
    EXAMPLES_DEFAULT,
    EXAMPLES_UPDATED,
    GENERAL_PREFIX,
    SYS_PROMPT_BILLS,
    SYS_PROMPT_DEFAULT,
    SYS_PROMPT_FOR_ACTIVATION_VALUES,
    SYS_PROMPT_NO_COT,
    SYS_PROMPT_SUFFIX,
    SYS_PROMPT_UPDATED,
)
from explanations.explanations import ActivationSign, ExplanationGenerationMetadata
from explanations.few_shot_examples import BILLS_EXAMPLES, format_example
from pydantic import BaseModel
from util.types import ChatMessage


def add_brackets(token_str: str, left_bracket: str, right_bracket: str) -> str:
    """If there are spaces within tokens, push them outside the brackets."""
    if not token_str.lstrip(" ").rstrip(" "):
        return f"{left_bracket}{token_str}{right_bracket}"

    left = 0
    while token_str[left] == " ":
        left += 1

    right = len(token_str) - 1
    while token_str[right] == " ":
        right -= 1

    upd_token_str = (
        token_str[:left]
        + left_bracket
        + token_str[left : right + 1]
        + right_bracket
        + token_str[right + 1 :]
    )
    return upd_token_str


def process_tokens(
    tokens: List[tuple[str, ActivationSign | None]]
) -> tuple[str, Dict[ActivationSign, set[str]]]:
    # Merge consecutively activating tokens of the same activating sign.
    upd_tokens: List[tuple[str, ActivationSign | None]] = [("", None)]
    for token, act_type in tokens:
        prev_token, prev_act_type = upd_tokens[-1]
        if act_type is not None and act_type == prev_act_type:
            upd_tokens[-1] = (prev_token + token, act_type)
        else:
            upd_tokens.append((token, act_type))

    # Add brackets around consecutively activating tokens.
    example_str = ""
    activating_merged_tokens: Dict[ActivationSign, Set[str]] = {
        act_sign: set() for act_sign in ActivationSign
    }
    for token, act_type in upd_tokens:
        if act_type == ActivationSign.POS:
            example_str += add_brackets(token, "<<", ">>")
            activating_merged_tokens[act_type].add(token)
        elif act_type == ActivationSign.NEG:
            example_str += add_brackets(token, "{{", "}}")
            activating_merged_tokens[act_type].add(token)
        else:
            example_str += token

    return example_str, activating_merged_tokens


def format_activation_record_for_pos(
    act_rec: ActivationRecord, act_thresh: float
) -> tuple[str, set[str]]:
    elements: List[tuple[str, Optional[ActivationSign]]] = []
    for token, act in zip(act_rec.tokens, act_rec.activations):
        if act > 0 and act >= act_thresh:
            # Using "negative" because I like {{}} better than <<>>.
            # TODO(damichoi): update code so that we pass brackets directly to process_tokens.
            elements.append((token, ActivationSign.NEG))
        else:
            elements.append((token, None))
    example_str, activating_merged_tokens_i = process_tokens(elements)
    return example_str, activating_merged_tokens_i[ActivationSign.NEG]


def format_activation_record_for_neg(
    act_rec: ActivationRecord, act_thresh: float
) -> tuple[str, set[str]]:
    elements: List[tuple[str, Optional[ActivationSign]]] = []
    for token, act in zip(act_rec.tokens, act_rec.activations):
        if act < 0 and act <= act_thresh:
            elements.append((token, ActivationSign.NEG))
        else:
            elements.append((token, None))
    example_str, activating_merged_tokens_i = process_tokens(elements)
    return example_str, activating_merged_tokens_i[ActivationSign.NEG]


def format_activation_records_for_one_act_sign(
    act_recs: List[ActivationRecord], act_thresh: float, sign: ActivationSign
) -> tuple[List[str], set[str]]:
    example_strs: List[str] = []
    activating_merged_tokens: set[str] = set()
    for act_rec in act_recs:
        example_str, activating_merged_tokens_i = (
            format_activation_record_for_pos(act_rec, act_thresh)
            if sign == ActivationSign.POS
            else format_activation_record_for_neg(act_rec, act_thresh)
        )
        example_strs.append(example_str)
        activating_merged_tokens.update(activating_merged_tokens_i)
    return example_strs, activating_merged_tokens


def get_explainer_prompt_for_one_act_sign(
    act_recs: List[ActivationRecord],
    activation_percentiles: Dict[float, float],
    sign: ActivationSign,
    min_highlights: int = 2,
) -> tuple[List[str], Set[str], float]:
    example_strs: List[str] = []
    activating_tokens: Set[str] = set()
    act_thresh: float = 0.0
    # Change quantile threshold if the number of highlighted tokens are less than min_highlights.
    if sign == ActivationSign.POS:
        q_idx = len(QUANTILE_KEYS) - 1

        while QUANTILE_KEYS[q_idx] > 0.5:
            act_thresh = activation_percentiles.get(QUANTILE_KEYS[q_idx], float("inf"))
            example_strs, activating_tokens = format_activation_records_for_one_act_sign(
                act_recs, act_thresh, sign
            )
            if len(activating_tokens) >= min_highlights:
                break
            q_idx -= 1
    else:
        q_idx = 0
        while QUANTILE_KEYS[q_idx] < 0.5:
            act_thresh = activation_percentiles.get(QUANTILE_KEYS[q_idx], float("-inf"))
            example_strs, activating_tokens = format_activation_records_for_one_act_sign(
                act_recs, act_thresh, sign
            )
            if len(activating_tokens) >= min_highlights:
                break
            q_idx += 1
    return example_strs, activating_tokens, act_thresh


class ExemplarsForExplanationGeneration:
    def __init__(
        self,
        neuron_exemplars: NeuronExemplars,
        act_sign: ActivationSign,
        exem_idxs: List[int],
        exemplar_split: ExemplarSplit = ExemplarSplit.TRAIN,
        permute_exemplars: bool = False,
        num_exemplars_range: Optional[tuple[int, int]] = None,
        fix_exemplars: bool = True,
    ):
        extype = ExemplarType.MAX if act_sign == ActivationSign.POS else ExemplarType.MIN

        all_act_recs = neuron_exemplars.activation_records[exemplar_split][extype]
        norm_act_recs = neuron_exemplars.get_normalized_act_records(
            exemplar_split, mask_opposite_sign=True
        )[extype]

        if num_exemplars_range is not None:
            max_num_exemplars = len(exem_idxs)
            # In this case, we use all exemplars and there is no sampling.
            if num_exemplars_range == (max_num_exemplars, max_num_exemplars):
                num_exemplars_range = None
            else:
                assert num_exemplars_range[0] > 0 and num_exemplars_range[1] <= len(
                    exem_idxs
                ), "Invalid num exemplars range!"

        self.all_activation_records = all_act_recs
        self.all_norm_activation_records = norm_act_recs
        self.activation_percentiles = neuron_exemplars.activation_percentiles
        self.act_sign = act_sign
        self.exem_idxs = exem_idxs
        # Parameters related to sampling exemplars.
        self.permute_exemplars = permute_exemplars
        self.num_exemplars_range = num_exemplars_range
        self.fix_exemplars = fix_exemplars
        self.exemplars_idx: List[int] | None = None

    def get_activation_records(
        self,
        rng: random.Random,
        normalize: bool = False,
        include_ranks: bool = False,
    ) -> Tuple[List[ActivationRecord] | List[tuple[int, ActivationRecord]], List[int]]:
        if self.num_exemplars_range is not None:
            # First check if we are using fixed exemplars. This can only happen if
            # self.num_exemplars_range[0] == self.num_exemplars_range[1].
            can_use_fixed = self.num_exemplars_range[0] == self.num_exemplars_range[1]
            if self.fix_exemplars and can_use_fixed:
                if self.exemplars_idx is None:
                    self.exemplars_idx = rng.sample(self.exem_idxs, k=self.num_exemplars_range[0])
                indices = self.exemplars_idx
            else:
                num_exemplars = rng.randint(
                    a=self.num_exemplars_range[0], b=self.num_exemplars_range[1]
                )
                indices = rng.sample(self.exem_idxs, k=num_exemplars)
            indices = sorted(indices)
        else:
            indices = self.exem_idxs
        if self.permute_exemplars:
            rng.shuffle(indices)

        all_act_recs = (
            self.all_norm_activation_records if normalize else self.all_activation_records
        )
        act_recs = [all_act_recs[idx] for idx in indices]

        # Optionally include ranks.
        if include_ranks:
            return [(idx, act_rec) for idx, act_rec in zip(indices, act_recs)], indices
        else:
            return act_recs, indices


class ExplainerPromptFormatter(BaseModel):
    # Whether to place examples in the user message or in the system prompt (or not at all).
    examples_placement: Literal["fewshot", "sysprompt", "no_examples"]
    # Whether to permute the order of examples.
    permute_examples: bool = False
    # Number of examples to include. If None, include all examples.
    num_examples: Optional[int] = None
    # For a fixed neuron and activation sign, whether to use fixed or different examples per prompt.
    # This is only relevant if num_examples < total number of examples.
    fix_examples: bool = True
    # If sample_examples is False, keep in track of the examples used the first time, and keep
    # using them in subsequent prompts. This is assuming that a new ExplainerPromptFormatter
    # is created for each neuron and activation sign.
    examples_idx: Optional[List[int]] = None

    def get_system_prompt(self) -> str:
        raise NotImplementedError("Subclass must implement this!")

    def get_all_examples(self) -> List[List[ChatMessage]]:
        """Returns all examples in a fixed order."""
        raise NotImplementedError("Subclass must implement this!")

    def get_examples(self, rng: Optional[random.Random] = None) -> List[List[ChatMessage]]:
        """Maybe returns a random sample of examples."""
        all_examples: List[List[ChatMessage]] = self.get_all_examples()
        if self.num_examples is not None and self.num_examples < len(all_examples):
            assert rng is not None
            if self.fix_examples:
                if self.examples_idx is None:
                    self.examples_idx = rng.sample(range(len(all_examples)), k=self.num_examples)
                examples_idx = self.examples_idx
            else:
                examples_idx = rng.sample(range(len(all_examples)), k=self.num_examples)
            examples_idx = sorted(examples_idx)
            sampled_examples = [all_examples[i] for i in examples_idx[: self.num_examples]]
        else:
            sampled_examples = all_examples

        if self.permute_examples:
            assert rng is not None
            rng.shuffle(sampled_examples)
        return sampled_examples

    def format_prompt(
        self,
        exemplars: ExemplarsForExplanationGeneration,
        rng: random.Random,
        **kwargs: Any,
    ) -> Tuple[List[ChatMessage], List[int]]:
        raise NotImplementedError("Subclass must implement this!")


class ExplainerPromptFormatterWithHighlights(ExplainerPromptFormatter):
    min_highlights: int

    def format_prompt(
        self,
        exemplars: ExemplarsForExplanationGeneration,
        rng: random.Random,
        dataaug_func: Optional[
            Callable[[List[ActivationRecord], random.Random], List[ActivationRecord]]
        ] = None,
        **kwargs: Any,
    ) -> Tuple[List[ChatMessage], List[int]]:
        # First, check whether this neuron's activation records in the training set has
        # at least one token with the correct activation sign.
        # We raise errors because it doesn't make sense to explain a sign that doesn't exist.
        # TODO(damichoi): implement ignoring an activation sign if it doesn't exist.
        act_recs = [exemplars.all_activation_records[idx] for idx in exemplars.exem_idxs]
        if exemplars.act_sign == ActivationSign.POS and all(
            act_rec.all_negative() for act_rec in act_recs
        ):
            raise ValueError("No positive activation records in the training set!")
        elif exemplars.act_sign == ActivationSign.NEG and all(
            act_rec.all_positive() for act_rec in act_recs
        ):
            raise ValueError("No negative activation records in the training set!")

        # Keep sampling until we get activation records with at least one token with the correct
        # activation sign.
        while True:
            act_recs, exem_indices = exemplars.get_activation_records(rng=rng)
            act_recs = cast(List[ActivationRecord], act_recs)
            if dataaug_func is not None:
                act_recs = dataaug_func(act_recs, rng)
            if exemplars.act_sign == ActivationSign.POS:
                if any(act_rec.any_positive() for act_rec in act_recs):
                    break
            else:
                if any(act_rec.any_negative() for act_rec in act_recs):
                    break

        _, _, act_thresh = get_explainer_prompt_for_one_act_sign(
            act_recs,
            exemplars.activation_percentiles,
            exemplars.act_sign,
            self.min_highlights,
        )

        sys_prompt: str = self.get_system_prompt()

        if self.examples_placement == "no_examples":
            messages = [ChatMessage(role="system", content=sys_prompt)]
            user_message = ""
        else:
            examples = self.get_examples(rng)

            if self.examples_placement == "sysprompt":
                sys_prompt += f"\n\n## Excerpt{'s' if len(examples) > 1 else ''}:\n"
                for idx, example_messages in enumerate(examples):
                    sys_prompt += f"\nNeuron {idx + 1}:\n\n"
                    assert len(example_messages) == 2  # Only one round of interaction.
                    sys_prompt += example_messages[0]["content"] + "\n\n"  # user
                    sys_prompt += "The following is a sample expected response:\n\n"
                    sys_prompt += example_messages[1]["content"] + "\n"  # assistant
                sys_prompt += f"## End excerpt{'s' if len(examples) > 1 else ''}\n\n"
            sys_prompt += SYS_PROMPT_SUFFIX
            messages = [ChatMessage(role="system", content=sys_prompt)]

            if self.examples_placement == "fewshot":
                for idx, example_messages in enumerate(examples):
                    ex_msgs: List[ChatMessage] = copy.deepcopy(example_messages)
                    assert ex_msgs[0]["role"] == "user"
                    ex_msgs[0]["content"] = f"Neuron {idx + 1}:\n\n" + ex_msgs[0]["content"]
                    messages.extend(ex_msgs)

            user_message = f"Neuron {len(examples) + 1}:\n\n"

        exemplar_strs, _ = format_activation_records_for_one_act_sign(
            act_recs, act_thresh, exemplars.act_sign
        )
        for idx, exemplar_str in enumerate(exemplar_strs):
            user_message += f"Excerpt {idx + 1}:{exemplar_str}\n"
        messages.append(ChatMessage(role="user", content=user_message))
        return messages, exem_indices


class DefaultExplainerPromptFormatter(ExplainerPromptFormatterWithHighlights):
    def get_system_prompt(self):
        return SYS_PROMPT_DEFAULT

    def get_all_examples(self):
        return EXAMPLES_DEFAULT


class UpdatedExplainerPromptFormatter(ExplainerPromptFormatterWithHighlights):
    def get_system_prompt(self):
        return SYS_PROMPT_UPDATED

    def get_all_examples(self):
        return EXAMPLES_UPDATED


class NoCoTExplainerPromptFormatter(ExplainerPromptFormatterWithHighlights):
    def get_system_prompt(self):
        return SYS_PROMPT_NO_COT

    def get_all_examples(self) -> List[List[ChatMessage]]:
        """Get Bills et al's puzzles as few-shot examples."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(script_dir, "puzzles.json"), "r") as f:
            puzzle_dicts = json.loads(f.read())

        # For reproducibility, we sort the puzzles.
        puzzle_names = sorted(puzzle_dicts.keys())

        examples_no_cot: List[List[ChatMessage]] = []
        for puzzle_name in puzzle_names:
            puzzle = puzzle_dicts[puzzle_name]
            sentences: List[List[str | List[str]]] = puzzle["sentences"]
            examples: List[str] = []
            for i, sentence in enumerate(sentences):
                tokens: List[Tuple[str, ActivationSign | None]] = []
                for token in sentence:
                    if isinstance(token, list):
                        tokens.append((str(token[0]), ActivationSign.NEG))
                    else:
                        tokens.append((str(token), None))
                example_str, _ = process_tokens(tokens)
                examples.append(f"Excerpt {i + 1}:  {example_str}")

            user_message = "\n".join(examples)
            assistant_message = f"{GENERAL_PREFIX}: {puzzle['explanation']}"
            examples_no_cot.append(
                [
                    ChatMessage(role="user", content=user_message),
                    ChatMessage(role="assistant", content=assistant_message),
                ]
            )
        return examples_no_cot


class ExplainerPromptFormatterWithActivationValues(ExplainerPromptFormatter):
    examples_placement: Literal["fewshot", "sysprompt", "no_examples"] = "no_examples"
    round_to_int: bool

    def get_system_prompt(self):
        return SYS_PROMPT_FOR_ACTIVATION_VALUES

    def format_prompt(
        self,
        exemplars: ExemplarsForExplanationGeneration,
        rng: random.Random,
        dataaug_func: Optional[
            Callable[[List[ActivationRecord], random.Random], List[ActivationRecord]]
        ] = None,
        **kwargs: Any,
    ) -> Tuple[List[ChatMessage], List[int]]:
        act_recs, exem_indices = exemplars.get_activation_records(rng=rng, normalize=True)
        act_recs = cast(List[ActivationRecord], act_recs)
        if dataaug_func is not None:
            act_recs = dataaug_func(act_recs, rng)

        messages = [ChatMessage(role="system", content=self.get_system_prompt())]

        act_rec_strs: List[str] = []
        for act_rec in act_recs:
            tokens = act_rec.tokens
            acts = act_rec.activations
            entries: List[str] = []
            for token, act in zip(tokens, acts):
                act_str = str(int(act)) if self.round_to_int else f"{act:.2f}"
                entries.append(f"{token}\t{act_str}")
            act_rec_strs.append("\n".join(entries))
        act_recs_str = "\n<start>\n" + "\n<end>\n<start>\n".join(act_rec_strs) + "\n<end>\n"

        user_message = "Activations:" + act_recs_str
        messages.append({"role": "user", "content": user_message})
        return messages, exem_indices


class BillsEtAlExplainerPromptFormatter(ExplainerPromptFormatter):
    use_puzzle_as_examples: bool = True

    def get_system_prompt(self):
        return SYS_PROMPT_BILLS

    def format_normalized_act_recs(
        self, normalized_act_recs: List[ActivationRecord], explanation: Optional[str] = None
    ) -> List[ChatMessage]:
        token_activation_pairs_list: List[List[Tuple[str, int]]] = []
        for act_rec in normalized_act_recs:
            activations = act_rec.activations
            discretized_activations = [min(10, math.floor(10 * act)) for act in activations]
            token_activation_pairs_list.append(list(zip(act_rec.tokens, discretized_activations)))

        user_message = f"Activations:{format_example(token_activation_pairs_list)}"

        # We repeat the non-zero activations only if it was requested and if the proportion of
        # non-zero activations isn't too high.
        non_zero_activations_count = 0
        total_activations_count = 0
        for token_activation_pairs in token_activation_pairs_list:
            for _, act in token_activation_pairs:
                if act > 0:
                    non_zero_activations_count += 1
                elif act < 0:
                    raise ValueError("This shouldn't happen!")
                total_activations_count += 1
        non_zero_activation_proportion = non_zero_activations_count / total_activations_count
        if non_zero_activation_proportion < 0.2:
            reiteration_str = format_example(token_activation_pairs_list, omit_zeros=True)
            if reiteration_str:
                user_message += (
                    f"\nSame activations, but with all zeros filtered out:"
                    f"{format_example(token_activation_pairs_list, omit_zeros=True)}"
                )
        if explanation is None:
            return [ChatMessage(role="user", content=user_message)]
        else:
            return [
                ChatMessage(role="user", content=user_message),
                ChatMessage(role="assistant", content=f" {explanation}."),
            ]

    def normalize_activations(
        self, activation_record: ActivationRecord, max_activation: float
    ) -> ActivationRecord:
        """Convert raw neuron activations be in range [0, 1]."""
        activations = activation_record.activations
        if max_activation <= 0:
            activations = [0.0 for _ in activations]
        else:
            activations = [max(0, act / max_activation) for act in activations]
        return ActivationRecord(tokens=activation_record.tokens, activations=activations)

        # return [min(10, math.floor(10 * relu(x) / max_activation)) for x in activation_record]

    def format_examples(
        self, examples: List[Tuple[str, List[ActivationRecord]]]
    ) -> List[List[ChatMessage]]:
        messages_list: List[List[ChatMessage]] = []
        for explanation, act_recs in examples:
            # Normalize the activation records.
            max_activation = max(calculate_max_activation(act_recs), 0)
            normalized_act_recs = [
                self.normalize_activations(act_rec, max_activation) for act_rec in act_recs
            ]
            messages_list.append(self.format_normalized_act_recs(normalized_act_recs, explanation))
        return messages_list

    def get_all_examples(self) -> List[List[ChatMessage]]:
        if self.use_puzzle_as_examples:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            with open(os.path.join(script_dir, "puzzles.json"), "r") as f:
                puzzle_dicts = json.loads(f.read())

            # For reproducibility, we sort the puzzles.
            puzzle_names = sorted(puzzle_dicts.keys())

            examples: List[Tuple[str, List[ActivationRecord]]] = []
            for puzzle_name in puzzle_names:
                puzzle = puzzle_dicts[puzzle_name]
                explanation = puzzle["explanation"]
                sentences: List[List[str | List[str | int]]] = puzzle["sentences"]

                act_recs: List[ActivationRecord] = []
                for sentence in sentences:
                    tokens: List[str] = []
                    activations: List[float] = []
                    for token_and_act in sentence:
                        if isinstance(token_and_act, list):
                            token = str(token_and_act[0])
                            activation = int(token_and_act[1])
                        else:
                            token, activation = token_and_act, 0
                        tokens.append(token)
                        activations.append(float(activation))
                    act_recs.append(ActivationRecord(tokens=tokens, activations=activations))
                examples.append((explanation, act_recs))
        else:
            examples = BILLS_EXAMPLES
        return self.format_examples(examples)

    def format_prompt(
        self,
        exemplars: ExemplarsForExplanationGeneration,
        rng: random.Random,
        dataaug_func: Optional[
            Callable[[List[ActivationRecord], random.Random], List[ActivationRecord]]
        ] = None,
        **kwargs: Any,
    ) -> Tuple[List[ChatMessage], List[int]]:
        assert self.examples_placement in ("fewshot", "no_examples")

        act_recs, exem_indices = exemplars.get_activation_records(rng=rng, normalize=True)
        act_recs = cast(List[ActivationRecord], act_recs)
        if dataaug_func is not None:
            act_recs = dataaug_func(act_recs, rng)

        messages = [ChatMessage(role="system", content=self.get_system_prompt())]

        num_fewshot = 0
        if self.examples_placement == "fewshot":
            examples_messages_list = self.get_examples(rng)
            for idx, example_messages in enumerate(examples_messages_list):
                assert len(example_messages) == 2  # Only one round of interaction.
                # Update few-shot example messages with their neuron indices.
                user_message = example_messages[0]["content"]
                upd_user_message = (
                    f"\n\nNeuron {idx + 1}\n"
                    + user_message
                    + f"\nExplanation of neuron {idx + 1} behavior: "
                    + "the main thing this neuron does is find"
                )
                example_messages[0]["content"] = upd_user_message
                example_messages[1]["content"] = example_messages[1]["content"].format(idx=idx + 1)
                messages.extend(example_messages)
            num_fewshot = len(examples_messages_list)

        user_message = self.format_normalized_act_recs(normalized_act_recs=act_recs)
        assert len(user_message) == 1
        user_message[0]["content"] = f"\n\nNeuron {num_fewshot + 1}\n" + user_message[0]["content"]
        messages.extend(user_message)
        return messages, exem_indices


def get_prompt_formatter(system_prompt_type: str, **kwargs: Any) -> ExplainerPromptFormatter:
    if system_prompt_type == "default_separate":
        return DefaultExplainerPromptFormatter(**kwargs)
    elif system_prompt_type == "updated_separate":
        return UpdatedExplainerPromptFormatter(**kwargs)
    elif system_prompt_type == "no_cot":
        return NoCoTExplainerPromptFormatter(**kwargs)
    elif system_prompt_type == "default_activation":
        return ExplainerPromptFormatterWithActivationValues(**kwargs)
    elif system_prompt_type == "bills":
        return BillsEtAlExplainerPromptFormatter(**kwargs)
    raise ValueError(f"Invalid explainer system_prompt_type {system_prompt_type}!")


def postprocess_response(response: str) -> str | None:
    if GENERAL_PREFIX in response:
        return response.split(GENERAL_PREFIX)[-1].strip("\n :")
    else:
        return None


def generate_explanations_for_one_act_sign(
    exemplars: ExemplarsForExplanationGeneration,
    prompt_formatter: ExplainerPromptFormatter,
    num_expl_samples: int,
    explainer: Explainer,
    rng: random.Random,
) -> tuple[List[str], ExplanationGenerationMetadata]:
    explanation_strs: List[str] = []
    exem_indices_for_expls: List[List[int]] = []
    response_strs: Dict[str, list[str]] = {"succ": [], "fail": []}
    num_refusals: int = 0
    num_format_failures: int = 0
    num_iterations: int = 0
    all_messages: List[List[ChatMessage]] = []
    while len(explanation_strs) < num_expl_samples:
        if num_iterations > 10:
            break
        if num_iterations > 0:
            print(f"retrying... ({num_iterations})")
        messages_list: List[List[ChatMessage]] = []
        exem_indices_list: List[List[int]] = []
        for _ in range(num_expl_samples):
            messages, exem_indices = prompt_formatter.format_prompt(exemplars, rng=rng)
            messages_list.append(messages)
            exem_indices_list.append(exem_indices)
        try:
            responses, refusals = explainer.get_chat_completions(
                messages_list=messages_list, num_samples=1
            )
        except Exception as e:
            print(e)
            num_iterations += 1
            continue
        for response, exem_indices in zip(responses, exem_indices_list):
            if isinstance(prompt_formatter, BillsEtAlExplainerPromptFormatter):
                explanation_str = response
            else:
                explanation_str = postprocess_response(response)
            if explanation_str is None:
                num_format_failures += 1
                response_strs["fail"].append(response)
            else:
                explanation_strs.append(explanation_str)
                response_strs["succ"].append(response)
                exem_indices_for_expls.append(exem_indices)
        num_refusals += refusals
        num_iterations += 1
        all_messages.extend(messages_list)

    return explanation_strs, ExplanationGenerationMetadata(
        ranks=exemplars.exem_idxs,
        messages=all_messages[0],
        num_refusals=num_refusals,
        num_format_failures=num_format_failures,
        num_iterations={num_expl_samples: num_iterations},
        exem_indices_for_explanations=exem_indices_for_expls,
        responses=response_strs,
        exemplars_idx=exemplars.exemplars_idx,
        examples_idx=prompt_formatter.examples_idx,
    )
