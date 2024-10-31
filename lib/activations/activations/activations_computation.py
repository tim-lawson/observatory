from typing import Literal

import torch
from util.subject import Subject


def get_activations_computing_func(subject: Subject, activation_type: Literal["MLP"], layer: int):
    """
    Returns a function that computes activations for a given input:
    input_ids: torch.Tensor
    attn_mask: torch.Tensor

    """
    if activation_type == "MLP":
        mlp_acts_for_layer = subject.w_outs[layer]

        def get_mlp_activations(input_ids: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
            with torch.no_grad():
                with subject.model.trace(
                    {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
                ):
                    acts = mlp_acts_for_layer.input.save()
            return acts

        return get_mlp_activations
    else:
        raise ValueError(f"Unknown activation type: {activation_type}")
