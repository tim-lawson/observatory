from typing import Literal

import torch
from util.subject import Subject


# TODO(timl): reduce duplication
def get_activations_computing_func(
    subject: Subject,
    # TODO(timl): move to enum?
    activation_type: Literal["resid", "mlp_in", "mlp_out", "attn_out", "neurons"],
    layer: int,
):
    """
    Returns a function that computes activations for a given input:
    input_ids: torch.Tensor
    attn_mask: torch.Tensor

    """
    if activation_type == "resid":
        acts_for_layer = subject.layers[layer]

        def get_activations(input_ids: torch.Tensor, attn_mask: torch.Tensor):
            with torch.no_grad():
                with subject.model.trace(
                    {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
                ):
                    acts = acts_for_layer.output.save()
            return acts

        return get_activations

    if activation_type == "mlp_in":
        acts_for_layer = subject.mlps[layer]

        def get_activations(input_ids: torch.Tensor, attn_mask: torch.Tensor):
            with torch.no_grad():
                with subject.model.trace(
                    {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
                ):
                    acts = acts_for_layer.input.save()
            return acts

        return get_activations

    if activation_type == "mlp_out":
        acts_for_layer = subject.mlps[layer]

        def get_activations(input_ids: torch.Tensor, attn_mask: torch.Tensor):
            with torch.no_grad():
                with subject.model.trace(
                    {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
                ):
                    acts = acts_for_layer.output.save()
            return acts

        return get_activations

    if activation_type == "attn_out":
        acts_for_layer = subject.attns[layer]

        def get_activations(input_ids: torch.Tensor, attn_mask: torch.Tensor):
            with torch.no_grad():
                with subject.model.trace(
                    {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
                ):
                    acts = acts_for_layer.output[0].save()
            return acts

        return get_activations

    if activation_type == "neurons":
        acts_for_layer = subject.w_outs[layer]

        def get_activations(input_ids: torch.Tensor, attn_mask: torch.Tensor):
            with torch.no_grad():
                with subject.model.trace(
                    {"input_ids": input_ids, "attention_mask": attn_mask}  # type: ignore
                ):
                    acts = acts_for_layer.input.save()
            return acts

        return get_activations

    else:
        raise ValueError(f"Unknown activation type: {activation_type}")
