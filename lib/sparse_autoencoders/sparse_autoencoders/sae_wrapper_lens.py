import re
from typing import Callable

import torch
from activations.activations_computation import ActivationType, get_activations_computing_func
from sae_lens import SAE  # type: ignore
from util.subject import Subject

from .sae_wrapper import SAEWrapper, SAEWrapperConfig


def get_activation_type(hook_name: str) -> ActivationType:
    """Convert hook names of supported SAEs: https://jbloomaus.github.io/SAELens/sae_table/"""
    if re.match(r"blocks.(\d+).hook_resid_", hook_name) is not None:
        return ActivationType.RESID
    if re.match(r"blocks.(\d+).hook_mlp_out", hook_name) is not None:
        return ActivationType.MLP_OUT
    if re.match(r"blocks.(\d+).hook_attn_out", hook_name) is not None:
        return ActivationType.ATTN_OUT
    raise ValueError(f"Unsupported hook name: {hook_name}")


class SAELensWrapper(SAEWrapper):
    sae: SAE

    @classmethod
    def from_pretrained(
        cls, release: str, sae_id: str, device: torch.device | str, dtype: str
    ) -> "SAELensWrapper":
        sae, _cfg_dict, _sparsity = SAE.from_pretrained(
            release=release, sae_id=sae_id, device=str(device)
        )
        print(sae)

        sae_wrapper_config = SAEWrapperConfig(
            sae_id=sae_id,
            activation_type=get_activation_type(sae.cfg.hook_name),
            layer=sae.cfg.hook_layer,
        )

        return cls(sae_wrapper_config, sae)

    def __init__(self, sae_wrapper_config: SAEWrapperConfig, sae: SAE) -> None:
        self.config = sae_wrapper_config
        self.sae = sae

    def get_activations_computing_func(self, subject: Subject) -> Callable[..., torch.Tensor]:
        activations_computing_func_ = get_activations_computing_func(
            subject=subject, activation_type=self.config.activation_type, layer=self.config.layer
        )

        def activations_computing_func(
            input_ids: torch.Tensor, attn_mask: torch.Tensor
        ) -> torch.Tensor:
            with torch.no_grad():
                return self.sae.encode(activations_computing_func_(input_ids, attn_mask))

        return activations_computing_func
