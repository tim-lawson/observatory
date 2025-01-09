from enum import Enum

import einops
import torch
from jacobian_saes.sae_pair import SAEPair
from jacobian_saes.training.mlp_with_act_grads import MLPWithActGrads
from transformer_lens import HookedTransformer  # type: ignore
from util.subject import Subject

from .sae_wrapper import SAEWrapper, SAEWrapperConfig
from .util import get_torch_dtype


class JacobianSAEActivationType(str, Enum):
    INPUT = "input"
    OUTPUT = "output"
    JACOBIAN = "jacobian"


class JacobianSAEWrapper(SAEWrapper):
    sae_pair: SAEPair
    mlp_with_act_grads: MLPWithActGrads

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        sae_id: str,
        activation_type: JacobianSAEActivationType,
        device: torch.device | str,
        dtype: str,
    ) -> "JacobianSAEWrapper":
        sae_pair = SAEPair.load_from_pretrained(path, device=str(device), dtype=dtype)

        hooked_transformer = HookedTransformer.from_pretrained(  # type: ignore
            sae_pair.cfg.model_name, device=device, dtype=dtype
        )

        mlp: torch.nn.Module = hooked_transformer.blocks[sae_pair.cfg.hook_layer].mlp
        mlp_with_act_grads = MLPWithActGrads(mlp.cfg)
        mlp_with_act_grads.load_state_dict(mlp.state_dict())
        mlp_with_act_grads.to(device=device, dtype=get_torch_dtype(dtype))

        sae_wrapper_config = SAEWrapperConfig(
            sae_id=sae_id,
            activation_type=activation_type,
            layer=sae_pair.cfg.hook_layer,
        )

        return cls(sae_wrapper_config, sae_pair, mlp_with_act_grads)

    def __init__(
        self,
        sae_wrapper_config: SAEWrapperConfig,
        sae_pair: SAEPair,
        mlp_with_act_grads: MLPWithActGrads,
    ) -> None:
        self.config = sae_wrapper_config
        self.sae_pair = sae_pair
        self.mlp_with_act_grads = mlp_with_act_grads

        self.w_dec_in_LI = sae_pair.get_W_dec(is_output_sae=False) @ self.mlp_with_act_grads.W_in
        w_out_enc_IL = self.mlp_with_act_grads.W_out @ sae_pair.get_W_enc(is_output_sae=True)
        self.w_out_enc_LI = w_out_enc_IL.permute(1, 0)

    def get_jacobian(
        self,
        mlp_act_grads_BTI: torch.Tensor,
        sae_indices_in_BTK: torch.Tensor,
        sae_indices_out_BTK: torch.Tensor,
    ) -> torch.Tensor:
        return einops.einsum(
            self.w_dec_in_LI[sae_indices_in_BTK],
            mlp_act_grads_BTI,
            self.w_out_enc_LI[sae_indices_out_BTK],
            "B T K1 I, B T I, B T K2 I -> B T K1 K2",
        )

    def get_activations_computing_func(self, subject: Subject):
        def activations_computing_func(
            input_ids: torch.Tensor, attn_mask: torch.Tensor
        ) -> torch.Tensor:
            with torch.no_grad():
                with subject.model.trace({"input_ids": input_ids, "attention_mask": attn_mask}):  # type: ignore
                    mlp_acts_in_BTD: torch.Tensor = subject.model.gpt_neox.layers[
                        self.config.layer
                    ].post_attention_layernorm.output.save()  # type: ignore
                    mlp_acts_out_BTD: torch.Tensor = subject.model.gpt_neox.layers[
                        self.config.layer
                    ].mlp.output.save()  # type: ignore

                input_sae_acts_BTL, input_sae_indices_BTK = self.sae_pair.encode(
                    mlp_acts_in_BTD,
                    is_output_sae=False,
                    return_topk_indices=True,  # type: ignore
                )

            mlp_acts_out_BTD, mlp_act_grads_BTI = self.mlp_with_act_grads.forward(mlp_acts_in_BTD)
            mlp_act_grads_BTI = mlp_act_grads_BTI.detach()  # requires grad...

            with torch.no_grad():  # ...but the rest doesn't
                sae_acts_out_BTL, sae_indices_out_BTK = self.sae_pair.encode(
                    mlp_acts_out_BTD,
                    is_output_sae=True,
                    return_topk_indices=True,  # type: ignore
                )

                if self.config.activation_type == JacobianSAEActivationType.INPUT:
                    return input_sae_acts_BTL
                if self.config.activation_type == JacobianSAEActivationType.OUTPUT:
                    return sae_acts_out_BTL
                else:
                    jacobian_BTKK = self.get_jacobian(
                        mlp_act_grads_BTI, input_sae_indices_BTK, sae_indices_out_BTK
                    )
                    return jacobian_BTKK  # TODO: figure out what to return here...

        return activations_computing_func
