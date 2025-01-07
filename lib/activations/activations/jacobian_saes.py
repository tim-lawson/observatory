import einops
import torch
from jacobian_saes.sae_pair import SAEPair
from jacobian_saes.training.mlp_with_act_grads import MLPWithActGrads
from transformer_lens import HookedTransformer  # type: ignore


class JacobianSAEs:
    sae: SAEPair
    mlp: MLPWithActGrads

    def __init__(self, sae_pair: SAEPair, mlp_with_act_grads: MLPWithActGrads):
        self.sae = sae_pair
        self.mlp = mlp_with_act_grads

        self.w_dec_in_LI = sae_pair.get_W_dec(is_output_sae=False) @ self.mlp.W_in
        self.w_out_enc_LI = (self.mlp.W_out @ sae_pair.get_W_enc(is_output_sae=True)).permute(1, 0)

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

    @classmethod
    def load(cls, path: str, device: torch.device | str) -> "JacobianSAEs":
        sae_pair = SAEPair.load_from_pretrained(path, device=str(device), dtype="float32")

        hooked_transformer = HookedTransformer.from_pretrained(  # type: ignore
            sae_pair.cfg.model_name, device=device, dtype="float32"
        )

        mlp: torch.nn.Module = hooked_transformer.blocks[sae_pair.cfg.hook_layer].mlp
        mlp_with_act_grads = MLPWithActGrads(mlp.cfg)
        mlp_with_act_grads.load_state_dict(mlp.state_dict())
        mlp_with_act_grads.to(device=device, dtype=torch.float32)

        return cls(sae_pair, mlp_with_act_grads)
