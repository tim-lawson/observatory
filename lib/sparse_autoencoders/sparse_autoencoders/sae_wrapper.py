from abc import ABC
from typing import Any, Callable

from activations.activations_computation import ActivationType, get_activations_computing_func
from pydantic import BaseModel
from util.subject import Subject


class SAEWrapperConfig(BaseModel):
    sae_id: str
    """Arbitrary id. Used to determine the folder name in ExemplarsWrapper."""

    activation_type: ActivationType | str

    layer: int


class SAEWrapper(ABC):
    def __init__(self, sae_wrapper_config: SAEWrapperConfig) -> None:
        self.config = sae_wrapper_config

    def get_activations_computing_func(self, subject: Subject) -> Callable[..., Any]:
        """Returns a function that computes activations for a given input."""
        return get_activations_computing_func(
            subject=subject, activation_type=self.config.activation_type, layer=self.config.layer
        )
