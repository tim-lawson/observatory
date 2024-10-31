from typing import List, Optional, Sequence

from pydantic import BaseModel


class ActivationRecord(BaseModel):
    """A sequence of tokens and their corresponding activations for a single neuron."""

    tokens: List[str]
    """Tokens in a sequence."""
    activations: List[float]
    """Raw activation values for the neuron corresponding to each token in the sequence."""
    token_ids: Optional[List[int]] = None
    """Token IDs for the tokens in the sequence."""

    def all_positive(self) -> bool:
        return all(act > 0 for act in self.activations)

    def any_positive(self) -> bool:
        return any(act > 0 for act in self.activations)

    def all_negative(self) -> bool:
        return all(act < 0 for act in self.activations)

    def any_negative(self) -> bool:
        return any(act < 0 for act in self.activations)


def calculate_max_activation(activation_records: Sequence[ActivationRecord]) -> float:
    return max(max(rec.activations) for rec in activation_records)


def calculate_min_activation(activation_records: Sequence[ActivationRecord]) -> float:
    return min(min(rec.activations) for rec in activation_records)
