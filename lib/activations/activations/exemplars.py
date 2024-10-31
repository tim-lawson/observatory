from enum import Enum
from typing import Dict, List

from activations.activations import (
    ActivationRecord,
    calculate_max_activation,
    calculate_min_activation,
)
from pydantic import BaseModel


class NeuronId(BaseModel):
    """Identifier for a neuron in an artificial neural network."""

    layer_index: int
    """The index of layer the neuron is in. The first layer used during inference has index 0."""
    neuron_index: int
    """The neuron's index within in its layer. Indices start from 0 in each layer."""


class ExemplarType(str, Enum):
    MAX = "max"
    MIN = "min"


class ExemplarSplit(str, Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    RANDOM_TRAIN = "random_train"
    RANDOM_VALID = "random_valid"
    RANDOM_TEST = "random_test"


class NeuronExemplars:
    """Exemplars for a neuron, for a specific split (one of train, valid, test)"""

    def __init__(
        self,
        activation_records: dict[ExemplarSplit, dict[ExemplarType, List[ActivationRecord]]],
        activation_percentiles: dict[float, float],
        dataset_names: dict[ExemplarSplit, dict[ExemplarType, List[str]]],
    ):
        extrema: Dict[ExemplarType, float] = {}
        for extype in ExemplarType:
            extrema[extype] = -float("inf") if extype == ExemplarType.MAX else float("inf")
            for split in activation_records:
                act_recs = activation_records[split][extype]
                if extype == ExemplarType.MAX:
                    extremum = calculate_max_activation(act_recs)
                    extrema[extype] = max(extremum, extrema[extype])
                else:  # extype == ExemplarType.MIN:
                    extremum = calculate_min_activation(act_recs)
                    extrema[extype] = min(extremum, extrema[extype])

        self.activation_records = activation_records
        self.activation_percentiles = activation_percentiles
        self.dataset_names = dataset_names
        self.extrema = extrema

    def get_normalized_act_records(
        self, exemplar_split: ExemplarSplit, mask_opposite_sign: bool = False
    ) -> Dict[ExemplarType, List[ActivationRecord]]:
        normalized_act_records: Dict[ExemplarType, List[ActivationRecord]] = {}
        for extype in self.activation_records[exemplar_split]:
            extremum = self.extrema[extype]
            normalized_act_records[extype] = []
            for act_rec in self.activation_records[exemplar_split][extype]:
                norm_acts: List[float] = []
                for act in act_rec.activations:
                    if extype == ExemplarType.MAX and extremum <= 0:
                        norm_acts.append(0)
                    elif extype == ExemplarType.MIN and extremum >= 0:
                        norm_acts.append(0)
                    else:
                        norm_act = act / extremum
                        norm_acts.append(max(0, norm_act) if mask_opposite_sign else norm_act)
                normalized_act_records[extype].append(
                    ActivationRecord(
                        tokens=act_rec.tokens,
                        token_ids=act_rec.token_ids,
                        activations=norm_acts,
                    )
                )
        return normalized_act_records
