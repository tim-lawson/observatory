from typing import Any, Literal, NamedTuple, TypedDict

import numpy as np
import torch
from numpy.typing import NDArray

NDFloatArray = NDArray[np.floating[Any]]
NDIntArray = NDArray[np.integer[Any]]


class ChatMessage(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class GenerateOutput(NamedTuple):
    output_ids_BT: NDIntArray
    logits_BV: torch.Tensor
    tokenwise_log_probs: list[tuple[NDIntArray, NDFloatArray]]
    continuations: list[str]


class TopKResult(NamedTuple):
    indices: list[int]
    probs: list[float]
