import numpy as np
from activations.activations import ActivationRecord


def get_nonzero_activation_record(activation_records: list[ActivationRecord]) -> ActivationRecord:
    nonzero = ActivationRecord(tokens=[], activations=[], token_ids=[])
    assert nonzero.token_ids is not None
    for activation_record in activation_records:
        for k, v in enumerate(activation_record.activations):
            if v > 0.0:
                nonzero.tokens.append(activation_record.tokens[k])
                nonzero.activations.append(v)
                if activation_record.token_ids is not None:
                    nonzero.token_ids.append(activation_record.token_ids[k])
    return nonzero


def get_token_activation_distribution(
    activation_records: list[ActivationRecord],
) -> dict[int, float]:
    nonzero = get_nonzero_activation_record(activation_records)
    assert nonzero.token_ids is not None
    return {
        k: sum(v for k_, v in zip(nonzero.token_ids, nonzero.activations) if k_ == k)
        for k in nonzero.token_ids
    }


def get_entropy(distribution: dict[int, float]) -> float:
    total = sum(distribution.values())
    return -sum(v / total * np.log(v / total) for v in distribution.values() if v > 0)


def format_activation_record(activation_record: ActivationRecord) -> str:
    output = ""
    for k, v in enumerate(activation_record.activations):
        if v > 0.0:
            output += f"{{{{{activation_record.tokens[k]}}}}}"
        else:
            output += activation_record.tokens[k]
    return output
