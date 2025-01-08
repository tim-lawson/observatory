import numpy as np
from activations.activations import ActivationRecord
from activations.dataset import fineweb_1m_dset_config
from activations.exemplars import ExemplarSplit, ExemplarType
from activations.exemplars_wrapper import ExemplarConfig, ExemplarsWrapper
from util.subject import Subject, get_subject_config

data_dir = "data/jsae-pythia-70m-deduped"

hf_model_id = "EleutherAI/pythia-70m-deduped"
subject_config = get_subject_config(hf_model_id)
subject = Subject(subject_config, nnsight_lm_kwargs={"dispatch": True})

exemplar_config = ExemplarConfig(
    hf_model_id=subject.lm_config.hf_model_id,
    hf_dataset_configs=(fineweb_1m_dset_config,),
    num_seqs=50_000,
    seq_len=95,
    k=100,
    batch_size=1,
)

layer_paths = {
    # 0: "/user/work/qr23940/git/jacobian-saes/checkpoints/gvk1fexn/final_300003328",
    1: "/user/work/qr23940/git/jacobian-saes/checkpoints/oebuvz45/final_300003328",
    # 2: "/user/work/qr23940/git/jacobian-saes/checkpoints/knpagr7x/final_300003328",
    # 3: "/user/work/qr23940/git/jacobian-saes/checkpoints/nfcqpb7p/final_300003328",
    # 4: "/user/work/qr23940/git/jacobian-saes/checkpoints/p70uhzud/final_300003328",
    # 5: "/user/work/qr23940/git/jacobian-saes/checkpoints/f7fcq59d/final_300003328",
}


def get_nonzero_activation_record(activation_records: list[ActivationRecord]):
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


def get_token_ids_distribution(activation_records: list[ActivationRecord]) -> dict[int, float]:
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


for layer, path in layer_paths.items():
    exemplar_split = ExemplarSplit.TRAIN
    exemplar_type = ExemplarType.MAX
    min_entropy = 4.0

    exemplars_wrapper = ExemplarsWrapper(
        data_dir=data_dir,
        config=exemplar_config,
        subject=subject,
    )

    neuron_entropies: list[float] = []

    for neuron_idx in range(32768):
        neuron_exemplars = exemplars_wrapper.get_neuron_exemplars(layer, neuron_idx)
        activation_records = neuron_exemplars.activation_records[exemplar_split][exemplar_type]
        token_ids_distribution = get_token_ids_distribution(activation_records)
        entropy = get_entropy(token_ids_distribution)
        neuron_entropies.append(entropy)
        print(f"layer={layer}, neuron={neuron_idx}, entropy={entropy:.3f}")
        if entropy > min_entropy:
            exemplars_wrapper.visualize_neuron_exemplars(
                layer, neuron_idx, exemplar_split, exemplar_type
            )
            for activation_record in activation_records:
                print(format_activation_record(activation_record))

    print(f"entropy mean={np.mean(neuron_entropies):.3f}, std={np.std(neuron_entropies):.3f}")
