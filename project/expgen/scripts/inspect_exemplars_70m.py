from activations.dataset import fineweb_1m_dset_config
from activations.exemplars import ExemplarSplit
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
    0: "/user/work/qr23940/git/jacobian-saes/checkpoints/gvk1fexn/final_300003328",
    1: "/user/work/qr23940/git/jacobian-saes/checkpoints/oebuvz45/final_300003328",
    2: "/user/work/qr23940/git/jacobian-saes/checkpoints/knpagr7x/final_300003328",
    3: "/user/work/qr23940/git/jacobian-saes/checkpoints/nfcqpb7p/final_300003328",
    4: "/user/work/qr23940/git/jacobian-saes/checkpoints/p70uhzud/final_300003328",
    5: "/user/work/qr23940/git/jacobian-saes/checkpoints/f7fcq59d/final_300003328",
}

for layer, path in layer_paths.items():
    exemplars_wrapper = ExemplarsWrapper(
        data_dir=data_dir,
        config=exemplar_config,
        subject=subject,
    )

    for neuron_idx in range(100):
        exemplars_wrapper.visualize_neuron_exemplars(layer, neuron_idx, ExemplarSplit.TRAIN)

    break
