# Roughly equivalent to expgen/notebooks/demo.ipynb

from typing import Mapping

# from activations.activations_computation import ActivationType
from activations.activations_computation import ActivationType
from activations.dataset import fineweb_1m_dset_config
from activations.exemplars import ExemplarSplit, ExemplarType
from activations.exemplars_computation import compute_exemplars_for_layer
from activations.exemplars_wrapper import ExemplarConfig, ExemplarsWrapper
from activations.jacobian_saes import JSAE
from explanations.explanations import (
    ActivationSign,
    NeuronExplanation,
    NeuronExplanations,
    simulate_and_score,
)
from explanations.explanations_wrapper import ExplanationConfig, ExplanationsWrapper
from explanations.simulation_utils import FinetunedSimulator
from sae_lens import SAE  # type: ignore
from util.subject import Subject, get_subject_config

data_dir = "data/jsae-demo/"

layer = 3
neuron_idxs = [2183, 14249]

# subject = Subject(llama31_8B_instruct_config)
hf_model_id = "EleutherAI/pythia-70m-deduped"
subject = Subject(get_subject_config(hf_model_id))

activation_type = ActivationType.RESID
sae_release = "llama_scope_lxr_8x"
sae_id = "l5r_8x"
# sae, _cfg_dict, _sparsity = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device="cuda")

jacobian_saes = JSAE.from_pretrained(
    path="/user/work/qr23940/git/jacobian-saes/checkpoints/nfcqpb7p/final_300003328",
    device="cuda:0",
    dtype="float16",
)

exemplar_config = ExemplarConfig(
    hf_model_id=subject.lm_config.hf_model_id,
    hf_dataset_configs=(fineweb_1m_dset_config,),
    num_seqs=50_000,
    seq_len=95,
    k=100,
    sae_id="jsae_l3_64x",
    batch_size=1,
)

exemplars_wrapper = ExemplarsWrapper(
    data_dir=data_dir,
    config=exemplar_config,
    subject=subject,
    # sae=sae,
    jsae=jacobian_saes,
)

compute_exemplars_for_layer(
    exemplars_wrapper=exemplars_wrapper, layer=layer, split=ExemplarSplit.TRAIN
)
compute_exemplars_for_layer(
    exemplars_wrapper=exemplars_wrapper, layer=layer, split=ExemplarSplit.VALID
)

explanation_config = ExplanationConfig(
    exemplar_config=exemplar_config,
    exem_slice_for_exp=(0, 20, 1),
    explainer_model_name="Transluce/llama_8b_explainer",
    examples_placement="no_examples",
    num_explanation_samples=50,
    simulator_model_name="Transluce/llama_8b_simulator",
)

explanations_wrapper = ExplanationsWrapper(
    save_path=data_dir,
    config=explanation_config,
    exemplars_data_dir=data_dir,
    subject=subject,
)

explanations_wrapper.initialize_explainer()

simulator = FinetunedSimulator.setup(
    model_path="Transluce/llama_8b_simulator",
    add_special_tokens=True,
    gpu_idx=1,
)

for neuron_idx in neuron_idxs:
    explanations_wrapper.generate_explanations_for_neuron(layer, neuron_idx)

    explanations = explanations_wrapper.get_explanations_for_neuron(
        layer, neuron_idx, exem_splits=[ExemplarSplit.VALID]
    )
    for i, exp in enumerate(explanations["negative"]):  # type: ignore
        print(f"description {i + 1}: {exp[0]}")

    neuron_explanations = explanations_wrapper.get_neuron_scored_explanations(layer, neuron_idx)
    assert neuron_explanations is not None

    explanations = neuron_explanations.explanations
    split_exemplars = explanations_wrapper.get_split_neuron_exemplars(
        True, ExemplarSplit.VALID, layer, neuron_idx
    )

    results: Mapping[ActivationSign, list[NeuronExplanation]] = {
        act_sign: [] for act_sign in ActivationSign
    }
    for act_sign, explanations_list in explanations.items():
        extype = ExemplarType.MAX if act_sign == ActivationSign.POS else ExemplarType.MIN
        if explanations_list is not None and len(explanations_list) > 0:
            results[act_sign] = simulate_and_score(
                split_exemplars=split_exemplars,
                explanations=explanations_list,
                exemplar_type=extype,
                simulator=simulator,
            )

    scored_neuron_explanations = NeuronExplanations(
        neuron_id=neuron_explanations.neuron_id,
        explanations=results,
    )

    best_explanations = scored_neuron_explanations.get_best_explanations(
        exemplar_splits=[ExemplarSplit.VALID]
    )
    for act_sign, neuron_expl in best_explanations.items():
        print(f"Top explanation for {act_sign}: ")
        explanation_str = neuron_expl.explanation
        score = neuron_expl.get_preferred_score(exemplar_splits=[ExemplarSplit.VALID])
        print(f"score: {score:.2f}: {explanation_str}")
