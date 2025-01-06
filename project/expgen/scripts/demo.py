# Roughly equivalent to expgen/notebooks/demo.ipynb

from activations.activations_computation import ActivationType

# from activations.dataset import fineweb_dset_config, ultrachat_dset_config
from activations.dataset import lmsys_dset_config
from activations.exemplars import ExemplarSplit, ExemplarType
from activations.exemplars_computation import compute_exemplars_for_layer
from activations.exemplars_wrapper import ExemplarConfig, ExemplarsWrapper
from explanations.explanations import ActivationSign, NeuronExplanations, simulate_and_score
from explanations.explanations_wrapper import ExplanationConfig, ExplanationsWrapper
from explanations.simulation_utils import FinetunedSimulator
from sae_lens import SAE  # type: ignore
from util.subject import Subject, llama31_8B_instruct_config

# Here are some neurons to get you started:
layer, neuron_idx = 5, 2183  # RL-related
layer, neuron_idx = 5, 14249  # East/west coast

# Subject model which contains the neuron of interest.
subject = Subject(llama31_8B_instruct_config)

activation_type = ActivationType.RESID
sae_release = "llama_scope_lxr_8x"
sae_id = "l5r_8x"
sae, _cfg_dict, _sparsity = SAE.from_pretrained(release=sae_release, sae_id=sae_id, device="cuda")

exemplar_config = ExemplarConfig(
    hf_model_id=subject.lm_config.hf_model_id,
    # The two datasets we are going to sample sequences from
    # hf_dataset_configs=(fineweb_dset_config, ultrachat_dset_config),
    # sampling_ratios=[0.9, 0.1],
    hf_dataset_configs=(lmsys_dset_config,),
    # We are going to go through 20,000 sequences per polarity (+/-).
    num_seqs=50_000,
    # Each sequence will be at least 95 tokens long.
    seq_len=95,
    # We are going to keep the top and bottom 100 sequences per neuron.
    k=100,
    activation_type=activation_type,
    sae_release=sae_release,
    batch_size=8,
)

# This will download FineWeb and UltraChat 200k if they are not already present.
# It will take a couple of minutes to download the datasets.
exemplars_wrapper = ExemplarsWrapper(
    data_dir="single_neuron_experiment/", config=exemplar_config, subject=subject, sae=sae
)

# This will download the subject model (Llama-3.1-8B-Instruct) if it is not already present.
compute_exemplars_for_layer(
    exemplars_wrapper=exemplars_wrapper, layer=layer, split=ExemplarSplit.TRAIN
)
compute_exemplars_for_layer(
    exemplars_wrapper=exemplars_wrapper, layer=layer, split=ExemplarSplit.VALID
)

# exemplars_wrapper.visualize_neuron_exemplars(
#     layer=layer,
#     neuron_idx=neuron_idx,
#     exemplar_split=ExemplarSplit.TRAIN,
#     indices=list(range(10)),  # Change this to visualize specific ranks in [0, 100].
# )

explanation_config = ExplanationConfig(
    exemplar_config=exemplar_config,
    # Include the top 20 exemplars in the prompt to the explainer.
    exem_slice_for_exp=(0, 20, 1),
    # Explainer model is on HuggingFace.
    explainer_model_name="Transluce/llama_8b_explainer",
    examples_placement="no_examples",
    # We sample 50 explanations.
    num_explanation_samples=50,
    # Simulator model is also on HuggingFace. More on this later.
    simulator_model_name="Transluce/llama_8b_simulator",
)

explanations_wrapper = ExplanationsWrapper(
    save_path="single_neuron_experiment/",
    config=explanation_config,
    exemplars_data_dir="single_neuron_experiment/",
    subject=subject,
)

# Load the explainer model.
# This will download the finetuned explainer model if it is not already present.
explanations_wrapper.initialize_explainer()

# Generate 50 descriptions using our fine-tuned explainer for both max/minimally activating exemplars.
explanations_wrapper.generate_explanations_for_neuron(layer, neuron_idx)

explanations = explanations_wrapper.get_explanations_for_neuron(
    layer, neuron_idx, exem_splits=[ExemplarSplit.VALID]
)
for i, exp in enumerate(explanations["negative"]):
    print(f"description {i + 1}: {exp[0]}")

# This will download the finetuned simulator model if it is not already present.
simulator = FinetunedSimulator.setup(
    model_path="Transluce/llama_8b_simulator",
    add_special_tokens=True,
    gpu_idx=1,  # If there are multiple GPUs, set this to > 0.
)

neuron_explanations = explanations_wrapper.get_neuron_scored_explanations(layer, neuron_idx)

explanations = neuron_explanations.explanations
split_exemplars = explanations_wrapper.get_split_neuron_exemplars(
    True, ExemplarSplit.VALID, layer, neuron_idx
)

results = {act_sign: [] for act_sign in ActivationSign}
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
