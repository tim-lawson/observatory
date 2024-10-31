# Neuron Descriptions Generation
This project contains code for automatically generating descriptions of neurons in Llama-3.1-8B-Instruct.

The process is broken down into 3 steps:

1. Generating Exemplars
2. Generating Descriptions
3. Scoring Descriptions

See [Scaling Automatic Neuron Description](https://transluce.org/neuron-descriptions) for more details on each of the steps.

## Setup

Follow the instructions in the [Installation section of the README](../../README.md#installation) in the root of the repo. Then install and activate the `expgen` environment by running
```bash
luce install expgen
luce activate expgen
```
This will cd to the expgen project folder `project/expgen`, which is where all scripts should be executed from (e.g. the commands in Step 1 and Step 2).

## Overview of the steps
[`notebooks/demo.ipynb`](notebooks/demo.ipynb) contains step-by-step instructions to generate descriptions for a single neuron in Llama-3.1-8B -Instruct.
To run the notebook, follow the instructions [here](../../README.md#using-jupyter-notebooks):
```bash
luce nb register expgen
luce nb start --port <port>
```
You can get the notebook server URL from the readout of the command above. Make sure to select the proper kernel in the notebook interface.


If you would like to generate descriptions at a larger scale, follow the steps below.

## Step 1. Generating Exemplars
Generating exemplars for all neurons take aroud 64 hours using 2 A100 GPUs.

We uploaded our exemplars to s3, which can be downloaded with the following command after installing [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html)
```bash
aws s3 sync s3://transluce-public/exemplars .
```
The total size is 427GB.

To generate your own exemplars from scratch, use `scripts/compute_exemplars.py`. An example command for computing exemplars for all neurons in layer 0 using the train dataset split is
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m scripts.compute_exemplars --data_dir <data_dir> --hf_datasets fineweb lmsys --num_seqs 1_000_000 --seq_len 95 --k 100 --batch_size 512 --sampling_ratios 0.5 0.5 --layer_indices 0 --split train
```
This requires at least 2 GPUs.

## Step 2. Generating and scoring Descriptions

To generate and score descriptions for many neurons, use `scripts/generate_and_score_explanation.py`. An example command is
```bash
CUDA_VISIBLE_DEVICES=0,1 python scripts/generate_and_score_explanation.py --config_path <path_to_config> --neurons_file <path_to_file>
```
An example config file is `configs/distilled_llama.yaml`.

# FAQ
1. How can I extend this to describe neurons in a different model?

Currently, the code requires the subject model (the model being explained) and the simulator model (used for scoring descriptions) to share the same tokenizer. So as long as this condition is satisfied (we are planning to release code for finetuning the simulator soon), the pipeline can be applied to other models.
The logic for doing simulation using a fine-tuned simulator is in the `FinetunedSimulator` class in `activations.simulation_utils`.

2. Can I describe things other than neurons? e.g. SAE features?

In terms of generating exemplars, one can add to get_activations_computing_func in `activations.activations_computation` to compute activations features other than MLP neurons.
The rest of the pipeline relies on indexing features by their layer and neuron_idx, so if this indexing is generalized, it's possible to describe other types of features.
