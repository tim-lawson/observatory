"""Utilities for computing exemplars."""

import math
import random
from functools import partial
from typing import Any, Dict, Generator, List, Tuple

import numpy as np
import torch
from activations.activations_computation import get_activations_computing_func
from activations.dataset import MultiSourceDataset
from activations.exemplars import ExemplarSplit, ExemplarType
from activations.exemplars_wrapper import ExemplarsWrapper
from torch.utils.data import DataLoader
from tqdm import tqdm
from util.types import NDFloatArray, NDIntArray


def collate_fn(batch: List[torch.Tensor], pad_id: int, max_length: int) -> Dict[str, torch.Tensor]:
    # Get sequence lengths
    lengths = torch.tensor([seq.size(0) for seq in batch])

    # Pad sequences to max_length
    padded_batch = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for i, seq in enumerate(batch):
        if lengths[i] > 0:
            padded_batch[i, -lengths[i] :] = seq

    # Create attention mask for left padding
    attn_mask = torch.arange(max_length).unsqueeze(0) >= (max_length - lengths).unsqueeze(1)
    return {"input_ids": padded_batch, "attention_mask": attn_mask.int()}


def collate_fn_with_dataset_ids(
    batch: List[Tuple[torch.Tensor, int]], pad_id: int, max_length: int
) -> Dict[str, torch.Tensor]:
    lengths: List[int] = []
    dataset_ids: List[int] = []
    for seq, dataset_id in batch:
        lengths.append(seq.size(0))
        dataset_ids.append(dataset_id)
    lengths_tensor = torch.tensor(lengths)

    # Pad sequences to max_length
    padded_batch = torch.full((len(batch), max_length), pad_id, dtype=torch.long)
    for i, (seq, _) in enumerate(batch):
        if lengths_tensor[i] > 0:
            padded_batch[i, -lengths_tensor[i] :] = seq

    # Create attention mask for left padding
    attn_mask = torch.arange(max_length).unsqueeze(0) >= (max_length - lengths_tensor).unsqueeze(1)
    return {
        "input_ids": padded_batch,
        "attention_mask": attn_mask.int(),
        "dataset_ids": torch.tensor(dataset_ids),
    }


def update_top_acts_and_starts(
    acts: torch.Tensor,  # (batch_size, seq_len, num_neurons)
    input_ids: torch.Tensor,  # (batch_size, seq_len)
    attn_mask: torch.Tensor,  # (batch_size, seq_len)
    dataset_idx: int,
    top_acts: torch.Tensor | None,  # (num_top_acts_to_save, num_neurons)
    topk_seq_acts: torch.Tensor | None,  # (k, seq_len, num_neurons)
    topk_seq_ids: torch.Tensor | None,  # (k, seq_len, num_neurons)
    topk_dataset_ids: torch.Tensor | None,  # (k, num_neurons)
    k: int,
    num_top_acts_to_save: int,
    largest: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Repeat to take into account the neuron dimension.
    input_ids = input_ids.unsqueeze(-1).repeat(1, 1, acts.shape[2])
    dataset_ids = (
        torch.ones(acts.shape[0], acts.shape[2], dtype=torch.int32, device=acts.device)
        * dataset_idx
    )

    # Mask acts according to attn_mask.
    mask_value = torch.inf * (-1 if largest else 1)
    masked_acts = acts.masked_fill(~attn_mask.bool()[:, :, None], mask_value)

    # Update top activations, exemplars and sequence token ids.
    # First, compute the max/min activations per sequence.
    max_or_min_acts, _ = (
        torch.max(masked_acts, dim=1) if largest else torch.min(masked_acts, dim=1)
    )  # (batch_size, num_neurons)
    if top_acts is None:
        top_acts = max_or_min_acts
        topk_seq_acts = masked_acts
        topk_seq_ids = input_ids
        topk_dataset_ids = dataset_ids
    else:
        assert (
            topk_seq_acts is not None and topk_seq_ids is not None and topk_dataset_ids is not None
        )
        if top_acts.shape[0] > k:
            top_acts = torch.cat(
                [top_acts[:k], max_or_min_acts, top_acts[k:]], dim=0
            )  # (num_top_acts_to_save + batch_size, num_neurons)
        else:
            top_acts = torch.cat(
                [top_acts, max_or_min_acts], dim=0
            )  # (num_top_acts_to_save + batch_size, num_neurons)
        topk_seq_acts = torch.cat(
            [topk_seq_acts, masked_acts], dim=0
        )  # (k + batch_size, seq_len, num_neurons)
        topk_seq_ids = torch.cat(
            [topk_seq_ids, input_ids], dim=0
        )  # (k + batch_size, seq_len, num_neurons)
        topk_dataset_ids = torch.cat(
            [topk_dataset_ids, dataset_ids], dim=0
        )  # (k + batch_size, num_neurons)

    top_acts, top_ix = torch.topk(
        top_acts, k=min(num_top_acts_to_save, top_acts.shape[0]), dim=0, largest=largest
    )
    # Get the top k activations.
    topk_ix = top_ix[:k]
    topk_dataset_ids = torch.gather(topk_dataset_ids, dim=0, index=topk_ix)
    # Expand here instead of repeat since we're just using the indices.
    topk_ix = torch.broadcast_to(
        topk_ix[:, None, :],
        (topk_ix.shape[0], topk_seq_acts.shape[1], topk_ix.shape[1]),
    )
    topk_seq_acts = torch.gather(topk_seq_acts, dim=0, index=topk_ix)
    topk_seq_ids = torch.gather(topk_seq_ids, dim=0, index=topk_ix)
    return top_acts, topk_seq_acts, topk_seq_ids, topk_dataset_ids


def get_generators(
    dataloader: DataLoader[Any],
) -> Generator[Dict[str, torch.Tensor], None, None]:
    def generator() -> Generator[Dict[str, torch.Tensor], None, None]:
        for batch in dataloader:
            yield batch

    return generator()


def compute_exemplars_for_layer(
    exemplars_wrapper: ExemplarsWrapper,
    layer: int,
    split: ExemplarSplit,
    save_every: int = 100,
) -> None:
    assert split in (ExemplarSplit.TRAIN, ExemplarSplit.VALID, ExemplarSplit.TEST)

    # Set the random seed.
    rng = random.Random(exemplars_wrapper.config.seed)

    config = exemplars_wrapper.config
    subject = exemplars_wrapper.subject
    num_iters = config.num_seqs // config.batch_size

    # Keep track of the top-k max and min activations per neuron.
    top_acts: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (num_top_acts_to_save, num_neurons)
    topk_seq_acts: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (k, seq_len, num_neurons)
    topk_seq_ids: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (k, seq_len, num_neurons)
    topk_dataset_ids: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (k, num_neurons)

    checkpoint = exemplars_wrapper.load_layer_checkpoint(layer, split)
    if checkpoint is None:
        start_step = 0
        num_tokens_seen = 0
    else:
        loaded_acts, loaded_seq_acts, loaded_ids, loaded_dataset_ids, step, num_tokens_seen = (
            checkpoint
        )
        assert step is not None and num_tokens_seen is not None
        print(f"Loading checkpoint from step {step}")

        # Set the start step and random state.
        if step + 1 >= num_iters:
            return
        start_step = step + 1

        # Convert numpy arrays to torch tensors.
        for extype in ExemplarType:
            top_acts[extype] = torch.tensor(
                loaded_acts[extype].astype(np.float32).transpose(1, 0)
            ).to("cuda", non_blocking=True)
            topk_seq_acts[extype] = torch.tensor(
                loaded_seq_acts[extype].astype(np.float32).transpose(1, 2, 0)
            ).to("cuda", non_blocking=True)
            topk_seq_ids[extype] = torch.tensor(
                loaded_ids[extype].astype(np.int32).transpose(1, 2, 0)
            ).to("cuda", non_blocking=True)
            topk_dataset_ids[extype] = torch.tensor(
                loaded_dataset_ids[extype].astype(np.int32).transpose(1, 0)
            ).to("cuda", non_blocking=True)

    pad_id = subject.tokenizer.pad_token_id
    assert pad_id is not None and isinstance(pad_id, int)
    assert split.value in ("train", "valid", "test")  # for type checking
    datasets = exemplars_wrapper.get_datasets(split.value)

    data_generators: List[Generator[Dict[str, torch.Tensor], None, None]] = []
    for dataset in datasets:
        dataloader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(collate_fn, pad_id=pad_id, max_length=config.seq_len),
        )
        data_generators.append(get_generators(dataloader))

    get_acts = get_activations_computing_func(
        subject=subject, activation_type=config.activation_type, layer=layer
    )

    def save(curr_step: int, num_tokens_seen_so_far: int):
        acts_to_save: Dict[ExemplarType, NDFloatArray] = {}
        seq_acts_to_save: Dict[ExemplarType, NDFloatArray] = {}
        token_ids_to_save: Dict[ExemplarType, NDIntArray] = {}
        dataset_ids_to_save: Dict[ExemplarType, NDIntArray] = {}
        for extype in ExemplarType:
            top_acts_tensor = top_acts[extype]
            topk_seq_acts_tensor = topk_seq_acts[extype]
            topk_seq_ids_tensor = topk_seq_ids[extype]
            topk_dataset_ids_tensor = topk_dataset_ids[extype]
            assert (
                top_acts_tensor is not None
                and topk_seq_acts_tensor is not None
                and topk_seq_ids_tensor is not None
                and topk_dataset_ids_tensor is not None
            )

            acts_to_save[extype] = (
                top_acts_tensor.permute(1, 0).float().cpu().numpy()  # type: ignore
            )  # (act_dim, num_top_acts_to_save)
            seq_acts_to_save[extype] = (
                topk_seq_acts_tensor.permute(2, 0, 1).float().cpu().numpy()  # type: ignore
            )  # (act_dim, k, seq_len)
            token_ids_to_save[extype] = (
                topk_seq_ids_tensor.permute(2, 0, 1).int().cpu().numpy()  # type: ignore
            )  # (act_dim, k, seq_len)
            dataset_ids_to_save[extype] = (
                topk_dataset_ids_tensor.permute(1, 0).int().cpu().numpy()  # type: ignore
            )  # (act_dim, k)

        exemplars_wrapper.save_layer_checkpoint(
            layer=layer,
            split=split,
            acts=acts_to_save,
            seq_acts=seq_acts_to_save,
            token_ids=token_ids_to_save,
            dataset_ids=dataset_ids_to_save,
            step=curr_step,
            num_tokens_seen=num_tokens_seen_so_far,
        )

    step = None
    for step in tqdm(range(num_iters)):
        # First, sample the dataset.
        dataset_idx = rng.randint(a=0, b=len(data_generators) - 1)
        gen = data_generators[dataset_idx]
        # Get batch of token ids.
        try:
            batch = next(gen)
        except StopIteration:
            print("Ran out of data from at least one dataset.")
            break

        num_tokens_seen += int(batch["attention_mask"].sum().item())

        # This is for reproducibility when resuming from a checkpoint.
        if step < start_step:
            continue

        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        acts = get_acts(input_ids=batch["input_ids"], attn_mask=batch["attention_mask"]).to(
            "cuda:0"
        )  # (batch_size, seq_len, act_dim)

        for key, largest in [(ExemplarType.MAX, True), (ExemplarType.MIN, False)]:
            (
                top_acts[key],
                topk_seq_acts[key],
                topk_seq_ids[key],
                topk_dataset_ids[key],
            ) = update_top_acts_and_starts(
                acts=acts,
                input_ids=batch["input_ids"],
                attn_mask=batch["attention_mask"],
                dataset_idx=dataset_idx,
                top_acts=top_acts[key],
                topk_seq_acts=topk_seq_acts[key],
                topk_seq_ids=topk_seq_ids[key],
                topk_dataset_ids=topk_dataset_ids[key],
                k=config.k,
                num_top_acts_to_save=config.num_top_acts_to_save,
                largest=largest,
            )
        if step > 0 and step % save_every == 0:
            print(f"Saving... ({step})")
            save(step, num_tokens_seen)
    print(f"Final save")
    assert step is not None
    save(step, num_tokens_seen)
    print(f"Num tokens seen: {num_tokens_seen}")


def compute_exemplars_for_neuron(
    exemplars_wrapper: ExemplarsWrapper,
    layer: int,
    neuron_idx: int,
    split: ExemplarSplit,
) -> None:
    assert split in (ExemplarSplit.TRAIN, ExemplarSplit.VALID, ExemplarSplit.TEST)

    # Set the random seed.
    rng = random.Random(exemplars_wrapper.config.seed)

    config = exemplars_wrapper.config
    subject = exemplars_wrapper.subject
    num_iters = config.num_seqs // config.batch_size

    # Keep track of the top-k max and min activations.
    top_acts: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (num_top_acts_to_save, num_neurons)
    topk_seq_acts: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (k, seq_len, num_neurons)
    topk_seq_ids: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (k, seq_len, num_neurons)
    topk_dataset_ids: Dict[ExemplarType, torch.Tensor | None] = {
        extype: None for extype in ExemplarType
    }  # (k, num_neurons)

    pad_id = subject.tokenizer.pad_token_id
    assert pad_id is not None and isinstance(pad_id, int)
    assert split.value in ("train", "valid", "test")  # for type checking
    datasets = exemplars_wrapper.get_datasets(split.value)

    data_generators: List[Generator[Dict[str, torch.Tensor], None, None]] = []
    for dataset in datasets:
        dataloader: DataLoader[Any] = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(collate_fn, pad_id=pad_id, max_length=config.seq_len),
        )
        data_generators.append(get_generators(dataloader))

    get_acts = get_activations_computing_func(
        subject=subject, activation_type=config.activation_type, layer=layer
    )

    num_tokens_seen = 0
    step = None
    for step in tqdm(range(num_iters)):
        # First, sample the dataset.
        dataset_idx = rng.randint(a=0, b=len(data_generators) - 1)
        gen = data_generators[dataset_idx]
        # Get batch of token ids.
        try:
            batch = next(gen)
        except StopIteration:
            print("Ran out of data from at least one dataset.")
            break
        num_tokens_seen += int(batch["attention_mask"].sum().item())

        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        acts = get_acts(input_ids=batch["input_ids"], attn_mask=batch["attention_mask"]).to(
            "cuda:0"
        )  # (batch_size, seq_len, act_dim)
        neuron_acts = acts[:, :, [neuron_idx]]  # (batch_size, seq_len, 1)
        for key, largest in [(ExemplarType.MAX, True), (ExemplarType.MIN, False)]:
            (
                top_acts[key],
                topk_seq_acts[key],
                topk_seq_ids[key],
                topk_dataset_ids[key],
            ) = update_top_acts_and_starts(
                acts=neuron_acts,
                input_ids=batch["input_ids"],
                attn_mask=batch["attention_mask"],
                dataset_idx=dataset_idx,
                top_acts=top_acts[key],
                topk_seq_acts=topk_seq_acts[key],
                topk_seq_ids=topk_seq_ids[key],
                topk_dataset_ids=topk_dataset_ids[key],
                k=config.k,
                num_top_acts_to_save=config.num_top_acts_to_save,
                largest=largest,
            )
    assert step is not None
    print(f"Num tokens seen: {num_tokens_seen}")
    # Save results.
    acts_to_save: Dict[ExemplarType, NDFloatArray] = {}
    seq_acts_to_save: Dict[ExemplarType, NDFloatArray] = {}
    token_ids_to_save: Dict[ExemplarType, NDIntArray] = {}
    dataset_ids_to_save: Dict[ExemplarType, NDIntArray] = {}
    for extype in ExemplarType:
        top_acts_tensor = top_acts[extype]
        topk_seq_acts_tensor = topk_seq_acts[extype]
        topk_seq_ids_tensor = topk_seq_ids[extype]
        topk_dataset_ids_tensor = topk_dataset_ids[extype]
        assert (
            top_acts_tensor is not None
            and topk_seq_acts_tensor is not None
            and topk_seq_ids_tensor is not None
            and topk_dataset_ids_tensor is not None
        )

        acts_to_save[extype] = (
            top_acts_tensor.permute(1, 0).float().cpu().numpy()  # type: ignore
        )  # (1, num_top_acts_to_save)
        seq_acts_to_save[extype] = (
            topk_seq_acts_tensor.permute(2, 0, 1).float().cpu().numpy()  # type: ignore
        )  # (1, k, seq_len)
        token_ids_to_save[extype] = (
            topk_seq_ids_tensor.permute(2, 0, 1).int().cpu().numpy()  # type: ignore
        )  # (1, k, seq_len)
        dataset_ids_to_save[extype] = (
            topk_dataset_ids_tensor.permute(1, 0).int().cpu().numpy()  # type: ignore
        )  # (1, k)

    exemplars_wrapper.save_neuron_checkpoint(
        layer=layer,
        neuron_idx=neuron_idx,
        split=split,
        seq_acts=seq_acts_to_save,
        token_ids=token_ids_to_save,
        dataset_ids=dataset_ids_to_save,
        acts=acts_to_save,
        num_tokens_seen=num_tokens_seen,
    )


def save_random_seqs_for_layer(
    exemplars_wrapper: ExemplarsWrapper,
    layer: int,
    split: ExemplarSplit,
):
    assert split in (
        ExemplarSplit.RANDOM_TRAIN,
        ExemplarSplit.RANDOM_VALID,
        ExemplarSplit.RANDOM_TEST,
    )
    config = exemplars_wrapper.config
    subject = exemplars_wrapper.subject
    rand_seqs = config.rand_seqs
    k = rand_seqs * 2  # we need rand_seqs random sequences for each exemplar type (max and min).

    num_neurons_per_step = config.batch_size // k
    num_exemplars_per_step = num_neurons_per_step * k
    num_total_exemplars_needed = subject.I * k
    num_iters = math.ceil(num_total_exemplars_needed / num_exemplars_per_step)

    # First, check if data already exists.
    start_iter = 0
    seq_acts_so_far: NDFloatArray | None = None
    seq_ids_so_far: NDIntArray | None = None
    dataset_ids_so_far: NDIntArray | None = None

    checkpoint = exemplars_wrapper.load_layer_checkpoint(layer, split)
    if checkpoint is not None:
        _, loaded_seq_acts, loaded_ids, loaded_dataset_ids, _, _ = checkpoint

        # We're done with this layer.
        if loaded_seq_acts[ExemplarType.MAX].shape[0] >= subject.I:
            print(f"Skipping layer {layer}.")
            return

        start_iter = loaded_seq_acts[ExemplarType.MAX].shape[0] // num_neurons_per_step

        seq_acts_list: List[NDFloatArray] = []
        seq_ids_list: List[NDIntArray] = []
        dataset_ids_list: List[NDIntArray] = []
        for extype in ExemplarType:
            seq_acts_list.append(loaded_seq_acts[extype])
            seq_ids_list.append(loaded_ids[extype])
            dataset_ids_list.append(loaded_dataset_ids[extype])
        seq_acts_so_far = np.stack(seq_acts_list, axis=1)  # (num_neurons, k, seq_len)
        seq_ids_so_far = np.stack(seq_ids_list, axis=1)  # (num_neurons, k, seq_len)
        dataset_ids_so_far = np.stack(dataset_ids_list, axis=1)  # (num_neurfons, k)
        print(f"Resuming layer {layer} from iter {start_iter}.")

    pad_id = subject.tokenizer.pad_token_id
    assert pad_id is not None and isinstance(pad_id, int)
    dataset_split = split.value.removeprefix("random_")
    assert dataset_split in ("train", "valid", "test")  # for type checking
    datasets = exemplars_wrapper.get_datasets(dataset_split=dataset_split)

    # Combine datasets and create a dataloader.
    combined_dataset = MultiSourceDataset(
        datasets, weights=config.sampling_ratios, seed=config.seed
    )
    dataloader: DataLoader[Any] = DataLoader(
        combined_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=partial(collate_fn_with_dataset_ids, pad_id=pad_id, max_length=config.seq_len),
    )
    data_generator = get_generators(dataloader)

    get_acts = get_activations_computing_func(
        subject=subject, activation_type=config.activation_type, layer=layer
    )

    all_seq_ids: List[NDIntArray] = []
    all_seq_acts: List[NDFloatArray] = []
    all_dataset_ids: List[NDIntArray] = []
    for step in tqdm(range(num_iters)):
        batch = next(data_generator)

        if step < start_iter:
            continue

        dataset_ids = batch.pop("dataset_ids")
        batch = {k: v.to("cuda", non_blocking=True) for k, v in batch.items()}
        acts = get_acts(input_ids=batch["input_ids"], attn_mask=batch["attention_mask"]).to(
            "cuda:0"
        )  # (batch_size, seq_len, act_dim)
        acts = acts[
            : num_neurons_per_step * k,
            :,
            step * num_neurons_per_step : (step + 1) * num_neurons_per_step,
        ]
        if acts.shape[2] < num_neurons_per_step:
            num_neurons_this_step = acts.shape[2]
            acts = acts[: num_neurons_this_step * k]
        else:
            num_neurons_this_step = num_neurons_per_step

        # assign each random sequence to a neuron.
        input_ids = batch["input_ids"][: num_neurons_this_step * k].reshape(
            num_neurons_this_step, k, -1
        )

        # assign activations to each neuron accordingly.
        acts_reshaped = acts.view(num_neurons_this_step, k, config.seq_len, num_neurons_this_step)
        i_indices = torch.arange(num_neurons_this_step)
        acts = acts_reshaped[i_indices, :, :, i_indices]

        all_seq_ids.append(input_ids.int().cpu().numpy())  # type: ignore (num_neurons_this_step, k, seq_len)
        all_seq_acts.append(acts.float().cpu().numpy())  # type: ignore (num_neurons_this_step, k, seq_len)

        # assign dataset idxs to each neuron accordingly.
        dataset_ids = dataset_ids[: num_neurons_this_step * k].reshape(num_neurons_this_step, k)
        all_dataset_ids.append(dataset_ids.int().numpy())  # type: ignore

    seq_ids = np.concatenate(all_seq_ids, axis=0).reshape(-1, 2, k // 2, config.seq_len)
    seq_acts = np.concatenate(all_seq_acts, axis=0).reshape(-1, 2, k // 2, config.seq_len)
    dataset_ids = np.concatenate(all_dataset_ids, axis=0).reshape(-1, 2, k // 2)

    # Concatenate with previously saved data.
    if seq_ids_so_far is not None:
        assert seq_acts_so_far is not None and dataset_ids_so_far is not None
        seq_ids = np.concatenate([seq_ids_so_far, seq_ids], axis=0)
        seq_acts = np.concatenate([seq_acts_so_far, seq_acts], axis=0)
        dataset_ids = np.concatenate([dataset_ids_so_far, dataset_ids], axis=0)

    print(seq_ids.shape)
    assert seq_ids.shape[0] == subject.I

    # Save results.
    seq_acts_to_save: Dict[ExemplarType, NDFloatArray] = {}
    seq_ids_to_save: Dict[ExemplarType, NDIntArray] = {}
    dataset_ids_to_save: Dict[ExemplarType, NDIntArray] = {}
    for i, extype in enumerate(ExemplarType):
        seq_acts_to_save[extype] = seq_acts[:, i, :, :]
        seq_ids_to_save[extype] = seq_ids[:, i, :, :]
        dataset_ids_to_save[extype] = dataset_ids[:, i, :]

    exemplars_wrapper.save_layer_checkpoint(
        layer=layer,
        split=split,
        seq_acts=seq_acts_to_save,
        token_ids=seq_ids_to_save,
        dataset_ids=dataset_ids_to_save,
    )
