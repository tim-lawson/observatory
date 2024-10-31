import os
import unittest

import numpy as np
from activations.exemplars import ExemplarSplit, ExemplarType
from activations.exemplars_computation import (
    compute_exemplars_for_layer,
    save_random_seqs_for_layer,
)
from activations.exemplars_wrapper import ExemplarsWrapper, fineweb_lmsys_llama31_8b_instruct_config
from util.subject import Subject, llama31_8B_instruct_config


class TestExemplarsComputation(unittest.TestCase):
    def setUp(self):
        self.layer = 0

    def test_reproducibility(self):
        exemplar_config = fineweb_lmsys_llama31_8b_instruct_config.model_copy()
        exemplar_config.num_seqs = 5120
        subject = Subject(llama31_8B_instruct_config)

        tmp_dir1 = "test_reproducibility_1"
        os.makedirs(tmp_dir1, exist_ok=True)

        exemplars_wrapper1 = ExemplarsWrapper(
            data_dir=tmp_dir1,
            config=exemplar_config,
            subject=subject,
        )

        tmp_dir2 = "test_reproducibility_2"
        os.makedirs(tmp_dir2, exist_ok=True)
        exemplars_wrapper2 = ExemplarsWrapper(
            data_dir=tmp_dir2,
            config=exemplar_config,
            subject=subject,
        )
        for split in [ExemplarSplit.TRAIN, ExemplarSplit.VALID, ExemplarSplit.TEST]:
            print(f"Testing exemplar-generating reproducibility for {split.value} split...")
            if split.value.startswith("random"):
                save_random_seqs_for_layer(exemplars_wrapper1, self.layer, split)
                save_random_seqs_for_layer(exemplars_wrapper2, self.layer, split)
            else:
                compute_exemplars_for_layer(exemplars_wrapper1, self.layer, split)
                compute_exemplars_for_layer(exemplars_wrapper2, self.layer, split)
        seq_acts1, token_ids1, dataset_ids1, act_percs1 = exemplars_wrapper1.get_layer_data(
            layer=self.layer
        )

        seq_acts2, token_ids2, dataset_ids2, act_percs2 = exemplars_wrapper2.get_layer_data(
            layer=self.layer
        )

        for split in [ExemplarSplit.TRAIN, ExemplarSplit.VALID, ExemplarSplit.TEST]:
            for extype in ExemplarType:
                np.testing.assert_allclose(seq_acts1[split][extype], seq_acts2[split][extype])
                np.testing.assert_allclose(token_ids1[split][extype], token_ids2[split][extype])
                np.testing.assert_allclose(
                    dataset_ids1[split][extype],
                    dataset_ids2[split][extype],
                )
            for key in act_percs1:
                np.testing.assert_allclose(act_percs1[key], act_percs2[key])

    def test_checkpointing(self):
        subject = Subject(llama31_8B_instruct_config)

        split = ExemplarSplit.TRAIN
        print(f"Testing exemplar-generating checkpointing for {split.value} split...")
        # First, run for 10 steps.
        tmp_dir = f"test_checkpointing_10"
        os.makedirs(tmp_dir, exist_ok=True)
        exemplar_config = fineweb_lmsys_llama31_8b_instruct_config.model_copy()
        exemplar_config.num_seqs = 5120
        exemplars_wrapper_full = ExemplarsWrapper(
            data_dir=tmp_dir,
            config=exemplar_config,
            subject=subject,
        )
        compute_exemplars_for_layer(exemplars_wrapper_full, layer=self.layer, split=split)

        # Next, run for 5 steps, and then 5 more steps.
        tmp_dir = "test_checkpointing_5_5"
        os.makedirs(tmp_dir, exist_ok=True)
        exemplar_config = fineweb_lmsys_llama31_8b_instruct_config.model_copy()
        exemplar_config.num_seqs = 2560
        exemplars_wrapper_half = ExemplarsWrapper(
            data_dir=tmp_dir,
            config=exemplar_config,
            subject=subject,
        )
        exemplar_config = fineweb_lmsys_llama31_8b_instruct_config.model_copy()
        exemplar_config.num_seqs = 5120
        exemplars_wrapper_half = ExemplarsWrapper(
            data_dir=tmp_dir,
            config=exemplar_config,
            subject=subject,
        )
        compute_exemplars_for_layer(exemplars_wrapper_half, layer=self.layer, split=split)

        seq_acts, token_ids, dataset_ids, act_percs = exemplars_wrapper_full.get_layer_data(
            layer=self.layer
        )
        seq_acts_5_5, token_ids_5_5, dataset_ids_5_5, act_percs_5_5 = (
            exemplars_wrapper_half.get_layer_data(layer=self.layer)
        )

        for extype in ExemplarType:
            np.testing.assert_allclose(seq_acts[split][extype], seq_acts_5_5[split][extype])
            np.testing.assert_allclose(token_ids[split][extype], token_ids_5_5[split][extype])
            np.testing.assert_allclose(
                dataset_ids[split][extype],
                dataset_ids_5_5[split][extype],
            )
        for key in act_percs:
            np.testing.assert_allclose(act_percs[key], act_percs_5_5[key])


if __name__ == "__main__":
    unittest.main()
