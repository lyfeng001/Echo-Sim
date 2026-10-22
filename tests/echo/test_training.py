"""Numerical contracts against frozen historical sampling and loss examples."""
import unittest
from pathlib import Path

import numpy as np
import torch

from echo.models.baseline import reduce_action_error
from echo.config import learning_rate, load_training
from echo.data.chunks import convert_actions, signal_targets
from echo.models.model import EchoModelConfig, make_echo_attention_mask, masked_echo_loss, signal_time
from echo.data.sampling import EchoWeightedSampler
from echo.evaluation.selection import SCENE_FIELDS, make_jobs
from echo.io import read_json

ROOT = Path(__file__).resolve().parents[2]


class TrainingContracts(unittest.TestCase):
    def test_three_types_and_terminal_padding(self):
        for kind in ("success", "recovery", "failure"):
            pad, ss, sm, fs, fm, am, ab, sb, fb = signal_targets(kind, 520, 519)
            self.assertEqual(pad.tolist(), [False]+[True]*49)
            np.testing.assert_array_equal(ss, pad)
            self.assertTrue(sm.all() and sb.all() and fb.all())
            self.assertEqual(bool(am.any()), kind != "failure")
            if kind == "recovery":
                self.assertFalse(fs.any())
                np.testing.assert_array_equal(ab, ~pad)
            elif kind == "failure":
                np.testing.assert_array_equal(fs, pad)
                self.assertTrue(fm.all())
        self.assertFalse(signal_targets("failure", 520, 0)[4].any())
        self.assertFalse(signal_targets("recovery", 1, 0)[3].any())

    def test_recovery_rounding_preserves_half_even(self):
        # 50*(1/20)=2.5, Python round -> 2; at 3/20 it is 7.5 -> 8.
        for local, zeros in ((1, 2), (3, 8)):
            fs = signal_targets("recovery", 21, local)[3]
            np.testing.assert_array_equal(fs, [0]*zeros+[1]*(50-zeros))

    def test_masked_loss_retains_padding_in_denominator(self):
        pad, _, sm, _, fm, am, ab, sb, fb = signal_targets("recovery", 3, 2)
        tensors = [torch.from_numpy(x[None]) for x in (am, sm[:, None], fm[:, None], pad, ab, sb, fb)]
        result = masked_echo_loss(torch.ones(1, 50, 7), torch.ones(1, 50, 1), torch.ones(1, 50, 1),
                                  *tensors[:4], EchoModelConfig(), *tensors[4:])
        self.assertEqual(result["action_count"].item(), 350)
        self.assertEqual(result["action_numerator"].item(), 7)
        self.assertAlmostEqual(result["loss"].item(), 107/450)
        baseline = reduce_action_error(torch.ones(1, 50, 32), tensors[4])
        self.assertAlmostEqual(baseline["loss"].item(), 1/50)

    def test_forcing_attention_and_gripper(self):
        torch.testing.assert_close(signal_time(torch.tensor([0., .7, .85, 1.]), .7), torch.tensor([0., 0., .5, 1.]))
        mask = make_echo_attention_mask(torch.tensor([[True, False]]), 50)[0]
        self.assertFalse(mask[:, 1].any())
        self.assertTrue(mask[2:52, 2:].all())
        self.assertFalse(mask[52:102, 102:].any())
        self.assertFalse(mask[102:, 52:102].any())
        action = np.zeros((3, 7), np.float32); action[:, 6] = [0, .5, 1]
        np.testing.assert_array_equal(convert_actions(action, "dataset_binary_01")[:, 6], [1, 0, -1])
        np.testing.assert_array_equal(action[:, 6], [0, .5, 1])

    def test_historical_sampler_golden_and_resume(self):
        class Dataset:
            segments = [{"segment_type": k} for k in ["success", "failure", "recovery", "success", "recovery"]]
            offsets = [0, 4, 10, 15, 23, 32]
        # Captured from the original experiment sampler, independent of this module.
        golden = [
            [24,24,23,30,20,15,16,15,22,22,19,19,0,3,0,1,10,14,11,11],
            [21,21,21,16,22,18,18,19,3,1,1,2,10,14,13,14,0,1,1,1],
            [4,4,5,8,3,2,2,2,20,17,21,17,15,18,17,20,1,1,1,2],
            [28,23,24,29,23,30,23,31,3,0,0,2,20,16,22,16,3,2,2,0],
        ]
        for rank in range(4):
            sampler = EchoWeightedSampler(Dataset(), 4, rank, 42, 80, block_size=4)
            self.assertEqual(list(sampler), golden[rank])
            sampler.set_epoch(0, batch_offset=2, batch_size=3)
            self.assertEqual(list(sampler), golden[rank][6:])

    def test_all_historical_evaluation_ids_and_seeds(self):
        for seed in (7, 17, 27):
            manifest = read_json(ROOT/f"manifests/evaluation_seed{seed}.json")
            seen, scenes = set(), []
            for job in manifest["jobs"]:
                if job["path"] not in seen:
                    scenes.append({k: job[k] for k in SCENE_FIELDS}); seen.add(job["path"])
            self.assertEqual(make_jobs(scenes, seed, 5), manifest["jobs"])

    def test_configurations_and_lr(self):
        for name in ("echo_tau0", "echo_tau07", "baseline"):
            config = load_training(ROOT/f"configs/{name}.json")
            self.assertAlmostEqual(learning_rate(0, config), config["lr"]/125)
            self.assertEqual(learning_rate(125, config), config["lr"])
            self.assertEqual(learning_rate(2500, config), config["min_lr"])


if __name__ == "__main__":
    unittest.main()
