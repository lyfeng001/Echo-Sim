"""Success latch, full rollout and frame-level signal accuracy."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from echo.evaluation.evaluate import initialize_run, rollout
from echo.evaluation.signals import SignalMetricConfig, signal_accuracy, whole_chunk


class EvaluationContracts(unittest.TestCase):
    def test_rollout_keeps_real_actions_after_transient_success(self):
        class Env:
            count = 0
            def step(self, action):
                self.count += 1
                self.actions.append(action)
                return observation, 0, self.count == 3, {}
            actions = []
        class Policy:
            last_signals = None
            calls = 0
            def infer(self, obs, instruction):
                self.calls += 1
                self.last_signals = {"ss": np.full(50, self.calls, np.float32), "fs": np.zeros(50, np.float32)}
                return np.ones((50, 7), np.float32)*self.calls
        observation = {name: np.zeros((8, 8, 3), np.uint8) for name in
                       ("agentview_image", "robot0_eye_in_hand_image")}
        env, policy = Env(), Policy()
        with patch("echo.evaluation.evaluate.state_vector", return_value=np.zeros(8, np.float32)):
            row = rollout(env, observation, policy, "task", {"max_steps": 520, "execute_horizon": 8})
        self.assertEqual(env.count, 520)
        self.assertEqual(policy.calls, 65)
        self.assertEqual(row["first_success_step"], 3)
        self.assertTrue(row["success"])
        self.assertFalse(row["terminal_success"])
        np.testing.assert_array_equal(row["actions"][-1], [65]*7)
        self.assertEqual(row["ss"].shape, (65, 50))

    def test_signal_window_and_pre_action_frame_alignment(self):
        events = np.zeros(16, bool); events[7] = True
        config = SignalMetricConfig(success_window_seconds=.1, control_hz=20)
        row = signal_accuracy(events, [0, 8], [.1, .9], [.9, .1], config)
        self.assertEqual((row["frame_correct"], row["frame_total"]), (11, 11))
        self.assertEqual((row["ongoing_frame_total"], row["success_frame_total"]), (8, 3))
        events[:] = False; events[-1] = True
        row = signal_accuracy(events, [0, 8], [.1, .1], [.9, .1], config)
        self.assertTrue(row["missing_post_success_query"])
        self.assertEqual((row["frame_correct"], row["frame_total"]), (16, 16))
        self.assertEqual(row["success_frame_total"], 0)

    def test_timeout_threshold_equality_and_fs(self):
        events = np.zeros(10, bool)
        row = signal_accuracy(events, [0, 8], [.1, .5], [1, 0])
        self.assertEqual(row["threshold_equal_queries"], 1)
        self.assertEqual(row["frame_correct"], 8)
        events[7] = True
        row = signal_accuracy(events, [0, 8], [.1, .9], [0, .5])
        self.assertEqual((row["frame_correct"], row["frame_total"]), (8, 10))
        self.assertEqual((row["success_frame_correct"], row["success_frame_total"]), (0, 2))

    def test_entire_horizon_mean_without_clipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            ss = np.zeros((1, 50), np.float32); ss[:, 8:] = 2
            path = Path(temporary)/"signals.npz"
            np.savez(path, query_controller_step=np.array([0]), ss=ss, fs=np.full((1,50), -.5))
            _, stop, failure = whole_chunk(path, 520)
            self.assertAlmostEqual(stop[0], 1.68)
            self.assertEqual(failure[0], -.5)

    def test_writer_layout_is_frozen(self):
        with tempfile.TemporaryDirectory() as temporary:
            initialize_run(temporary, {"model": "test"}, 2)
            initialize_run(temporary, {"model": "test"}, 2)
            with self.assertRaises(ValueError):
                initialize_run(temporary, {"model": "test"}, 3)
            with self.assertRaises(ValueError):
                initialize_run(temporary, {"model": "changed"}, 2)


if __name__ == "__main__":
    unittest.main()
