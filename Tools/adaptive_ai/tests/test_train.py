from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import train


class FakeEnvironment:
    def __init__(self, port: int, timeout: float):
        self.steps = 0
        self.closed = False

    def reset(self, seed: int | None = None):
        self.steps = 0
        return 0, {}

    def step(self, action: int):
        self.steps += 1
        return self.steps % 8, 1.0, self.steps == 2, False, {}

    def close(self):
        self.closed = True


class TrainerTests(unittest.TestCase):
    def test_ppo_updates_and_saves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "adaptive_combat.pt"
            args = Namespace(
                seed=7,
                learning_rate=3e-4,
                resume=None,
                checkpoint=str(checkpoint_path),
                port=47852,
                timeout=3.0,
                timesteps=4,
                rollout_steps=4,
                max_episode_steps=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                update_epochs=1,
                minibatch_size=2,
                value_coefficient=0.5,
                entropy_coefficient=0.01,
                max_grad_norm=0.5,
            )

            with patch.object(train, "OvergrowthCombatEnv", FakeEnvironment):
                train.train(args)

            self.assertTrue(checkpoint_path.is_file())
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["steps"], 4)
            with checkpoint_path.with_suffix(".csv").open(newline="", encoding="utf-8") as log_file:
                self.assertEqual(len(list(csv.reader(log_file))), 3)


if __name__ == "__main__":
    unittest.main()
