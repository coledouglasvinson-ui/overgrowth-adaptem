from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Categorical

from env import OvergrowthCombatEnv


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(nn.Embedding(8, 32), nn.Flatten(), nn.Linear(32, 64), nn.Tanh())
        self.actor = nn.Linear(64, 4)
        self.critic = nn.Linear(64, 1)

    def forward(self, observation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(observation)
        return self.actor(features), self.critic(features).squeeze(-1)


def save_checkpoint(path: Path, model: ActorCritic, optimizer: torch.optim.Optimizer, steps: int) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": steps},
        temporary_path,
    )
    os.replace(temporary_path, path)


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device("cpu")
    model = ActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    total_steps = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        total_steps = int(checkpoint["steps"])

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = checkpoint_path.with_suffix(".csv")
    write_header = not log_path.exists()
    episode_return = 0.0
    episode_steps = 0
    observation = None
    environment = OvergrowthCombatEnv(port=args.port, timeout=args.timeout)
    print(f"Listening on 127.0.0.1:{args.port}; launch Overgrowth into a sandbox level.")

    try:
        observation, _ = environment.reset(seed=args.seed)
        while total_steps < args.timesteps:
            observations: list[int] = []
            actions: list[int] = []
            log_probabilities: list[torch.Tensor] = []
            rewards: list[float] = []
            values: list[torch.Tensor] = []
            next_values: list[torch.Tensor] = []
            terminated_flags: list[bool] = []
            episode_boundaries: list[bool] = []

            for _ in range(min(args.rollout_steps, args.timesteps - total_steps)):
                current = torch.tensor([observation], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits, value = model(current)
                    distribution = Categorical(logits=logits)
                    action = distribution.sample()

                next_observation, reward, terminated, _, _ = environment.step(int(action.item()))
                episode_steps += 1
                truncated = episode_steps >= args.max_episode_steps and not terminated
                next_tensor = torch.tensor([next_observation], dtype=torch.long, device=device)
                with torch.no_grad():
                    _, next_value = model(next_tensor)

                observations.append(observation)
                actions.append(int(action.item()))
                log_probabilities.append(distribution.log_prob(action).squeeze(0))
                rewards.append(float(reward))
                values.append(value.squeeze(0))
                next_values.append(next_value.squeeze(0))
                terminated_flags.append(terminated)
                episode_boundaries.append(terminated or truncated)
                episode_return += reward
                total_steps += 1
                observation = next_observation

                if terminated or truncated:
                    with log_path.open("a", newline="", encoding="utf-8") as log_file:
                        writer = csv.writer(log_file)
                        if write_header:
                            writer.writerow(("steps", "episode_return", "episode_steps"))
                            write_header = False
                        writer.writerow((total_steps, episode_return, episode_steps))
                    print(
                        f"steps={total_steps} episode_return={episode_return:.3f} "
                        f"episode_steps={episode_steps}"
                    )
                    episode_return = 0.0
                    episode_steps = 0
                    observation, _ = environment.reset()

            advantages = torch.zeros(len(rewards), dtype=torch.float32, device=device)
            next_advantage = torch.tensor(0.0, device=device)
            for index in reversed(range(len(rewards))):
                nonterminal = 0.0 if terminated_flags[index] else 1.0
                delta = rewards[index] + args.gamma * next_values[index] * nonterminal - values[index]
                continuation = 0.0 if episode_boundaries[index] else 1.0
                next_advantage = delta + args.gamma * args.gae_lambda * continuation * next_advantage
                advantages[index] = next_advantage

            returns = advantages + torch.stack(values)
            batch_observations = torch.tensor(observations, dtype=torch.long, device=device)
            batch_actions = torch.tensor(actions, dtype=torch.long, device=device)
            batch_old_log_probabilities = torch.stack(log_probabilities).detach()
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            for _ in range(args.update_epochs):
                permutation = torch.randperm(len(rewards), device=device)
                for start in range(0, len(rewards), args.minibatch_size):
                    indices = permutation[start : start + args.minibatch_size]
                    logits, predicted_values = model(batch_observations[indices])
                    distribution = Categorical(logits=logits)
                    new_log_probabilities = distribution.log_prob(batch_actions[indices])
                    ratio = torch.exp(new_log_probabilities - batch_old_log_probabilities[indices])
                    unclipped = ratio * advantages[indices]
                    clipped = torch.clamp(ratio, 1.0 - args.clip_range, 1.0 + args.clip_range) * advantages[indices]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = (predicted_values - returns[indices]).square().mean()
                    entropy_loss = distribution.entropy().mean()
                    loss = policy_loss + args.value_coefficient * value_loss - args.entropy_coefficient * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()

            save_checkpoint(checkpoint_path, model, optimizer, total_steps)
    finally:
        environment.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Overgrowth combat strategy with PPO.")
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--rollout-steps", type=int, default=128)
    parser.add_argument("--max-episode-steps", type=int, default=256)
    parser.add_argument("--port", type=int, default=47852)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--checkpoint", default="runs/adaptive_combat.pt")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=64)
    parser.add_argument("--value-coefficient", type=float, default=0.5)
    parser.add_argument("--entropy-coefficient", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    args = parser.parse_args()
    if args.timesteps < 1 or args.rollout_steps < 1 or args.max_episode_steps < 1:
        parser.error("timesteps, rollout-steps, and max-episode-steps must be positive")
    if args.minibatch_size < 1 or args.update_epochs < 1:
        parser.error("minibatch-size and update-epochs must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    if args.timeout <= 0 or args.learning_rate <= 0:
        parser.error("timeout and learning-rate must be positive")
    if not 0 < args.gamma <= 1 or not 0 < args.gae_lambda <= 1:
        parser.error("gamma and gae-lambda must be in (0, 1]")
    if args.clip_range <= 0:
        parser.error("clip-range must be positive")
    return args


if __name__ == "__main__":
    train(parse_args())
