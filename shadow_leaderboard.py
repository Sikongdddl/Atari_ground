
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shadow_leaderboard.py
Backend-only "Shadow Match" leaderboard for Atari Pong (old Gym API).
- Reuses DQN and PongWrapper from dqn.py
- Collects ALE snapshots at the start of points ("state bank")
- Evaluates multiple models on the same snapshot set for fair comparison
- Outputs a CSV leaderboard and prints a summary

Usage (examples):
    # Collect 50 snapshots with scripted driver, evaluate baseline model
    python shadow_leaderboard.py \
        --env PongNoFrameskip-v4 \
        --bank-size 50 \
        --csv ./shadow_leaderboard.csv \
        --model baseline=./pong_dqn.pth

    # Use a trained policy to drive snapshot collection
    python shadow_leaderboard.py \
        --env PongNoFrameskip-v4 \
        --bank-size 50 \
        --driver policy \
        --driver-weights ./pong_dqn.pth \
        --csv ./shadow_leaderboard.csv \
        --model baseline=./pong_dqn.pth \
        --model student=./finetune_ckpts/pong_dqn_ft_epoch1.pth
"""

import os
import csv
import time
import argparse
import random
import numpy as np
import torch
import gym

from typing import Optional

# Import user's old-Gym DQN & wrappers
from dqn import DQN, PongWrapper, make_env_oldgym  # type: ignore


class DQNPolicy:
    """Thin inference wrapper for .pth -> act(state)->int (0/1/2)."""
    def __init__(self, weights_path: str, action_dim: int, device: str = "cpu"):
        self.net = DQN(action_dim).to(device)
        sd = torch.load(weights_path, map_location=device)
        self.net.load_state_dict(sd)
        self.net.eval()
        self.device = device

    @torch.no_grad()
    def act(self, state_tensor: torch.Tensor) -> int:
        # state_tensor: (4,84,84) float32
        q = self.net(state_tensor.unsqueeze(0).to(self.device))
        return int(q.argmax(1).item())


def make_env_legacy(env_name: str, seed = None) -> PongWrapper:
    """Build old-Gym Atari env and wrap with user's PongWrapper."""
    env = gym.make(env_name)
    if seed is not None:
        try:
            env.seed(seed)  # old API
        except Exception:
            pass
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    return PongWrapper(env)


class StateBank:
    """
    Collect ALE emulator snapshots as common evaluation starting points.
    Strategy: after a point ends, reset and step a few NOOPs, then cloneState().
    """
    def __init__(self, env_name: str = "PongNoFrameskip-v4", seed: int = 123, noop_after_reset: int = 3):
        self.env_name = env_name
        self.seed = seed
        self.noop_after_reset = noop_after_reset

    def _scripted_action(self, tick: int) -> int:
        # Simple driver: mostly NOOP, occasionally toggle UP/DOWN to move the game forward.
        r = random.random()
        if r < 0.85:
            return 0  # NOOP
        return 1 if (tick // 7) % 2 == 0 else 2

    def collect(self, num_states: int = 50, driver=None ,driver_policy= None):
        env = make_env_legacy(self.env_name, seed=self.seed)
        ale = getattr(env.unwrapped, "ale", None)
        has_clone = hasattr(ale, "cloneState") if ale is not None else False

        snapshots = []
        if not has_clone:
            print("[Shadow] Warning: ALE lacks cloneState/restoreState; will fallback to seeded-first-point evaluation.")
            env.close()
            return snapshots

        obs = env.reset()
        tick = 0
        while len(snapshots) < num_states:
            done = False
            # Play until a point ends (reward != 0)
            while not done:
                if driver == "policy" and driver_policy is not None:
                    a = driver_policy.act(obs)
                else:
                    a = self._scripted_action(tick)
                obs, r, done, _ = env.step(a)
                tick += 1
                if r != 0:
                    break
            # New point
            obs = env.reset()
            for _ in range(self.noop_after_reset):
                obs, _, _, _ = env.step(0)  # NOOP a few frames so ball leaves the serve origin
            # Take snapshot
            snapshots.append(ale.cloneState())

        env.close()
        print(f"[Shadow] Collected {len(snapshots)} snapshots.")
        return snapshots


class ShadowArena:
    """
    Evaluate multiple models on the same snapshot bank.
    - After restore, prefill frame stack via NOOPs (>= stack size) for PongWrapper sync
    - Roll until current point ends (reward != 0) or max steps
    """
    def __init__(self, env_name: str = "PongNoFrameskip-v4", device: str = "cpu",
                 prefill_noop: int = 4, max_steps_per_point: int = 4000):
        self.env_name = env_name
        self.device = device
        self.prefill_noop = prefill_noop
        self.max_steps_per_point = max_steps_per_point

    @torch.no_grad()
    def _play_one_point_from_snapshot(self, env: PongWrapper, policy: DQNPolicy, snapshot):
        ale = getattr(env.unwrapped, "ale", None)
        env.reset()
        if snapshot is not None and ale is not None:
            ale.restoreState(snapshot)
            # Prefill frames to sync PongWrapper's deque with the ALE screen
            for _ in range(self.prefill_noop):
                obs, _, _, _ = env.step(0)
        else:
            obs = env.reset()
            for _ in range(self.prefill_noop):
                obs, _, _, _ = env.step(0)

        steps = 0
        rew_sum = 0.0
        while steps < self.max_steps_per_point:
            a = policy.act(obs)  # 0/1/2
            obs, r, done, _ = env.step(a)
            rew_sum += float(r)
            steps += 1
            # In Pong, non-zero reward marks end of the point
            if r != 0 or done:
                break
        return rew_sum, steps

    def evaluate_models(self, weights_dict, snapshots, csv_path: str = "./shadow_leaderboard.csv",
                        seed: int = 777, fallback_points: int = 50):
        env = make_env_legacy(self.env_name, seed=seed)
        env.reset()
        action_dim = len(env.valid_actions)

        ale = getattr(env.unwrapped, "ale", None)
        has_clone = hasattr(ale, "cloneState") if ale is not None else False
        use_snapshots = has_clone and (len(snapshots) > 0)

        # Prepare players
        players = {name: DQNPolicy(path, action_dim, device=self.device)
                   for name, path in weights_dict.items()}

        # CSV header
        header = ["model", "points_won", "points_lost", "ties", "net_points", "avg_steps_per_point", "total_points"]
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:   # 文件不存在时才写表头
                writer.writerow(header)

        results = {}
        for name, policy in players.items():
            print(f"[Shadow] Evaluating: {name}")
            won = lost = tie = 0
            steps_all = []

            if use_snapshots:
                for snap in snapshots:
                    rew, steps = self._play_one_point_from_snapshot(env, policy, snap)
                    steps_all.append(steps)
                    if rew > 0:   won += 1
                    elif rew < 0: lost += 1
                    else:         tie += 1
            else:
                # Fallback (S1): evaluate first point from different seeds
                n_points = fallback_points if (len(snapshots) == 0) else len(snapshots)
                for k in range(n_points):
                    env.close()
                    env = make_env_legacy(self.env_name, seed=seed + k)
                    rew, steps = self._play_one_point_from_snapshot(env, policy, snapshot=None)
                    steps_all.append(steps)
                    if rew > 0:   won += 1
                    elif rew < 0: lost += 1
                    else:         tie += 1

            total = won + lost + tie
            avg_steps = float(np.mean(steps_all)) if steps_all else 0.0
            net = won - lost
            results[name] = {
                "points_won": won,
                "points_lost": lost,
                "ties": tie,
                "net_points": net,
                "avg_steps_per_point": avg_steps,
                "total_points": total,
            }
            with open(csv_path, "a", newline="") as f:
                csv.writer(f).writerow([name, won, lost, tie, net, round(avg_steps, 2), total])

        env.close()

        # Print sorted summary
        print("\n===== Shadow Leaderboard =====")
        ranked = sorted(results.items(), key=lambda kv: (kv[1]["net_points"], kv[1]["points_won"]), reverse=True)
        for i, (name, stat) in enumerate(ranked, 1):
            print(f"{i:>2}. {name:20s}  "
                  f"W:{stat['points_won']:>3}  L:{stat['points_lost']:>3}  T:{stat['ties']:>3}  "
                  f"Net:{stat['net_points']:+3d}  AvgSteps/pt:{stat['avg_steps_per_point']:.1f}")
        print(f"CSV written to: {csv_path}")
        return results


def main():
    parser = argparse.ArgumentParser(description="Shadow leaderboard for Atari Pong (old Gym API).")
    parser.add_argument("--env", type=str, default="PongNoFrameskip-v4", help="Environment name (old Gym Atari).")
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--seed", type=int, default=123, help="Seed for snapshot collection.")
    parser.add_argument("--bank-size", type=int, default=50, help="Number of snapshots to collect.")
    parser.add_argument("--noop-after-reset", type=int, default=3, help="NOOP steps after reset before snapshot.")
    parser.add_argument("--prefill-noop", type=int, default=4, help="NOOP steps after restore to sync frame stack.")
    parser.add_argument("--max-steps", type=int, default=4000, help="Max env steps per point.")
    parser.add_argument("--csv", type=str, default="./shadow_leaderboard.csv", help="Output CSV path.")
    parser.add_argument("--driver", type=str, choices=["scripted", "policy"], default="scripted", help="Snapshot driver.")
    parser.add_argument("--driver-weights", type=str, default=None, help="Policy weights for 'policy' driver.")
    parser.add_argument("--model", type=str, action="append", default=[],
                        help="Model spec name=path (can appear multiple times).")
    parser.add_argument("--fallback-points", type=int, default=50, help="Points to evaluate in fallback mode (no ALE clone).")

    args = parser.parse_args()

    # Parse models
    weights_dict = {}
    for spec in args.model:
        if "=" not in spec:
            raise ValueError(f"--model expects name=path, got: {spec}")
        name, path = spec.split("=", 1)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Model weights not found: {path}")
        weights_dict[name] = path
    if not weights_dict:
        raise ValueError("No models provided. Use --model name=path (can repeat).")

    # Prepare optional driver policy
    driver_policy = None
    if args.driver == "policy":
        # Need an env to infer action_dim for DQN head
        tmp_env = make_env_legacy(args.env, seed=args.seed)
        action_dim = len(tmp_env.valid_actions)
        tmp_env.close()
        if not args.driver_weights:
            raise ValueError("--driver policy requires --driver-weights")
        driver_policy = DQNPolicy(args.driver_weights, action_dim, device=args.device)

    # Collect snapshot bank
    bank = StateBank(env_name=args.env, seed=args.seed, noop_after_reset=args.noop_after_reset)
    snapshots = bank.collect(num_states=args.bank_size, driver=args.driver, driver_policy=driver_policy)

    # Evaluate
    arena = ShadowArena(env_name=args.env, device=args.device,
                        prefill_noop=args.prefill_noop, max_steps_per_point=args.max_steps)
    arena.evaluate_models(weights_dict, snapshots, csv_path=args.csv,
                          seed=args.seed + 654, fallback_points=args.fallback_points)


if __name__ == "__main__":
    main()