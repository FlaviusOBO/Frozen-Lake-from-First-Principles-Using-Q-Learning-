"""
evaluate.py
===========

Evaluate a trained Frozen Lake Q-Learning agent.

The agent acts *greedily* (pure exploitation, no exploration) over a fixed
number of episodes (default: 100) and the following metrics are reported:

    * Success Rate (%)
    * Average Reward
    * Number of Failures
    * Number of Successful Runs

The trained Q-table is loaded from ``results/q_table.npy`` (produced by
``train.py``). Run with::

    python evaluate.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

from agent import QLearningAgent
from environment import FrozenLakeEnv

NUM_EVAL_EPISODES = 100
MAX_STEPS_PER_EPISODE = 200
RESULTS_DIR = "results"
Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.npy")
SEED = 123  # different from training seed so evaluation is an honest test


def evaluate(
    q_table_path: str = Q_TABLE_PATH,
    num_episodes: int = NUM_EVAL_EPISODES,
    render_first: bool = False,
) -> dict:
    """Run greedy evaluation episodes and return a metrics dictionary."""
    if not os.path.exists(q_table_path):
        print(
            f"[error] Q-table not found at '{q_table_path}'.\n"
            f"        Run 'python train.py' first to train and save the agent."
        )
        sys.exit(1)

    env = FrozenLakeEnv(is_slippery=False, seed=SEED)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        seed=SEED,
    )
    agent.load(q_table_path)

    rewards = []
    successes = 0

    for episode in range(1, num_episodes + 1):
        state = env.reset()
        episode_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state, greedy=True)  # exploit only
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
            if done:
                break

        rewards.append(episode_reward)
        if episode_reward > 0.0:
            successes += 1

        if render_first and episode == 1:
            print("Final state of the first evaluation episode:")
            env.render()

    failures = num_episodes - successes
    success_rate = 100.0 * successes / num_episodes
    average_reward = float(np.mean(rewards))

    metrics = {
        "episodes": num_episodes,
        "success_rate_percent": success_rate,
        "average_reward": average_reward,
        "successful_runs": successes,
        "failures": failures,
    }

    print("=" * 60)
    print(f"Evaluation over {num_episodes} episodes (greedy / no exploration)")
    print("=" * 60)
    print(f"Success Rate          : {success_rate:.2f}%")
    print(f"Average Reward        : {average_reward:.4f}")
    print(f"Number of Failures    : {failures}")
    print(f"Number of Successful Runs : {successes}")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    evaluate(render_first=True)
