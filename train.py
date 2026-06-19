"""
train.py
========

Training entry point for the Frozen Lake Q-Learning agent.

Responsibilities
----------------
1. Build the environment and agent.
2. Run the training loop for a configurable number of episodes.
3. Record per-episode statistics: reward, success flag, epsilon, steps.
4. Extract the learned greedy policy and print it as an arrow grid.
5. (Bonus Option B) Plot the moving-average success rate and the epsilon decay
   curve, saving the figures into the ``results/`` directory.
6. Persist the trained Q-table to ``results/q_table.npy`` for evaluation.

Run with::

    python train.py
"""

from __future__ import annotations

import os
from typing import List

import numpy as np

from agent import QLearningAgent
from environment import FrozenLakeEnv

# --------------------------------------------------------------------------- #
# Hyperparameters (feel free to experiment with these values)
# --------------------------------------------------------------------------- #
NUM_EPISODES = 5000
MAX_STEPS_PER_EPISODE = 200      # safety cap to prevent infinite wandering

ALPHA = 0.1                      # learning rate
GAMMA = 0.99                     # discount factor
EPSILON_START = 1.0              # initial exploration probability
EPSILON_MIN = 0.01               # minimum exploration probability
EPSILON_DECAY = 0.9995           # per-episode multiplicative decay

SEED = 42                        # for reproducibility
RESULTS_DIR = "results"
Q_TABLE_PATH = os.path.join(RESULTS_DIR, "q_table.npy")


def moving_average(values: List[float], window: int) -> np.ndarray:
    """Compute a simple moving average over ``values`` with the given window."""
    if window <= 1 or len(values) < window:
        return np.asarray(values, dtype=float)
    cumulative = np.cumsum(np.insert(values, 0, 0.0))
    return (cumulative[window:] - cumulative[:-window]) / float(window)


def extract_policy(agent: QLearningAgent, env: FrozenLakeEnv) -> np.ndarray:
    """Derive the greedy policy (best action per state) from the Q-table."""
    return np.argmax(agent.q_table, axis=1)


def render_policy_grid(agent: QLearningAgent, env: FrozenLakeEnv) -> str:
    """Build a human-readable arrow grid for the learned policy.

    Terminal cells are shown as their map symbol (``H`` for holes, ``G`` for the
    goal) rather than an action arrow, since no action is taken from them.
    """
    policy = extract_policy(agent, env)
    lines = []
    for r in range(env.n_rows):
        symbols = []
        for c in range(env.n_cols):
            state = r * env.n_cols + c
            if state == env.goal_state:
                symbols.append("G")
            elif state in env.hole_states:
                symbols.append("H")
            else:
                symbols.append(env.ACTION_ARROWS[int(policy[state])])
        lines.append("  ".join(symbols))
    return "\n".join(lines)


def plot_results(
    success_history: List[int],
    epsilon_history: List[float],
    window: int = 100,
) -> None:
    """Bonus Option B: plot success-rate moving average and epsilon decay.

    Saved as ``results/success_rate.png`` and ``results/epsilon_decay.png``.
    matplotlib is imported lazily so the rest of the pipeline still runs even
    if matplotlib is not installed.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless-safe backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warning] matplotlib not available - skipping plots.")
        return

    # --- Success-rate moving average ---
    success_ma = moving_average(success_history, window)
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(success_ma)), success_ma, color="#c52a6b", linewidth=1.8)
    plt.title(f"Success Rate (moving average, window={window})")
    plt.xlabel("Episode")
    plt.ylabel("Success rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    success_path = os.path.join(RESULTS_DIR, "success_rate.png")
    plt.savefig(success_path, dpi=120)
    plt.close()

    # --- Epsilon decay ---
    plt.figure(figsize=(9, 5))
    plt.plot(range(len(epsilon_history)), epsilon_history, color="#1f77b4", linewidth=1.8)
    plt.title("Epsilon Decay Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Epsilon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    epsilon_path = os.path.join(RESULTS_DIR, "epsilon_decay.png")
    plt.savefig(epsilon_path, dpi=120)
    plt.close()

    print(f"Saved plots to '{success_path}' and '{epsilon_path}'.")


def train() -> QLearningAgent:
    """Run the full training procedure and return the trained agent."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    env = FrozenLakeEnv(is_slippery=False, seed=SEED)
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=ALPHA,
        gamma=GAMMA,
        epsilon=EPSILON_START,
        epsilon_min=EPSILON_MIN,
        epsilon_decay=EPSILON_DECAY,
        seed=SEED,
    )

    # --- Statistics trackers ---
    reward_history: List[float] = []
    success_history: List[int] = []   # 1 if the goal was reached, else 0
    epsilon_history: List[float] = []
    steps_history: List[int] = []

    print("=" * 60)
    print("Training Q-Learning agent on 8x8 Frozen Lake")
    print("=" * 60)
    print(
        f"Episodes={NUM_EPISODES} | alpha={ALPHA} | gamma={GAMMA} | "
        f"epsilon={EPSILON_START}->{EPSILON_MIN} (decay={EPSILON_DECAY})\n"
    )

    for episode in range(1, NUM_EPISODES + 1):
        state = env.reset()
        episode_reward = 0.0
        reached_goal = False

        for step in range(MAX_STEPS_PER_EPISODE):
            action = agent.choose_action(state, greedy=False)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

            if done:
                reached_goal = reward > 0.0
                break

        agent.decay_epsilon()

        reward_history.append(episode_reward)
        success_history.append(1 if reached_goal else 0)
        epsilon_history.append(agent.epsilon)
        steps_history.append(step + 1)

        # Periodic progress report.
        if episode % 500 == 0:
            recent_success = np.mean(success_history[-500:])
            print(
                f"Episode {episode:>5} | "
                f"recent success rate (last 500): {recent_success:6.2%} | "
                f"epsilon: {agent.epsilon:.4f}"
            )

    # --- Summary ---
    total_successes = int(np.sum(success_history))
    print("\n" + "-" * 60)
    print("Training complete.")
    print(f"Total successful episodes : {total_successes} / {NUM_EPISODES}")
    print(f"Overall success rate      : {total_successes / NUM_EPISODES:.2%}")
    print(f"Final epsilon             : {agent.epsilon:.4f}")
    print("-" * 60 + "\n")

    # --- Learned policy ---
    print("Learned policy (greedy action per state):")
    print("Legend:  \u2190 Left   \u2193 Down   \u2192 Right   \u2191 Up   H Hole   G Goal\n")
    print(render_policy_grid(agent, env))
    print()

    # --- Persist Q-table and plots ---
    agent.save(Q_TABLE_PATH)
    print(f"Saved trained Q-table to '{Q_TABLE_PATH}'.")
    plot_results(success_history, epsilon_history, window=100)

    return agent


if __name__ == "__main__":
    train()
