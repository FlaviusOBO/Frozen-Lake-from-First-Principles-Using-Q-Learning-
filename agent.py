"""
agent.py
========

A from-scratch tabular Q-Learning agent.

The agent maintains a Q-table of shape (n_states, n_actions) and learns using
the standard temporal-difference update:

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Exploration uses an epsilon-greedy policy with exponential epsilon decay.
"""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-Learning agent with an epsilon-greedy behaviour policy."""

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.9995,
        seed: int | None = None,
    ) -> None:
        """Initialise the agent and its Q-table.

        Parameters
        ----------
        n_states, n_actions:
            Dimensions of the Q-table.
        alpha:
            Learning rate (step size) in [0, 1].
        gamma:
            Discount factor in [0, 1].
        epsilon:
            Initial exploration probability.
        epsilon_min:
            Lower bound that epsilon decays towards.
        epsilon_decay:
            Multiplicative decay factor applied to epsilon each episode.
        seed:
            Optional seed for reproducible action selection.
        """
        self.n_states = n_states
        self.n_actions = n_actions

        self.alpha = alpha
        self.gamma = gamma

        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table initialised to zeros: optimistic-neutral start.
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float64)

        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ #
    # Action selection
    # ------------------------------------------------------------------ #
    def _argmax_random_tiebreak(self, state: int) -> int:
        """Return the greedy action, breaking ties uniformly at random.

        Plain ``np.argmax`` always returns the first maximal index, which biases
        the policy early in training when many Q-values are still equal (zero).
        """
        q_values = self.q_table[state]
        best = np.flatnonzero(q_values == q_values.max())
        return int(self._rng.choice(best))

    def choose_action(self, state: int, greedy: bool = False) -> int:
        """Select an action for ``state``.

        Parameters
        ----------
        state:
            Current integer state index.
        greedy:
            If True, always exploit (used during evaluation). If False, use the
            epsilon-greedy rule.
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))  # explore
        return self._argmax_random_tiebreak(state)          # exploit

    # ------------------------------------------------------------------ #
    # Learning
    # ------------------------------------------------------------------ #
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        """Apply the Q-Learning update for one transition.

        Implements exactly:

            Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

        For terminal transitions the bootstrap term ``max_a' Q(s', a')`` is
        dropped (there is no future return after a terminal state).
        """
        best_next = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

    def decay_epsilon(self) -> None:
        """Multiplicatively decay epsilon, clamped at ``epsilon_min``."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        """Persist the Q-table to disk as a NumPy ``.npy`` file."""
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        """Load a Q-table previously saved with :meth:`save`."""
        self.q_table = np.load(path)
