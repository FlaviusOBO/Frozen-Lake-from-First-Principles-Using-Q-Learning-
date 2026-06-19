"""
environment.py
==============

A from-scratch implementation of the 8x8 Frozen Lake grid-world environment.

No external reinforcement-learning frameworks (Gymnasium, OpenAI Gym, Stable
Baselines, RLlib, etc.) are used. The environment is built entirely with
standard Python and NumPy.

Map legend
----------
    S : Start state
    F : Frozen (safe) state
    H : Hole (terminal state, episode fails)
    G : Goal (terminal state, episode succeeds)

Action encoding
---------------
    0 = Left
    1 = Down
    2 = Right
    3 = Up

State representation
--------------------
States are encoded as single integer indices in the range [0, n_states - 1],
where:

    state_index = row * n_cols + col

This is convenient for indexing the Q-table directly.
"""

from __future__ import annotations

import random
from typing import List, Tuple


class FrozenLakeEnv:
    """A deterministic (optionally stochastic) 8x8 Frozen Lake environment."""

    # ---- Action constants ---------------------------------------------------
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    ACTIONS = (LEFT, DOWN, RIGHT, UP)

    # ---- The standard 8x8 map specified in the assignment -------------------
    DEFAULT_MAP: List[str] = [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ]

    # Human-readable arrows for rendering a policy.
    ACTION_ARROWS = {LEFT: "\u2190", DOWN: "\u2193", RIGHT: "\u2192", UP: "\u2191"}

    def __init__(
        self,
        grid: List[str] | None = None,
        is_slippery: bool = False,
        slip_prob: float = 1.0 / 3.0,
        seed: int | None = None,
    ) -> None:
        """Initialise the environment.

        Parameters
        ----------
        grid:
            Optional list of equal-length strings describing the map. Defaults
            to the standard 8x8 assignment map.
        is_slippery:
            If True, transitions become stochastic (Bonus Option A). The agent
            moves in the intended direction with probability ``1 - slip_prob``
            and slips to one of the two perpendicular directions otherwise.
        slip_prob:
            Total probability of slipping when ``is_slippery`` is True.
        seed:
            Optional seed for reproducible stochastic transitions.
        """
        self.grid = grid if grid is not None else list(self.DEFAULT_MAP)
        self.n_rows = len(self.grid)
        self.n_cols = len(self.grid[0])

        # Validate that the map is rectangular.
        if any(len(row) != self.n_cols for row in self.grid):
            raise ValueError("All rows in the map must have equal length.")

        self.n_states = self.n_rows * self.n_cols
        self.n_actions = len(self.ACTIONS)

        self.is_slippery = is_slippery
        self.slip_prob = slip_prob

        self._rng = random.Random(seed)

        # Locate the start, goal and hole cells once at construction time.
        self.start_state = self._find_tile("S")
        self.goal_state = self._find_tile("G")
        self.hole_states = {
            self._to_state(r, c)
            for r in range(self.n_rows)
            for c in range(self.n_cols)
            if self.grid[r][c] == "H"
        }

        # Current agent state (set properly by reset()).
        self.state: int = self.start_state

    # ------------------------------------------------------------------ #
    # Coordinate / state helpers
    # ------------------------------------------------------------------ #
    def _to_state(self, row: int, col: int) -> int:
        """Convert (row, col) coordinates into a single integer state index."""
        return row * self.n_cols + col

    def _to_coords(self, state: int) -> Tuple[int, int]:
        """Convert an integer state index back into (row, col) coordinates."""
        return divmod(state, self.n_cols)

    def _find_tile(self, symbol: str) -> int:
        """Return the integer state index of the first cell matching ``symbol``."""
        for r in range(self.n_rows):
            for c in range(self.n_cols):
                if self.grid[r][c] == symbol:
                    return self._to_state(r, c)
        raise ValueError(f"Tile '{symbol}' not found in the map.")

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def reset(self) -> int:
        """Reset the agent to the start state and return that state."""
        self.state = self.start_state
        return self.state

    def get_state(self) -> int:
        """Return the agent's current integer state index."""
        return self.state

    def is_terminal(self, state: int | None = None) -> bool:
        """Return True if ``state`` (default: current state) is terminal."""
        if state is None:
            state = self.state
        return state == self.goal_state or state in self.hole_states

    def _move(self, state: int, action: int) -> int:
        """Apply a single deterministic move, respecting grid boundaries.

        If a move would leave the grid, the agent stays in place.
        """
        row, col = self._to_coords(state)
        if action == self.LEFT:
            col = max(col - 1, 0)
        elif action == self.DOWN:
            row = min(row + 1, self.n_rows - 1)
        elif action == self.RIGHT:
            col = min(col + 1, self.n_cols - 1)
        elif action == self.UP:
            row = max(row - 1, 0)
        else:
            raise ValueError(f"Invalid action: {action}")
        return self._to_state(row, col)

    def _resolve_action(self, action: int) -> int:
        """Resolve the executed action, accounting for slipperiness.

        Under the slippery model, the intended action is taken with probability
        ``1 - slip_prob``; otherwise the agent slips to one of the two
        directions perpendicular to the intended one (each equally likely).
        """
        if not self.is_slippery:
            return action

        if self._rng.random() >= self.slip_prob:
            return action

        # Perpendicular directions: Left/Right are perpendicular to Up/Down.
        if action in (self.LEFT, self.RIGHT):
            perpendicular = (self.UP, self.DOWN)
        else:
            perpendicular = (self.LEFT, self.RIGHT)
        return self._rng.choice(perpendicular)

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        """Take ``action`` from the current state.

        Returns
        -------
        next_state : int
            The resulting integer state index.
        reward : float
            +1.0 for reaching the goal, 0.0 otherwise.
        done : bool
            True if the episode has ended (goal or hole reached).
        info : dict
            Auxiliary diagnostic information.
        """
        if action not in self.ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        # If we are already in a terminal state, stepping is a no-op.
        if self.is_terminal(self.state):
            return self.state, 0.0, True, {"terminal": True}

        executed_action = self._resolve_action(action)
        next_state = self._move(self.state, executed_action)
        self.state = next_state

        if next_state == self.goal_state:
            reward, done = 1.0, True
        elif next_state in self.hole_states:
            reward, done = 0.0, True
        else:
            reward, done = 0.0, False

        info = {"executed_action": executed_action, "intended_action": action}
        return next_state, reward, done, info

    # ------------------------------------------------------------------ #
    # Rendering
    # ------------------------------------------------------------------ #
    def render(self, mode: str = "human") -> str:
        """Render the grid with the agent's position marked by ``*``.

        Returns the rendered string (also prints it when ``mode == 'human'``).
        """
        agent_row, agent_col = self._to_coords(self.state)
        lines = []
        for r in range(self.n_rows):
            row_chars = []
            for c in range(self.n_cols):
                if r == agent_row and c == agent_col:
                    row_chars.append("*")
                else:
                    row_chars.append(self.grid[r][c])
            lines.append(" ".join(row_chars))
        rendered = "\n".join(lines)
        if mode == "human":
            print(rendered)
            print()
        return rendered


if __name__ == "__main__":
    # Quick manual sanity check of the environment.
    env = FrozenLakeEnv()
    print(f"States: {env.n_states}, Actions: {env.n_actions}")
    print(f"Start: {env.start_state}, Goal: {env.goal_state}")
    print(f"Holes: {sorted(env.hole_states)}\n")
    env.reset()
    env.render()
    s, r, d, info = env.step(FrozenLakeEnv.RIGHT)
    print(f"After moving RIGHT -> state={s}, reward={r}, done={d}, info={info}")
    env.render()
