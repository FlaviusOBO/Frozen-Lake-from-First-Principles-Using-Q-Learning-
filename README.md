# Frozen Lake — Q-Learning from First Principles

A complete, framework-free implementation of tabular **Q-Learning** that solves
the 8×8 **Frozen Lake** grid-world. The environment, agent, training loop, and
evaluation are all written from scratch in pure Python (with NumPy for array
math and matplotlib for plotting). No Gymnasium, OpenAI Gym, Stable Baselines,
or RLlib is used.

---

## Introduction

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a branch of machine learning in which an **agent**
learns to make decisions by interacting with an **environment**. At each step the
agent observes a **state**, chooses an **action**, and receives a **reward** plus
the next state. The agent's objective is to learn a **policy** — a mapping from
states to actions — that maximises the expected cumulative (discounted) reward
over time. Unlike supervised learning, there are no labelled correct answers;
the agent must discover good behaviour through trial, error, and delayed
feedback.

### What is Frozen Lake?

Frozen Lake is a classic grid-world problem. An agent must cross a frozen lake
from a **Start** tile to a **Goal** tile without falling into any **Holes**.
The map used here is the standard 8×8 layout:

```
S F F F F F F F
F F F F F F F F
F F F H F F F F
F F F H F F F F
F F F H F F F F
F H H F F F H F
F H F F H F H F
F F F H F F F G
```

| Symbol | Meaning                         |
|:------:|---------------------------------|
| `S`    | Start state                     |
| `F`    | Frozen (safe) state             |
| `H`    | Hole (terminal — episode fails) |
| `G`    | Goal (terminal — episode wins)  |

---

## Environment Design

### State representation

States are encoded as **single integer indices** in the range `[0, 63]`:

```
state_index = row * n_cols + col
```

This maps each of the 64 cells of the 8×8 grid to a row of the Q-table.

### Action representation

Four discrete actions are available:

| Action | Index |
|--------|:-----:|
| Left   | 0     |
| Down   | 1     |
| Right  | 2     |
| Up     | 3     |

Movements that would leave the grid are clipped, so the agent simply stays in
place at the boundary.

### Reward structure

| Event                  | Reward |
|------------------------|:------:|
| Reaching the Goal (`G`)| +1.0   |
| Falling into a Hole (`H`)| 0.0  |
| Any other (safe) step  | 0.0    |

Episodes terminate on reaching either a hole or the goal. With a discount factor
`gamma < 1`, the agent is implicitly encouraged to reach the goal in as few
steps as possible, since later rewards are discounted.

---

## Q-Learning Algorithm

### Description of Q-Learning

Q-Learning is a **model-free, off-policy, value-based** RL algorithm. It learns
an action-value function `Q(s, a)` — the expected discounted return of taking
action `a` in state `s` and behaving optimally thereafter. Once `Q` is learned,
the optimal policy is simply to pick the action with the highest Q-value in each
state.

### Explanation of the update equation

After each transition `(s, a, r, s')`, the Q-table is updated using the
temporal-difference rule:

```
Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]
```

- `alpha` (learning rate) — how strongly new information overrides the old
  estimate.
- `gamma` (discount factor) — how much future rewards are valued relative to
  immediate ones.
- `r + gamma * max_a' Q(s', a')` — the **TD target**, a bootstrapped estimate of
  the true value.
- The bracketed term is the **TD error**: the difference between the target and
  the current estimate.

For terminal transitions the bootstrap term `max_a' Q(s', a')` is dropped, since
there is no future return after a terminal state.

### Exploration strategy

The agent uses an **epsilon-greedy** policy: with probability `epsilon` it picks
a random action (exploration), and otherwise it picks the greedy/best action
(exploitation). `epsilon` starts high and **decays** exponentially each episode
toward a small floor, shifting the agent from broad exploration early on to
confident exploitation later. Ties between equal Q-values are broken uniformly
at random to avoid early directional bias.

---

## Training Procedure

### Hyperparameters used

| Hyperparameter        | Value   |
|-----------------------|---------|
| Episodes              | 5000    |
| Max steps per episode | 200     |
| Learning rate (alpha) | 0.1     |
| Discount factor (gamma)| 0.99   |
| Initial epsilon       | 1.0     |
| Minimum epsilon       | 0.01    |
| Epsilon decay (per episode) | 0.9995 |
| Random seed           | 42      |

These are defined at the top of `train.py` and are easy to experiment with, as
the assignment requires.

### Number of episodes

The agent trains for **5000 episodes**. The exponential epsilon schedule means
the agent explores heavily at the start and gradually commits to the policy it
has learned.

---

## Results

### Final success rate

On the deterministic map the learned **greedy policy solves the lake reliably**.
Evaluating with pure exploitation over 100 episodes yields:

```
Success Rate          : 100.00%
Average Reward        : 1.0000
Number of Failures    : 0
Number of Successful Runs : 100
```

(The success rate *during training* is lower — around the low-to-mid 90s in the
final 500 episodes — because the agent is still exploring with a non-zero
epsilon while it trains.)

### Learned policy

`train.py` prints the greedy policy as an arrow grid, for example:

```
Legend:  ← Left   ↓ Down   → Right   ↑ Up   H Hole   G Goal

↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
→  →  →  →  →  ↓  ↓  ↓
↑  ↑  ↑  H  →  →  ↓  ↓
↑  ↑  ↑  H  →  →  →  ↓
↑  ↑  ↑  H  →  →  →  ↓
↑  H  H  →  →  ↑  H  ↓
↑  H  ←  ←  H  ↓  H  ↓
←  ←  ←  H  ←  →  →  G
```

Holes and the goal are shown as `H`/`G` since no action is taken from a terminal
state. Arrows on unreachable / "don't care" cells are simply the argmax of an
untrained row and can be ignored.

### Discussion of performance

The TD-error magnitude shrinks as training progresses and the policy stabilises
into a safe corridor that routes around the column-3 holes and the cluster in
the bottom-left, then descends the right-hand side to the goal. The bonus plots
(`results/success_rate.png` and `results/epsilon_decay.png`) show the
moving-average success rate climbing steadily as epsilon decays, illustrating
the classic exploration-to-exploitation transition.

---

## Project Structure

```
frozen-lake-qlearning/
├── environment.py     # FrozenLakeEnv: reset / step / render / get_state / is_terminal
├── agent.py           # QLearningAgent: Q-table, epsilon-greedy, epsilon decay, update
├── train.py           # training loop, policy extraction, bonus plots, Q-table saving
├── evaluate.py        # greedy evaluation over 100 episodes
├── requirements.txt   # numpy, matplotlib
├── README.md          # this file
└── results/           # saved Q-table and generated plots
```

---

## Execution Instructions

1. **(Optional) create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate      # Windows: .venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the agent** (saves the Q-table and plots into `results/`)
   ```bash
   python train.py
   ```

4. **Evaluate the trained agent** (100 greedy episodes)
   ```bash
   python evaluate.py
   ```

You can also run `python environment.py` on its own for a quick sanity check of
the grid and a sample step.

---

## Bonus Task Implemented

**Option B — Visualisation.** `train.py` produces two figures via matplotlib:

- `results/success_rate.png` — moving-average success rate over episodes.
- `results/epsilon_decay.png` — epsilon value over episodes.

> The environment also ships with an optional `is_slippery` flag
> (`FrozenLakeEnv(is_slippery=True)`) that introduces stochastic transitions,
> providing a hook for Bonus **Option A** as well.
