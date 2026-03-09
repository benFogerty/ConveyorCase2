# MCTS Tote-Sequence Solver

This folder contains a Monte Carlo Tree Search (MCTS) solver that optimizes **tote loading order** to minimize **last-order completion time (makespan)**. It plugs into the existing scheduler and exports inputs compatible with the **M3 example format**.

## What It Optimizes

**Primary decision (MCTS actions):**
- The next tote to load (tote loading order).

**Deterministic decoding (default):**
- **Order sequence:** LPT (largest total item count first).
- **Conveyor assignment:** travel-aware mapping (pos 0→conv 3, 1→2, 2→1, 3→0, repeat).
- **Item order within tote:** sorted by earliest order position, then shape id.

This keeps branching manageable while still optimizing tote order with rollouts.

## How It Works

- **Selection:** UCB1 (minimization handled by reward transform).
- **Expansion:** top-K tote candidates by heuristic score (earliest order, farthest conveyor). Default `K=20`.
- **Simulation:** greedy-eval rollout (choose next tote by lowest evaluated makespan) by default; heuristic or random available.
- **Backpropagation:** reward is `-makespan` by default (or reciprocal mode).
- **Budget:** supports `n_iterations` and/or `time_limit_seconds`.

## Usage (Integrated)

MCTS is included in `scheduler/compare_methods.py`.

```bash
python scheduler/compare_methods.py data_generator --out-dir results
```

To also write playbooks (including M3 example-format input):

```bash
python scheduler/compare_methods.py data_generator --out-dir results --save-playbook --playbook-method MCTS
```

Playbooks include:
- `input.csv` (per-order FIFO format)
- `input_conveyor.csv` (M3 example format)
- `tote_and_item_order.txt` (manual tote loading sequence)

## Usage (Direct)

```python
from bens_methods.mcts_solver import solve, export_m3_input
from scheduler.joint_solution import evaluate_solution

# orders, tote_contents from generator files
# evaluate_solution is the deterministic simulator wrapper

sol, ms = solve(orders, tote_contents, evaluate=lambda s: evaluate_solution(s, orders, tote_contents))
export_m3_input(sol, orders, "input_conveyor.csv", format="per_conveyor")
```

## Configuration

You can override defaults via `MCTSConfig`:

```python
from bens_methods.mcts_solver import MCTSConfig

cfg = MCTSConfig(
    c_ucb=1.4,
    top_k_candidates=20,
    reward_mode="negative",  # or "reciprocal"
    rollout_policy="greedy_eval",  # or "heuristic" / "random"
    log_improvements=True,
)
```

## Notes on Compatibility

- **Example-format input:** use `export_m3_input(..., format="per_conveyor")` to write `conv_num,cirle,pentagon,...`.
- **FIFO evaluation:** `input.csv` (per-order) is still required for accurate order completion metrics.

## Files

- `mcts_solver.py` — MCTS implementation and export helper.
- `README.md` — this file.
