# Conveyor Simulator (MSE433 Module 3)

Conveyor simulation and scheduling for the IDEAS Clinic case. The system models a ring of four conveyors: orders require items (eight shapes: circle, pentagon, trapezoid, triangle, star, moon, heart, cross) that are loaded from totes onto the conveyor and picked at conveyor stations. The goal is to minimize **last-order completion time** (time until the last order is fully picked) by choosing order sequence, conveyor assignment, tote loading order, and item order within totes.

---

## Repository structure

| Path | Description |
|------|-------------|
| `conveyor_sim.py` | Core simulator: reads conveyor input CSV, runs greedy event simulation, writes output (conv_num, shape, time). |
| `scheduler/` | Optimization and comparison: LPT, method comparison, visualizations, playbooks. |
| `data_generator/` | Output from the data generator notebook (each subfolder = one instance with `generated/` containing CSVs). |
| `example_files/` | Example input/output CSV format. |
| `results/` | Method comparison outputs (CSVs, summary, plots). Instance folders with "sample" in the name are excluded from comparisons. |

---

## Requirements

- Python 3
- Optional: `matplotlib` for comparison and line plots (`pip install matplotlib`)

---

## How to run

### 1. Conveyor simulator (standalone)

Input CSV must have columns: `conv_num`, `circle`, `pentagon`, `trapezoid`, `triangle`, `star`, `moon`, `heart`, `cross`. (Legacy files with `cirle` are also accepted.)

```bash
# Default: items staggered per conveyor (from input)
python3 conveyor_sim.py input.csv output.csv

# All items load at conveyor 0, 2.5 s apart; makespan = time to last pick
python3 conveyor_sim.py input.csv output.csv --all-load-at-conveyor-0
```

**Simulation rules:** Four conveyors in a ring; moving to the next conveyor takes 5 s; full loop 20 s. The scheduler is greedy (smallest-time event next). With `--all-load-at-conveyor-0`, the clock starts when the first item is loaded (t=0); each item is added `--load-spacing` seconds apart (default 2.5).

### 2. Generate data

Use the data generator notebook (`MSE433_M3_data_generator.ipynb`) to produce per-instance folders under `data_generator/<timestamp>/generated/` with:

- `order_itemtypes.csv` — item type IDs per order (row index = order ID)
- `order_quantities.csv` — quantities per (order, type)
- `orders_totes.csv` — tote ID per (order, type)

The scheduler turns these into conveyor-level input and runs the simulator to evaluate solutions.

### 3. Build conveyor input from generated data (LPT)

```bash
python scheduler/core.py path/to/generated/ output_input.csv
python3 conveyor_sim.py output_input.csv output_events.csv --all-load-at-conveyor-0
```

### 4. Compare all methods (recommended)

Runs all methods on every **non-sample** instance under `data_generator/` (folders whose names contain "sample" are skipped). Outputs go to `results/`.

```bash
python scheduler/compare_methods.py data_generator --out-dir results
```

Optional: better SimulatedAnnealing with restarts and hill-climb polish:

```bash
python scheduler/compare_methods.py data_generator --out-dir results --joint-restarts 2 --polish
```

Save a playbook (input.csv, events, tote/item order) per instance:

```bash
python scheduler/compare_methods.py data_generator --out-dir results --save-playbook
python scheduler/compare_methods.py data_generator --out-dir results --save-playbook --playbook-method GeneticAlgorithm
```

### 5. Single-instance joint optimization

Optimize order sequence, conveyor assignment, tote order, and item order for one generated folder (requires `orders_totes.csv`):

```bash
python scheduler/joint_solution.py path/to/generated/ --objective last_order --max-evals 800 --save-best
```

### 6. Visualizations

```bash
# Single run
python scheduler/visualize.py sim_output.csv conveyor_input.csv --out-dir scheduler

# Compare LPT vs fixed order for one generated folder
python scheduler/visualize.py --compare data_generator/<timestamp>/generated --out-dir scheduler

# Order/tote flow view
python scheduler/visualize.py --flow data_generator/<timestamp>/generated sim_output.csv --out-dir scheduler
```

### 7. Order-sequence search (hill or SA)

```bash
python scheduler/search_orders.py path/to/generated/ --method hill --save-best
python scheduler/search_orders.py path/to/generated/ --method sa --max-evals 500 --seed 42 --save-best
```

---

## Models tested

All methods minimize **last_order** (time until the last order is complete). The problem has four decision dimensions: (1) order sequence, (2) conveyor per order, (3) tote loading order, (4) item order within each tote.

| Method | Description |
|--------|-------------|
| **Baseline** | Identity order (0, 1, 2, …). Travel-aware conveyor (pos 0→conv 4, 1→3, 2→2, 3→1). Default tote/item order. |
| **BaselineRR** | Same identity order. Round-robin conveyor (pos 0→1, 1→2, 2→3, 3→4). Default tote/item order. |
| **GreedyMakespanInsertion** | Order sequence only. Greedy constructive: at each step append the order that minimizes last_order. Travel-aware conveyor; default tote/item. |
| **OrderToteHill** | Order sequence + tote loading order. First-improvement hill climb from LPT start (no conveyor or item-order moves). |
| **FullHillClimb** | All four dimensions. First-improvement hill climb from LPT (order, tote, conveyor, item-order neighborhoods). |
| **SimulatedAnnealing** | All four dimensions. Simulated annealing (optional restarts and hill-climb polish). |
| **GeneticAlgorithm** | All four dimensions. Genetic algorithm (order + conveyor + tote in chromosome; crossover, mutation, tournament selection). |
| **TabuSearch** | All four dimensions. Tabu list + aspiration; same neighborhoods as FullHillClimb. |
| **IteratedLocalSearch** | All four dimensions. Perturb solution, then full hill climb; repeat. |
| **BeamSearch** | Order sequence only. Constructive beam search (keep best B sequences per depth). Travel-aware conveyor; default tote/item. |

---

## Results folder (after running compare_methods)

| File | Description |
|------|-------------|
| `comparison.csv` | Per-instance last_order (seconds) for each method. |
| `summary.txt` | Average last_order per method, % improvement vs Baseline, wins, average moves (evaluations), average playbook events (LOAD, PICK, HOP). |
| `comparison.png` | Bar chart of average last_order by method. |
| `comparison_line.png` | Line graph of last_order by instance for Baseline, GreedyMakespanInsertion, FullHillClimb, SimulatedAnnealing, GeneticAlgorithm. |
| `comparison_moves.csv` | Per-instance evaluation/move counts per method. |
| `comparison_events.csv` | Per-instance LOAD, PICK, HOP counts per method. |
| `<method>/<instance_id>/` | (Only with `--save-playbook`) input.csv, events_playbook.txt, tote_and_item_order.txt. |

---

## Input/output format

- **Simulator input columns:** `conv_num`, `circle`, `pentagon`, `trapezoid`, `triangle`, `star`, `moon`, `heart`, `cross` (or legacy `cirle` for the first shape).
- **Simulator output columns:** `conv_num`, `shape`, `time`.
- The scheduler can write **one row per order** with `order_id`, `conv_num`, and the eight shape columns; the simulator accepts that format.

---

## Generated data format

The generator writes three CSVs (no headers). Row index = order ID. For each order row `i`, column position `k` is aligned across files:

- **order_itemtypes.csv:** `order_itemtypes[i][k]` = item type ID.
- **order_quantities.csv:** `order_quantities[i][k]` = quantity for that type.
- **orders_totes.csv:** `orders_totes[i][k]` = tote ID for that (order, type).

Totes and their contents are fixed by the data; optimization only chooses the **order** of orders, conveyors, totes, and items within totes.

---

## Optimization focus

The main optimization is how to turn generated order+tote data into conveyor-level input and load sequence. The **scheduler/** package does that and evaluates solutions with the simulator. All methods are compared on the same objective (last_order completion) and the same instances (excluding names containing "sample").
