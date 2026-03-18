# LPT (Longest Processing Time First) Approach

## Optimization objective

The **purpose of the optimization** is to choose:

1. **Optimal order sequence** — In what order should orders be processed (which order is 1st, 2nd, 3rd, …)? Each conveyor gets orders in round-robin (conv 0 → orders at positions 0, 4, 8, …; conv 1 → 1, 5, 9, …; etc.).
2. **Optimal tote loading order** — In what order should totes be loaded onto the conveyor? (Which tote first, second, … so that items arrive in a good sequence for the sim.)

**Totes are fixed by the data** (orders_totes.csv): which items are in which tote is given. We do not create or change totes; we only **order the totes** (which tote to empty first, second, …) and **order the items within each tote**.

**Goal:** Minimize **makespan** (time until the last order is complete).

**What this implementation does:**  
- **Order sequence:** We use **LPT** (Longest Processing Time first) as a heuristic: sort orders by total item count **descending**, then assign in that order to conveyors 0–3 in round-robin. That fixes which order is 1st, 2nd, … and thus which conveyor each order’s items go to. You can replace or wrap LPT with search (e.g. local search, simulated annealing) to improve makespan.  
- **Tote loading order:** When **orders_totes.csv** is present, the simulator is given an **explicit load sequence**: we order totes by earliest LPT position, then by farthest conveyor (feed conv 3 first), then tote ID; one tote is fully emptied before the next. That sequence is passed to the sim so the real load order matches our tote order.  
- **Item order within each tote:** For each tote we output items in the order they appear in the data (you can override this via `item_order_per_tote` in `build_load_sequence` to optimize the order items from a tote are placed on the belt).

---

## What’s in this folder

An LPT-based scheduler that converts data generator output into M3_Example input format, plus **visualizations** to evaluate makespan and flow.

- **Input:** Data generator CSVs (`order_itemtypes.csv`, `order_quantities.csv`, and optionally `orders_totes.csv` for tote/loading view).
- **Output:** Conveyor input CSV in M3_Example format (`conv_num` + 8 shape columns).
- **Rule (LPT):** Orders are sorted by total item count (processing time) **descending**, then assigned to conveyors in round-robin order. Each conveyor’s shape counts are the sum of its assigned orders’ demands.
- **Travel-time aware:** Items entering at conveyor 0 take longest to reach conveyor 3 (3 hops × 5 s). So we assign **position 0 (largest order) → conveyor 3**, position 1 → conv 2, position 2 → conv 1, position 3 → conv 0, then repeat. That way the largest orders get the farthest conveyor and their items start circulating soonest. **Tote loading order** also prefers totes that feed conveyor 3 (then 2, 1, 0) so those items are loaded and emptied first.

---

## How to tell if the algorithm is good

1. **Compare to a baseline** — Run LPT vs **fixed order** (orders 0,1,2,…) on the same data. Lower makespan = better. Use the comparison visualizer (see below).
2. **Look at the plots:**
   - **Cumulative picks over time** — Steeper curve = more picks per second; makespan is the time when the curve flattens.
   - **Cumulative picks per conveyor** — Conveyors that finish earlier free up “slots”; balanced curves often mean better makespan.
   - **Load per conveyor** — LPT aims to balance load; similar bar heights suggest balanced assignment.
3. **Run on multiple instances** — Generate several data sets (different seeds), run LPT and baseline on each, and compare average makespan and improvement %.

**Improving the optimization:**  
- **Order sequence:** Use `search_orders.py` (hill or SA) or add your own search over the order permutation.  
- **Tote loading order:** The sim now accepts an explicit load sequence (used when `orders_totes.csv` exists). You can try other tote orderings (e.g. different tie-breaks) and pass the resulting sequence to the sim.  
- **Item order within tote:** Pass `item_order_per_tote[tote_id]` = list of (order_id, shape, qty) in desired load order to `build_load_sequence`; then search over permutations of items within each tote to minimize makespan.

---

## Visualizations

From project root:

**Single run (one sim output):**  
Plots cumulative picks over time, picks per conveyor over time, and load balance from the input CSV.

```bash
python scheduler/visualize.py scheduler/simulated_output.csv scheduler/conveyor_input.csv --out-dir scheduler
```

Output: `scheduler/visualization_single.png`

**Compare LPT vs fixed order (baseline):**  
Builds both inputs from a generated folder, runs the sim for both (using **all load at conveyor 0**, 2.5 s spacing), and plots makespan comparison + cumulative-pick curves.

```bash
python scheduler/visualize.py --compare data_generator/2026-02-26_16-18-01/generated --out-dir scheduler
```

Output: `scheduler/visualization_compare.png` and printed makespan improvement %.

**Order-sequence search (improve beyond LPT):**  
Hill climbing or simulated annealing over order permutations; each sequence is evaluated with the sim (all load at conv 0, 2.5s spacing).

```bash
python scheduler/search_orders.py path/to/generated/ --method hill --save-best
python scheduler/search_orders.py path/to/generated/ --method sa --max-evals 500 --seed 42 --save-best
```

Output: best makespan, best order sequence, and optionally `scheduler/search_best_input.csv`.

**Method comparison (same objective: last_order):**  
Run Baseline, BaselineRR, GreedyMakespanInsertion, OrderToteHill, FullHillClimb, SimulatedAnnealing, GeneticAlgorithm, TabuSearch, IteratedLocalSearch, and BeamSearch on every instance; all evaluated with **last_order completion**. All outputs go into the **results/** folder.

`BranchAndBound` is excluded by default because it is much slower than the other methods.

```bash
python scheduler/compare_methods.py data_generator --out-dir results
# Better SimulatedAnnealing (e.g. 92.5s): add --joint-restarts 2 --polish
python scheduler/compare_methods.py data_generator --out-dir results --joint-restarts 2 --polish
# Include BranchAndBound only when you explicitly want it
python scheduler/compare_methods.py data_generator --out-dir results --include-branch-and-bound
```

Defaults: 800 evals per joint method, 2 restarts for SA (take best), optional `--polish` (hill climb after SA). Outputs in `results/`: `comparison.csv`, `summary.txt`, `comparison.png`. If you use `--save-playbook`, each playbook folder includes `input_conveyor.csv` in the M3 example format.

**Joint optimization (order + conveyor + tote order + item order):**  
Search over all four decision layers to minimize **last-order completion time**. Requires `orders_totes.csv`.

```bash
python scheduler/joint_solution.py path/to/generated/ --objective last_order --max-evals 800 --save-best
```

Output: best objective value, best order sequence and tote loading order; optionally `joint_best_input.csv`, `joint_best_playbook.txt`, and `joint_best_solution.txt`.

**Multi-instance validation:**  
Run **`compare_methods.py`** on every `data_generator/<timestamp>/generated/` folder; it aggregates makespans and prints summary (avg per method, % improvement vs Baseline). See Method comparison above.

**Order/tote flow (sequence, assignment, when orders complete, totes):**  
Shows (1) **LPT order sequence** — which order is 1st, 2nd, … and which conveyor it’s on; (2) **Orders per conveyor**; (3) **Gantt chart** — estimated completion time per order; (4) **Tote view** (if `orders_totes.csv` exists): **which tote** helps which orders and which conveyor(s) those items go to, and **loading by LPT position** — at each position, which totes you load for that order and what items they contain.

```bash
python scheduler/visualize.py --flow data_generator/2026-02-26_16-18-01/generated scheduler/simulated_output.csv --out-dir scheduler
```

Output: `scheduler/visualization_flow.png` and `scheduler/tote_and_loading_summary.txt`.

*(Requires `matplotlib`: `pip install matplotlib`)*

**Full event playbook (for validation):**  
Writes every event (LOAD, HOP, PICK) in chronological order so you can validate timing and logic.

```bash
python scheduler/write_event_playbook.py path/to/generated/ --out scheduler/event_playbook.txt --csv scheduler/event_playbook.csv
```

See **`scheduler/VALIDATION_PLAYBOOK.md`** for event types, order rules, and a validation checklist.

---

## Contents of this folder

**Code:** `core.py`, `compare_methods.py`, `search_orders.py`, `joint_solution.py`, `write_event_playbook.py`, `visualize.py`, `__init__.py`

**Docs:** `README.md`, `VALIDATION_PLAYBOOK.md`

**Generated when you run the scripts:** With `--out-dir scheduler` (or default): conveyor input CSVs, sim output, visualizations, playbooks, etc. With `--out-dir results`: **`compare_methods.py`** writes comparison outputs to **`results/`** (`comparison.csv`, `summary.txt`, `comparison.png`) and playbooks to **`results/<method>/<instance_id>/`**.

---

## Commands (from project root)

```bash
# Generate M3 input from a data-generator run
python scheduler/core.py path/to/generated/ scheduler/conveyor_input.csv

# Run the conveyor sim on that input (use --all-load-at-conveyor-0 to match comparison/flow model)
python conveyor_sim.py scheduler/conveyor_input.csv scheduler/simulated_output.csv --all-load-at-conveyor-0

# Visualize (single, compare, or flow — see above)
python scheduler/visualize.py scheduler/simulated_output.csv scheduler/conveyor_input.csv --out-dir scheduler
python scheduler/visualize.py --compare path/to/generated/ --out-dir scheduler
python scheduler/visualize.py --flow path/to/generated/ scheduler/simulated_output.csv --out-dir scheduler

# Order-sequence search (hill or SA)
python scheduler/search_orders.py path/to/generated/ --method hill --save-best

# Method comparison (default set excludes BranchAndBound because it is slow)
python scheduler/compare_methods.py data_generator --out-dir results

# Joint optimization (order + conveyor + tote + item order)
python scheduler/joint_solution.py path/to/generated/ --objective last_order --save-best
```
