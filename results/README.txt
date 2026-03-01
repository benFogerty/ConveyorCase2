This folder holds outputs from the method comparison (same objective: last_order completion).

================================================================================
METHODS (what each optimizes)
================================================================================
All methods minimize last_order (time until the last order is complete). The
problem has four decision dimensions: (1) order sequence, (2) conveyor per order,
(3) tote loading order, (4) item order within each tote.

  Baseline              Identity order (0,1,2,...). Travel-aware conveyor assignment
                        (pos 0→conv 3, 1→2, 2→1, 3→0). Default tote/item order.

  BaselineRR            Same identity order (0,1,2,...). Round-robin conveyor assignment
                        (pos 0→conv 0, 1→1, 2→2, 3→0). Default tote/item order.

  OrderToteHill         Order sequence + tote loading order only (no conveyor
                        or item-order moves). First-improvement hill climb from LPT start.

  FullHillClimb         All four dimensions. First-improvement hill climb from LPT
                        (order, tote, conveyor, item-order neighborhoods).

  SimulatedAnnealing    All four dimensions. Simulated annealing (optional
                        restarts and hill-climb polish).

  GeneticAlgorithm      All four dimensions. Genetic algorithm (order + conveyor in
                        chromosome; tote/item order from decoding).

  TabuSearch             All four dimensions. Tabu search (tabu list + aspiration;
                        same neighborhoods as FullHillClimb).

  IteratedLocalSearch   All four dimensions. Iterated local search (perturb
                        solution, then full hill climb; repeat).

================================================================================
HOW TO RUN
================================================================================
  python scheduler/compare_methods.py data_generator --out-dir results

For best SimulatedAnnealing results: use --joint-restarts 2 and --polish:
  python scheduler/compare_methods.py data_generator --out-dir results --joint-restarts 2 --polish

To generate a playbook folder per instance (input.csv, events, tote/item order):
  python scheduler/compare_methods.py data_generator --out-dir results --save-playbook
  python scheduler/compare_methods.py data_generator --out-dir results --save-playbook --playbook-method GeneticAlgorithm

================================================================================
FILES
================================================================================
  comparison.csv   - Per-instance last_order (s) for each method
  summary.txt      - Average per method, % improvement, wins
  comparison.png   - Bar chart of average last_order by method

  <method>/<instance_id>/ - (only when --save-playbook) Playbooks grouped by method:
    e.g. results/GeneticAlgorithm/2026-02-26_16-18-01/
    input.csv              - Conveyor input in M3_Example format (conv_num + 8 shape columns)
    events_playbook.txt    - Chronological LOAD, HOP, PICK events
    tote_and_item_order.txt - Order totes are loaded; items within each tote
