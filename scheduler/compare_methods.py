#!/usr/bin/env python3
"""
Compare multiple solution methods under the same objective (last_order completion).
All outputs go into a single results folder.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scheduler.beam_search import beam_search_order_sequence

from conveyor_sim import SHAPE_COLUMNS, load_conveyors, simulate_greedy
from scheduler.core import (
    build_load_sequence,
    load_generator_data,
    load_tote_data,
    write_m3_input_by_order,
)
from scheduler.joint_solution import (
    Solution,
    solution_from_order_sequence,
    initial_solution,
    evaluate_solution,
    hill_climb_order_and_tote,
    hill_climb_full,
    simulated_annealing,
    genetic_algorithm,
    tabu_search,
    iterated_local_search,
    order_demand,
    orders_on_conveyor,
)

from scheduler.search_orders import (
    greedy_makespan_insertion,
)
from scheduler.branch_and_bound import (
    BranchAndBoundConfig,
    solve as branch_and_bound_solve,
    export_m3_input,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


DEFAULT_METHODS = [
    "Baseline",
    "BaselineRR",
    "GreedyMakespanInsertion",
    "OrderToteHill",
    "FullHillClimb",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "TabuSearch",
    "IteratedLocalSearch",
    "BeamSearch",
]

OPTIONAL_METHODS = [
    "BranchAndBound",
]

ALL_METHODS = DEFAULT_METHODS + OPTIONAL_METHODS


def get_event_counts(
    orders: list,
    tote_contents: dict,
    solution: Solution,
) -> dict[str, int]:
    """Return counts of LOAD, PICK, HOP events for the given solution. Keys: 'load', 'pick', 'hop'."""
    _, _, trace = evaluate_solution(
        solution, orders, tote_contents, objective="last_order", return_trace=True
    )
    if trace is None:
        return {"load": 0, "pick": 0, "hop": 0}
    load_count = sum(1 for _, ev, _ in trace if ev == "LOAD")
    pick_count = sum(1 for _, ev, _ in trace if ev == "PICK")
    hop_count = sum(1 for _, ev, _ in trace if ev == "HOP")
    return {"load": load_count, "pick": pick_count, "hop": hop_count}


def write_playbook_folder(
    orders: list,
    tote_contents: dict,
    solution: Solution,
    method_name: str,
    instance_id: str,
    last_order_value: float,
    out_dir: Path,
) -> None:
    """Write a playbook folder under out_dir/method_name/instance_id/ with input.csv, input_conveyor.csv, events_playbook.txt, tote_and_item_order.txt."""
    folder = out_dir / method_name / instance_id
    folder.mkdir(parents=True, exist_ok=True)

    # One row per order: order_id, conv_num, shape columns (each conveyor's assigned orders explicit)
    write_m3_input_by_order(
        orders, solution.order_to_conveyor, solution.order_sequence, folder / "input.csv"
    )
    # Per-conveyor example format (conv_num + shape columns)
    export_m3_input(solution, orders, folder / "input_conveyor.csv", format="per_conveyor")

    load_seq = build_load_sequence(
        solution.order_sequence,
        tote_contents,
        travel_aware=True,
        item_order_per_tote=solution.item_order_per_tote,
        tote_loading_order=solution.tote_loading_order,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        tmp = Path(f.name)
    try:
        write_m3_input_by_order(
            orders, solution.order_to_conveyor, solution.order_sequence, tmp
        )
        conveyors, orders_per_conv = load_conveyors(tmp)
        results, trace_events = simulate_greedy(
            conveyors,
            all_load_at_conveyor_0=True,
            load_spacing=2.5,
            load_sequence=load_seq,
            return_trace=True,
            orders_per_conveyor=orders_per_conv,
        )
    finally:
        tmp.unlink(missing_ok=True)

    # Compute which event index completes each order (FIFO pick assignment per conveyor)
    conv_orders = orders_on_conveyor(solution.order_to_conveyor, solution.order_sequence)
    demand = order_demand(orders)
    remaining = {oid: dict(d) for oid, d in demand.items()}
    order_complete_at_event: dict[int, tuple[float, int, int]] = {}  # order_id -> (t, event_index, conveyor)
    for i, (t, ev_type, payload) in enumerate(trace_events):
        if ev_type != "PICK":
            continue
        _item_id, conv, shape = payload
        order_list = conv_orders.get(conv, [])
        for oid in order_list:
            if remaining.get(oid, {}).get(shape, 0) > 0:
                remaining[oid][shape] -= 1
                if remaining[oid][shape] == 0:
                    del remaining[oid][shape]
                if not remaining[oid]:
                    order_complete_at_event[oid] = (t, i, conv)
                break

    # Count event types (LOAD, PICK, HOP) for playbook summary
    load_count = sum(1 for _, ev, _ in trace_events if ev == "LOAD")
    pick_count = sum(1 for _, ev, _ in trace_events if ev == "PICK")
    hop_count = sum(1 for _, ev, _ in trace_events if ev == "HOP")

    shape_names = list(SHAPE_COLUMNS)
    with (folder / "events_playbook.txt").open("w", encoding="utf-8") as f:
        f.write("EVENT PLAYBOOK (chronological)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Instance: {instance_id}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Last-order completion: {last_order_value:.2f}s\n")
        f.write(f"Total events: {len(trace_events)}  (LOAD: {load_count}, PICK: {pick_count}, HOP: {hop_count})\n")
        f.write("LOAD item_id = index of that item in the load sequence; see tote_and_item_order.txt for which tote/item each id corresponds to.\n")
        f.write("When an order receives its last pick, an ORDER COMPLETE line is shown (with completion time and conveyor).\n")
        f.write("=" * 60 + "\n\n")
        for i, (t, ev_type, payload) in enumerate(trace_events):
            if ev_type == "LOAD":
                item_id, shape, conv = payload
                sname = shape_names[shape] if shape < len(shape_names) else str(shape)
                f.write(f"{i+1:5}.  t={t:7.2f}s  LOAD   item_id={item_id:3}  shape={shape} ({sname})  at conveyor {conv}\n")
            elif ev_type == "HOP":
                item_id, from_c, to_c = payload
                f.write(f"{i+1:5}.  t={t:7.2f}s  HOP    item_id={item_id:3}  conveyor {from_c} -> {to_c}\n")
            else:
                item_id, conv, shape = payload
                sname = shape_names[shape] if shape < len(shape_names) else str(shape)
                f.write(f"{i+1:5}.  t={t:7.2f}s  PICK   item_id={item_id:3}  shape={shape} ({sname})  at conveyor {conv}\n")
                # If this PICK completed one or more orders (FIFO can only complete one per pick)
                for oid, (ct, ei, cconv) in list(order_complete_at_event.items()):
                    if ei == i:
                        f.write(f"       >>> ORDER {oid} COMPLETE at t={ct:.2f}s at conveyor {cconv}\n")
                        break

    item_id = 0
    with (folder / "tote_and_item_order.txt").open("w", encoding="utf-8") as f:
        f.write("TOTE LOADING ORDER AND ITEMS PER TOTE\n")
        f.write("=" * 60 + "\n")
        f.write("Tote contents (which items are in which tote) come from the data (orders_totes.csv).\n")
        f.write("This file shows the chosen ORDER of totes (first tote emptied completely, then next, etc.)\n")
        f.write("and the ORDER of items within each tote as loaded onto the conveyor.\n")
        f.write("Each line is one (order, shape, qty) from the data; qty = number of physical units.\n")
        f.write("So a tote can have more than 3 physical items (e.g. two lines with qty 3 and qty 2 = 5 units).\n")
        f.write("Item IDs match the LOAD events in events_playbook.txt (item_id in that file).\n\n")
        pos = 0
        for tote_id in solution.tote_loading_order:
            if tote_id not in tote_contents:
                continue
            pos += 1
            items_in_tote = solution.item_order_per_tote.get(tote_id) or tote_contents.get(tote_id, [])
            n_items = sum(qty for _, _, qty in items_in_tote)
            start_id = item_id
            item_id += n_items
            end_id = item_id - 1
            id_range = f"  (item_id {start_id}-{end_id})" if n_items > 0 else ""
            f.write(f"  {pos}. Tote {tote_id}{id_range}:\n")
            for order_id, shape, qty in items_in_tote:
                sname = shape_names[shape] if shape < len(shape_names) else str(shape)
                f.write(f"       order {order_id}  shape {shape} ({sname})  qty {qty}\n")
            f.write("\n")
        if item_id != len(load_seq):
            f.write(f"  [Warning: total items {item_id} != load sequence length {len(load_seq)}]\n")


def active_methods(include_branch_and_bound: bool) -> list[str]:
    methods = list(DEFAULT_METHODS)
    if include_branch_and_bound:
        methods.extend(OPTIONAL_METHODS)
    return methods


def get_solution_for_method(
    orders: list,
    tote_contents: dict,
    method: str,
    seed: int | None,
    joint_max_evals: int,
    joint_restarts: int,
    polish: bool,
    polish_max_evals: int,
    verbose: bool = False,
) -> tuple[Solution, float]:
    """Run the given method and return (solution, last_order value)."""
    n = len(orders)
    if method == "Baseline":
        sol = solution_from_order_sequence(orders, tote_contents, list(range(n)), travel_aware=True)
        val, _, _ = evaluate_solution(sol, orders, tote_contents, objective="last_order")
        return sol, val
    if method == "BaselineRR":
        sol = solution_from_order_sequence(orders, tote_contents, list(range(n)), travel_aware=False)
        val, _, _ = evaluate_solution(sol, orders, tote_contents, objective="last_order")
        return sol, val
    if method == "GreedyMakespanInsertion":
        # Order-only greedy construction
        seq, ms = greedy_makespan_insertion(
            orders,
            travel_aware=True,
            tote_contents=tote_contents,
            order_to_totes=None,
        )
        sol = solution_from_order_sequence(
            orders, tote_contents, seq, travel_aware=True
        )
        return sol, ms
    if method == "BranchAndBound":
        def _eval(sol: Solution):
            return evaluate_solution(sol, orders, tote_contents, objective="last_order")
        sol, val = branch_and_bound_solve(
            orders,
            tote_contents,
            evaluate=_eval,
            n_iterations=joint_max_evals * 2,
            seed=seed,
            config=BranchAndBoundConfig(log_improvements=verbose),
        )
        return sol, val
    if method == "OrderToteHill":
        start = initial_solution(orders, tote_contents, travel_aware=True)
        sol, val, _ = hill_climb_order_and_tote(
            orders, tote_contents, start,
            objective="last_order", max_evals=joint_max_evals, verbose=False,
        )
        return sol, val
    if method == "FullHillClimb":
        start = initial_solution(orders, tote_contents, travel_aware=True)
        sol, val, _ = hill_climb_full(
            orders, tote_contents, start,
            objective="last_order", max_evals=joint_max_evals, verbose=False,
        )
        return sol, val
    if method == "SimulatedAnnealing":
        best_sol = None
        best_val = float("inf")
        for r in range(joint_restarts):
            s = (seed + r) if seed is not None else None
            if s is not None:
                random.seed(s)
            sol, val, _ = simulated_annealing(
                orders, tote_contents,
                objective="last_order", max_evals=joint_max_evals, seed=s, verbose=False,
            )
            if val < best_val:
                best_val = val
                best_sol = sol
        if best_sol is not None and polish:
            best_sol, best_val, _ = hill_climb_full(
                orders, tote_contents, best_sol,
                objective="last_order", max_evals=polish_max_evals, verbose=False,
            )
        return (best_sol, best_val) if best_sol is not None else (initial_solution(orders, tote_contents, travel_aware=True), float("inf"))
    if method == "GeneticAlgorithm":
        sol, val, _ = genetic_algorithm(
            orders, tote_contents,
            objective="last_order", max_evals=joint_max_evals, seed=seed, verbose=False,
        )
        return sol, val
    if method == "TabuSearch":
        sol, val, _ = tabu_search(
            orders, tote_contents,
            objective="last_order", max_evals=joint_max_evals, seed=seed, verbose=False,
        )
        return sol, val
    if method == "IteratedLocalSearch":
        sol, val, _ = iterated_local_search(
            orders, tote_contents,
            objective="last_order", max_evals=joint_max_evals, seed=seed, verbose=False,
        )
        return sol, val
    raise ValueError(f"Unknown method: {method}")


def run_one_instance(
    generated_path: Path,
    methods: list[str],
    seed: int | None,
    joint_max_evals: int,
    joint_restarts: int,
    polish: bool,
    polish_max_evals: int,
    verbose: bool,
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    """Run all methods on one instance. Return (method_name -> last_order value, method_name -> {load, pick, hop} counts)."""
    orders = load_generator_data(generated_path)
    if not orders:
        raise ValueError("No orders")
    tote_contents, order_to_totes = load_tote_data(generated_path, orders)
    if not tote_contents or not order_to_totes:
        raise ValueError("Tote data required (orders_totes.csv)")

    n = len(orders)
    results: dict[str, float] = {}
    solutions: dict[str, Solution] = {}

    # Baseline: order 0,1,2,..., travel-aware conveyor (pos 0→conv 3, 1→2, 2→1, 3→0), default tote/item
    baseline_sol = solution_from_order_sequence(
        orders, tote_contents, list(range(n)), travel_aware=True
    )
    solutions["Baseline"] = baseline_sol
    results["Baseline"], _, _ = evaluate_solution(baseline_sol, orders, tote_contents, objective="last_order")

    # BaselineRR: same order 0,1,2,..., round-robin conveyor (pos 0→conv 0, 1→1, 2→2, 3→0), default tote/item
    baseline_rr_sol = solution_from_order_sequence(
        orders, tote_contents, list(range(n)), travel_aware=False
    )
    solutions["BaselineRR"] = baseline_rr_sol
    results["BaselineRR"], _, _ = evaluate_solution(baseline_rr_sol, orders, tote_contents, objective="last_order")

    # MCT: greedy makespan insertion (order-only)
    seq, mct_ms = greedy_makespan_insertion(
        orders, travel_aware=True, tote_contents=tote_contents, order_to_totes=None,
    )
    results["GreedyMakespanInsertion"] = mct_ms
    solutions["GreedyMakespanInsertion"] = solution_from_order_sequence(orders, tote_contents, seq, travel_aware=True)

    # ---- Beam Search ----
    seq_beam, beam_ms = beam_search_order_sequence(
        orders,
        beam_width=5,
        candidate_pool=12,
        travel_aware=True,
        tote_contents=tote_contents,
        order_to_totes=None,
    )
    results["BeamSearch"] = beam_ms
    solutions["BeamSearch"] = solution_from_order_sequence(orders, tote_contents, seq_beam, travel_aware=True)

    if "BranchAndBound" in methods:
        # BranchAndBound: tote-sequence branch-and-bound with deterministic decode
        def _eval(sol: Solution):
            return evaluate_solution(sol, orders, tote_contents, objective="last_order")
        bnb_sol, results["BranchAndBound"] = branch_and_bound_solve(
            orders,
            tote_contents,
            evaluate=_eval,
            n_iterations=joint_max_evals * 2,
            seed=seed,
            config=BranchAndBoundConfig(log_improvements=verbose),
        )
        solutions["BranchAndBound"] = bnb_sol

    # OrderToteHill: hill climb over order sequence and tote loading order only (no conveyor, no item order)
    order_tote_start = initial_solution(orders, tote_contents, travel_aware=True)
    ot_sol, results["OrderToteHill"], _ = hill_climb_order_and_tote(
        orders, tote_contents, order_tote_start,
        objective="last_order", max_evals=joint_max_evals, verbose=False,
    )
    solutions["OrderToteHill"] = ot_sol

    # FullHillClimb: full hill climb from LPT — same neighborhoods, same budget
    lpt_full = initial_solution(orders, tote_contents, travel_aware=True)
    fh_sol, results["FullHillClimb"], _ = hill_climb_full(
        orders, tote_contents, lpt_full,
        objective="last_order", max_evals=joint_max_evals, verbose=False,
    )
    solutions["FullHillClimb"] = fh_sol

    # SimulatedAnnealing: full SA (with optional restarts), then optional polish
    best_joint = None
    best_val = float("inf")
    for r in range(joint_restarts):
        s = (seed + r) if seed is not None else None
        if s is not None:
            random.seed(s)
        sol, val, _ = simulated_annealing(
            orders, tote_contents,
            objective="last_order",
            max_evals=joint_max_evals,
            seed=s,
            verbose=verbose and joint_restarts == 1,
        )
        if val < best_val:
            best_val = val
            best_joint = sol
    if best_joint is not None and polish:
        best_joint, best_val, _ = hill_climb_full(
            orders, tote_contents, best_joint,
            objective="last_order", max_evals=polish_max_evals, verbose=False,
        )
    results["SimulatedAnnealing"] = best_val
    solutions["SimulatedAnnealing"] = best_joint if best_joint is not None else solutions["Baseline"]

    # GeneticAlgorithm: OX crossover, mutation, selection by last_order
    ga_sol, results["GeneticAlgorithm"], _ = genetic_algorithm(
        orders, tote_contents,
        objective="last_order",
        max_evals=joint_max_evals,
        seed=seed,
        verbose=False,
    )
    solutions["GeneticAlgorithm"] = ga_sol

    # TabuSearch: tabu list + aspiration, same neighborhoods as FullHillClimb
    ts_sol, results["TabuSearch"], _ = tabu_search(
        orders, tote_contents,
        initial=None,
        objective="last_order",
        max_evals=joint_max_evals,
        seed=seed,
        verbose=False,
    )
    solutions["TabuSearch"] = ts_sol

    # IteratedLocalSearch: perturb + hill climb, repeat
    ils_sol, results["IteratedLocalSearch"], _ = iterated_local_search(
        orders, tote_contents,
        objective="last_order",
        max_evals=joint_max_evals,
        perturb_strength=3,
        seed=seed,
        verbose=False,
    )
    solutions["IteratedLocalSearch"] = ils_sol

    # Event counts (LOAD, PICK, HOP) per method for this instance
    event_counts: dict[str, dict[str, int]] = {}
    for method in methods:
        event_counts[method] = get_event_counts(orders, tote_contents, solutions[method])

    return results, event_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the default method set under the last_order objective and write all outputs to the results folder. BranchAndBound is opt-in via --include-branch-and-bound because it is slow.",
    )
    parser.add_argument(
        "base_dir",
        type=Path,
        nargs="?",
        default=Path("data_generator"),
        help="Base directory with <timestamp>/generated/ folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results"),
        help="Output folder for comparison CSV, summary, and plot (default: results)",
    )
    parser.add_argument(
        "--include-branch-and-bound",
        action="store_true",
        help="Include BranchAndBound in the comparison run (excluded by default because it is slow)",
    )
    parser.add_argument("--joint-max-evals", type=int, default=800, help="Max evals for SimulatedAnnealing per instance (default 800)")
    parser.add_argument("--joint-restarts", type=int, default=2, help="Run SimulatedAnnealing this many times (different seeds), take best (default 2)")
    parser.add_argument("--polish", action="store_true", help="After SA, run hill climb on order/tote/conveyor to polish")
    parser.add_argument("--polish-max-evals", type=int, default=300, help="Max evals for polish hill climb (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SimulatedAnnealing (restarts use seed, seed+1, ...)")
    parser.add_argument("--save-playbook", action="store_true", help="For each instance, write a playbook folder under out_dir/<instance_id>/ (input.csv, events_playbook.txt, tote_and_item_order.txt)")
    parser.add_argument("--playbook-method", type=str, default=None, choices=ALL_METHODS, help="Method to use for playbook (default: best active method for that instance)")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less per-instance output")
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not base_dir.exists():
        raise SystemExit(f"Base directory not found: {base_dir}")

    generated_dirs = sorted(
        d for d in base_dir.iterdir()
        if d.is_dir() and (d / "generated").is_dir()
    )
    if not generated_dirs:
        raise SystemExit(f"No generated folders under {base_dir}")

    methods = active_methods(args.include_branch_and_bound)

    if args.playbook_method == "BranchAndBound" and "BranchAndBound" not in methods:
        raise SystemExit("BranchAndBound playbooks require --include-branch-and-bound")

    instance_ids = [d.name for d in generated_dirs]
    rows: list[dict] = []
    event_counts_rows: list[dict] = []

    for inst_id, gen_dir in zip(instance_ids, generated_dirs, strict=True):
        path = gen_dir / "generated"
        try:
            res, event_counts = run_one_instance(
                path,
                methods=methods,
                seed=args.seed,
                joint_max_evals=args.joint_max_evals,
                joint_restarts=args.joint_restarts,
                polish=args.polish,
                polish_max_evals=args.polish_max_evals,
                verbose=not args.quiet,
            )
        except Exception as e:
            print(f"Skip {inst_id}: {e}", file=sys.stderr)
            continue
        row = {"instance": inst_id, **res}
        rows.append(row)
        # Flatten event counts for this instance: instance, Baseline_load, Baseline_pick, Baseline_hop, ...
        ec_row: dict = {"instance": inst_id}
        for m in methods:
            ec_row[f"{m}_load"] = event_counts[m]["load"]
            ec_row[f"{m}_pick"] = event_counts[m]["pick"]
            ec_row[f"{m}_hop"] = event_counts[m]["hop"]
        event_counts_rows.append(ec_row)
        if not args.quiet:
            print(f"  {inst_id}: " + " ".join(f"{m}={res[m]:.2f}" for m in methods))

        if getattr(args, "save_playbook", False):
            try:
                playbook_method = args.playbook_method or min(methods, key=lambda m: res[m])
                orders = load_generator_data(path)
                tote_contents, _ = load_tote_data(path, orders)
                if orders and tote_contents:
                    sol, val = get_solution_for_method(
                        orders, tote_contents, playbook_method,
                        seed=args.seed,
                        joint_max_evals=args.joint_max_evals,
                        joint_restarts=args.joint_restarts,
                        polish=args.polish,
                        polish_max_evals=args.polish_max_evals,
                        verbose=not args.quiet,
                    )
                    write_playbook_folder(
                        orders, tote_contents, sol, playbook_method, inst_id, val, out_dir,
                    )
                    if not args.quiet:
                        print(f"    Playbook: {out_dir / playbook_method / inst_id} (method={playbook_method})")
            except Exception as e:
                print(f"Skip playbook for {inst_id}: {e}", file=sys.stderr)

    if not rows:
        raise SystemExit("No instances processed.")

    # Summary
    N = len(rows)
    print("\n--- Summary (objective: last_order, lower is better) ---")
    for col in methods:
        avg = sum(r[col] for r in rows) / N
        best = min(r[col] for r in rows)
        print(f"  {col}: avg={avg:.2f}s  best={best:.2f}s")
    baseline_avg = sum(r["Baseline"] for r in rows) / N
    print(f"  BaselineRR vs Baseline: {(1 - sum(r['BaselineRR'] for r in rows) / N / baseline_avg) * 100:.1f}% (avg; positive = RR better)")
    for name in methods:
        if name in {"Baseline", "BaselineRR"}:
            continue
        avg = sum(r[name] for r in rows) / N
        print(f"  {name} vs Baseline: {(1 - avg / baseline_avg) * 100:.1f}% improvement (avg)")

    # Wins
    for col in methods:
        wins = sum(1 for r in rows if r[col] == min(r[m] for m in methods))
        print(f"  {col} wins: {wins}/{N}")

    # Write comparison CSV
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instance"] + methods, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")

    # Average event counts (LOAD, PICK, HOP) per method
    event_fieldnames = ["instance"] + [f"{m}_load" for m in methods] + [f"{m}_pick" for m in methods] + [f"{m}_hop" for m in methods]
    avg_load = {m: sum(r[f"{m}_load"] for r in event_counts_rows) / N for m in methods}
    avg_pick = {m: sum(r[f"{m}_pick"] for r in event_counts_rows) / N for m in methods}
    avg_hop = {m: sum(r[f"{m}_hop"] for r in event_counts_rows) / N for m in methods}
    events_csv_path = out_dir / "comparison_events.csv"
    with events_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=event_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(event_counts_rows)
    print(f"Saved {events_csv_path}")

    # Write summary text
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Method comparison (objective: last_order completion, seconds)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Methods: " + ", ".join(methods) + "\n\n")
        for col in methods:
            avg = sum(r[col] for r in rows) / N
            f.write(f"  {col}: avg={avg:.2f}s\n")
        f.write("\n")
        f.write(f"BaselineRR vs Baseline: {(1 - sum(r['BaselineRR'] for r in rows) / N / baseline_avg) * 100:.1f}% (avg; positive = RR better)\n")
        for name in methods:
            if name in {"Baseline", "BaselineRR"}:
                continue
            avg = sum(r[name] for r in rows) / N
            f.write(f"{name} vs Baseline: {(1 - avg / baseline_avg) * 100:.1f}% improvement (avg)\n")
        f.write("\nAverage playbook events per method (LOAD, PICK, HOP):\n")
        for m in methods:
            f.write(f"  {m}: LOAD={avg_load[m]:.1f}, PICK={avg_pick[m]:.1f}, HOP={avg_hop[m]:.1f}\n")
        f.write("\nPer-instance results: see comparison.csv\n")
        f.write("Per-instance event counts: see comparison_events.csv\n")
    print(f"Saved {summary_path}")

    # Bar chart (average per method)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(11, 5))
        avgs = [sum(r[m] for r in rows) / N for m in methods]
        colors = [
            "#95a5a6", "#3498db", "#1abc9c", "#2ecc71", "#e67e22",
            "#e74c3c", "#9b59b6", "#f39c12", "#34495e", "#16a085",
            "#7f8c8d",
        ]
        bars = ax.bar(range(len(methods)), avgs, color=colors[: len(methods)], edgecolor="black")
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha="right")
        ax.set_ylabel("Avg last_order (s)")
        ax.set_title("Method comparison (same objective: last_order completion)")
        for b, v in zip(bars, avgs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1f}s", ha="center", fontsize=10)
        plt.tight_layout()
        plot_path = out_dir / "comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")

        # Line graph: selected methods across instances
        line_methods = [
            "Baseline",
            "GreedyMakespanInsertion",
            "FullHillClimb",
            "SimulatedAnnealing",
            "GeneticAlgorithm",
        ]
        line_colors = ["#95a5a6", "#3498db", "#2ecc71", "#e74c3c", "#9b59b6"]
        fig2, ax2 = plt.subplots(figsize=(11, 5))
        x = range(N)
        for method, color in zip(line_methods, line_colors):
            y = [r[method] for r in rows]
            ax2.plot(x, y, marker="o", markersize=4, label=method, color=color, linewidth=1.5)
        ax2.set_xlabel("Instance (index)")
        ax2.set_ylabel("Last-order completion (s)")
        ax2.set_title("Last-order completion by instance: Baseline, GreedyMakespanInsertion, FullHillClimb, SimulatedAnnealing, GeneticAlgorithm")
        ax2.legend(loc="best", fontsize=9)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        line_plot_path = out_dir / "comparison_line.png"
        plt.savefig(line_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {line_plot_path}")
    else:
        print("Install matplotlib to generate comparison.png: pip install matplotlib")


if __name__ == "__main__":
    main()
