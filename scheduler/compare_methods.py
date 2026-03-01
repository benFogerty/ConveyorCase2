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

from conveyor_sim import SHAPE_COLUMNS, load_conveyors, simulate_greedy
from scheduler.core import (
    build_conveyor_input_from_assignment,
    build_load_sequence,
    load_generator_data,
    load_tote_data,
    write_m3_input,
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

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


METHODS = [
    "Baseline",
    "BaselineRR",
    "OrderToteHill",
    "FullHillClimb",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "TabuSearch",
    "IteratedLocalSearch",
]


def write_playbook_folder(
    orders: list,
    tote_contents: dict,
    solution: Solution,
    method_name: str,
    instance_id: str,
    last_order_value: float,
    out_dir: Path,
) -> None:
    """Write a playbook folder under out_dir/method_name/instance_id/ with input.csv, events_playbook.txt, tote_and_item_order.txt."""
    folder = out_dir / method_name / instance_id
    folder.mkdir(parents=True, exist_ok=True)

    counts = build_conveyor_input_from_assignment(orders, solution.order_to_conveyor)
    write_m3_input(counts, folder / "input.csv")

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
        write_m3_input(counts, tmp)
        conveyors = load_conveyors(tmp)
        results, trace_events = simulate_greedy(
            conveyors,
            all_load_at_conveyor_0=True,
            load_spacing=2.5,
            load_sequence=load_seq,
            return_trace=True,
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

    shape_names = list(SHAPE_COLUMNS)
    with (folder / "events_playbook.txt").open("w", encoding="utf-8") as f:
        f.write("EVENT PLAYBOOK (chronological)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Instance: {instance_id}\n")
        f.write(f"Method: {method_name}\n")
        f.write(f"Last-order completion: {last_order_value:.2f}s\n")
        f.write(f"Total events (LOAD+HOP+PICK): {len(trace_events)}\n")
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


def get_solution_for_method(
    orders: list,
    tote_contents: dict,
    method: str,
    seed: int | None,
    joint_max_evals: int,
    joint_restarts: int,
    polish: bool,
    polish_max_evals: int,
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
    seed: int | None,
    joint_max_evals: int,
    joint_restarts: int,
    polish: bool,
    polish_max_evals: int,
    verbose: bool,
) -> dict[str, float]:
    """Run all methods on one instance. Return dict method_name -> last_order value."""
    orders = load_generator_data(generated_path)
    if not orders:
        raise ValueError("No orders")
    tote_contents, order_to_totes = load_tote_data(generated_path, orders)
    if not tote_contents or not order_to_totes:
        raise ValueError("Tote data required (orders_totes.csv)")

    n = len(orders)
    results: dict[str, float] = {}

    # Baseline: order 0,1,2,..., travel-aware conveyor (pos 0→conv 3, 1→2, 2→1, 3→0), default tote/item
    baseline_sol = solution_from_order_sequence(
        orders, tote_contents, list(range(n)), travel_aware=True
    )
    results["Baseline"], _, _ = evaluate_solution(baseline_sol, orders, tote_contents, objective="last_order")

    # BaselineRR: same order 0,1,2,..., round-robin conveyor (pos 0→conv 0, 1→1, 2→2, 3→0), default tote/item
    baseline_rr_sol = solution_from_order_sequence(
        orders, tote_contents, list(range(n)), travel_aware=False
    )
    results["BaselineRR"], _, _ = evaluate_solution(baseline_rr_sol, orders, tote_contents, objective="last_order")

    # OrderToteHill: hill climb over order sequence and tote loading order only (no conveyor, no item order)
    order_tote_start = initial_solution(orders, tote_contents, travel_aware=True)
    _, results["OrderToteHill"], _ = hill_climb_order_and_tote(
        orders, tote_contents, order_tote_start,
        objective="last_order", max_evals=joint_max_evals, verbose=False,
    )

    # FullHillClimb: full hill climb from LPT — same neighborhoods, same budget
    lpt_full = initial_solution(orders, tote_contents, travel_aware=True)
    _, results["FullHillClimb"], _ = hill_climb_full(
        orders, tote_contents, lpt_full,
        objective="last_order", max_evals=joint_max_evals, verbose=False,
    )

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

    # GeneticAlgorithm: OX crossover, mutation, selection by last_order
    _, results["GeneticAlgorithm"], _ = genetic_algorithm(
        orders, tote_contents,
        objective="last_order",
        max_evals=joint_max_evals,
        seed=seed,
        verbose=False,
    )

    # TabuSearch: tabu list + aspiration, same neighborhoods as FullHillClimb
    _, results["TabuSearch"], _ = tabu_search(
        orders, tote_contents,
        initial=None,
        objective="last_order",
        max_evals=joint_max_evals,
        seed=seed,
        verbose=False,
    )

    # IteratedLocalSearch: perturb + hill climb, repeat
    _, results["IteratedLocalSearch"], _ = iterated_local_search(
        orders, tote_contents,
        objective="last_order",
        max_evals=joint_max_evals,
        perturb_strength=3,
        seed=seed,
        verbose=False,
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Baseline, BaselineRR, OrderToteHill, FullHillClimb, SimulatedAnnealing, GeneticAlgorithm, TabuSearch, IteratedLocalSearch under last_order objective; write all outputs to results folder.",
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
    parser.add_argument("--joint-max-evals", type=int, default=800, help="Max evals for SimulatedAnnealing per instance (default 800)")
    parser.add_argument("--joint-restarts", type=int, default=2, help="Run SimulatedAnnealing this many times (different seeds), take best (default 2)")
    parser.add_argument("--polish", action="store_true", help="After SA, run hill climb on order/tote/conveyor to polish")
    parser.add_argument("--polish-max-evals", type=int, default=300, help="Max evals for polish hill climb (default 300)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for SimulatedAnnealing (restarts use seed, seed+1, ...)")
    parser.add_argument("--save-playbook", action="store_true", help="For each instance, write a playbook folder under out_dir/<instance_id>/ (input.csv, events_playbook.txt, tote_and_item_order.txt)")
    parser.add_argument("--playbook-method", type=str, default=None, choices=METHODS, help="Method to use for playbook (default: best for that instance)")
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

    instance_ids = [d.name for d in generated_dirs]
    rows: list[dict] = []

    for inst_id, gen_dir in zip(instance_ids, generated_dirs, strict=True):
        path = gen_dir / "generated"
        try:
            res = run_one_instance(
                path,
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
        if not args.quiet:
            print(f"  {inst_id}: " + " ".join(f"{m}={res[m]:.2f}" for m in METHODS))

        if getattr(args, "save_playbook", False):
            playbook_method = args.playbook_method or min(METHODS, key=lambda m: res[m])
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
                )
                write_playbook_folder(
                    orders, tote_contents, sol, playbook_method, inst_id, val, out_dir,
                )
                if not args.quiet:
                    print(f"    Playbook: {out_dir / playbook_method / inst_id} (method={playbook_method})")

    if not rows:
        raise SystemExit("No instances processed.")

    # Summary
    N = len(rows)
    print("\n--- Summary (objective: last_order, lower is better) ---")
    for col in METHODS:
        avg = sum(r[col] for r in rows) / N
        best = min(r[col] for r in rows)
        print(f"  {col}: avg={avg:.2f}s  best={best:.2f}s")
    baseline_avg = sum(r["Baseline"] for r in rows) / N
    print(f"  BaselineRR vs Baseline: {(1 - sum(r['BaselineRR'] for r in rows) / N / baseline_avg) * 100:.1f}% (avg; positive = RR better)")
    for name in ["OrderToteHill", "FullHillClimb", "SimulatedAnnealing", "GeneticAlgorithm", "TabuSearch", "IteratedLocalSearch"]:
        avg = sum(r[name] for r in rows) / N
        print(f"  {name} vs Baseline: {(1 - avg / baseline_avg) * 100:.1f}% improvement (avg)")

    # Wins
    for col in METHODS:
        wins = sum(1 for r in rows if r[col] == min(r[m] for m in METHODS))
        print(f"  {col} wins: {wins}/{N}")

    # Write comparison CSV
    csv_path = out_dir / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["instance"] + METHODS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved {csv_path}")

    # Write summary text
    summary_path = out_dir / "summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Method comparison (objective: last_order completion, seconds)\n")
        f.write("=" * 60 + "\n\n")
        f.write("Methods: " + ", ".join(METHODS) + "\n\n")
        for col in METHODS:
            avg = sum(r[col] for r in rows) / N
            f.write(f"  {col}: avg={avg:.2f}s\n")
        f.write("\n")
        f.write(f"BaselineRR vs Baseline: {(1 - sum(r['BaselineRR'] for r in rows) / N / baseline_avg) * 100:.1f}% (avg; positive = RR better)\n")
        for name in ["OrderToteHill", "FullHillClimb", "SimulatedAnnealing", "GeneticAlgorithm", "TabuSearch", "IteratedLocalSearch"]:
            avg = sum(r[name] for r in rows) / N
            f.write(f"{name} vs Baseline: {(1 - avg / baseline_avg) * 100:.1f}% improvement (avg)\n")
        f.write("\nPer-instance results: see comparison.csv\n")
    print(f"Saved {summary_path}")

    # Bar chart (average per method)
    if plt is not None:
        fig, ax = plt.subplots(figsize=(11, 5))
        avgs = [sum(r[m] for r in rows) / N for m in METHODS]
        colors = [
            "#95a5a6", "#3498db", "#1abc9c", "#2ecc71", "#e67e22",
            "#e74c3c", "#9b59b6", "#f39c12", "#34495e",
        ]
        bars = ax.bar(range(len(METHODS)), avgs, color=colors[: len(METHODS)], edgecolor="black")
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHODS, rotation=45, ha="right")
        ax.set_ylabel("Avg last_order (s)")
        ax.set_title("Method comparison (same objective: last_order completion)")
        for b, v in zip(bars, avgs):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3, f"{v:.1f}s", ha="center", fontsize=10)
        plt.tight_layout()
        plot_path = out_dir / "comparison.png"
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {plot_path}")
    else:
        print("Install matplotlib to generate comparison.png: pip install matplotlib")


if __name__ == "__main__":
    main()
