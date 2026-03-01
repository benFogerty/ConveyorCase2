#!/usr/bin/env python3
"""
Order-sequence search: improve beyond LPT using local search or simulated annealing.
Evaluates makespan via the conveyor sim (all load at conveyor 0, 2.5s spacing).
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conveyor_sim import load_conveyors, simulate_greedy
from scheduler.core import build_conveyor_input, build_load_sequence, load_generator_data, lpt_order, write_m3_input


def makespan_for_sequence(
    orders,
    order_sequence: list[int],
    travel_aware: bool = True,
    tote_contents: dict | None = None,
    order_to_totes: dict | None = None,
    item_order_per_tote: dict | None = None,
) -> float:
    """Return makespan (s) for the given order sequence. Uses all load at conv 0, 2.5s spacing.
    If tote_contents and order_to_totes are provided, load order = tote order + item order within tote."""
    counts = build_conveyor_input(orders, order_sequence, travel_aware=travel_aware)
    load_seq = None
    if tote_contents and order_to_totes:
        load_seq = build_load_sequence(
            order_sequence, tote_contents, travel_aware, item_order_per_tote=item_order_per_tote
        )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        path = Path(f.name)
    try:
        write_m3_input(counts, path)
        conveyors = load_conveyors(path)
        results = simulate_greedy(
            conveyors, all_load_at_conveyor_0=True, load_spacing=2.5, load_sequence=load_seq
        )
        return max(t for _, _, t in results) if results else 0.0
    finally:
        path.unlink(missing_ok=True)


def neighborhood_swaps(seq: list[int]) -> list[list[int]]:
    """All sequences obtained by swapping two distinct positions."""
    out = []
    for i in range(len(seq)):
        for j in range(i + 1, len(seq)):
            new_seq = copy.copy(seq)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            out.append(new_seq)
    return out


def random_neighbor(seq: list[int]) -> list[int]:
    """One random swap."""
    i, j = random.sample(range(len(seq)), 2)
    new_seq = copy.copy(seq)
    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
    return new_seq


def hill_climb(
    orders,
    initial_seq: list[int],
    travel_aware: bool = True,
    verbose: bool = True,
    tote_contents: dict | None = None,
    order_to_totes: dict | None = None,
    item_order_per_tote: dict | None = None,
):
    """First-improvement hill climbing using swap neighborhood. Returns (best_seq, best_makespan, evals).
    Pass tote_contents/order_to_totes to use tote + item load order in the sim."""
    current_seq = list(initial_seq)
    current_ms = makespan_for_sequence(
        orders, current_seq, travel_aware,
        tote_contents=tote_contents, order_to_totes=order_to_totes,
        item_order_per_tote=item_order_per_tote,
    )
    evals = 1
    while True:
        improved = False
        for neighbor in neighborhood_swaps(current_seq):
            ms = makespan_for_sequence(
                orders, neighbor, travel_aware,
                tote_contents=tote_contents, order_to_totes=order_to_totes,
                item_order_per_tote=item_order_per_tote,
            )
            evals += 1
            if ms < current_ms:
                current_seq = neighbor
                current_ms = ms
                improved = True
                if verbose:
                    print(f"  Improvement: makespan = {current_ms:.2f}s (evals={evals})")
                break
        if not improved:
            break
    return current_seq, current_ms, evals


def simulated_annealing(
    orders,
    initial_seq: list[int],
    travel_aware: bool = True,
    max_evals: int = 500,
    T0: float = 10.0,
    cooling: float = 0.995,
    verbose: bool = True,
    tote_contents: dict | None = None,
    order_to_totes: dict | None = None,
    item_order_per_tote: dict | None = None,
):
    """Simulated annealing with swap neighborhood. Returns (best_seq, best_makespan, evals).
    Pass tote_contents/order_to_totes to use tote + item load order in the sim."""
    current_seq = list(initial_seq)
    current_ms = makespan_for_sequence(
        orders, current_seq, travel_aware,
        tote_contents=tote_contents, order_to_totes=order_to_totes,
        item_order_per_tote=item_order_per_tote,
    )
    best_seq = list(current_seq)
    best_ms = current_ms
    T = T0
    evals = 1
    for _ in range(max_evals - 1):
        neighbor = random_neighbor(current_seq)
        neighbor_ms = makespan_for_sequence(
            orders, neighbor, travel_aware,
            tote_contents=tote_contents, order_to_totes=order_to_totes,
            item_order_per_tote=item_order_per_tote,
        )
        evals += 1
        delta = neighbor_ms - current_ms
        if delta <= 0 or random.random() < (1.0 if delta <= 0 else math.exp(-delta / T)):
            current_seq = neighbor
            current_ms = neighbor_ms
            if current_ms < best_ms:
                best_seq = list(current_seq)
                best_ms = current_ms
                if verbose:
                    print(f"  New best: makespan = {best_ms:.2f}s (evals={evals}, T={T:.4f})")
        T *= cooling
    return best_seq, best_ms, evals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search for better order sequence than LPT; minimize makespan (all load at conv 0, 2.5s spacing).",
    )
    parser.add_argument("generated_dir", type=Path, help="Path to folder with order_itemtypes.csv, order_quantities.csv")
    parser.add_argument(
        "--method",
        choices=["hill", "sa"],
        default="hill",
        help="hill = first-improvement hill climbing; sa = simulated annealing",
    )
    parser.add_argument("--max-evals", type=int, default=500, help="Max sim evaluations for SA (default 500)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for SA")
    parser.add_argument("--out-dir", type=Path, default=Path("lpt"), help="Write best input CSV here")
    parser.add_argument("--save-best", action="store_true", help="Save best solution as conveyor input CSV")
    parser.add_argument("-q", "--quiet", action="store_true", help="Less output")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    from scheduler.core import load_tote_data

    generated_dir = Path(args.generated_dir)
    orders = load_generator_data(generated_dir)
    if not orders:
        raise SystemExit("No orders loaded.")

    tote_contents, order_to_totes = load_tote_data(generated_dir, orders)
    use_tote = bool(tote_contents and order_to_totes)

    n = len(orders)
    lpt_seq = lpt_order(orders)
    lpt_ms = makespan_for_sequence(
        orders, lpt_seq, travel_aware=True,
        tote_contents=tote_contents if use_tote else None,
        order_to_totes=order_to_totes if use_tote else None,
    )
    verbose = not args.quiet

    if verbose:
        print(f"Orders: {n}. LPT (travel-aware) makespan = {lpt_ms:.2f}s", end="")
        if use_tote:
            print(" (load sequence: tote order + item order within tote)")
        else:
            print()
        print(f"Running {args.method} search...")

    kw = {"tote_contents": tote_contents if use_tote else None, "order_to_totes": order_to_totes if use_tote else None}
    if args.method == "hill":
        best_seq, best_ms, evals = hill_climb(orders, lpt_seq, travel_aware=True, verbose=verbose, **kw)
    else:
        best_seq, best_ms, evals = simulated_annealing(
            orders, lpt_seq, travel_aware=True, max_evals=args.max_evals, verbose=verbose, **kw
        )

    if verbose:
        print(f"Best makespan = {best_ms:.2f}s (evals = {evals})")
        if best_ms < lpt_ms:
            print(f"Improvement over LPT: {lpt_ms - best_ms:.2f}s ({(1 - best_ms / lpt_ms) * 100:.1f}%)")
        else:
            print("No improvement over LPT (LPT was already best in neighborhood).")

    if args.save_best or args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "search_best_input.csv"
        counts = build_conveyor_input(orders, best_seq, travel_aware=True)
        write_m3_input(counts, path)
        if verbose:
            print(f"Saved best input to {path}")

    # Print sequence for reproducibility (order indices in position 1st, 2nd, ...)
    if verbose:
        print("Best order sequence (positions 1..n):", best_seq)


if __name__ == "__main__":
    main()
