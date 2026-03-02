#!/usr/bin/env python3
"""
Joint optimization over the four decision points to minimize time when the last order finishes.
Decisions: (1) order sequence, (2) conveyor per order, (3) tote loading order, (4) item order within each tote.
All four are optimized together.
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conveyor_sim import load_conveyors, simulate_greedy
from scheduler.core import (
    NUM_CONVEYORS,
    build_conveyor_input_from_assignment,
    build_load_sequence,
    load_generator_data,
    load_tote_data,
    lpt_order,
    write_m3_input,
    write_m3_input_by_order,
)


def _derived_order_to_conveyor(order_sequence: List[int], travel_aware: bool) -> Dict[int, int]:
    """Map order_id -> conveyor from sequence + travel_aware (for initial solution)."""
    return {
        oid: (NUM_CONVEYORS - 1 - pos % NUM_CONVEYORS) if travel_aware else (pos % NUM_CONVEYORS)
        for pos, oid in enumerate(order_sequence)
    }


@dataclass
class Solution:
    """
    Full solution. Totes themselves (which items are in which tote) are fixed by
    orders_totes.csv—we do NOT create or change totes. We only decide:
    - order_sequence / order_to_conveyor: which order goes to which conveyor (for pick assignment).
    - tote_loading_order: ORDER of totes (which tote we empty first, second, ...).
    - item_order_per_tote: ORDER of items within each tote (same set of items as data; we only reorder).
    """

    order_sequence: List[int]
    order_to_conveyor: Dict[int, int]  # order_id -> conveyor (0..3)
    tote_loading_order: List[int]  # order we empty totes (algorithm choice; totes are from data)
    item_order_per_tote: Dict[int, List[Tuple[int, int, int]]]  # per-tote item ORDER (same items as data)

    def copy(self) -> "Solution":
        return Solution(
            order_sequence=list(self.order_sequence),
            order_to_conveyor=dict(self.order_to_conveyor),
            tote_loading_order=list(self.tote_loading_order),
            item_order_per_tote={t: list(items) for t, items in self.item_order_per_tote.items()},
        )


def orders_on_conveyor(
    order_to_conveyor: Dict[int, int],
    order_sequence: List[int],
) -> Dict[int, List[int]]:
    """conv -> list of order_ids on that conveyor, in order_sequence order (for FIFO pick assignment)."""
    conv_to_orders: Dict[int, List[int]] = {c: [] for c in range(NUM_CONVEYORS)}
    for oid in order_sequence:
        c = order_to_conveyor.get(oid, 0)
        if 0 <= c < NUM_CONVEYORS:
            conv_to_orders[c].append(oid)
    return conv_to_orders


def order_demand(orders: List[Tuple[int, List[Tuple[int, int]]]]) -> Dict[int, Dict[int, int]]:
    """order_id -> {shape -> qty}."""
    out: Dict[int, Dict[int, int]] = {}
    for oid, pairs in orders:
        out[oid] = {}
        for s, q in pairs:
            out[oid][s] = out[oid].get(s, 0) + q
    return out


def compute_last_order_completion(
    picks: List[Tuple[int, int, float]],
    order_to_conveyor: Dict[int, int],
    order_sequence: List[int],
    orders: List[Tuple[int, List[Tuple[int, int]]]],
) -> Tuple[float, Dict[int, float]]:
    """
    Assign each (conv, shape, time) pick to an order on that conveyor (FIFO by order_sequence).
    Return (makespan = max over order completion times, order_id -> completion time).
    """
    conv_orders = orders_on_conveyor(order_to_conveyor, order_sequence)
    demand = order_demand(orders)
    # remaining[order_id][shape] = count still needed
    remaining: Dict[int, Dict[int, int]] = {oid: dict(d) for oid, d in demand.items()}
    order_completion: Dict[int, float] = {oid: 0.0 for oid, _ in orders}

    for conv, shape, t in picks:
        order_list = conv_orders.get(conv, [])
        assigned = False
        for oid in order_list:
            if remaining.get(oid, {}).get(shape, 0) > 0:
                remaining[oid][shape] -= 1
                if remaining[oid][shape] == 0:
                    del remaining[oid][shape]
                order_completion[oid] = max(order_completion[oid], t)
                assigned = True
                break
        if not assigned:
            raise ValueError(f"Pick (conv={conv}, shape={shape}, t={t}) could not be assigned to any order")
    makespan = max(order_completion.values()) if order_completion else 0.0
    return makespan, order_completion


def evaluate_solution(
    solution: Solution,
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    objective: str = "last_order",
    return_trace: bool = False,
):
    """
    Run sim for this solution. objective in ("last_item", "last_order").
    Returns (objective_value, results, trace_events or None).
    """
    load_seq = build_load_sequence(
        solution.order_sequence,
        tote_contents,
        travel_aware=True,  # unused when tote_loading_order is explicit
        item_order_per_tote=solution.item_order_per_tote or None,
        tote_loading_order=solution.tote_loading_order,
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        path = Path(f.name)
    try:
        write_m3_input_by_order(
            orders, solution.order_to_conveyor, solution.order_sequence, path
        )
        conveyors, orders_per_conv = load_conveyors(path)
        out = simulate_greedy(
            conveyors,
            all_load_at_conveyor_0=True,
            load_spacing=2.5,
            load_sequence=load_seq,
            return_trace=return_trace,
            orders_per_conveyor=orders_per_conv,
        )
    finally:
        path.unlink(missing_ok=True)

    if return_trace:
        results, trace = out
    else:
        results = out
        trace = None

    if objective == "last_item":
        value = max(t for _, _, t in results) if results else 0.0
        return (value, results, trace)

    # last_order: assign picks to orders and take max completion time
    picks = [(c, s, t) for c, s, t in results]
    value, _ = compute_last_order_completion(
        picks, solution.order_to_conveyor, solution.order_sequence, orders
    )
    return (value, results, trace)


def solution_from_order_sequence(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    order_sequence: List[int],
    travel_aware: bool = True,
) -> Solution:
    """Build a Solution from order_sequence. Totes are from data (tote_contents). We only set the ORDER of totes (by earliest order position, etc.) and use data order for items within each tote."""
    o2c = _derived_order_to_conveyor(order_sequence, travel_aware)
    order_pos = {oid: pos for pos, oid in enumerate(order_sequence)}
    tote_to_earliest = {}
    tote_to_max_conv = {}
    for tote_id, items in tote_contents.items():
        tote_to_earliest[tote_id] = min(order_pos.get(o, 999) for o, _, _ in items)
        convs = [o2c.get(o, -1) for o, _, _ in items if o2c.get(o, -1) >= 0]
        tote_to_max_conv[tote_id] = max(convs) if convs else 0
    tote_order = sorted(
        tote_contents.keys(),
        key=lambda t: (tote_to_earliest[t], -tote_to_max_conv[t], t),
    )
    # Same items per tote as data; we only (later) may reorder within tote
    item_order = {tote_id: list(items) for tote_id, items in tote_contents.items()}
    return Solution(
        order_sequence=list(order_sequence),
        order_to_conveyor=o2c,
        tote_loading_order=tote_order,
        item_order_per_tote=item_order,
    )


def solution_from_order_and_tote(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    order_sequence: List[int],
    tote_loading_order: List[int],
    travel_aware: bool = True,
) -> Solution:
    """Build a Solution from explicit order_sequence and tote_loading_order. Totes (contents) from data; we only supply the ORDER of totes and use data order for items within each tote."""
    o2c = _derived_order_to_conveyor(order_sequence, travel_aware)
    # Same items per tote as data (we only reorder within tote when optimizing)
    item_order = {tote_id: list(items) for tote_id, items in tote_contents.items()}
    return Solution(
        order_sequence=list(order_sequence),
        order_to_conveyor=o2c,
        tote_loading_order=list(tote_loading_order),
        item_order_per_tote=item_order,
    )


def initial_solution(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    travel_aware: bool = True,
) -> Solution:
    """LPT order, travel-aware conveyor assignment. Totes from data; we only set ORDER of totes (earliest pos, farthest conv, tote_id) and ORDER of items in totes (default = data order)."""
    order_seq = lpt_order(orders)
    return solution_from_order_sequence(orders, tote_contents, order_seq, travel_aware)


def hill_climb_order_only(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    initial_order_sequence: List[int],
    travel_aware: bool = True,
    verbose: bool = False,
) -> Tuple[Solution, float, int]:
    """First-improvement hill climb over order sequence only (swap neighborhood). Returns (best_solution, best_value, evals)."""
    current = solution_from_order_sequence(orders, tote_contents, initial_order_sequence, travel_aware)
    current_val, _, _ = evaluate_solution(current, orders, tote_contents, objective="last_order")
    evals = 1
    n = len(current.order_sequence)
    while True:
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                new_seq = list(current.order_sequence)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                neighbor = solution_from_order_sequence(orders, tote_contents, new_seq, travel_aware)
                val, _, _ = evaluate_solution(neighbor, orders, tote_contents, objective="last_order")
                evals += 1
                if val < current_val:
                    current = neighbor
                    current_val = val
                    improved = True
                    if verbose:
                        print(f"  Order hill climb: last_order={current_val:.2f}s (evals={evals})")
                    break
            if improved:
                break
        if not improved:
            break
    return current, current_val, evals


# --- Neighborhoods ---


def neighbor_swap_orders(solution: Solution) -> Solution:
    """Swap two positions in order_sequence."""
    n = len(solution.order_sequence)
    if n < 2:
        return solution.copy()
    i, j = random.sample(range(n), 2)
    s = solution.copy()
    s.order_sequence[i], s.order_sequence[j] = s.order_sequence[j], s.order_sequence[i]
    return s


def neighbor_insert_order(solution: Solution) -> Solution:
    """Move one order from position i to position j (insert move)."""
    n = len(solution.order_sequence)
    if n < 2:
        return solution.copy()
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return solution.copy()
    s = solution.copy()
    seq = list(s.order_sequence)
    o = seq.pop(i)
    seq.insert(j, o)
    s.order_sequence = seq
    return s


def neighbor_2opt_order(solution: Solution) -> Solution:
    """2-opt: reverse a contiguous segment of order_sequence (i inclusive, j inclusive)."""
    n = len(solution.order_sequence)
    if n < 2:
        return solution.copy()
    i = random.randrange(n)
    j = random.randrange(n)
    if i > j:
        i, j = j, i
    if i == j:
        return solution.copy()
    s = solution.copy()
    seq = list(s.order_sequence)
    seq[i : j + 1] = reversed(seq[i : j + 1])
    s.order_sequence = seq
    return s


def neighbor_swap_totes(solution: Solution) -> Solution:
    """Swap two totes in tote_loading_order."""
    n = len(solution.tote_loading_order)
    if n < 2:
        return solution.copy()
    i, j = random.sample(range(n), 2)
    s = solution.copy()
    s.tote_loading_order[i], s.tote_loading_order[j] = s.tote_loading_order[j], s.tote_loading_order[i]
    return s


def neighbor_insert_tote(solution: Solution) -> Solution:
    """Move one tote from position i to position j in tote_loading_order."""
    n = len(solution.tote_loading_order)
    if n < 2:
        return solution.copy()
    i = random.randrange(n)
    j = random.randrange(n)
    if i == j:
        return solution.copy()
    s = solution.copy()
    totes = list(s.tote_loading_order)
    t = totes.pop(i)
    totes.insert(j, t)
    s.tote_loading_order = totes
    return s


def neighbor_2opt_tote(solution: Solution) -> Solution:
    """2-opt: reverse a contiguous segment of tote_loading_order."""
    n = len(solution.tote_loading_order)
    if n < 2:
        return solution.copy()
    i = random.randrange(n)
    j = random.randrange(n)
    if i > j:
        i, j = j, i
    if i == j:
        return solution.copy()
    s = solution.copy()
    totes = list(s.tote_loading_order)
    totes[i : j + 1] = reversed(totes[i : j + 1])
    s.tote_loading_order = totes
    return s


def neighbor_swap_items_in_tote(solution: Solution) -> Solution:
    """Pick one tote and swap two items in its item_order_per_tote."""
    s = solution.copy()
    tote_ids = [t for t in s.tote_loading_order if len(s.item_order_per_tote.get(t, [])) >= 2]
    if not tote_ids:
        return s
    t = random.choice(tote_ids)
    items = s.item_order_per_tote[t]
    i, j = random.sample(range(len(items)), 2)
    items = list(items)
    items[i], items[j] = items[j], items[i]
    s.item_order_per_tote[t] = items
    return s


def neighbor_swap_conveyors(solution: Solution) -> Solution:
    """Swap conveyor assignment of two orders."""
    s = solution.copy()
    if len(s.order_sequence) < 2:
        return s
    i, j = random.sample(range(len(s.order_sequence)), 2)
    o1, o2 = s.order_sequence[i], s.order_sequence[j]
    c1, c2 = s.order_to_conveyor[o1], s.order_to_conveyor[o2]
    s.order_to_conveyor[o1], s.order_to_conveyor[o2] = c2, c1
    return s


def random_neighbor(solution: Solution) -> Solution:
    """Choose one of: order (swap/insert/2-opt), tote (swap/insert/2-opt), item swap, conveyor swap."""
    r = random.random()
    if r < 1 / 8:
        return neighbor_swap_orders(solution)
    if r < 2 / 8:
        return neighbor_insert_order(solution)
    if r < 3 / 8:
        return neighbor_2opt_order(solution)
    if r < 4 / 8:
        return neighbor_swap_totes(solution)
    if r < 5 / 8:
        return neighbor_insert_tote(solution)
    if r < 6 / 8:
        return neighbor_2opt_tote(solution)
    if r < 7 / 8:
        return neighbor_swap_items_in_tote(solution)
    return neighbor_swap_conveyors(solution)


def simulated_annealing(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    objective: str = "last_order",
    max_evals: int = 1000,
    T0: float = 15.0,
    cooling: float = 0.995,
    seed: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[Solution, float, int]:
    """Run SA over full solution. Returns (best_solution, best_value, num_evals)."""
    if seed is not None:
        random.seed(seed)
    current = initial_solution(orders, tote_contents, travel_aware=True)
    current_val, _, _ = evaluate_solution(current, orders, tote_contents, objective=objective)
    best = current.copy()
    best_val = current_val
    T = T0
    evals = 1
    for _ in range(max_evals - 1):
        neighbor = random_neighbor(current)
        neighbor_val, _, _ = evaluate_solution(neighbor, orders, tote_contents, objective=objective)
        evals += 1
        delta = neighbor_val - current_val
        if delta <= 0 or random.random() < math.exp(-delta / T):
            current = neighbor
            current_val = neighbor_val
            if current_val < best_val:
                best = current.copy()
                best_val = current_val
                if verbose:
                    print(f"  New best: {objective}={best_val:.2f}s (evals={evals}, T={T:.4f})")
        T *= cooling
    return best, best_val, evals


def _order_crossover(parent_a: List[int], parent_b: List[int], rng: random.Random) -> List[int]:
    """Order crossover (OX): copy a segment from parent_a, fill rest from parent_b in order of appearance."""
    n = len(parent_a)
    if n <= 1:
        return list(parent_a)
    i = rng.randrange(n)
    j = rng.randrange(n)
    if i > j:
        i, j = j, i
    if i == j:
        return list(parent_a)
    segment = set(parent_a[i : j + 1])
    child = [None] * n
    child[i : j + 1] = parent_a[i : j + 1]
    remaining = [x for x in parent_b if x not in segment]
    idx = 0
    for k in range(n):
        if child[k] is None:
            child[k] = remaining[idx]
            idx += 1
    return child


def genetic_algorithm(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    objective: str = "last_order",
    max_evals: int = 1000,
    pop_size: int = 30,
    tournament_size: int = 3,
    p_crossover: float = 0.8,
    p_mutate: float = 0.2,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Solution, float, int]:
    """Genetic algorithm: population of (order_sequence, tote_loading_order), OX crossover, swap/insert mutation, selection by last_order. Returns (best_solution, best_value, evals)."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()

    order_ids = [o[0] for o in orders]
    tote_ids = list(tote_contents.keys())
    n_ord = len(order_ids)
    n_tote = len(tote_ids)

    def random_order_seq() -> List[int]:
        out = list(order_ids)
        rng.shuffle(out)
        return out

    def random_tote_order() -> List[int]:
        out = list(tote_ids)
        rng.shuffle(out)
        return out

    def eval_sol(sol: Solution) -> float:
        v, _, _ = evaluate_solution(sol, orders, tote_contents, objective=objective)
        return v

    # Initial population: one LPT-based, rest random
    init_sol = initial_solution(orders, tote_contents, travel_aware=True)
    population: List[Tuple[List[int], List[int]]] = [
        (list(init_sol.order_sequence), list(init_sol.tote_loading_order))
    ]
    for _ in range(pop_size - 1):
        population.append((random_order_seq(), random_tote_order()))

    best_sol = init_sol.copy()
    best_val = eval_sol(best_sol)
    evals = 1

    while evals < max_evals:
        # Evaluate population (cache values)
        pop_sols: List[Solution] = []
        pop_vals: List[float] = []
        for ord_seq, tot_seq in population:
            sol = solution_from_order_and_tote(orders, tote_contents, ord_seq, tot_seq, travel_aware=True)
            v = eval_sol(sol)
            evals += 1
            pop_sols.append(sol)
            pop_vals.append(v)
            if v < best_val:
                best_val = v
                best_sol = sol.copy()
                if verbose:
                    print(f"  GA new best: {objective}={best_val:.2f}s (evals={evals})")
            if evals >= max_evals:
                return best_sol, best_val, evals

        # Tournament selection (minimize)
        def select_one() -> int:
            cands = rng.sample(range(pop_size), min(tournament_size, pop_size))
            return min(cands, key=lambda i: pop_vals[i])

        next_pop: List[Tuple[List[int], List[int]]] = []
        for _ in range(pop_size):
            p1 = select_one()
            p2 = select_one()
            if rng.random() < p_crossover:
                ord_seq = _order_crossover(population[p1][0], population[p2][0], rng)
                tot_seq = _order_crossover(population[p1][1], population[p2][1], rng)
            else:
                ord_seq = list(population[p1][0])
                tot_seq = list(population[p1][1])
            # Mutation
            if rng.random() < p_mutate and n_ord >= 2:
                if rng.random() < 0.5:
                    i, j = rng.sample(range(n_ord), 2)
                    ord_seq[i], ord_seq[j] = ord_seq[j], ord_seq[i]
                else:
                    i, j = rng.sample(range(n_ord), 2)
                    o = ord_seq.pop(i)
                    ord_seq.insert(j, o)
            if rng.random() < p_mutate and n_tote >= 2:
                if rng.random() < 0.5:
                    i, j = rng.sample(range(n_tote), 2)
                    tot_seq[i], tot_seq[j] = tot_seq[j], tot_seq[i]
                else:
                    i, j = rng.sample(range(n_tote), 2)
                    t = tot_seq.pop(i)
                    tot_seq.insert(j, t)
            next_pop.append((ord_seq, tot_seq))
        population = next_pop

    return best_sol, best_val, evals


def hill_climb_full(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    initial: Solution,
    objective: str = "last_order",
    max_evals: int = 500,
    verbose: bool = False,
) -> Tuple[Solution, float, int]:
    """First-improvement hill climb: order (swap, insert, 2-opt), tote (swap, insert, 2-opt), conveyor swap, item order within totes (swap). Returns (best_solution, best_value, evals)."""
    current = initial.copy()
    current_val, _, _ = evaluate_solution(current, orders, tote_contents, objective=objective)
    evals = 1
    no_improve = 0
    while evals < max_evals and no_improve < 4:
        improved = False
        # Order swaps (keep conveyor, tote, item from current)
        n = len(current.order_sequence)
        for i in range(n):
            for j in range(i + 1, n):
                s = current.copy()
                s.order_sequence[i], s.order_sequence[j] = s.order_sequence[j], s.order_sequence[i]
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current = s
                    current_val = val
                    improved = True
                    no_improve = 0
                    if verbose:
                        print(f"  Polish (order): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Order insert (move position i to position j)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = current.copy()
                seq = list(s.order_sequence)
                o = seq.pop(i)
                seq.insert(j, o)
                s.order_sequence = seq
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current = s
                    current_val = val
                    improved = True
                    no_improve = 0
                    if verbose:
                        print(f"  Polish (order insert): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Order 2-opt (reverse segment [i:j+1])
        for i in range(n):
            for j in range(i + 1, n):
                s = current.copy()
                seq = list(s.order_sequence)
                seq[i : j + 1] = reversed(seq[i : j + 1])
                s.order_sequence = seq
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current = s
                    current_val = val
                    improved = True
                    no_improve = 0
                    if verbose:
                        print(f"  Polish (order 2-opt): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Tote swaps
        totes = current.tote_loading_order
        for i in range(len(totes)):
            for j in range(i + 1, len(totes)):
                s = current.copy()
                s.tote_loading_order[i], s.tote_loading_order[j] = s.tote_loading_order[j], s.tote_loading_order[i]
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current = s
                    current_val = val
                    improved = True
                    no_improve = 0
                    if verbose:
                        print(f"  Polish (tote): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Tote insert (move tote from position i to j)
        nt = len(totes)
        for i in range(nt):
            for j in range(nt):
                if i == j:
                    continue
                s = current.copy()
                tlist = list(s.tote_loading_order)
                t = tlist.pop(i)
                tlist.insert(j, t)
                s.tote_loading_order = tlist
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current = s
                    current_val = val
                    improved = True
                    no_improve = 0
                    if verbose:
                        print(f"  Polish (tote insert): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Tote 2-opt (reverse segment of totes)
        for i in range(nt):
            for j in range(i + 1, nt):
                s = current.copy()
                tlist = list(s.tote_loading_order)
                tlist[i : j + 1] = reversed(tlist[i : j + 1])
                s.tote_loading_order = tlist
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current = s
                    current_val = val
                    improved = True
                    no_improve = 0
                    if verbose:
                        print(f"  Polish (tote 2-opt): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Conveyor swap (two orders)
        oids = list(current.order_to_conveyor.keys())
        for i in range(len(oids)):
            for j in range(i + 1, len(oids)):
                o1, o2 = oids[i], oids[j]
                s = current.copy()
                s.order_to_conveyor[o1], s.order_to_conveyor[o2] = s.order_to_conveyor[o2], s.order_to_conveyor[o1]
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current = s
                    current_val = val
                    improved = True
                    no_improve = 0
                    if verbose:
                        print(f"  Polish (conv): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Item order within totes (swap two items in one tote)
        for tote_id in current.tote_loading_order:
            items = current.item_order_per_tote.get(tote_id, [])
            if len(items) < 2:
                continue
            for i in range(len(items)):
                for j in range(i + 1, len(items)):
                    s = current.copy()
                    ilist = list(s.item_order_per_tote[tote_id])
                    ilist[i], ilist[j] = ilist[j], ilist[i]
                    s.item_order_per_tote[tote_id] = ilist
                    val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                    evals += 1
                    if val < current_val:
                        current = s
                        current_val = val
                        improved = True
                        no_improve = 0
                        if verbose:
                            print(f"  Polish (item): {objective}={current_val:.2f}s")
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
    return current, current_val, evals


def hill_climb_order_and_tote(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    initial: Solution,
    objective: str = "last_order",
    max_evals: int = 500,
    verbose: bool = False,
) -> Tuple[Solution, float, int]:
    """First-improvement hill climb over order sequence and tote loading order only (no conveyor, no item order). Returns (best_solution, best_value, evals)."""
    current = initial.copy()
    current_val, _, _ = evaluate_solution(current, orders, tote_contents, objective=objective)
    evals = 1
    no_improve = 0
    while evals < max_evals and no_improve < 4:
        improved = False
        n = len(current.order_sequence)
        # Order swap
        for i in range(n):
            for j in range(i + 1, n):
                s = current.copy()
                s.order_sequence[i], s.order_sequence[j] = s.order_sequence[j], s.order_sequence[i]
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current, current_val, improved, no_improve = s, val, True, 0
                    if verbose:
                        print(f"  OrderTote (order swap): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Order insert
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                s = current.copy()
                seq = list(s.order_sequence)
                o = seq.pop(i)
                seq.insert(j, o)
                s.order_sequence = seq
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current, current_val, improved, no_improve = s, val, True, 0
                    if verbose:
                        print(f"  OrderTote (order insert): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Order 2-opt
        for i in range(n):
            for j in range(i + 1, n):
                s = current.copy()
                seq = list(s.order_sequence)
                seq[i : j + 1] = reversed(seq[i : j + 1])
                s.order_sequence = seq
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current, current_val, improved, no_improve = s, val, True, 0
                    if verbose:
                        print(f"  OrderTote (order 2-opt): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        totes = current.tote_loading_order
        nt = len(totes)
        # Tote swap
        for i in range(nt):
            for j in range(i + 1, nt):
                s = current.copy()
                s.tote_loading_order[i], s.tote_loading_order[j] = s.tote_loading_order[j], s.tote_loading_order[i]
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current, current_val, improved, no_improve = s, val, True, 0
                    if verbose:
                        print(f"  OrderTote (tote swap): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Tote insert
        for i in range(nt):
            for j in range(nt):
                if i == j:
                    continue
                s = current.copy()
                tlist = list(s.tote_loading_order)
                t = tlist.pop(i)
                tlist.insert(j, t)
                s.tote_loading_order = tlist
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current, current_val, improved, no_improve = s, val, True, 0
                    if verbose:
                        print(f"  OrderTote (tote insert): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if improved:
            continue
        # Tote 2-opt
        for i in range(nt):
            for j in range(i + 1, nt):
                s = current.copy()
                tlist = list(s.tote_loading_order)
                tlist[i : j + 1] = reversed(tlist[i : j + 1])
                s.tote_loading_order = tlist
                val, _, _ = evaluate_solution(s, orders, tote_contents, objective=objective)
                evals += 1
                if val < current_val:
                    current, current_val, improved, no_improve = s, val, True, 0
                    if verbose:
                        print(f"  OrderTote (tote 2-opt): {objective}={current_val:.2f}s")
                    break
            if improved:
                break
        if not improved:
            no_improve += 1
    return current, current_val, evals


def _apply_move(sol: Solution, move: Tuple) -> Solution:
    """Apply a move (move_type, *params) to solution; return new solution."""
    kind = move[0]
    s = sol.copy()
    if kind == "swap_order":
        i, j = move[1], move[2]
        s.order_sequence[i], s.order_sequence[j] = s.order_sequence[j], s.order_sequence[i]
    elif kind == "insert_order":
        i, j = move[1], move[2]
        seq = list(s.order_sequence)
        o = seq.pop(i)
        seq.insert(j, o)
        s.order_sequence = seq
    elif kind == "2opt_order":
        i, j = move[1], move[2]
        seq = list(s.order_sequence)
        seq[i : j + 1] = reversed(seq[i : j + 1])
        s.order_sequence = seq
    elif kind == "swap_tote":
        i, j = move[1], move[2]
        s.tote_loading_order[i], s.tote_loading_order[j] = s.tote_loading_order[j], s.tote_loading_order[i]
    elif kind == "insert_tote":
        i, j = move[1], move[2]
        tlist = list(s.tote_loading_order)
        t = tlist.pop(i)
        tlist.insert(j, t)
        s.tote_loading_order = tlist
    elif kind == "2opt_tote":
        i, j = move[1], move[2]
        tlist = list(s.tote_loading_order)
        tlist[i : j + 1] = reversed(tlist[i : j + 1])
        s.tote_loading_order = tlist
    elif kind == "swap_conv":
        o1, o2 = move[1], move[2]
        s.order_to_conveyor[o1], s.order_to_conveyor[o2] = s.order_to_conveyor[o2], s.order_to_conveyor[o1]
    elif kind == "swap_item":
        tote_id, i, j = move[1], move[2], move[3]
        items = list(s.item_order_per_tote.get(tote_id, []))
        if len(items) >= 2 and 0 <= i < len(items) and 0 <= j < len(items):
            items[i], items[j] = items[j], items[i]
            s.item_order_per_tote[tote_id] = items
    return s


def _sample_moves(solution: Solution, n_sample: int, rng: random.Random) -> List[Tuple]:
    """Sample n_sample random moves (order, tote, conveyor, item-order). Returns list of (move_type, *params)."""
    moves: List[Tuple] = []
    n = len(solution.order_sequence)
    nt = len(solution.tote_loading_order)
    oids = list(solution.order_to_conveyor.keys())
    totes_with_two = [t for t in solution.tote_loading_order if len(solution.item_order_per_tote.get(t, [])) >= 2]
    for _ in range(n_sample):
        r = rng.random()
        if r < 1 / 8 and n >= 2:
            i, j = rng.sample(range(n), 2)
            if i > j:
                i, j = j, i
            moves.append(("swap_order", i, j))
        elif r < 2 / 8 and n >= 2:
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                moves.append(("insert_order", i, j))
        elif r < 3 / 8 and n >= 2:
            i, j = rng.randrange(n), rng.randrange(n)
            if i > j:
                i, j = j, i
            if i != j:
                moves.append(("2opt_order", i, j))
        elif r < 4 / 8 and nt >= 2:
            i, j = rng.sample(range(nt), 2)
            if i > j:
                i, j = j, i
            moves.append(("swap_tote", i, j))
        elif r < 5 / 8 and nt >= 2:
            i, j = rng.randrange(nt), rng.randrange(nt)
            if i != j:
                moves.append(("insert_tote", i, j))
        elif r < 6 / 8 and nt >= 2:
            i, j = rng.randrange(nt), rng.randrange(nt)
            if i > j:
                i, j = j, i
            if i != j:
                moves.append(("2opt_tote", i, j))
        elif r < 7 / 8 and n >= 2:
            i, j = rng.sample(range(len(oids)), 2)
            o1, o2 = oids[i], oids[j]
            moves.append(("swap_conv", o1, o2))
        elif totes_with_two:
            t = rng.choice(totes_with_two)
            items = solution.item_order_per_tote[t]
            i, j = rng.sample(range(len(items)), 2)
            if i > j:
                i, j = j, i
            moves.append(("swap_item", t, i, j))
    return moves


def tabu_search(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    initial: Optional[Solution] = None,
    objective: str = "last_order",
    max_evals: int = 1000,
    tabu_tenure: int = 10,
    neighbors_per_iter: int = 80,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Solution, float, int]:
    """Hill climb with tabu list: best non-tabu neighbor (or aspiration). Same neighborhoods as hill_climb. Returns (best_solution, best_value, evals)."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    current = (initial or initial_solution(orders, tote_contents, travel_aware=True)).copy()
    current_val, _, _ = evaluate_solution(current, orders, tote_contents, objective=objective)
    evals = 1
    best_sol = current.copy()
    best_val = current_val
    tabu_list: deque = deque(maxlen=tabu_tenure)

    while evals < max_evals:
        moves = _sample_moves(current, min(neighbors_per_iter, max_evals - evals), rng)
        candidates: List[Tuple[Tuple, Solution, float]] = []
        for move in moves:
            neighbor = _apply_move(current, move)
            val, _, _ = evaluate_solution(neighbor, orders, tote_contents, objective=objective)
            evals += 1
            candidates.append((move, neighbor, val))
            if evals >= max_evals:
                break
        if not candidates:
            break
        candidates.sort(key=lambda x: x[2])
        chosen = None
        for move, sol, val in candidates:
            if move not in tabu_list:
                chosen = (move, sol, val)
                break
            if val < best_val:
                chosen = (move, sol, val)
                break
        if chosen is None:
            chosen = candidates[0]
        move, new_sol, new_val = chosen
        tabu_list.append(move)
        current = new_sol
        current_val = new_val
        if current_val < best_val:
            best_val = current_val
            best_sol = current.copy()
            if verbose:
                print(f"  Tabu new best: {objective}={best_val:.2f}s (evals={evals})")

    return best_sol, best_val, evals


def iterated_local_search(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    objective: str = "last_order",
    max_evals: int = 1000,
    perturb_strength: int = 3,
    seed: Optional[int] = None,
    verbose: bool = False,
) -> Tuple[Solution, float, int]:
    """Iterated local search: hill climb to local opt, perturb (k random moves), repeat. Returns (best_solution, best_value, evals)."""
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    current = initial_solution(orders, tote_contents, travel_aware=True)
    best, best_val, total_evals = hill_climb_full(
        orders, tote_contents, current,
        objective=objective, max_evals=max_evals, verbose=False,
    )
    current = best.copy()

    while total_evals < max_evals:
        # Perturb: apply k random moves (no extra evals)
        for _ in range(perturb_strength):
            moves = _sample_moves(current, 1, rng)
            if moves:
                current = _apply_move(current, moves[0])
        remaining = max_evals - total_evals
        if remaining <= 0:
            break
        current, current_val, used = hill_climb_full(
            orders, tote_contents, current,
            objective=objective, max_evals=remaining, verbose=False,
        )
        total_evals += used
        if current_val < best_val:
            best_val = current_val
            best = current.copy()
            if verbose:
                print(f"  ILS new best: {objective}={best_val:.2f}s (evals={total_evals})")

    return best, best_val, total_evals


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Joint optimization: order sequence, tote order, item order within totes; minimize last-order completion.",
    )
    parser.add_argument("generated_dir", type=Path, help="Path to generated/ folder")
    parser.add_argument("--out-dir", type=Path, default=Path("scheduler"), help="Output directory")
    parser.add_argument("--objective", choices=["last_order", "last_item"], default="last_order")
    parser.add_argument("--max-evals", type=int, default=800)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-best", action="store_true", help="Save best conveyor input CSV and playbook")
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    orders = load_generator_data(generated_dir)
    if not orders:
        raise SystemExit("No orders loaded.")
    tote_contents, order_to_totes = load_tote_data(generated_dir, orders)
    if not tote_contents or not order_to_totes:
        raise SystemExit("Joint optimization requires orders_totes.csv (tote data).")

    verbose = not args.quiet
    if verbose:
        print(f"Objective: {args.objective}. Running SA (max_evals={args.max_evals})...")

    best_sol, best_val, evals = simulated_annealing(
        orders,
        tote_contents,
        objective=args.objective,
        max_evals=args.max_evals,
        seed=args.seed,
        verbose=verbose,
    )

    if verbose:
        print(f"Best {args.objective} = {best_val:.2f}s (evals={evals})")
        print("Best order_sequence:", best_sol.order_sequence)
        print("Best order_to_conveyor (order_id -> conv):", best_sol.order_to_conveyor)
        print("Best tote_loading_order (first 10):", best_sol.tote_loading_order[:10])

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.save_best:
        write_m3_input_by_order(
            orders, best_sol.order_to_conveyor, best_sol.order_sequence,
            out_dir / "joint_best_input.csv",
        )
        if verbose:
            print(f"Saved {out_dir / 'joint_best_input.csv'}")
        # Write playbook for best solution: need to run sim with trace and dump events
        load_seq = build_load_sequence(
            best_sol.order_sequence,
            tote_contents,
            travel_aware=True,
            item_order_per_tote=best_sol.item_order_per_tote,
            tote_loading_order=best_sol.tote_loading_order,
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            tmp = Path(f.name)
        try:
            write_m3_input_by_order(
                orders, best_sol.order_to_conveyor, best_sol.order_sequence, tmp
            )
            conveyors, orders_per_conv = load_conveyors(tmp)
            results, trace = simulate_greedy(
                conveyors,
                all_load_at_conveyor_0=True,
                load_spacing=2.5,
                load_sequence=load_seq,
                return_trace=True,
                orders_per_conveyor=orders_per_conv,
            )
        finally:
            tmp.unlink(missing_ok=True)
        playbook_path = out_dir / "joint_best_playbook.txt"
        with playbook_path.open("w", encoding="utf-8") as f:
            f.write("JOINT BEST SOLUTION – EVENT PLAYBOOK\n")
            f.write(f"Objective: {args.objective} = {best_val:.2f}s\n")
            f.write("=" * 60 + "\n\n")
            for i, (t, ev, payload) in enumerate(trace):
                if ev == "LOAD":
                    f.write(f"{i+1:5}.  t={t:7.2f}s  LOAD   item_id={payload[0]:3}  shape={payload[1]}  conv={payload[2]}\n")
                elif ev == "HOP":
                    f.write(f"{i+1:5}.  t={t:7.2f}s  HOP    item_id={payload[0]:3}  {payload[1]} -> {payload[2]}\n")
                else:
                    f.write(f"{i+1:5}.  t={t:7.2f}s  PICK   item_id={payload[0]:3}  shape={payload[1]}  conv={payload[2]}\n")
        if verbose:
            print(f"Saved {playbook_path}")

    # Save solution summary (order sequence, conveyor assignment, tote order)
    summary_path = out_dir / "joint_best_solution.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"Objective: {args.objective}\nBest value: {best_val:.2f}s\n\n")
        f.write("Order sequence (position -> order_id):\n")
        for pos, oid in enumerate(best_sol.order_sequence):
            conv = best_sol.order_to_conveyor.get(oid, -1)
            f.write(f"  {pos+1}. order {oid} -> conveyor {conv}\n")
        f.write("\nOrder -> conveyor:\n")
        for oid in sorted(best_sol.order_to_conveyor.keys()):
            f.write(f"  order {oid} -> conv {best_sol.order_to_conveyor[oid]}\n")
        f.write("\nTote loading order:\n")
        for i, t in enumerate(best_sol.tote_loading_order):
            f.write(f"  {i+1}. tote {t}\n")
    if verbose:
        print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
