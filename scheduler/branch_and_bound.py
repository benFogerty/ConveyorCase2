#!/usr/bin/env python3
"""
Budgeted branch-and-bound solver for tote loading order.
"""

from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from conveyor_sim import HOP_SECONDS, LOAD_SPACING_SECONDS
from scheduler.core import (
    NUM_CONVEYORS,
    build_conveyor_input_from_assignment,
    lpt_order,
    write_m3_input,
    write_m3_input_by_order,
)
from scheduler.joint_solution import Solution, evaluate_solution

Order = Tuple[int, List[Tuple[int, int]]]
ToteContents = Dict[int, List[Tuple[int, int, int]]]
EvaluateFn = Callable[[Solution], float | Tuple[float, object, object]]


@dataclass
class BranchAndBoundConfig:
    top_k_candidates: int = 20
    completion_policy: str = "greedy_eval"  # "heuristic", "random", or "greedy_eval"
    log_improvements: bool = True


@dataclass
class State:
    scheduled_totes: Tuple[int, ...]
    remaining_totes: Tuple[int, ...]
    prefix_units: int
    last_load_index_per_order: Dict[int, int]
    remaining_units_per_order: Dict[int, int]

    def is_terminal(self) -> bool:
        return len(self.remaining_totes) == 0


@dataclass(order=True)
class QueueNode:
    priority: Tuple[float, int, int]
    state: State = field(compare=False)
    lower_bound: float = field(compare=False)


class BranchAndBound:
    def __init__(
        self,
        orders: List[Order],
        tote_contents: ToteContents,
        evaluate: EvaluateFn,
        config: Optional[BranchAndBoundConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.orders = orders
        self.tote_contents = tote_contents
        self.evaluate = evaluate
        self.config = config or BranchAndBoundConfig()
        self.rng = random.Random(seed)
        self.eval_count = 0
        self.best_solution: Optional[Solution] = None
        self.best_makespan = float("inf")
        self._eval_cache: Dict[Tuple[int, ...], float] = {}

        self.order_sequence = lpt_order(orders)
        self.order_pos = {oid: i for i, oid in enumerate(self.order_sequence)}
        self.order_to_conveyor = self._derived_order_to_conveyor(self.order_sequence)
        self.item_order = self._build_item_order()
        self.tote_scores = self._compute_tote_scores()
        self.total_units_by_order = self._units_by_order(
            item for items in self.item_order.values() for item in items
        )
        self.min_travel_by_order = {
            oid: (self.order_to_conveyor.get(oid, 1) - 1) * HOP_SECONDS
            for oid, _ in orders
        }

    def _derived_order_to_conveyor(
        self,
        order_sequence: List[int],
        travel_aware: bool = True,
    ) -> Dict[int, int]:
        if travel_aware:
            return {
                oid: (NUM_CONVEYORS - (pos % NUM_CONVEYORS))
                for pos, oid in enumerate(order_sequence)
            }
        return {oid: ((pos % NUM_CONVEYORS) + 1) for pos, oid in enumerate(order_sequence)}

    def _build_item_order(self) -> Dict[int, List[Tuple[int, int, int]]]:
        item_order: Dict[int, List[Tuple[int, int, int]]] = {}
        for tote_id, items in self.tote_contents.items():
            item_order[tote_id] = sorted(
                items,
                key=lambda it: (self.order_pos.get(it[0], 10**9), it[1], it[0]),
            )
        return item_order

    def _compute_tote_scores(self) -> Dict[int, Tuple[int, int, int]]:
        scores: Dict[int, Tuple[int, int, int]] = {}
        for tote_id, items in self.item_order.items():
            earliest = min(self.order_pos.get(o, 10**9) for o, _, _ in items)
            convs = [self.order_to_conveyor.get(o, 1) for o, _, _ in items]
            max_conv = max(convs) if convs else 1
            scores[tote_id] = (earliest, -max_conv, tote_id)
        return scores

    def _units_by_order(self, items) -> Dict[int, int]:
        units: Dict[int, int] = {}
        for order_id, _shape, qty in items:
            units[order_id] = units.get(order_id, 0) + qty
        return units

    def _action_candidates(self, remaining: Tuple[int, ...]) -> List[int]:
        if not remaining:
            return []
        scored = sorted(remaining, key=lambda t: self.tote_scores.get(t, (10**9, 0, t)))
        k = self.config.top_k_candidates
        if k is None or k <= 0 or k >= len(scored):
            return scored
        return scored[:k]

    def _decode_solution(self, tote_order: List[int] | Tuple[int, ...]) -> Solution:
        return Solution(
            order_sequence=list(self.order_sequence),
            order_to_conveyor=dict(self.order_to_conveyor),
            tote_loading_order=list(tote_order),
            item_order_per_tote={t: list(items) for t, items in self.item_order.items()},
        )

    def _evaluate_solution(self, solution: Solution) -> float:
        try:
            out = self.evaluate(solution)
        except Exception:
            self.eval_count += 1
            return float("inf")
        value = out[0] if isinstance(out, tuple) else out
        self.eval_count += 1
        return float(value)

    def _evaluate_tote_order(self, tote_order: List[int] | Tuple[int, ...]) -> float:
        key = tuple(tote_order)
        if key in self._eval_cache:
            return self._eval_cache[key]
        sol = self._decode_solution(key)
        ms = self._evaluate_solution(sol)
        self._eval_cache[key] = ms
        return ms

    def _initial_state(self) -> State:
        return State(
            scheduled_totes=tuple(),
            remaining_totes=tuple(sorted(self.item_order.keys())),
            prefix_units=0,
            last_load_index_per_order={},
            remaining_units_per_order=dict(self.total_units_by_order),
        )

    def _make_child(self, state: State, tote_id: int) -> State:
        last_load_index = dict(state.last_load_index_per_order)
        remaining_units = dict(state.remaining_units_per_order)
        next_load_index = state.prefix_units

        for order_id, _shape, qty in self.item_order.get(tote_id, []):
            if qty <= 0:
                continue
            remaining_units[order_id] = max(0, remaining_units.get(order_id, 0) - qty)
            last_load_index[order_id] = next_load_index + qty - 1
            next_load_index += qty

        return State(
            scheduled_totes=state.scheduled_totes + (tote_id,),
            remaining_totes=tuple(t for t in state.remaining_totes if t != tote_id),
            prefix_units=next_load_index,
            last_load_index_per_order=last_load_index,
            remaining_units_per_order=remaining_units,
        )

    def _lower_bound(self, state: State) -> float:
        lower = (state.prefix_units - 1) * LOAD_SPACING_SECONDS if state.prefix_units > 0 else 0.0
        for order_id, _ in self.orders:
            remaining = state.remaining_units_per_order.get(order_id, 0)
            last_prefix_idx = state.last_load_index_per_order.get(order_id, -1)
            if remaining > 0:
                last_required_idx = max(last_prefix_idx, state.prefix_units + remaining - 1)
            else:
                if last_prefix_idx < 0:
                    continue
                last_required_idx = last_prefix_idx
            lower = max(
                lower,
                last_required_idx * LOAD_SPACING_SECONDS + self.min_travel_by_order.get(order_id, 0.0),
            )
        return lower

    def _complete_sequence(self, state: State) -> List[int]:
        if self.config.completion_policy == "random":
            remaining = list(state.remaining_totes)
            self.rng.shuffle(remaining)
            return list(state.scheduled_totes) + remaining
        if self.config.completion_policy == "heuristic":
            remaining = sorted(
                state.remaining_totes,
                key=lambda t: self.tote_scores.get(t, (10**9, 0, t)),
            )
            return list(state.scheduled_totes) + remaining
        return self._complete_sequence_greedy(state)

    def _complete_sequence_greedy(self, state: State) -> List[int]:
        scheduled = list(state.scheduled_totes)
        remaining = list(state.remaining_totes)
        while remaining:
            candidates = self._action_candidates(tuple(remaining)) or list(remaining)
            best_tote = candidates[0]
            best_ms = float("inf")
            for tote_id in candidates:
                suffix = [t for t in remaining if t != tote_id]
                suffix.sort(key=lambda t: self.tote_scores.get(t, (10**9, 0, t)))
                full = scheduled + [tote_id] + suffix
                ms = self._evaluate_tote_order(full)
                if ms < best_ms:
                    best_ms = ms
                    best_tote = tote_id
            scheduled.append(best_tote)
            remaining.remove(best_tote)
        return scheduled

    def _consider_completion(self, state: State) -> None:
        tote_order = self._complete_sequence(state)
        ms = self._evaluate_tote_order(tote_order)
        if ms < self.best_makespan:
            self.best_makespan = ms
            self.best_solution = self._decode_solution(tote_order)
            if self.config.log_improvements:
                print(f"  BranchAndBound new best: last_order={ms:.2f}s (evals={self.eval_count})")

    def run(
        self,
        time_limit_seconds: Optional[float] = None,
        n_iterations: Optional[int] = None,
    ) -> Tuple[Solution, float]:
        root = self._initial_state()
        self._consider_completion(root)

        frontier: List[QueueNode] = []
        root_lb = self._lower_bound(root)
        heapq.heappush(frontier, QueueNode((root_lb, 0, 0), root, root_lb))

        expanded = 0
        push_index = 1
        start = time.time()

        while frontier:
            if n_iterations is not None and expanded >= n_iterations:
                break
            if time_limit_seconds is not None and (time.time() - start) >= time_limit_seconds:
                break

            node = heapq.heappop(frontier)
            state = node.state
            if node.lower_bound >= self.best_makespan:
                continue
            if state.is_terminal():
                ms = self._evaluate_tote_order(state.scheduled_totes)
                if ms < self.best_makespan:
                    self.best_makespan = ms
                    self.best_solution = self._decode_solution(state.scheduled_totes)
                continue

            expanded += 1
            for tote_id in self._action_candidates(state.remaining_totes):
                child = self._make_child(state, tote_id)
                child_lb = self._lower_bound(child)
                if child_lb >= self.best_makespan:
                    continue
                self._consider_completion(child)
                if child.is_terminal():
                    continue
                heapq.heappush(
                    frontier,
                    QueueNode(
                        (child_lb, -len(child.scheduled_totes), push_index),
                        child,
                        child_lb,
                    ),
                )
                push_index += 1

        if self.best_solution is None:
            fallback_order = self._complete_sequence(root)
            self.best_solution = self._decode_solution(fallback_order)
            self.best_makespan = self._evaluate_tote_order(fallback_order)

        return self.best_solution, self.best_makespan


def solve(
    orders: List[Order],
    tote_contents: ToteContents,
    evaluate: Optional[EvaluateFn] = None,
    time_limit_seconds: Optional[float] = None,
    n_iterations: Optional[int] = 800,
    seed: Optional[int] = None,
    config: Optional[BranchAndBoundConfig] = None,
) -> Tuple[Solution, float]:
    if evaluate is None:
        def _eval(sol: Solution):
            return evaluate_solution(sol, orders, tote_contents, objective="last_order")
        evaluate = _eval
    solver = BranchAndBound(orders, tote_contents, evaluate, config=config, seed=seed)
    return solver.run(time_limit_seconds=time_limit_seconds, n_iterations=n_iterations)


def export_m3_input(
    solution: Solution,
    orders: List[Order],
    out_path: str | Path,
    format: str = "per_conveyor",
) -> None:
    out_path = Path(out_path)
    if format == "per_conveyor":
        counts = build_conveyor_input_from_assignment(orders, solution.order_to_conveyor)
        write_m3_input(counts, out_path)
    elif format == "per_order":
        write_m3_input_by_order(orders, solution.order_to_conveyor, solution.order_sequence, out_path)
    else:
        raise ValueError("format must be 'per_conveyor' or 'per_order'")
