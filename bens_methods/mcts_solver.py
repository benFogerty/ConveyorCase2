#!/usr/bin/env python3
"""
Monte Carlo Tree Search (MCTS) solver for tote loading order.
Primary actions: choose next tote to load. Other decisions are decoded deterministically.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

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
class MCTSConfig:
    c_ucb: float = 1.4
    top_k_candidates: int = 20
    reward_mode: str = "negative"  # "negative" or "reciprocal"
    rollout_policy: str = "greedy_eval"  # "heuristic", "random", or "greedy_eval"
    log_improvements: bool = True


@dataclass
class State:
    scheduled_totes: List[int]
    remaining_totes: List[int]

    def is_terminal(self) -> bool:
        return len(self.remaining_totes) == 0


@dataclass
class Node:
    state: State
    parent: Optional["Node"] = None
    action: Optional[int] = None
    children: Dict[int, "Node"] = field(default_factory=dict)
    visit_count: int = 0
    total_value: float = 0.0
    best_value: float = float("-inf")
    untried_actions: List[int] = field(default_factory=list)

    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0


class MCTS:
    def __init__(
        self,
        orders: List[Order],
        tote_contents: ToteContents,
        evaluate: EvaluateFn,
        config: Optional[MCTSConfig] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.orders = orders
        self.tote_contents = tote_contents
        self.evaluate = evaluate
        self.config = config or MCTSConfig()
        self.rng = random.Random(seed)
        self.eval_count = 0
        self.best_solution: Optional[Solution] = None
        self.best_makespan = float("inf")
        self._eval_cache: Dict[Tuple[int, ...], float] = {}

        # Precompute deterministic decode helpers
        self.order_sequence = lpt_order(orders)
        self.order_pos = {oid: i for i, oid in enumerate(self.order_sequence)}
        self.order_to_conveyor = self._derived_order_to_conveyor(self.order_sequence)
        self.tote_scores = self._compute_tote_scores()

    def _derived_order_to_conveyor(self, order_sequence: List[int], travel_aware: bool = True) -> Dict[int, int]:
        if travel_aware:
            return {
                oid: (NUM_CONVEYORS - 1 - pos % NUM_CONVEYORS)
                for pos, oid in enumerate(order_sequence)
            }
        return {oid: (pos % NUM_CONVEYORS) for pos, oid in enumerate(order_sequence)}

    def _compute_tote_scores(self) -> Dict[int, Tuple[int, int, int]]:
        # Score tuple: (earliest_order_pos, -max_conveyor, tote_id) for stable ordering
        scores: Dict[int, Tuple[int, int, int]] = {}
        for tote_id, items in self.tote_contents.items():
            earliest = min(self.order_pos.get(o, 10**9) for o, _, _ in items)
            convs = [self.order_to_conveyor.get(o, 0) for o, _, _ in items]
            max_conv = max(convs) if convs else 0
            scores[tote_id] = (earliest, -max_conv, tote_id)
        return scores

    def _action_candidates(self, remaining: List[int]) -> List[int]:
        if not remaining:
            return []
        # Rank by heuristic score; take top-K
        scored = sorted(remaining, key=lambda t: self.tote_scores.get(t, (10**9, 0, t)))
        k = self.config.top_k_candidates
        if k is None or k <= 0 or k >= len(scored):
            return scored
        return scored[:k]

    def _select_child(self, node: Node) -> Node:
        assert node.children
        log_parent = math.log(node.visit_count) if node.visit_count > 0 else 0.0
        best_score = float("-inf")
        best_child = None
        for child in node.children.values():
            if child.visit_count == 0:
                ucb = float("inf")
            else:
                exploit = child.total_value / child.visit_count
                explore = self.config.c_ucb * math.sqrt(log_parent / child.visit_count)
                ucb = exploit + explore
            if ucb > best_score:
                best_score = ucb
                best_child = child
        return best_child or next(iter(node.children.values()))

    def _rollout_sequence(self, state: State) -> List[int]:
        if self.config.rollout_policy == "random":
            return self._rollout_sequence_random(state)
        if self.config.rollout_policy == "greedy_eval":
            return self._rollout_sequence_greedy(state)
        return self._rollout_sequence_heuristic(state)

    def _rollout_sequence_random(self, state: State) -> List[int]:
        remaining = list(state.remaining_totes)
        self.rng.shuffle(remaining)
        return list(state.scheduled_totes) + remaining

    def _rollout_sequence_heuristic(self, state: State) -> List[int]:
        remaining = list(state.remaining_totes)
        remaining.sort(key=lambda t: self.tote_scores.get(t, (10**9, 0, t)))
        return list(state.scheduled_totes) + remaining

    def _evaluate_tote_order(self, tote_order: List[int]) -> float:
        key = tuple(tote_order)
        if key in self._eval_cache:
            return self._eval_cache[key]
        sol = self._decode_solution(tote_order)
        ms = self._evaluate_solution(sol)
        self._eval_cache[key] = ms
        return ms

    def _rollout_sequence_greedy(self, state: State) -> List[int]:
        scheduled = list(state.scheduled_totes)
        remaining = list(state.remaining_totes)
        while remaining:
            candidates = self._action_candidates(remaining)
            if not candidates:
                candidates = list(remaining)
            best_tote = None
            best_ms = float("inf")
            for t in candidates:
                rest = [x for x in remaining if x != t]
                rest.sort(key=lambda x: self.tote_scores.get(x, (10**9, 0, x)))
                full = scheduled + [t] + rest
                ms = self._evaluate_tote_order(full)
                if ms < best_ms:
                    best_ms = ms
                    best_tote = t
            if best_tote is None:
                best_tote = candidates[0]
            scheduled.append(best_tote)
            remaining.remove(best_tote)
        return scheduled

    def _decode_solution(self, tote_order: List[int]) -> Solution:
        # Item order within each tote: earliest order priority, then shape
        item_order: Dict[int, List[Tuple[int, int, int]]] = {}
        for tote_id, items in self.tote_contents.items():
            ordered = sorted(
                items,
                key=lambda it: (
                    self.order_pos.get(it[0], 10**9),
                    it[1],
                    it[0],
                ),
            )
            item_order[tote_id] = ordered
        return Solution(
            order_sequence=list(self.order_sequence),
            order_to_conveyor=dict(self.order_to_conveyor),
            tote_loading_order=list(tote_order),
            item_order_per_tote=item_order,
        )

    def _reward(self, makespan: float) -> float:
        if self.config.reward_mode == "reciprocal":
            return 1.0 / (1.0 + makespan)
        return -makespan

    def _evaluate_solution(self, solution: Solution) -> float:
        out = self.evaluate(solution)
        value = out[0] if isinstance(out, tuple) else out
        self.eval_count += 1
        return float(value)

    def run(
        self,
        time_limit_seconds: Optional[float] = None,
        n_iterations: Optional[int] = None,
    ) -> Tuple[Solution, float]:
        totes = sorted(self.tote_contents.keys())
        root_state = State(scheduled_totes=[], remaining_totes=totes)
        root = Node(state=root_state)
        root.untried_actions = self._action_candidates(root_state.remaining_totes)

        # Initial heuristic solution
        init_tote_order = self._rollout_sequence(root_state)
        init_sol = self._decode_solution(init_tote_order)
        init_ms = self._evaluate_solution(init_sol)
        self.best_solution = init_sol
        self.best_makespan = init_ms

        start = time.time()
        it = 0
        while True:
            if n_iterations is not None and it >= n_iterations:
                break
            if time_limit_seconds is not None and (time.time() - start) >= time_limit_seconds:
                break
            it += 1

            node = root
            # Selection
            while not node.state.is_terminal() and node.is_fully_expanded() and node.children:
                node = self._select_child(node)

            # Expansion
            if not node.state.is_terminal() and node.untried_actions:
                action = self.rng.choice(node.untried_actions)
                node.untried_actions.remove(action)
                new_scheduled = list(node.state.scheduled_totes) + [action]
                new_remaining = [t for t in node.state.remaining_totes if t != action]
                child_state = State(scheduled_totes=new_scheduled, remaining_totes=new_remaining)
                child = Node(state=child_state, parent=node, action=action)
                child.untried_actions = self._action_candidates(child_state.remaining_totes)
                node.children[action] = child
                node = child

            # Simulation
            rollout_order = self._rollout_sequence(node.state)
            sol = self._decode_solution(rollout_order)
            ms = self._evaluate_solution(sol)
            reward = self._reward(ms)

            if ms < self.best_makespan:
                self.best_makespan = ms
                self.best_solution = sol
                if self.config.log_improvements:
                    print(f"  MCTS new best: last_order={ms:.2f}s (evals={self.eval_count})")

            # Backpropagation
            cur = node
            while cur is not None:
                cur.visit_count += 1
                cur.total_value += reward
                if reward > cur.best_value:
                    cur.best_value = reward
                cur = cur.parent

        return self.best_solution, self.best_makespan


def solve(
    orders: List[Order],
    tote_contents: ToteContents,
    evaluate: Optional[EvaluateFn] = None,
    time_limit_seconds: Optional[float] = None,
    n_iterations: Optional[int] = 800,
    seed: Optional[int] = None,
    config: Optional[MCTSConfig] = None,
) -> Tuple[Solution, float]:
    if evaluate is None:
        def _eval(sol: Solution):
            return evaluate_solution(sol, orders, tote_contents, objective="last_order")
        evaluate = _eval
    mcts = MCTS(orders, tote_contents, evaluate, config=config, seed=seed)
    return mcts.run(time_limit_seconds=time_limit_seconds, n_iterations=n_iterations)


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
