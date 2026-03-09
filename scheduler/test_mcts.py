#!/usr/bin/env python3
"""Minimal sanity test for MCTS solver."""

from __future__ import annotations

import csv
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bens_methods.mcts_solver import MCTSConfig, export_m3_input, solve
from scheduler.joint_solution import evaluate_solution


def synthetic_instance():
    # orders: list of (order_id, [(shape_id, qty), ...])
    orders = [
        (0, [(0, 2)]),
        (1, [(1, 1)]),
        (2, [(0, 1), (1, 1)]),
        (3, [(2, 1)]),
    ]
    # tote_contents: tote_id -> [(order_id, shape, qty), ...]
    tote_contents = {
        0: [(0, 0, 2)],
        1: [(1, 1, 1)],
        2: [(2, 0, 1)],
        3: [(2, 1, 1)],
        4: [(3, 2, 1)],
    }
    return orders, tote_contents


def _sorted_items(items):
    return sorted(items, key=lambda x: (x[0], x[1], x[2]))


def main() -> None:
    orders, tote_contents = synthetic_instance()

    def _eval(sol):
        return evaluate_solution(sol, orders, tote_contents, objective="last_order")

    cfg = MCTSConfig(log_improvements=False)
    sol, ms = solve(
        orders,
        tote_contents,
        evaluate=_eval,
        n_iterations=50,
        seed=123,
        config=cfg,
    )

    assert ms >= 0.0, "makespan should be non-negative"

    order_ids = [oid for oid, _ in orders]
    assert sorted(sol.order_sequence) == sorted(order_ids), "order sequence must be a permutation"
    assert len(sol.order_to_conveyor) == len(orders), "order_to_conveyor size mismatch"
    assert all(0 <= c <= 3 for c in sol.order_to_conveyor.values()), "invalid conveyor id"

    tote_ids = sorted(tote_contents.keys())
    assert sorted(sol.tote_loading_order) == tote_ids, "tote loading order must include all totes"

    for tote_id, items in tote_contents.items():
        got = sol.item_order_per_tote.get(tote_id, [])
        assert _sorted_items(got) == _sorted_items(items), "item order must be a permutation per tote"

    # Export per-conveyor M3 input and verify header
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "input_conveyor.csv"
        export_m3_input(sol, orders, out_path, format="per_conveyor")
        with out_path.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert header == [
                "conv_num",
                "cirle",
                "pentagon",
                "trapezoid",
                "triangle",
                "star",
                "moon",
                "heart",
                "cross",
            ], "unexpected header in per-conveyor export"

    print("MCTS test passed (synthetic instance).")


if __name__ == "__main__":
    main()
