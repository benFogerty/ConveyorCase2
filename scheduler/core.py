#!/usr/bin/env python3
"""
LPT (Longest Processing Time First) scheduler.

Optimization goal: choose the optimal ORDER SEQUENCE (which order is 1st, 2nd, ...)
and the optimal TOTE LOADING ORDER to minimize makespan. This module addresses
the order-sequence part: it reads data generator output (order_itemtypes,
order_quantities), sorts orders by total item count descending (LPT), assigns
them to conveyors 0..3 in round-robin order, and writes an M3_Example input CSV.
Tote loading order is currently implied by the order sequence (see README).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Must match conveyor_sim.SHAPE_COLUMNS
SHAPE_COLUMNS = [
    "cirle",
    "pentagon",
    "trapezoid",
    "triangle",
    "star",
    "moon",
    "heart",
    "cross",
]
NUM_CONVEYORS = 4
NUM_SHAPES = len(SHAPE_COLUMNS)


def load_generator_data(generated_dir: Path) -> List[Tuple[int, List[Tuple[int, int]]]]:
    """
    Load order_itemtypes and order_quantities from generated_dir (no headers).
    Returns list of (order_idx, [(shape_id, qty), ...]).
    """
    types_path = generated_dir / "order_itemtypes.csv"
    quant_path = generated_dir / "order_quantities.csv"
    if not types_path.exists():
        raise FileNotFoundError(f"Missing {types_path}")
    if not quant_path.exists():
        raise FileNotFoundError(f"Missing {quant_path}")

    def parse_row(path: Path) -> List[List[float]]:
        rows: List[List[float]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cells = [c.strip() for c in line.split(",")]
                row: List[float] = []
                for c in cells:
                    if c == "" or c.lower() == "nan":
                        row.append(float("nan"))
                    else:
                        try:
                            row.append(float(c))
                        except ValueError:
                            row.append(float("nan"))
                rows.append(row)
        return rows

    type_rows = parse_row(types_path)
    quant_rows = parse_row(quant_path)
    if len(type_rows) != len(quant_rows):
        raise ValueError(
            f"Row count mismatch: itemtypes {len(type_rows)} vs quantities {len(quant_rows)}"
        )

    orders: List[Tuple[int, List[Tuple[int, int]]]] = []
    for order_idx, (trow, qrow) in enumerate(zip(type_rows, quant_rows)):
        pairs: List[Tuple[int, int]] = []
        for t, q in zip(trow, qrow):
            if t != t or q != q:  # NaN check
                continue
            shape_id = int(t)
            qty = int(round(q))
            if qty <= 0 or shape_id < 0:
                continue
            if shape_id >= NUM_SHAPES:
                continue  # skip invalid shape
            pairs.append((shape_id, qty))
        orders.append((order_idx, pairs))
    return orders


def load_tote_data(
    generated_dir: Path,
    orders: List[Tuple[int, List[Tuple[int, int]]]],
) -> Tuple[dict[int, List[Tuple[int, int, int]]], dict[int, List[Tuple[int, int, int]]]]:
    """
    Load orders_totes.csv: which tote each (order, item) is in. Totes are NOT created here—
    the data defines which items are in which tote (per the case: "pick items into totes
    as determined by your random data"). We only use this to build tote_contents; ordering
    of totes and of items within totes is decided by the algorithm later.
    Returns:
      tote_contents[tote_id] = [(order_id, shape, qty), ...]  (fixed by data)
      order_to_totes[order_id] = [(tote_id, shape, qty), ...]
    If orders_totes.csv is missing, returns ({}, {}).
    """
    tote_path = generated_dir / "orders_totes.csv"
    if not tote_path.exists():
        return {}, {}

    def parse_row(path: Path) -> List[List[float]]:
        rows: List[List[float]] = []
        with path.open("r", newline="", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cells = [c.strip() for c in line.split(",")]
                row: List[float] = []
                for c in cells:
                    if c == "" or c.lower() == "nan":
                        row.append(float("nan"))
                    else:
                        try:
                            row.append(float(c))
                        except ValueError:
                            row.append(float("nan"))
                rows.append(row)
        return rows

    tote_rows = parse_row(tote_path)
    tote_contents: dict[int, List[Tuple[int, int, int]]] = {}
    order_to_totes: dict[int, List[Tuple[int, int, int]]] = {}

    for order_idx, (_, pairs) in enumerate(orders):
        if order_idx >= len(tote_rows):
            continue
        trow = tote_rows[order_idx]
        order_to_totes[order_idx] = []
        for k, (shape, qty) in enumerate(pairs):
            if k >= len(trow) or trow[k] != trow[k]:  # NaN
                continue
            tote_id = int(round(trow[k]))
            tote_contents.setdefault(tote_id, []).append((order_idx, shape, qty))
            order_to_totes[order_idx].append((tote_id, shape, qty))

    return tote_contents, order_to_totes


def lpt_order(orders: List[Tuple[int, List[Tuple[int, int]]]]) -> List[int]:
    """Return order indices sorted by total item count descending (LPT)."""
    total_items = [(idx, sum(q for _, q in pairs)) for idx, pairs in orders]
    total_items.sort(key=lambda x: (-x[1], x[0]))  # descending by items, then by idx
    return [idx for idx, _ in total_items]


def spt_order(orders: List[Tuple[int, List[Tuple[int, int]]]]) -> List[int]:
    """Return order indices sorted by total item count ascending (SPT - Shortest Processing Time first)."""
    total_items = [(idx, sum(q for _, q in pairs)) for idx, pairs in orders]
    total_items.sort(key=lambda x: (x[1], x[0]))  # ascending by items, then by idx
    return [idx for idx, _ in total_items]


def build_load_sequence(
    order_sequence: List[int],
    tote_contents: Dict[int, List[Tuple[int, int, int]]],
    travel_aware: bool = True,
    item_order_per_tote: Optional[Dict[int, List[Tuple[int, int, int]]]] = None,
    tote_loading_order: Optional[List[int]] = None,
) -> List[int]:
    """
    Build the exact sequence of shapes (one per item) loaded onto conveyor 0.
    Tote contents (which items are in which tote) are fixed by data—we do NOT create totes.
    We only decide: (1) ORDER of totes (which tote to empty first, second, ...) and
    (2) ORDER of items within each tote (item_order_per_tote or else tote_contents order).
    If tote_loading_order is provided, use it; else compute from order_sequence
    (earliest order position, farthest conveyor, tote_id). Within each tote we use
    item_order_per_tote[tote_id] if provided, else tote_contents[tote_id].
    Returns a list of shape ids (length = total item count).
    """
    if tote_loading_order is not None:
        # Explicit tote order (e.g. from joint optimization)
        tote_order = tote_loading_order
    else:
        order_pos = {oid: pos for pos, oid in enumerate(order_sequence)}
        order_to_conv = {}
        for pos, oid in enumerate(order_sequence):
            order_to_conv[oid] = (NUM_CONVEYORS - 1 - pos % NUM_CONVEYORS) if travel_aware else (pos % NUM_CONVEYORS)
        tote_to_earliest_pos: Dict[int, int] = {}
        tote_to_max_conv: Dict[int, int] = {}
        for tote_id, items in tote_contents.items():
            positions = [order_pos.get(o, 999) for o, _, _ in items]
            tote_to_earliest_pos[tote_id] = min(positions)
            convs = [order_to_conv.get(o, -1) for o, _, _ in items if order_to_conv.get(o, -1) >= 0]
            tote_to_max_conv[tote_id] = max(convs) if convs else 0
        tote_order = sorted(
            tote_contents.keys(),
            key=lambda t: (tote_to_earliest_pos[t], -tote_to_max_conv[t], t),
        )
    out: List[int] = []
    for tote_id in tote_order:
        if tote_id not in tote_contents:
            continue
        items_in_tote = (item_order_per_tote or {}).get(tote_id) or tote_contents[tote_id]
        for _order_id, shape, qty in items_in_tote:
            if 0 <= shape < NUM_SHAPES and qty > 0:
                out.extend([shape] * qty)
    return out


def build_conveyor_input(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    order_sequence: List[int],
    travel_aware: bool = True,
) -> List[List[int]]:
    """
    order_sequence = order (first in list = 1st order, etc.).
    If travel_aware: position 0 -> conveyor 3, 1 -> 2, 2 -> 1, 3 -> 0 (farthest first).
    If not: position 0 -> conv 0, 1 -> 1, 2 -> 2, 3 -> 3 (round-robin).
    Returns shape counts per conveyor: counts[c][s] for conveyor c, shape s.
    """
    o2c = {
        oid: (NUM_CONVEYORS - 1 - pos % NUM_CONVEYORS) if travel_aware else (pos % NUM_CONVEYORS)
        for pos, oid in enumerate(order_sequence)
    }
    return build_conveyor_input_from_assignment(orders, o2c)


def build_conveyor_input_from_assignment(
    orders: List[Tuple[int, List[Tuple[int, int]]]],
    order_to_conveyor: Dict[int, int],
) -> List[List[int]]:
    """
    Build shape counts per conveyor from explicit order_id -> conveyor assignment.
    Returns counts[c][s] for conveyor c, shape s.
    """
    order_to_pairs = {idx: pairs for idx, pairs in orders}
    counts: List[List[int]] = [[0] * NUM_SHAPES for _ in range(NUM_CONVEYORS)]
    for oid, pairs in order_to_pairs.items():
        conv = order_to_conveyor.get(oid, 0)
        if conv < 0 or conv >= NUM_CONVEYORS:
            continue
        for shape_id, qty in pairs:
            if 0 <= shape_id < NUM_SHAPES:
                counts[conv][shape_id] += qty
    return counts


def write_m3_input(counts: List[List[int]], output_path: Path) -> None:
    """Write M3_Example input CSV: conv_num and 8 shape columns."""
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["conv_num", *SHAPE_COLUMNS])
        for conv in range(NUM_CONVEYORS):
            row = [conv] + [counts[conv][s] for s in range(NUM_SHAPES)]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LPT scheduler: generator data -> M3 conveyor input CSV."
    )
    parser.add_argument(
        "generated_dir",
        type=Path,
        help="Path to folder containing order_itemtypes.csv and order_quantities.csv",
    )
    parser.add_argument(
        "output_csv",
        type=Path,
        help="Path to write M3_Example input CSV",
    )
    args = parser.parse_args()

    orders = load_generator_data(args.generated_dir)
    if not orders:
        raise SystemExit("No orders loaded.")

    order_sequence = lpt_order(orders)
    counts = build_conveyor_input(orders, order_sequence)
    write_m3_input(counts, args.output_csv)
    print(f"Wrote {args.output_csv} ({NUM_CONVEYORS} conveyors, LPT order length {len(order_sequence)})")


if __name__ == "__main__":
    main()
