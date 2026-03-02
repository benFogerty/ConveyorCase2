#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# When load_conveyors reads one-row-per-order CSV: conv -> list of (order_id, demand[8]) in row order
OrdersPerConveyor = Optional[Dict[int, List[Tuple[int, List[int]]]]]

SHAPE_COLUMNS = [
    "cirle",  # source template typo kept intentionally
    "pentagon",
    "trapezoid",
    "triangle",
    "star",
    "moon",
    "heart",
    "cross",
]
NUM_CONVEYORS = 4
HOP_SECONDS = 5.0
FULL_LOOP_SECONDS = HOP_SECONDS * NUM_CONVEYORS
# When all items load at conveyor 0: time between loading consecutive items ("half a conveyor belt" = half of one hop)
LOAD_SPACING_SECONDS = 2.5
# Safety: abort sim if we process more events than this (avoids infinite loop from supply/order mismatch)
MAX_SIM_ITERATIONS_PER_ITEM = 10000  # ~2500 full conveyor loops per item

@dataclass
class ConveyorState:
    conv_num: int
    queue: List[int]
    index: int = 0


def _generic_initial_time(conv_num: int, item_count: int) -> float:
    # Backward-compatible helper (unused by circulation model).
    return 0.0


def _generic_gap(conv_num: int, emission_index: int, shape: int, circulation: int) -> float:
    # Backward-compatible helper (unused by circulation model).
    return 1.0


def load_conveyors(input_csv: Path) -> Tuple[Dict[int, ConveyorState], OrdersPerConveyor]:
    conveyors: Dict[int, ConveyorState] = {}
    orders_per_conv: OrdersPerConveyor = None
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        required = ["conv_num", *SHAPE_COLUMNS]
        missing = [c for c in required if c not in fieldnames]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # One row per order (order_id, conv_num, shapes): build per-order demand and aggregate
        if "order_id" in fieldnames:
            counts: List[List[int]] = [[0] * len(SHAPE_COLUMNS) for _ in range(NUM_CONVEYORS)]
            orders_per_conv = {c: [] for c in range(NUM_CONVEYORS)}
            for row in reader:
                conv_num = int(row["conv_num"].strip())
                if 0 <= conv_num < NUM_CONVEYORS:
                    demand = [int(row[col].strip()) for col in SHAPE_COLUMNS]
                    order_id = int(row["order_id"].strip())
                    orders_per_conv[conv_num].append((order_id, demand))
                    for shape_idx, qty in enumerate(demand):
                        counts[conv_num][shape_idx] += qty
            for conv_num in range(NUM_CONVEYORS):
                queue: List[int] = []
                for shape_idx in range(len(SHAPE_COLUMNS)):
                    queue.extend([shape_idx] * counts[conv_num][shape_idx])
                conveyors[conv_num] = ConveyorState(conv_num=conv_num, queue=queue)
            return (conveyors, orders_per_conv)

        # One row per conveyor (legacy)
        for row in reader:
            conv_num = int(row["conv_num"].strip())
            queue: List[int] = []
            for shape_idx, col in enumerate(SHAPE_COLUMNS):
                count = int(row[col].strip())
                queue.extend([shape_idx] * count)
            conveyors[conv_num] = ConveyorState(conv_num=conv_num, queue=queue)

    return (conveyors, None)


def simulate_greedy(
    conveyors: Dict[int, ConveyorState],
    all_load_at_conveyor_0: bool = False,
    load_spacing: float = LOAD_SPACING_SECONDS,
    load_sequence: Optional[Sequence[int]] = None,
    return_trace: bool = False,
    orders_per_conveyor: OrdersPerConveyor = None,
):
    if not conveyors:
        return ([] if not return_trace else ([], []))

    trace_events: List[Tuple[float, str, Tuple]] = []  # (time, "LOAD"|"HOP"|"PICK", payload)
    demands: Dict[int, List[int]] = {
        conv_num: [0] * len(SHAPE_COLUMNS) for conv_num in conveyors
    }
    items: List[Tuple[int, int, float, bool]] = []  # (shape, start_conv, next_time, picked)
    for conv_num, state in conveyors.items():
        for shape in state.queue:
            demands[conv_num][shape] += 1

    # Mutable per-order remaining demand when using strict "first order only" rule
    current_orders: Optional[Dict[int, List[Tuple[int, List[int]]]]] = None
    if orders_per_conveyor is not None:
        current_orders = {
            c: [(oid, list(rem)) for oid, rem in orders_per_conveyor[c]]
            for c in orders_per_conveyor
        }

    global_supply = [0] * len(SHAPE_COLUMNS)
    global_demand = [0] * len(SHAPE_COLUMNS)
    for conv_num in conveyors:
        for s in range(len(SHAPE_COLUMNS)):
            global_supply[s] += demands[conv_num][s]
            global_demand[s] += demands[conv_num][s]
    if global_supply != global_demand:
        raise ValueError("Infeasible input: global shape supply must equal global demand.")

    if all_load_at_conveyor_0:
        # All items load onto conveyor 0 only. Clock starts when first item is loaded (t=0).
        # Each item is added load_spacing apart (e.g. half a conveyor belt = 2.5s).
        # load_sequence: optional explicit order of shapes (enables tote order + item order within tote).
        if load_sequence is not None:
            if len(load_sequence) != sum(global_demand):
                raise ValueError(
                    f"load_sequence length {len(load_sequence)} != total items {sum(global_demand)}"
                )
            seq_counts = [0] * len(SHAPE_COLUMNS)
            for s in load_sequence:
                if 0 <= s < len(SHAPE_COLUMNS):
                    seq_counts[s] += 1
            if seq_counts != global_demand:
                raise ValueError(
                    "load_sequence shape counts do not match conveyor demand"
                )
            use_sequence = list(load_sequence)
        else:
            use_sequence = []
            for s in range(len(SHAPE_COLUMNS)):
                use_sequence.extend([s] * global_demand[s])
        pq: List[Tuple[float, int, int]] = []
        for item_id, shape in enumerate(use_sequence):
            t0 = item_id * load_spacing
            items.append((shape, 0, t0, False))
            heapq.heappush(pq, (t0, 0, item_id))
            if return_trace:
                trace_events.append((t0, "LOAD", (item_id, shape, 0)))
    else:
        # Original: items distributed along each local belt segment (staggered over 20s per conveyor).
        for conv_num, state in conveyors.items():
            n_local = len(state.queue)
            if n_local == 0:
                continue
            step = FULL_LOOP_SECONDS / n_local
            for i, shape in enumerate(state.queue):
                items.append((shape, conv_num, i * step, False))
        pq = []
        for item_id, (_, conv, t0, _) in enumerate(items):
            heapq.heappush(pq, (t0, conv, item_id))

    picked = [False] * len(items)
    total_items = len(items)
    results: List[Tuple[int, int, float]] = []
    max_iterations = total_items * MAX_SIM_ITERATIONS_PER_ITEM
    iterations = 0

    while pq and len(results) < total_items:
        iterations += 1
        if iterations > max_iterations:
            raise ValueError(
                "Simulation did not complete within iteration limit; possible supply/order mismatch or deadlock."
            )
        now, conv_num, item_id = heapq.heappop(pq)
        if picked[item_id]:
            continue

        shape, _, _, _ = items[item_id]
        do_pick = False
        if current_orders is not None:
            # Strict FIFO: only PICK if the current (first) order on this conveyor needs this shape
            orders_list = current_orders.get(conv_num, [])
            if orders_list:
                order_id, remaining = orders_list[0]
                if remaining[shape] > 0:
                    do_pick = True
                    remaining[shape] -= 1
                    if sum(remaining) == 0:
                        orders_list.pop(0)
            # else: no current order for this conv -> HOP
        else:
            do_pick = demands[conv_num][shape] > 0
            if do_pick:
                demands[conv_num][shape] -= 1

        if do_pick:
            picked[item_id] = True
            t_pick = round(now, 6)
            results.append((conv_num, shape, t_pick))
            if return_trace:
                trace_events.append((t_pick, "PICK", (item_id, conv_num, shape)))
            continue

        next_conv = (conv_num + 1) % NUM_CONVEYORS
        t_hop = now + HOP_SECONDS
        if return_trace:
            trace_events.append((t_hop, "HOP", (item_id, conv_num, next_conv)))
        heapq.heappush(pq, (t_hop, next_conv, item_id))

    if len(results) != total_items:
        raise ValueError("Simulation ended before all items were picked. Check demand assumptions.")

    if return_trace:
        trace_events.sort(key=lambda e: (e[0], {"LOAD": 0, "HOP": 1, "PICK": 2}[e[1]], e[2][0]))
        return results, trace_events
    return results


def write_output(rows: Sequence[Tuple[int, int, float]], output_csv: Path) -> None:
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["conv_num", "shape", "time"])
        for conv_num, shape, t in rows:
            writer.writerow([conv_num, shape, f"{t:.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Simulate the IDEAS Clinic conveyor with a greedy dispatcher."
        )
    )
    parser.add_argument("input_csv", type=Path, help="Path to conveyor input CSV")
    parser.add_argument("output_csv", type=Path, help="Path to write simulation output CSV")
    parser.add_argument(
        "--all-load-at-conveyor-0",
        action="store_true",
        help="All items load onto conveyor 0 only; clock starts at first load; items spaced load_spacing apart.",
    )
    parser.add_argument(
        "--load-spacing",
        type=float,
        default=LOAD_SPACING_SECONDS,
        help="Seconds between loading consecutive items at conveyor 0 (default: 2.5 = half a belt).",
    )
    args = parser.parse_args()

    conveyors, orders_per_conv = load_conveyors(args.input_csv)
    rows = simulate_greedy(
        conveyors,
        all_load_at_conveyor_0=args.all_load_at_conveyor_0,
        load_spacing=args.load_spacing,
        orders_per_conveyor=orders_per_conv,
    )
    write_output(rows, args.output_csv)


if __name__ == "__main__":
    main()
