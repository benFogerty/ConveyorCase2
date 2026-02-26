#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import heapq
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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


def load_conveyors(input_csv: Path) -> Dict[int, ConveyorState]:
    conveyors: Dict[int, ConveyorState] = {}
    with input_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["conv_num", *SHAPE_COLUMNS]
        missing = [c for c in required if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        for row in reader:
            conv_num = int(row["conv_num"].strip())
            queue: List[int] = []
            for shape_idx, col in enumerate(SHAPE_COLUMNS):
                count = int(row[col].strip())
                queue.extend([shape_idx] * count)
            conveyors[conv_num] = ConveyorState(conv_num=conv_num, queue=queue)

    return conveyors


def simulate_greedy(conveyors: Dict[int, ConveyorState]) -> List[Tuple[int, int, float]]:
    if not conveyors:
        return []

    demands: Dict[int, List[int]] = {
        conv_num: [0] * len(SHAPE_COLUMNS) for conv_num in conveyors
    }
    items: List[Tuple[int, int, float, bool]] = []  # (shape, conv, next_time, picked)
    for conv_num, state in conveyors.items():
        for shape in state.queue:
            demands[conv_num][shape] += 1

    # Items are distributed along each local belt segment at t=0 and then circulate.
    # This avoids all items appearing at exactly the same time.
    for conv_num, state in conveyors.items():
        n_local = len(state.queue)
        if n_local == 0:
            continue
        step = FULL_LOOP_SECONDS / n_local
        for i, shape in enumerate(state.queue):
            items.append((shape, conv_num, i * step, False))

    global_supply = [0] * len(SHAPE_COLUMNS)
    global_demand = [0] * len(SHAPE_COLUMNS)
    for conv_num in conveyors:
        for s in range(len(SHAPE_COLUMNS)):
            global_supply[s] += demands[conv_num][s]
            global_demand[s] += demands[conv_num][s]
    if global_supply != global_demand:
        raise ValueError("Infeasible input: global shape supply must equal global demand.")

    pq: List[Tuple[float, int, int]] = []
    for item_id, (_, conv, t0, _) in enumerate(items):
        heapq.heappush(pq, (t0, conv, item_id))

    picked = [False] * len(items)
    total_items = len(items)
    results: List[Tuple[int, int, float]] = []

    while pq and len(results) < total_items:
        now, conv_num, item_id = heapq.heappop(pq)
        if picked[item_id]:
            continue

        shape, _, _, _ = items[item_id]
        if demands[conv_num][shape] > 0:
            demands[conv_num][shape] -= 1
            picked[item_id] = True
            results.append((conv_num, shape, round(now, 6)))
            continue

        next_conv = (conv_num + 1) % NUM_CONVEYORS
        heapq.heappush(pq, (now + HOP_SECONDS, next_conv, item_id))

    if len(results) != total_items:
        raise ValueError("Simulation ended before all items were picked. Check demand assumptions.")

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
    args = parser.parse_args()

    conveyors = load_conveyors(args.input_csv)
    rows = simulate_greedy(conveyors)
    write_output(rows, args.output_csv)


if __name__ == "__main__":
    main()
