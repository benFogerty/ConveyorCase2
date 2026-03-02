#!/usr/bin/env python3
"""
Generate a full event playbook for validation: every LOAD, HOP, and PICK in chronological order.
Uses the same pipeline as lpt (order sequence, optional tote load sequence).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from conveyor_sim import SHAPE_COLUMNS, load_conveyors, simulate_greedy
from scheduler.core import (
    NUM_CONVEYORS,
    build_load_sequence,
    load_generator_data,
    load_tote_data,
    lpt_order,
    write_m3_input_by_order,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a chronological event playbook (every LOAD, HOP, PICK) for validation.",
    )
    parser.add_argument("generated_dir", type=Path, help="Path to generated/ folder")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("scheduler/event_playbook.txt"),
        help="Output playbook file (default: scheduler/event_playbook.txt)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Also write machine-readable event log CSV (columns: time, event, item_id, ...)",
    )
    parser.add_argument(
        "--order",
        choices=["lpt", "fixed", "spt"],
        default="lpt",
        help="Order sequence: lpt, fixed (0,1,2,...), or spt (default: lpt)",
    )
    parser.add_argument("--travel-aware", action="store_true", default=True, help="Use travel-aware conveyor assignment")
    parser.add_argument("--no-travel-aware", action="store_false", dest="travel_aware")
    args = parser.parse_args()

    generated_dir = Path(args.generated_dir)
    orders = load_generator_data(generated_dir)
    if not orders:
        raise SystemExit("No orders loaded.")

    n = len(orders)
    if args.order == "lpt":
        order_sequence = lpt_order(orders)
    elif args.order == "fixed":
        order_sequence = list(range(n))
    else:
        from scheduler.core import spt_order
        order_sequence = spt_order(orders)

    tote_contents, order_to_totes = load_tote_data(generated_dir, orders)
    use_tote = bool(tote_contents and order_to_totes)
    load_sequence = build_load_sequence(order_sequence, tote_contents, args.travel_aware) if use_tote else None

    order_to_conveyor = {
        oid: (NUM_CONVEYORS - 1 - pos % NUM_CONVEYORS) if args.travel_aware else (pos % NUM_CONVEYORS)
        for pos, oid in enumerate(order_sequence)
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
        tmp = Path(f.name)
    try:
        write_m3_input_by_order(orders, order_to_conveyor, order_sequence, tmp)
        conveyors, orders_per_conv = load_conveyors(tmp)
        results, trace_events = simulate_greedy(
            conveyors,
            all_load_at_conveyor_0=True,
            load_spacing=2.5,
            load_sequence=load_sequence,
            return_trace=True,
            orders_per_conveyor=orders_per_conv,
        )
    finally:
        tmp.unlink(missing_ok=True)

    makespan = max(t for _, _, t in results) if results else 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    shape_names = list(SHAPE_COLUMNS)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("EVENT PLAYBOOK (chronological)\n")
        f.write("=" * 60 + "\n")
        f.write(f"Generated from: {generated_dir}\n")
        f.write(f"Order sequence: {args.order} (travel_aware={args.travel_aware})\n")
        f.write(f"Load sequence: {'tote order + item order' if use_tote else 'default (all shape 0, then 1, ...)'}\n")
        f.write(f"Total items (LOADs): {len(results)}\n")
        f.write(f"Total events (LOAD+HOP+PICK): {len(trace_events)}\n")
        f.write(f"Makespan: {makespan:.2f}s\n")
        f.write("=" * 60 + "\n\n")

        for i, (t, ev_type, payload) in enumerate(trace_events):
            if ev_type == "LOAD":
                item_id, shape, conv = payload
                sname = shape_names[shape] if shape < len(shape_names) else str(shape)
                f.write(f"{i+1:5}.  t={t:7.2f}s  LOAD   item_id={item_id:3}  shape={shape} ({sname})  at conveyor {conv}\n")
            elif ev_type == "HOP":
                item_id, from_c, to_c = payload
                f.write(f"{i+1:5}.  t={t:7.2f}s  HOP    item_id={item_id:3}  conveyor {from_c} -> {to_c}\n")
            else:  # PICK
                item_id, conv, shape = payload
                sname = shape_names[shape] if shape < len(shape_names) else str(shape)
                f.write(f"{i+1:5}.  t={t:7.2f}s  PICK   item_id={item_id:3}  shape={shape} ({sname})  at conveyor {conv}\n")

    print(f"Wrote {out_path} ({len(trace_events)} events, makespan={makespan:.2f}s)")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv as csv_module
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv_module.writer(f)
            w.writerow(["time", "event", "item_id", "extra1", "extra2"])
            for t, ev_type, payload in trace_events:
                if ev_type == "LOAD":
                    w.writerow([t, "LOAD", payload[0], payload[1], payload[2]])
                elif ev_type == "HOP":
                    w.writerow([t, "HOP", payload[0], payload[1], payload[2]])
                else:
                    w.writerow([t, "PICK", payload[0], payload[1], payload[2]])
        print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
