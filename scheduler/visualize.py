#!/usr/bin/env python3
"""
Visualize conveyor sim output and (optionally) compare LPT vs baseline.

Single run:
  python scheduler/visualize.py sim_output.csv [input.csv]
  - Plots: cumulative picks over time, picks per conveyor, load balance from input.
  - Prints makespan and stats.

Compare LPT vs fixed order (baseline):
  python scheduler/visualize.py --compare path/to/generated/
  - Builds LPT and fixed-order inputs, runs sim for both, plots comparison.

Order/tote flow (sequence, assignment, completion):
  python scheduler/visualize.py --flow path/to/generated/ sim_output.csv
  - Shows LPT order sequence, which orders are on each conveyor, and estimated completion time per order (Gantt).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Project root for conveyor_sim import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")
except ImportError:
    plt = None


def load_sim_output(path: Path) -> list[tuple[int, int, float]]:
    """Load sim CSV; return list of (conv_num, shape, time)."""
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                int(row["conv_num"].strip()),
                int(row["shape"].strip()),
                float(row["time"].strip()),
            ))
    return rows


def load_input_counts(path: Path) -> list[list[int]]:
    """Load M3 input CSV; return counts[conv][shape] (list of 4 lists of 8 ints)."""
    shape_cols = [
        "cirle", "pentagon", "trapezoid", "triangle",
        "star", "moon", "heart", "cross",
    ]
    counts = [[0] * 8 for _ in range(4)]
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            c = int(row["conv_num"].strip())
            for i, col in enumerate(shape_cols):
                if i < 8 and col in row:
                    counts[c][i] = int(row[col].strip())
    return counts


def makespan(events: list[tuple[int, int, float]]) -> float:
    return max(t for _, _, t in events) if events else 0.0


def plot_single(
    events: list[tuple[int, int, float]],
    input_counts: list[list[int]] | None,
    output_dir: Path,
    title_prefix: str = "",
) -> None:
    """Plot cumulative picks, picks by conveyor over time, and (if input_counts) load balance."""
    if plt is None:
        print("matplotlib not installed; skipping plots. Install with: pip install matplotlib")
        return

    ms = makespan(events)
    total = len(events)

    n_plots = 3 if input_counts else 2
    fig, axes = plt.subplots(n_plots, 1, figsize=(8, 3 * n_plots))
    if n_plots == 1:
        axes = [axes]
    ax1, ax2 = axes[0], axes[1]

    # 1) Cumulative picks over time
    times = [t for _, _, t in events]
    times_sorted = sorted(times)
    cumul = list(range(1, len(times_sorted) + 1))
    ax1.step([0] + times_sorted, [0] + cumul, where="post", color="steelblue", linewidth=2)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Cumulative picks")
    ax1.set_title(f"{title_prefix}Cumulative picks over time (makespan = {ms:.1f}s, n = {total})")
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(left=0)

    # 2) Picks per conveyor over time (cumulative per conveyor)
    conv_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    for c in range(4):
        c_events = [(t, 1) for conv, _, t in events if conv == c]
        c_events.sort(key=lambda x: x[0])
        t_vals = [0] + [t for t, _ in c_events]
        cum = [0] + list(range(1, len(c_events) + 1))
        ax2.step(t_vals, cum, where="post", label=f"Conv {c}", color=conv_colors[c], linewidth=1.5)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cumulative picks")
    ax2.set_title("Cumulative picks per conveyor over time")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    # 3) Load balance from input (items per conveyor)
    if input_counts:
        ax3 = axes[2]
        conv_totals = [sum(input_counts[c]) for c in range(4)]
        bars = ax3.bar(
            [f"Conv {c}" for c in range(4)],
            conv_totals,
            color=conv_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax3.set_ylabel("Total items")
        ax3.set_title(f"{title_prefix}Load per conveyor (input)")
        for b, v in zip(bars, conv_totals):
            ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, str(v), ha="center", fontsize=10)

    plt.tight_layout()
    out_path = output_dir / "visualization_single.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_comparison(
    results_lpt: list[tuple[int, int, float]],
    results_baseline: list[tuple[int, int, float]],
    output_dir: Path,
) -> None:
    """Plot makespan bar chart and cumulative picks for LPT vs baseline."""
    if plt is None:
        print("matplotlib not installed; skipping comparison plots.")
        return

    ms_lpt = makespan(results_lpt)
    ms_base = makespan(results_baseline)
    total = len(results_lpt)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Makespan comparison
    ax1.bar(["LPT", "Fixed order"], [ms_lpt, ms_base], color=["#3498db", "#95a5a6"], edgecolor="black")
    ax1.set_ylabel("Makespan (s)")
    ax1.set_title("Makespan comparison\n(lower is better)")
    for i, v in enumerate([ms_lpt, ms_base]):
        ax1.text(i, v + 0.3, f"{v:.1f}s", ha="center", fontsize=11)
    improvement = (ms_base - ms_lpt) / ms_base * 100 if ms_base > 0 else 0
    ax1.text(0.5, max(ms_lpt, ms_base) * 0.5, f"LPT vs baseline:\n{improvement:+.1f}%", ha="center", fontsize=10)

    # Cumulative picks over time (both)
    def cumul_curve(events: list[tuple[int, int, float]]) -> tuple[list[float], list[int]]:
        times = sorted(t for _, _, t in events)
        return times, list(range(1, len(times) + 1))

    t_lpt, c_lpt = cumul_curve(results_lpt)
    t_base, c_base = cumul_curve(results_baseline)
    ax2.step([0] + t_lpt, [0] + c_lpt, where="post", label="LPT", color="#3498db", linewidth=2)
    ax2.step([0] + t_base, [0] + c_base, where="post", label="Fixed order", color="#95a5a6", linewidth=2)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Cumulative picks")
    ax2.set_title("Cumulative picks over time")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    out_path = output_dir / "visualization_compare.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")
    print(f"  LPT makespan: {ms_lpt:.2f}s  |  Fixed-order makespan: {ms_base:.2f}s  |  Improvement: {improvement:+.1f}%")


def _estimate_order_completion_times(
    events: list[tuple[int, int, float]],
    conv_orders: list[list[tuple[int, int]]],  # conv_orders[c] = [(order_id, num_items), ...]
) -> dict[int, float]:
    """
    Assign picks at each conveyor to orders in sequence; return order_id -> time of last pick for that order.
    (Estimate: sim doesn't label picks by order; we assume conveyor serves orders in assignment order.)
    """
    # Per conveyor: list of (time, ...) for each pick at that conveyor, sorted by time
    picks_per_conv: list[list[float]] = [[] for _ in range(4)]
    for conv, _, t in events:
        if conv < 4:
            picks_per_conv[conv].append(t)
    for c in range(4):
        picks_per_conv[c].sort()

    order_completion: dict[int, float] = {}
    for c in range(4):
        idx = 0
        for order_id, num_items in conv_orders[c]:
            # Next num_items picks on this conveyor belong to this order
            for _ in range(num_items):
                if idx < len(picks_per_conv[c]):
                    order_completion[order_id] = picks_per_conv[c][idx]
                    idx += 1
    return order_completion


def plot_flow(
    orders: list[tuple[int, list[tuple[int, int]]]],  # (order_idx, [(shape, qty), ...])
    lpt_sequence: list[int],
    events: list[tuple[int, int, float]],
    output_dir: Path,
    tote_contents: dict[int, list[tuple[int, int, int]]] | None = None,  # tote_id -> [(order_id, shape, qty)]
    order_to_totes: dict[int, list[tuple[int, int, int]]] | None = None,  # order_id -> [(tote_id, shape, qty)]
) -> None:
    """
    Plot: (1) LPT order sequence table, (2) Orders per conveyor, (3) Gantt of estimated order completion.
    If tote data is provided: (4) Tote table and loading-by-position view.
    """
    if plt is None:
        print("matplotlib not installed; skipping flow plot.")
        return

    n_orders = len(orders)
    order_to_items = {idx: sum(q for _, q in pairs) for idx, pairs in orders}
    # Travel-aware: position 0 -> conv 3, 1 -> conv 2, 2 -> conv 1, 3 -> conv 0 (must match scheduler.core build_conveyor_input)
    order_to_conv = {lpt_sequence[pos]: (3 - pos % 4) for pos in range(len(lpt_sequence))}

    # Conveyor assignment: travel-aware position 0 -> conv 3, 1 -> 2, 2 -> 1, 3 -> 0, 4 -> 3, ...
    conv_orders: list[list[tuple[int, int]]] = [[] for _ in range(4)]  # (order_id, num_items) per conv
    for position in range(len(lpt_sequence)):
        conv = 3 - position % 4
        order_id = lpt_sequence[position]
        num_items = order_to_items.get(order_id, 0)
        conv_orders[conv].append((order_id, num_items))

    order_completion = _estimate_order_completion_times(events, conv_orders)

    conv_colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]
    ms = makespan(events)

    has_tote = tote_contents is not None and order_to_totes is not None and len(tote_contents) > 0
    n_plots = 4 if has_tote else 3
    fig = plt.figure(figsize=(12, 4 * n_plots if has_tote else 8))
    gs = fig.add_gridspec(n_plots, 1, height_ratios=[0.9, 0.6, 1.2, 1.4] if has_tote else [0.9, 0.6, 1.2])

    # 1) Table: LPT sequence (position, order, items, conveyor)
    ax_table = fig.add_subplot(gs[0])
    ax_table.set_axis_off()
    rows = [["Position", "Order", "Items", "Conveyor"]]
    for pos, order_id in enumerate(lpt_sequence):
        items = order_to_items.get(order_id, 0)
        conv = 3 - pos % 4
        rows.append([str(pos + 1), f"Order {order_id}", str(items), f"Conv {conv}"])
    table = ax_table.table(
        cellText=rows,
        colLabels=None,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)
    for j in range(4):
        table[(0, j)].set_facecolor("#34495e")
        table[(0, j)].set_text_props(color="white", fontweight="bold")
    ax_table.set_title("LPT order sequence: position → order → items → conveyor", fontsize=11)

    # 2) Text: Orders per conveyor
    ax_txt = fig.add_subplot(gs[1])
    ax_txt.set_axis_off()
    lines = ["Orders (and item counts) on each conveyor:"]
    for c in range(4):
        parts = [f"Conv {c}:"] + [f"Order {oid} ({n})" for oid, n in conv_orders[c]]
        lines.append("  ".join(parts))
    ax_txt.text(0.5, 0.5, "\n".join(lines), transform=ax_txt.transAxes, fontsize=10,
                verticalalignment="center", horizontalalignment="center", family="monospace")

    # 3) Gantt: order completion (y = rows grouped by conveyor, x = time, bar from 0 to completion)
    ax_gantt = fig.add_subplot(gs[2])
    # Rows: Conv 0 orders, then Conv 1, then Conv 2, then Conv 3
    row_labels: list[str] = []
    row_conv: list[int] = []
    order_ids_by_row: list[int] = []
    for c in range(4):
        for order_id, n in conv_orders[c]:
            row_labels.append(f"Conv {c}: Order {order_id} ({n} items)")
            row_conv.append(c)
            order_ids_by_row.append(order_id)
    n_rows = len(row_labels)
    for i in range(n_rows):
        order_id = order_ids_by_row[i]
        c = row_conv[i]
        t_end = order_completion.get(order_id, 0.0)
        ax_gantt.barh(i, t_end, left=0, height=0.6, color=conv_colors[c], edgecolor="black", linewidth=0.5)
        ax_gantt.text(t_end + 0.2, i, f"{t_end:.1f}s", va="center", fontsize=8)
    ax_gantt.set_yticks(range(n_rows))
    ax_gantt.set_yticklabels(row_labels, fontsize=9)
    ax_gantt.set_xlabel("Time (s)")
    ax_gantt.set_xlim(left=0, right=ms * 1.08)
    ax_gantt.set_title("Estimated order completion time (bar = 0 → last pick for that order)")
    ax_gantt.grid(True, axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax_gantt.legend(handles=[Patch(facecolor=conv_colors[c], label=f"Conv {c}") for c in range(4)], loc="lower right")

    # 4) Tote view: global loading order (one tote fully emptied before next), then per-tote and per-position details
    if has_tote and tote_contents is not None and order_to_totes is not None:
        ax_tote = fig.add_subplot(gs[3])
        ax_tote.set_axis_off()
        # Build global tote loading order: each tote once. Order by earliest LPT position, then by farthest conveyor (feed conv 3 before conv 0 — travel time), then tote_id. One tote fully emptied before the next.
        order_pos = {oid: pos for pos, oid in enumerate(lpt_sequence)}
        tote_to_earliest_pos: dict[int, int] = {}
        tote_to_max_conv: dict[int, int] = {}  # farthest conveyor this tote feeds (3 = longest travel from entry)
        for tote_id, items in tote_contents.items():
            positions = [order_pos.get(o, 999) for o, _, _ in items]
            tote_to_earliest_pos[tote_id] = min(positions)
            convs = [order_to_conv.get(o, -1) for o, _, _ in items if order_to_conv.get(o, -1) >= 0]
            tote_to_max_conv[tote_id] = max(convs) if convs else 0
        tote_loading_sequence = sorted(tote_contents.keys(), key=lambda t: (tote_to_earliest_pos[t], -tote_to_max_conv[t], t))
        lines = [
            "TOTE LOADING ORDER (one tote fully emptied before the next):",
            "",
        ]
        for i, tote_id in enumerate(tote_loading_sequence, 1):
            items = tote_contents.get(tote_id, [])
            n_items = sum(q for _, _, q in items)
            orders_in_tote = sorted(set(o for o, _, _ in items))
            convs = sorted(set(order_to_conv.get(o, -1) for o in orders_in_tote))
            convs = [c for c in convs if c >= 0]
            content_str = ", ".join(f"{q}×s{s}" for _, s, q in items)
            lines.append(f"  {i}. Tote {tote_id} ({n_items} items) — Order(s) {orders_in_tote} → Conv(s) {convs}  [{content_str}]")
        lines.extend([
            "",
            "Totes: which orders they help fulfil, and which conveyor(s) (detail):",
            "",
        ])
        for tote_id in sorted(tote_contents.keys()):
            items = tote_contents[tote_id]
            orders_in_tote = sorted(set(o for o, _, _ in items))
            convs = sorted(set(order_to_conv.get(o, -1) for o in orders_in_tote))
            convs = [c for c in convs if c >= 0]
            item_str = ", ".join(f"Order{o}:{q}×s{s}" for o, s, q in items[:5])
            if len(items) > 5:
                item_str += f" ... (+{len(items)-5} more)"
            lines.append(f"  Tote {tote_id}: orders {orders_in_tote} → Conv(s) {convs}  [{item_str}]")
        lines.extend([
            "",
            "Loading by LPT position (which totes per order):",
            "",
        ])
        for pos in range(len(lpt_sequence)):
            order_id = lpt_sequence[pos]
            conv = pos % 4
            totes_info = order_to_totes.get(order_id, [])
            tote_ids = sorted(set(t for t, _, _ in totes_info))
            item_parts = [f"{q}×shape{s}" for t, s, q in totes_info]
            conv = 3 - pos % 4
            lines.append(f"  Position {pos+1}: Order {order_id} (Conv {conv}) — totes {tote_ids} — {', '.join(item_parts)}")
        ax_tote.text(0.02, 0.98, "\n".join(lines), transform=ax_tote.transAxes, fontsize=7,
                     verticalalignment="top", horizontalalignment="left", family="monospace",
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.3))

    plt.tight_layout()
    out_path = output_dir / "visualization_flow.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")

    # Also write text outputs: clear tote loading order + full summary (tote_loading_sequence already built above)
    if has_tote and tote_contents is not None and order_to_totes is not None:
        # Dedicated file: only the loading order (one tote fully emptied before the next)
        loading_order_path = output_dir / "tote_loading_order.txt"
        with loading_order_path.open("w", encoding="utf-8") as f:
            f.write("TOTE LOADING ORDER (one tote fully emptied before the next)\n")
            f.write("=" * 60 + "\n\n")
            for i, tote_id in enumerate(tote_loading_sequence, 1):
                items = tote_contents[tote_id]
                n_items = sum(q for _, _, q in items)
                orders_in_tote = sorted(set(o for o, _, _ in items))
                convs = sorted(set(order_to_conv.get(o, -1) for o in orders_in_tote))
                convs = [c for c in convs if c >= 0]
                content_str = ", ".join(f"{q}×shape{s}" for _, s, q in items)
                f.write(f"{i}. Tote {tote_id}  ({n_items} items)  Order(s) {orders_in_tote}  Conv(s) {convs}\n")
                f.write(f"   Contents: {content_str}\n\n")
        print(f"Saved {loading_order_path}")

        summary_path = output_dir / "tote_and_loading_summary.txt"
        with summary_path.open("w", encoding="utf-8") as f:
            f.write("TOTE LOADING ORDER (one tote fully emptied before the next)\n")
            f.write("=" * 60 + "\n\n")
            for i, tote_id in enumerate(tote_loading_sequence, 1):
                items = tote_contents[tote_id]
                n_items = sum(q for _, _, q in items)
                orders_in_tote = sorted(set(o for o, _, _ in items))
                convs = sorted(set(order_to_conv.get(o, -1) for o in orders_in_tote))
                convs = [c for c in convs if c >= 0]
                content_str = ", ".join(f"{q}×shape{s}" for _, s, q in items)
                f.write(f"{i}. Tote {tote_id} ({n_items} items) — Order(s) {orders_in_tote} → Conv(s) {convs}\n")
                f.write(f"   Contents: {content_str}\n\n")
            f.write("\nTOTE VIEW: which totes help which orders, and which conveyor(s)\n")
            f.write("=" * 60 + "\n\n")
            for tote_id in sorted(tote_contents.keys()):
                items = tote_contents[tote_id]
                orders_in_tote = sorted(set(o for o, _, _ in items))
                convs = sorted(set(order_to_conv.get(o, -1) for o in orders_in_tote))
                convs = [c for c in convs if c >= 0]
                f.write(f"Tote {tote_id}: helps orders {orders_in_tote} → items go to conveyor(s) {convs}\n")
                for o, s, q in items:
                    f.write(f"    Order {o}: {q}× shape {s}\n")
                f.write("\n")
            f.write("\nLOADING BY LPT POSITION: which totes to load for each order\n")
            f.write("=" * 60 + "\n\n")
            for pos in range(len(lpt_sequence)):
                order_id = lpt_sequence[pos]
                conv = 3 - pos % 4
                totes_info = order_to_totes.get(order_id, [])
                tote_ids = sorted(set(t for t, _, _ in totes_info))
                item_parts = [f"{q}×shape{s}" for t, s, q in totes_info]
                f.write(f"Position {pos+1}: Order {order_id} (Conv {conv}) — totes {tote_ids}\n")
                f.write(f"    Items: {', '.join(item_parts)}\n\n")
        print(f"Saved {summary_path}")


def run_flow(generated_dir: Path, sim_output_path: Path, output_dir: Path) -> None:
    """Load generator data, LPT sequence, sim output, and (if present) tote data; plot flow + tote/loading view."""
    from scheduler.core import load_generator_data, load_tote_data, lpt_order

    generated_dir = Path(generated_dir)
    sim_output_path = Path(sim_output_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not sim_output_path.exists():
        raise SystemExit(f"Sim output not found: {sim_output_path}")

    orders = load_generator_data(generated_dir)
    if not orders:
        raise SystemExit("No orders loaded.")

    lpt_seq = lpt_order(orders)
    events = load_sim_output(sim_output_path)
    tote_contents, order_to_totes = load_tote_data(generated_dir, orders)
    plot_flow(orders, lpt_seq, events, output_dir, tote_contents, order_to_totes)


def run_compare(generated_dir: Path, output_dir: Path) -> None:
    """Build LPT and fixed-order inputs, run sim, plot comparison."""
    from conveyor_sim import load_conveyors, simulate_greedy
    from scheduler.core import (
        load_generator_data,
        lpt_order,
        build_conveyor_input,
        write_m3_input,
        NUM_CONVEYORS,
    )

    generated_dir = Path(generated_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    orders = load_generator_data(generated_dir)
    if not orders:
        raise SystemExit("No orders loaded.")

    n = len(orders)
    # LPT sequence
    lpt_seq = lpt_order(orders)
    counts_lpt = build_conveyor_input(orders, lpt_seq)
    lpt_input = output_dir / "compare_lpt_input.csv"
    write_m3_input(counts_lpt, lpt_input)

    # Fixed order: 0, 1, 2, ..., n-1
    fixed_seq = list(range(n))
    counts_fixed = build_conveyor_input(orders, fixed_seq)
    fixed_input = output_dir / "compare_fixed_input.csv"
    write_m3_input(counts_fixed, fixed_input)

    # Run sim: all items load at conveyor 0, 2.5s spacing (half a belt)
    conveyors_lpt = load_conveyors(lpt_input)
    conveyors_fixed = load_conveyors(fixed_input)
    results_lpt = simulate_greedy(conveyors_lpt, all_load_at_conveyor_0=True, load_spacing=2.5)
    results_baseline = simulate_greedy(conveyors_fixed, all_load_at_conveyor_0=True, load_spacing=2.5)

    # Write sim outputs for reference
    from conveyor_sim import write_output
    write_output(results_lpt, output_dir / "compare_lpt_sim_output.csv")
    write_output(results_baseline, output_dir / "compare_fixed_sim_output.csv")

    plot_comparison(results_lpt, results_baseline, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize sim output and/or compare LPT vs baseline.")
    parser.add_argument("sim_output", nargs="?", type=Path, help="Sim output CSV (required for single/flow; for --flow use after --flow dir)")
    parser.add_argument("input_csv", nargs="?", type=Path, help="Optional M3 input CSV (for load balance plot)")
    parser.add_argument("--compare", type=Path, metavar="GENERATED_DIR", help="Compare LPT vs fixed order using this generated dir")
    parser.add_argument("--flow", type=Path, metavar="GENERATED_DIR", help="Show order/tote flow: LPT sequence, orders per conveyor, completion Gantt (pass sim_output as first positional)")
    parser.add_argument("--out-dir", type=Path, default=Path("scheduler"), help="Directory for plots (default: scheduler)")
    args = parser.parse_args()

    if args.compare is not None:
        run_compare(args.compare, args.out_dir)
        return

    if args.flow is not None:
        if not args.sim_output or not args.sim_output.exists():
            parser.error("--flow requires sim_output CSV: python scheduler/visualize.py --flow path/to/generated/ sim_output.csv")
            return
        run_flow(args.flow, args.sim_output, args.out_dir)
        return

    if not args.sim_output or not args.sim_output.exists():
        parser.error("Need sim_output CSV, or use --compare GENERATED_DIR or --flow GENERATED_DIR sim_output.csv")
        return

    events = load_sim_output(args.sim_output)
    ms = makespan(events)
    total = len(events)
    print(f"Makespan: {ms:.2f}s  |  Total picks: {total}")

    input_counts = None
    if args.input_csv and args.input_csv.exists():
        input_counts = load_input_counts(args.input_csv)
        conv_totals = [sum(input_counts[c]) for c in range(4)]
        print(f"Load per conveyor: {conv_totals}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_single(events, input_counts, out_dir)


if __name__ == "__main__":
    main()
