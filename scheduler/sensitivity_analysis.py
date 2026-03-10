from __future__ import annotations

import os
import sys
import random
import tempfile
from pathlib import Path
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import matplotlib.pyplot as plt

from conveyor_sim import load_conveyors, simulate_greedy
from scheduler.beam_search import beam_search_order_sequence
from scheduler.core import load_generator_data, load_tote_data
from scheduler.joint_solution import (
    solution_from_order_sequence,
    initial_solution,
    evaluate_solution,
    hill_climb_order_and_tote,
    hill_climb_full,
    simulated_annealing,
    genetic_algorithm,
    tabu_search,
    iterated_local_search,
)
from scheduler.search_orders import greedy_makespan_insertion


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

SHAPE_COLS = [
    "cirle",
    "pentagon",
    "trapezoid",
    "triangle",
    "star",
    "moon",
    "heart",
    "cross",
]

DEFAULT_METHODS = [
    "Baseline",
    "BaselineRR",
    "GreedyMakespanInsertion",
    "BeamSearch",
    "OrderToteHill",
    "FullHillClimb",
    "SimulatedAnnealing",
    "GeneticAlgorithm",
    "TabuSearch",
    "IteratedLocalSearch",
]

DEMAND_FACTORS = [0.8, 1.0, 1.2, 1.5]
SPEED_FACTORS = [0.8, 1.0, 1.2]

DEFAULT_SEED = 42
DEFAULT_JOINT_MAX_EVALS = 800
DEFAULT_JOINT_RESTARTS = 2
DEFAULT_POLISH = False
DEFAULT_POLISH_MAX_EVALS = 300


def validate_input_columns(df: pd.DataFrame) -> None:
    required = {"conv_num", *SHAPE_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")


def scale_demand(df: pd.DataFrame, factor: float) -> pd.DataFrame:
    scaled = df.copy()
    for col in SHAPE_COLS:
        scaled[col] = (scaled[col] * factor).round().clip(lower=0).astype(int)
    return scaled


def write_temp_csv(df: pd.DataFrame) -> str:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp_path = tmp.name
    tmp.close()
    df.to_csv(tmp_path, index=False)
    return tmp_path


def metrics_from_rows(rows: List[tuple]) -> Dict:
    makespan = max(t for _, _, t in rows) if rows else 0.0

    conv_finish_times = {}
    for conv_num, _, t in rows:
        conv_finish_times[conv_num] = max(conv_finish_times.get(conv_num, 0.0), t)

    avg_last_order = (
        sum(conv_finish_times.values()) / len(conv_finish_times)
        if conv_finish_times
        else 0.0
    )

    return {
        "makespan": makespan,
        "avg_last_order": avg_last_order,
        "total_orders": len(conv_finish_times),
    }


def run_greedy_csv_wrapper(input_csv_path: str, speed_multiplier: float = 1.0) -> Dict:
    conveyors, orders_per_conv = load_conveyors(Path(input_csv_path))

    base_load_spacing = 2.5
    effective_load_spacing = base_load_spacing / speed_multiplier

    rows = simulate_greedy(
        conveyors,
        all_load_at_conveyor_0=True,
        load_spacing=effective_load_spacing,
        orders_per_conveyor=orders_per_conv,
    )
    return metrics_from_rows(rows)


def run_algorithm_on_generated_instance(
    generated_path: str,
    method: str,
    seed: int = DEFAULT_SEED,
    joint_max_evals: int = DEFAULT_JOINT_MAX_EVALS,
    joint_restarts: int = DEFAULT_JOINT_RESTARTS,
    polish: bool = DEFAULT_POLISH,
    polish_max_evals: int = DEFAULT_POLISH_MAX_EVALS,
) -> Dict:
    generated_dir = Path(generated_path)
    orders = load_generator_data(generated_dir)
    if not orders:
        raise ValueError(f"No orders found in generated instance: {generated_dir}")

    tote_contents, order_to_totes = load_tote_data(generated_dir, orders)
    if not tote_contents or not order_to_totes:
        raise ValueError(f"Tote data required for generated instance: {generated_dir}")

    n = len(orders)

    if method == "Baseline":
        sol = solution_from_order_sequence(
            orders, tote_contents, list(range(n)), travel_aware=True
        )
        val, _, _ = evaluate_solution(sol, orders, tote_contents, objective="last_order")
        return {"makespan": val, "avg_last_order": val, "total_orders": len(orders)}

    if method == "BaselineRR":
        sol = solution_from_order_sequence(
            orders, tote_contents, list(range(n)), travel_aware=False
        )
        val, _, _ = evaluate_solution(sol, orders, tote_contents, objective="last_order")
        return {"makespan": val, "avg_last_order": val, "total_orders": len(orders)}

    if method == "GreedyMakespanInsertion":
        seq, ms = greedy_makespan_insertion(
            orders,
            travel_aware=True,
            tote_contents=tote_contents,
            order_to_totes=None,
        )
        return {"makespan": ms, "avg_last_order": ms, "total_orders": len(orders)}

    if method == "BeamSearch":
        seq_beam, beam_ms = beam_search_order_sequence(
            orders,
            beam_width=5,
            candidate_pool=12,
            travel_aware=True,
            tote_contents=tote_contents,
            order_to_totes=None,
        )
        return {"makespan": beam_ms, "avg_last_order": beam_ms, "total_orders": len(orders)}

    if method == "OrderToteHill":
        start = initial_solution(orders, tote_contents, travel_aware=True)
        sol, val, _ = hill_climb_order_and_tote(
            orders,
            tote_contents,
            start,
            objective="last_order",
            max_evals=joint_max_evals,
            verbose=False,
        )
        return {"makespan": val, "avg_last_order": val, "total_orders": len(orders)}

    if method == "FullHillClimb":
        start = initial_solution(orders, tote_contents, travel_aware=True)
        sol, val, _ = hill_climb_full(
            orders,
            tote_contents,
            start,
            objective="last_order",
            max_evals=joint_max_evals,
            verbose=False,
        )
        return {"makespan": val, "avg_last_order": val, "total_orders": len(orders)}

    if method == "SimulatedAnnealing":
        best_val = float("inf")
        best_sol = None
        for r in range(joint_restarts):
            s = seed + r if seed is not None else None
            if s is not None:
                random.seed(s)
            sol, val, _ = simulated_annealing(
                orders,
                tote_contents,
                objective="last_order",
                max_evals=joint_max_evals,
                seed=s,
                verbose=False,
            )
            if val < best_val:
                best_val = val
                best_sol = sol

        if best_sol is not None and polish:
            best_sol, best_val, _ = hill_climb_full(
                orders,
                tote_contents,
                best_sol,
                objective="last_order",
                max_evals=polish_max_evals,
                verbose=False,
            )

        return {"makespan": best_val, "avg_last_order": best_val, "total_orders": len(orders)}

    if method == "GeneticAlgorithm":
        sol, val, _ = genetic_algorithm(
            orders,
            tote_contents,
            objective="last_order",
            max_evals=joint_max_evals,
            seed=seed,
            verbose=False,
        )
        return {"makespan": val, "avg_last_order": val, "total_orders": len(orders)}

    if method == "TabuSearch":
        sol, val, _ = tabu_search(
            orders,
            tote_contents,
            initial=None,
            objective="last_order",
            max_evals=joint_max_evals,
            seed=seed,
            verbose=False,
        )
        return {"makespan": val, "avg_last_order": val, "total_orders": len(orders)}

    if method == "IteratedLocalSearch":
        sol, val, _ = iterated_local_search(
            orders,
            tote_contents,
            objective="last_order",
            max_evals=joint_max_evals,
            perturb_strength=3,
            seed=seed,
            verbose=False,
        )
        return {"makespan": val, "avg_last_order": val, "total_orders": len(orders)}

    raise ValueError(f"Unknown method: {method}")


def run_demand_sensitivity(
    base_csv_path: str,
    method: str = "Greedy"
) -> pd.DataFrame:
    if method != "Greedy":
        raise ValueError("Demand sensitivity currently supports only Greedy CSV mode.")

    base_df = pd.read_csv(base_csv_path)
    validate_input_columns(base_df)

    rows = []

    for factor in DEMAND_FACTORS:
        scaled_df = scale_demand(base_df, factor)
        tmp_csv = write_temp_csv(scaled_df)

        try:
            metrics = run_greedy_csv_wrapper(tmp_csv, speed_multiplier=1.0)
            rows.append({
                "analysis_type": "demand_scaling",
                "method": method,
                "input_source": base_csv_path,
                "demand_factor": factor,
                "speed_factor": 1.0,
                "makespan": metrics.get("makespan"),
                "avg_last_order": metrics.get("avg_last_order"),
                "total_orders": metrics.get("total_orders"),
            })
        finally:
            if os.path.exists(tmp_csv):
                os.remove(tmp_csv)

    return pd.DataFrame(rows)


def run_speed_sensitivity(
    base_csv_path: str,
    method: str = "Greedy"
) -> pd.DataFrame:
    if method != "Greedy":
        raise ValueError("Speed sensitivity currently supports only Greedy CSV mode.")

    rows = []

    for speed in SPEED_FACTORS:
        metrics = run_greedy_csv_wrapper(base_csv_path, speed_multiplier=speed)
        rows.append({
            "analysis_type": "conveyor_speed",
            "method": method,
            "input_source": base_csv_path,
            "demand_factor": 1.0,
            "speed_factor": speed,
            "makespan": metrics.get("makespan"),
            "avg_last_order": metrics.get("avg_last_order"),
            "total_orders": metrics.get("total_orders"),
        })

    return pd.DataFrame(rows)


def run_algorithm_comparison(
    generated_path: str,
    methods: List[str]
) -> pd.DataFrame:
    rows = []

    for method in methods:
        metrics = run_algorithm_on_generated_instance(generated_path, method=method)
        rows.append({
            "analysis_type": "algorithm_comparison",
            "method": method,
            "input_source": generated_path,
            "demand_factor": 1.0,
            "speed_factor": 1.0,
            "makespan": metrics.get("makespan"),
            "avg_last_order": metrics.get("avg_last_order"),
            "total_orders": metrics.get("total_orders"),
        })

    return pd.DataFrame(rows)


def save_plot_demand(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["demand_factor"], df["makespan"], marker="o")
    plt.xlabel("Demand scaling factor")
    plt.ylabel("Makespan")
    plt.title("Sensitivity Analysis: Demand Scaling vs Makespan")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_plot_speed(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(df["speed_factor"], df["makespan"], marker="o")
    plt.xlabel("Speed multiplier")
    plt.ylabel("Makespan")
    plt.title("Sensitivity Analysis: Conveyor Speed vs Makespan")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_plot_algorithms(df: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(12, 5))
    plt.bar(df["method"], df["makespan"])
    plt.xticks(rotation=40, ha="right")
    plt.ylabel("Makespan")
    plt.title("Sensitivity Analysis: Algorithm Comparison")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def summarize_results(
    demand_df: pd.DataFrame,
    speed_df: pd.DataFrame,
    algo_df: pd.DataFrame
) -> str:
    lines = []

    lines.append("SENSITIVITY ANALYSIS SUMMARY")
    lines.append("=" * 40)

    if not demand_df.empty:
        base = demand_df.loc[demand_df["demand_factor"] == 1.0, "makespan"].iloc[0]
        low = demand_df.loc[
            demand_df["demand_factor"] == demand_df["demand_factor"].min(),
            "makespan"
        ].iloc[0]
        high = demand_df.loc[
            demand_df["demand_factor"] == demand_df["demand_factor"].max(),
            "makespan"
        ].iloc[0]

        lines.append("\n1) Demand Scaling")
        lines.append(f"   Base makespan at 1.0x demand: {base:.2f}")
        lines.append(f"   Makespan at lowest demand:    {low:.2f}")
        lines.append(f"   Makespan at highest demand:   {high:.2f}")

        if high > base:
            pct = ((high - base) / base) * 100
            lines.append(
                f"   Interpretation: Increasing demand to the highest tested level increased makespan by {pct:.1f}%."
            )

    if not speed_df.empty:
        base = speed_df.loc[speed_df["speed_factor"] == 1.0, "makespan"].iloc[0]
        slow = speed_df.loc[
            speed_df["speed_factor"] == speed_df["speed_factor"].min(),
            "makespan"
        ].iloc[0]
        fast = speed_df.loc[
            speed_df["speed_factor"] == speed_df["speed_factor"].max(),
            "makespan"
        ].iloc[0]

        lines.append("\n2) Conveyor Speed")
        lines.append(f"   Base makespan at 1.0x speed: {base:.2f}")
        lines.append(f"   Makespan at slowest speed:   {slow:.2f}")
        lines.append(f"   Makespan at fastest speed:   {fast:.2f}")

        if fast < base:
            pct = ((base - fast) / base) * 100
            lines.append(
                f"   Interpretation: Increasing speed to the highest tested level reduced makespan by {pct:.1f}%."
            )

    if not algo_df.empty:
        best_idx = algo_df["makespan"].idxmin()
        worst_idx = algo_df["makespan"].idxmax()
        best_row = algo_df.loc[best_idx]
        worst_row = algo_df.loc[worst_idx]

        lines.append("\n3) Algorithm Comparison")
        lines.append(f"   Best method:  {best_row['method']} ({best_row['makespan']:.2f})")
        lines.append(f"   Worst method: {worst_row['method']} ({worst_row['makespan']:.2f})")

        if worst_row["makespan"] > 0:
            pct = ((worst_row["makespan"] - best_row["makespan"]) / worst_row["makespan"]) * 100
            lines.append(
                f"   Interpretation: The best method improved makespan by {pct:.1f}% versus the worst tested method."
            )

    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("python -m scheduler.sensitivity_analysis <input_csv_path> <generated_instance_path>")
        print('Example:')
        print('python -m scheduler.sensitivity_analysis "example_files/MSE433_M3_Example_input.csv" "data_generator/20260302_123456/generated"')
        sys.exit(1)

    base_csv_path = sys.argv[1]
    generated_path = sys.argv[2]

    demand_df = run_demand_sensitivity(base_csv_path, method="Greedy")
    speed_df = run_speed_sensitivity(base_csv_path, method="Greedy")
    algo_df = run_algorithm_comparison(generated_path, methods=DEFAULT_METHODS)

    all_results = pd.concat([demand_df, speed_df, algo_df], ignore_index=True)
    csv_path = RESULTS_DIR / "sensitivity_results.csv"
    all_results.to_csv(csv_path, index=False)

    save_plot_demand(demand_df, RESULTS_DIR / "sensitivity_demand.png")
    save_plot_speed(speed_df, RESULTS_DIR / "sensitivity_speed.png")
    save_plot_algorithms(algo_df, RESULTS_DIR / "sensitivity_algorithms.png")

    summary = summarize_results(demand_df, speed_df, algo_df)
    summary_path = RESULTS_DIR / "sensitivity_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")

    print("\nDone.")
    print(f"Saved results to: {csv_path}")
    print(f"Saved demand plot to: {RESULTS_DIR / 'sensitivity_demand.png'}")
    print(f"Saved speed plot to: {RESULTS_DIR / 'sensitivity_speed.png'}")
    print(f"Saved algorithm plot to: {RESULTS_DIR / 'sensitivity_algorithms.png'}")
    print(f"Saved summary to: {summary_path}")
    print("\n" + summary)


if __name__ == "__main__":
    main()