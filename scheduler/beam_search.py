#!/usr/bin/env python3
"""
Beam Search for ORDER SEQUENCE optimization.

Goal: find an order sequence that minimizes makespan as evaluated by the simulator
(using makespan_for_sequence from scheduler/search_orders.py).

Beam Search is constructive: it builds the sequence position-by-position, but
keeps the best B partial sequences at each depth (beam_width), instead of only 1
like greedy insertion.

This module is intended to be imported by compare_methods.py, e.g.:
  from scheduler.beam_search import beam_search_order_sequence
"""

from __future__ import annotations

from typing import List, Tuple, Optional, Dict


def _order_total_items(orders) -> List[int]:
    """
    orders format (from load_generator_data): List[(order_idx, [(shape_id, qty), ...])]
    Return list total_items[order_id] = sum qty for that order.
    """
    totals = [0] * len(orders)
    for oid, pairs in orders:
        totals[oid] = sum(q for _, q in pairs)
    return totals


def beam_search_order_sequence(
    orders,
    beam_width: int = 5,
    candidate_pool: int = 12,
    travel_aware: bool = True,
    tote_contents: Optional[dict] = None,
    order_to_totes: Optional[dict] = None,
    item_order_per_tote: Optional[dict] = None,
) -> Tuple[List[int], float]:
    """
    Beam search over order sequences.

    Args:
      orders: output of load_generator_data(...)
      beam_width: keep top-B partial sequences at each depth (B=1 -> greedy)
      candidate_pool: when expanding a partial sequence, only try the top-K remaining
                      orders by item count (K capped by remaining). This reduces runtime.
                      Set to a large number (e.g. 9999) to try all remaining.
      travel_aware, tote_contents, order_to_totes, item_order_per_tote:
        passed through to makespan_for_sequence.

    Returns:
      (best_sequence, best_makespan)
    """
    # Import here to avoid circular import issues when scheduler modules import each other
    from scheduler.search_orders import makespan_for_sequence

    n = len(orders)
    if n == 0:
        return [], 0.0

    beam_width = max(1, int(beam_width))
    candidate_pool = max(1, int(candidate_pool))

    all_order_ids = list(range(n))
    totals = _order_total_items(orders)

    # Start with empty sequence; we will evaluate partial sequences using the simulator.
    # beam entries: (sequence, makespan_value)
    beam: List[Tuple[List[int], float]] = [([], 0.0)]

    for depth in range(n):
        candidates: List[Tuple[List[int], float]] = []

        for seq, _ in beam:
            used = set(seq)
            remaining = [oid for oid in all_order_ids if oid not in used]
            if not remaining:
                continue

            # Heuristic shortlist of remaining orders to expand:
            # Try biggest remaining orders first (often most impactful for makespan).
            remaining_sorted = sorted(remaining, key=lambda oid: (-totals[oid], oid))
            trial_orders = remaining_sorted[: min(candidate_pool, len(remaining_sorted))]

            for oid in trial_orders:
                new_seq = seq + [oid]
                ms = makespan_for_sequence(
                    orders,
                    new_seq,
                    travel_aware,
                    tote_contents=tote_contents,
                    order_to_totes=order_to_totes,
                    item_order_per_tote=item_order_per_tote,
                )
                candidates.append((new_seq, ms))

        if not candidates:
            # Fallback: shouldn't happen, but keep things safe
            return beam[0][0], beam[0][1]

        # Keep best beam_width by makespan (lower is better), tie-break lexicographically for determinism
        candidates.sort(key=lambda x: (x[1], x[0]))
        beam = candidates[:beam_width]

    best_seq, best_ms = beam[0]
    return best_seq, best_ms


def recommend_params(num_orders: int) -> Dict[str, int]:
    """
    Helper to choose reasonable defaults based on n.
    Not required, but handy if you want to tune automatically.
    """
    if num_orders <= 18:
        return {"beam_width": 10, "candidate_pool": 9999}
    if num_orders <= 30:
        return {"beam_width": 6, "candidate_pool": 15}
    if num_orders <= 45:
        return {"beam_width": 4, "candidate_pool": 12}
    return {"beam_width": 3, "candidate_pool": 10}