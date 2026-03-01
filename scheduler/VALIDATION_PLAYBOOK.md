# Validation Playbook: Every Event in the System

This document defines **every event type** that can occur and the **order and rules** to validate a run.

---

## 1. Event types

| Event | When it happens | What to validate |
|-------|-----------------|------------------|
| **LOAD** | Item is placed onto conveyor 0 at its load time | Time = `item_id × load_spacing` (default 2.5 s). Conveyor is always 0. Order of items = load sequence (tote order + item order within tote if tote data exists; else all shape 0, then 1, …). |
| **HOP** | Item leaves one conveyor and will arrive at the next | Time increases by exactly `HOP_SECONDS` (5.0 s) from the previous time that item was at that conveyor. Conveyors are in a ring: 0 → 1 → 2 → 3 → 0. |
| **PICK** | Item is consumed at a conveyor (demand satisfied) | Conveyor had unmet demand for that shape. That conveyor’s demand for that shape decreases by 1. Item is removed from circulation. |

---

## 2. Order of events

- **Clock:** One global clock; all events have a non‑decreasing time.
- **LOADs:** For “all load at conveyor 0” mode, LOADs occur at t = 0, 2.5, 5.0, … (one per item, in load sequence order).
- **PICKs and HOPs:** The simulator processes events in time order. At each step it considers the next event (earliest time). If that item is at a conveyor that has demand for its shape → **PICK**. Otherwise → **HOP** (re-enter queue at same item, next conveyor, time + 5.0).
- **Makespan:** Time of the **last PICK** (time when the last item is picked).

---

## 3. Parameters (for validation)

| Parameter | Value | Meaning |
|-----------|--------|--------|
| `load_spacing` | 2.5 s | Time between consecutive LOADs at conveyor 0 (half a belt). |
| `HOP_SECONDS` | 5.0 s | Time to move from one conveyor to the next. |
| `NUM_CONVEYORS` | 4 | Conveyors 0, 1, 2, 3 in a ring (0→1→2→3→0). |

---

## 4. Load sequence (order of LOADs)

- **Without tote data:** Item order = all shape 0, then all shape 1, … shape 7 (by global demand).
- **With tote data:** Item order = from `build_load_sequence`: totes in order (earliest LPT position, then farthest conveyor, then tote_id); within each tote, items in `tote_contents[tote_id]` order (or `item_order_per_tote[tote_id]` if provided).

So: **event 1** = LOAD item_id=0 at t=0, **event 2** = LOAD item_id=1 at t=2.5, … then HOPs and PICKs interleaved by time.

---

## 5. How to generate the full event list

From project root:

```bash
# Full chronological playbook (LPT order, travel-aware, tote sequence if present)
python scheduler/write_event_playbook.py data_generator/<timestamp>/generated --out scheduler/event_playbook.txt

# Also write CSV for scripts
python scheduler/write_event_playbook.py data_generator/<timestamp>/generated --out scheduler/event_playbook.txt --csv scheduler/event_playbook.csv

# Fixed order or SPT
python scheduler/write_event_playbook.py data_generator/<timestamp>/generated --order fixed --out scheduler/event_playbook_fixed.txt
```

Output: one line per event in time order (LOAD, HOP, PICK) with time, item_id, shape/conveyor as applicable.

---

## 6. Validation checklist

1. **LOADs:** Count = total items. Times = 0, 2.5, 5.0, … for item_id 0, 1, 2, …. All at conveyor 0.
2. **Shapes:** Each item has a shape 0–7; total per shape matches conveyor input (sum over conveyors).
3. **PICKs:** Exactly one PICK per item; each (conv, shape) pair’s PICK count ≤ initial demand at that conveyor for that shape.
4. **HOPs:** Every HOP is from conv C to (C+1)%4; time step is 5.0 s from the event that brought the item to C (previous HOP or LOAD if C=0).
5. **Makespan:** Equals the time of the last PICK in the playbook.

---

## 7. Sim trace in code

The simulator can return a trace when called with `return_trace=True`:

```python
results, trace_events = simulate_greedy(
    conveyors,
    all_load_at_conveyor_0=True,
    load_spacing=2.5,
    load_sequence=my_sequence,
    return_trace=True,
)
# trace_events = [(time, "LOAD"|"HOP"|"PICK", payload), ...] in chronological order
# LOAD payload = (item_id, shape, conv)
# HOP payload = (item_id, from_conv, to_conv)
# PICK payload = (item_id, conv, shape)
```

This is what `write_event_playbook.py` uses to write the playbook file.
