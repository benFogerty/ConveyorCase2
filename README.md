# Conveyor Simulator (Case 3)

Greedy conveyor simulation for the IDEAS Clinic case.

## Run

```bash
# Default: items staggered per conveyor (from input)
python3 conveyor_sim.py input.csv output.csv

# All items load at conveyor 0, 2.5s apart (half a belt); makespan = total time to last pick
python3 conveyor_sim.py input.csv output.csv --all-load-at-conveyor-0
```

## Notes

- Input columns must be: `conv_num,cirle,pentagon,trapezoid,triangle,star,moon,heart,cross`
- Output columns are: `conv_num,shape,time`
- The scheduler is greedy: at each step it emits the next event with the smallest available time.
- The simulator uses only the input CSV and deterministic timing functions (no hardcoded expected output).
- Circulation assumptions:
  - Four conveyors in a ring.
  - Moving from one conveyor to the next takes `5.0` seconds.
  - One full loop takes `20.0` seconds.
  - If an item is not picked when it passes a conveyor scanner, it continues circulating.
- **Two modes:**
  - **Default:** Items are initially staggered along each conveyor segment (per input CSV) so they do not all arrive at the same time.
  - **`--all-load-at-conveyor-0`:** All items load onto conveyor 0 only. Clock starts when the first item is loaded (t=0). Each item is added `--load-spacing` seconds apart (default 2.5 = half a conveyor belt). Total time (makespan) = time until the last item is picked.

## How Generator Data Relates to This Simulator

- The data generator notebook outputs order-level files: `order_itemtypes.csv`, `order_quantities.csv`, and `orders_totes.csv`.
- This simulator does not read those files directly.
- The simulator expects a pre-organized conveyor input file (`conv_num` + shape counts), which means a preprocessing/organization step has already happened.

### Optimization Focus

- In this case study, the main optimization decision is how to transform generated order+tote data into conveyor-level input counts.
- That organization/mapping step is what you optimize.
- Then you run the simulator and evaluate resulting KPIs (for example: completion time, throughput, and circulation count).
- The **`scheduler/`** package contains the optimization and comparison code (LPT, joint methods, visualization). See `scheduler/README.md` for commands.

## Reading Generated Data

The generator writes three CSV files with no headers. Each row index is an order ID.

### `order_itemtypes.csv`

- Row `i`: the list of item/shape type IDs required by order `i`.
- Column position `k` in that row: the `k`-th distinct type needed by that order.
- Values are integer type IDs (for this case, typically `0..7`).

### `order_quantities.csv`

- Row `i`: quantities for order `i`.
- Column position `k`: quantity required for the item type at position `k` in `order_itemtypes.csv` row `i`.
- Values are positive integers.

### `orders_totes.csv`

- Row `i`: tote locations for order `i`.
- Column position `k`: tote ID where the item type at position `k` in `order_itemtypes.csv` row `i` is stored.
- Values are tote IDs (integers).

### Positional Alignment Rule

- For a fixed order row `i`, column position `k` means the same item across all three files.
- In other words:
  - `type = order_itemtypes[i][k]`
  - `qty = order_quantities[i][k]`
  - `tote = orders_totes[i][k]`

### Example Interpretation

- If row `i` is:
  - `order_itemtypes`: `3,1`
  - `order_quantities`: `3,2`
  - `orders_totes`: `0,0`
- Then order `i` needs:
  - 3 units of type 3 from tote 0
  - 2 units of type 1 from tote 0
