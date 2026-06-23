"""
Cost-Neutral Hedge Optimization Model
Implements the model from formulas.tex with optional constraints (III, IV, V),
all parameters settable at the top, and subset filtering using utils.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.optimize import minimize, Bounds, linprog
# Add Excel writing utility
from utils.write_excel import write_cost_neutral_hedge_results
from utils.load_data import apply_time_horizon, ensure_datetime_index, read_csv_time_window, find_latest_csv_with_substring

# --- PARAMETERS & FLAGS (set here) ---
# Subset filter (set to e.g. {"product_type": "year", "load_type": "base"} or None for all)
# SUBSET_FILTER = [
#     {"product_type": "day", "load_type": "base"},
#     {"product_type": "day", "load_type": "peak"},
#     {"product_type": "weekend", "load_type": "base"},
# ]

# Use this SUBSET_FILTER to run the optimization for year base and year peak products only
# SUBSET_FILTER = [
# SUBSET_FILTER = [     
#     {"product_type": "year", "load_type": "base"},
#     {"product_type": "year", "load_type": "peak"},
# ]

# Use this SUBSET_FILTER to run the optimization for quarter base and year peak products only
# SUBSET_FILTER = [
#     {"product_type": "quarter", "load_type": "base"},
#     {"product_type": "quarter", "load_type": "peak"},
# ]

# Use this SUBSET_FILTER to run the optimization for month base and year peak products only
SUBSET_FILTER = [
    {"product_type": "month", "load_type": "base"},
    {"product_type": "month", "load_type": "peak"},
]


# Flag to save the filtered subset coverage and metadata for reference/debugging
SAVE_FILTERED_SUBSET = True

# Model flags
ENFORCE_COVERAGE = False # Constraint III
HEDGE_RATIO_BOUNDS = True # Constraint IV
MIN_HEDGE_RATIO = True    # Constraint V

# Model parameters
EPSILON = 0.3            # Tolerance band for constraint III
HEDGE_RATIO_LB = 0.0     # Lower bound for a^j (IV)
HEDGE_RATIO_UB = 0.8    # Upper bound for a^j (IV)
H_MIN = 0.90               # Minimum hedge ratio (V)

# Optional time horizon (inclusive). Dates are filtered from the parquet coverage and CSV inputs.
# Leave either as None for unbounded range on that side.
# Example: "2026-01-01", "2026-12-31 23:00:00+01:00" (supports ISO with timezone offset).
# Warning: Requested dates must exist in all data sources (coverage, loads, prices) or alignment will fail.
START_DATE = "2028-01-01"
END_DATE = "2028-12-31 23:00:00+01:00"

# Solve strategy
# - "global": one full-horizon solve
# - "monthly": rolling monthly chunks with overlap, then optional global polish
SOLVE_MODE = "monthly" 
#SOLVE_MODE = "global"
CHUNK_OVERLAP_HOURS = 48
POLISH_GLOBAL_AFTER_CHUNKS = True

# Optimizer backend
# - "linprog": deterministic LP solve via HiGHS (recommended for this linear model)
# - "slsqp": nonlinear optimizer path (legacy)
#OPTIMIZER_BACKEND = "linprog"
OPTIMIZER_BACKEND = "slsqp"

# Data paths
COVERAGE_PATH = "utils/data/hedge_instruments_coverage.parquet"
METADATA_PATH = "utils/data/hedge_instruments_metadata.parquet"


def resolve_coverage_path_for_horizon(coverage_path, start_date=None, end_date=None):
    """
    Prefer a year-specific parquet when the requested horizon stays within one year.

    Falls back to the provided coverage_path if no matching yearly file exists or the
    horizon spans multiple years.
    """
    if start_date is None and end_date is None:
        return coverage_path

    try:
        start_year = pd.Timestamp(start_date).year if start_date is not None else None
        end_year = pd.Timestamp(end_date).year if end_date is not None else None
    except Exception:
        return coverage_path

    if start_year is None:
        target_year = end_year
    elif end_year is None:
        target_year = start_year
    elif start_year == end_year:
        target_year = start_year
    else:
        return coverage_path

    if target_year is None:
        return coverage_path

    coverage_file = Path(coverage_path)
    yearly_candidate = coverage_file.with_name(f"{coverage_file.stem}_{target_year}{coverage_file.suffix}")
    if yearly_candidate.exists():
        print(f"Using year-specific coverage file: {yearly_candidate}")
        return str(yearly_candidate)
    return coverage_path

try:
    LOADS_PATH = find_latest_csv_with_substring("con", data_dir="Data/current")  # e.g., consumption/load file
    PRICE_CSV_PATH = find_latest_csv_with_substring("pri", data_dir="Data/current")  # e.g., price file
except FileNotFoundError as e:
    print(f"[ERROR] {e}\nPlease ensure the correct CSV files are present in Data/current/.")
    raise SystemExit(1)


# Expose key variables for testing and use in objective
n_instruments = None
loads = None
spot_prices = None
instrument_coverage = None
forward_prices = None
objective_coeff = None
constraints = []
bounds = None

# --- OBJECTIVE (I) ---
def objective(a):
    """
    Objective function: minimize total hedging cost for the selected instruments.
    a: array of hedge ratios (length n_instruments)
    Implements: sum_j sum_i P^j * a^j * indicator(i in I^j) * L_i
    """
    # objective_coeff already contains P^j * sum_i(I_{i,j} * L_i)
    if objective_coeff is None:
        raise RuntimeError("objective_coeff is not initialized.")
    return float(np.dot(objective_coeff, a))


def build_problem_matrices(cov, loads_vec, spot_vec, fwd_vec):
    """
    Build vectorized matrices/vectors for objective and constraints.
    """
    covered_load_per_instr = cov.T @ loads_vec
    covered_hours_per_instr = cov.sum(axis=0)
    avg_load_per_instr = np.divide(
        covered_load_per_instr,
        covered_hours_per_instr,
        out=np.zeros_like(covered_load_per_instr, dtype=float),
        where=covered_hours_per_instr > 0,
    )
    hedge_matrix = cov * avg_load_per_instr[np.newaxis, :]
    obj_coeff = fwd_vec * covered_load_per_instr
    cost_neutral_vec = hedge_matrix.T @ spot_vec
    cost_neutral_rhs = float(loads_vec @ spot_vec)
    total_load = float(np.sum(loads_vec))
    min_ratio_vec = np.divide(
        covered_load_per_instr,
        total_load,
        out=np.zeros_like(covered_load_per_instr, dtype=float),
        where=total_load != 0,
    )
    return {
        "covered_hours_per_instr": covered_hours_per_instr,
        "covered_load_per_instr": covered_load_per_instr,
        "avg_load_per_instr": avg_load_per_instr,
        "hedge_matrix": hedge_matrix,
        "objective_coeff": obj_coeff,
        "cost_neutral_vec": cost_neutral_vec,
        "cost_neutral_rhs": cost_neutral_rhs,
        "min_ratio_vec": min_ratio_vec,
    }


def make_bounds(covered_hours_per_instr):
    if not HEDGE_RATIO_BOUNDS:
        return None
    lb = np.where(covered_hours_per_instr > 0, HEDGE_RATIO_LB, 0.0)
    ub = np.where(covered_hours_per_instr > 0, HEDGE_RATIO_UB, 0.0)
    return Bounds(lb, ub)


def solve_vectorized(cov, loads_vec, spot_vec, fwd_vec, a0=None, label="global"):
    mats = build_problem_matrices(cov, loads_vec, spot_vec, fwd_vec)
    local_bounds = make_bounds(mats["covered_hours_per_instr"])

    def local_objective(a):
        return float(np.dot(mats["objective_coeff"], a))

    constraints_local = []

    def cost_neutrality(a):
        return mats["cost_neutral_rhs"] - float(np.dot(mats["cost_neutral_vec"], a))

    constraints_local.append({"type": "eq", "fun": cost_neutrality})

    hedge_matrix = mats["hedge_matrix"]

    def hedge_profile_latex(a):
        return hedge_matrix @ a

    if ENFORCE_COVERAGE:
        def coverage_lower(a):
            return (hedge_matrix @ a) - (1 - EPSILON) * loads_vec

        def coverage_upper(a):
            return (1 + EPSILON) * loads_vec - (hedge_matrix @ a)

        constraints_local.append({"type": "ineq", "fun": coverage_lower})
        constraints_local.append({"type": "ineq", "fun": coverage_upper})

    if MIN_HEDGE_RATIO:
        def min_hedge_ratio(a):
            return float(np.dot(mats["min_ratio_vec"], a) - H_MIN)

        constraints_local.append({"type": "ineq", "fun": min_hedge_ratio})

    if a0 is None:
        a0 = np.full(cov.shape[1], 0.5)
    if local_bounds is not None:
        a0 = np.clip(a0, local_bounds.lb, local_bounds.ub)

    print(f"\nConstraint values at initial guess for {label}:")
    for i, c in enumerate(constraints_local):
        val = c["fun"](a0)
        if c["type"] == "eq":
            print(f"Constraint {i} (equality): value = {val}")
        else:
            minval = np.min(val) if hasattr(val, "__len__") else val
            print(f"Constraint {i} (inequality): min value = {minval}")

    print(f"Starting optimizer for {label} with backend={OPTIMIZER_BACKEND}...")
    if OPTIMIZER_BACKEND == "linprog":
        n_vars = cov.shape[1]
        c = mats["objective_coeff"]

        A_eq = np.atleast_2d(mats["cost_neutral_vec"])
        b_eq = np.array([mats["cost_neutral_rhs"]], dtype=float)

        A_ub_blocks = []
        b_ub_blocks = []

        if ENFORCE_COVERAGE:
            # (hedge_matrix @ a) - (1-eps)loads >= 0  ->  -hedge_matrix @ a <= -(1-eps)loads
            A_ub_blocks.append(-hedge_matrix)
            b_ub_blocks.append(-(1 - EPSILON) * loads_vec)
            # (1+eps)loads - (hedge_matrix @ a) >= 0  ->   hedge_matrix @ a <=  (1+eps)loads
            A_ub_blocks.append(hedge_matrix)
            b_ub_blocks.append((1 + EPSILON) * loads_vec)

        if MIN_HEDGE_RATIO:
            # dot(min_ratio_vec, a) - H_MIN >= 0  ->  -dot(min_ratio_vec, a) <= -H_MIN
            A_ub_blocks.append(-np.atleast_2d(mats["min_ratio_vec"]))
            b_ub_blocks.append(np.array([-H_MIN], dtype=float))

        if A_ub_blocks:
            A_ub = np.vstack(A_ub_blocks)
            b_ub = np.concatenate([np.ravel(x) for x in b_ub_blocks])
        else:
            A_ub = None
            b_ub = None

        if local_bounds is None:
            bounds_lp = [(None, None)] * n_vars
        else:
            bounds_lp = list(zip(local_bounds.lb.tolist(), local_bounds.ub.tolist()))

        lp_result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_lp,
            method="highs",
        )

        result = lp_result
    elif OPTIMIZER_BACKEND == "slsqp":
        result = minimize(
            local_objective,
            a0,
            method="SLSQP",
            bounds=local_bounds,
            constraints=constraints_local,
            options={"disp": True},
        )
    else:
        raise ValueError(f"Unsupported OPTIMIZER_BACKEND='{OPTIMIZER_BACKEND}'. Use 'linprog' or 'slsqp'.")

    return result, mats, constraints_local, local_bounds, hedge_profile_latex


def iter_month_chunks(index, overlap_hours):
    overlap = pd.Timedelta(hours=overlap_hours)
    tz = index.tz
    first = index.min()
    last = index.max()
    month_start = pd.Timestamp(first.year, first.month, 1, tz=tz)
    while month_start <= last:
        next_month = month_start + pd.offsets.MonthBegin(1)
        core_start = month_start
        core_end = next_month - pd.Timedelta(hours=1)
        chunk_start = core_start - overlap
        chunk_end = core_end + overlap
        core_mask = (index >= core_start) & (index <= core_end)
        chunk_mask = (index >= chunk_start) & (index <= chunk_end)
        yield core_start, core_end, chunk_mask, core_mask
        month_start = next_month


def monthly_chunk_initializer(cov, loads_vec, spot_vec, fwd_vec, hour_index):
    n_instr = cov.shape[1]
    accum = np.zeros(n_instr)
    weights = np.zeros(n_instr)
    prev_a = np.full(n_instr, 0.5)

    for core_start, core_end, chunk_mask, core_mask in iter_month_chunks(hour_index, CHUNK_OVERLAP_HOURS):
        if not np.any(core_mask) or not np.any(chunk_mask):
            continue

        label = f"chunk {core_start.strftime('%Y-%m')}"
        chunk_cov = cov[chunk_mask]
        chunk_loads = loads_vec[chunk_mask]
        chunk_spot = spot_vec[chunk_mask]

        result_chunk, _, _, bounds_chunk, _ = solve_vectorized(
            chunk_cov,
            chunk_loads,
            chunk_spot,
            fwd_vec,
            a0=prev_a,
            label=label,
        )

        if result_chunk.success:
            prev_a = result_chunk.x
        elif bounds_chunk is not None:
            prev_a = np.clip(prev_a, bounds_chunk.lb, bounds_chunk.ub)

        core_cov = cov[core_mask]
        core_weights = core_cov.sum(axis=0)
        accum += prev_a * core_weights
        weights += core_weights

    init_a = np.divide(accum, weights, out=prev_a.copy(), where=weights > 0)
    return init_a

def main():
    global forward_prices, instrument_coverage, loads, spot_prices, constraints, bounds, objective_coeff, n_instruments
    """
    Main workflow for cost-neutral hedge optimization.

    - Loads hedge instrument coverage and metadata.
    - Applies subset filtering based on product_type and load_type.
    - Calculates forward prices for the selected subset.
    - Sets up the optimization problem (objective and constraints).
    - Solves for optimal hedge ratios using scipy.optimize.minimize.
    - Prints results to the console (and optionally writes to Excel).
    """
    # --- DATA LOADING & SUBSET FILTERING ---
    from utils.filter_subset_selection_of_hedge_instruments import filter_hedge_instruments

    resolved_coverage_path = resolve_coverage_path_for_horizon(
        COVERAGE_PATH,
        start_date=START_DATE,
        end_date=END_DATE,
    )

    # Load price data once and reuse it for both forward-price enrichment and spot series.
    try:
        price_df = read_csv_time_window(
            PRICE_CSV_PATH,
            start_date=START_DATE,
            end_date=END_DATE,
            sep=';',
            decimal=',',
        )
        if price_df.empty:
            raise ValueError("Price CSV window produced no rows for the requested horizon.")
        price_value_col = price_df.columns[0]
        price_df[price_value_col] = pd.to_numeric(
            price_df[price_value_col].astype(str).str.replace(',', '.'),
            errors='raise',
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load spot prices from {PRICE_CSV_PATH}: {e}")

    # Filter subset and get coverage/metadata
    try:
        filtered_coverage, run_dir = filter_hedge_instruments(
            subset_filter=SUBSET_FILTER,
            coverage_path=resolved_coverage_path,
            mapping_path=METADATA_PATH,
            save=SAVE_FILTERED_SUBSET,
            price_csv_path=PRICE_CSV_PATH,
            price_df=price_df,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to filter hedge instruments: {e}")
    # Load the filtered metadata written by the filter step.
    # It already contains the subset selection and forward prices, so we avoid
    # re-reading the full metadata parquet and recomputing prices here.
    if isinstance(SUBSET_FILTER, list):
        suffix = "_and".join([f"{filt['product_type']}_{filt['load_type']}" for filt in SUBSET_FILTER])
    else:
        suffix = f"{SUBSET_FILTER['product_type']}_{SUBSET_FILTER['load_type']}"
    if run_dir is None:
        raise RuntimeError("filter_hedge_instruments did not return a run_dir, so filtered metadata cannot be loaded.")
    filtered_metadata_path = os.path.join(run_dir, f"filtered_metadata_{suffix}.parquet")
    try:
        metadata = pd.read_parquet(filtered_metadata_path)
    except Exception as e:
        raise RuntimeError(f"Failed to read filtered metadata parquet file {filtered_metadata_path}: {e}")
    subset_ids = metadata["instrument_id"].tolist()
    if not subset_ids:
        raise RuntimeError("No instruments found in filtered metadata. Check your subset filter and filtered output.")

    # --- MODEL SETUP ---
    # Extract data arrays
    # Parse coverage datetime to index and apply optional time horizon.
    # Timezone-aware datetimes are preserved for consistency with input data.
    coverage = ensure_datetime_index(filtered_coverage, datetime_col="datetime")
    coverage = apply_time_horizon(coverage, start_date=START_DATE, end_date=END_DATE)
    # Load the load data from LOADS_PATH (required)
    if LOADS_PATH is None:
        raise ValueError("LOADS_PATH must be set to a CSV file containing the true load (L_i) per hour. No fallback is allowed.")
    try:
        load_df = read_csv_time_window(
            LOADS_PATH,
            start_date=START_DATE,
            end_date=END_DATE,
            sep=';',
            decimal=',',
        )
        if load_df.empty:
            raise ValueError("Load CSV window produced no rows for the requested horizon.")
        load_value_col = load_df.columns[0]
        load_df[load_value_col] = pd.to_numeric(
            load_df[load_value_col].astype(str).str.replace(',', '.'),
            errors='raise',
        )
        # Reindex to coverage timestamps; will fail if any required timestamps are missing.
        load_series = load_df[load_value_col].reindex(coverage.index)
        if load_series.isna().any():
            missing = int(load_series.isna().sum())
            raise ValueError(f"Load data missing {missing} timestamps after horizon/index alignment.")
        loads = load_series.astype(float).values
    except Exception as e:
        raise RuntimeError(f"Failed to read or process load data from {LOADS_PATH}: {e}")
    # Reuse already-loaded spot prices and align to coverage timestamps.
    try:
        spot_series = price_df[price_value_col].reindex(coverage.index)
        if spot_series.isna().any():
            missing = int(spot_series.isna().sum())
            raise ValueError(f"Spot price data missing {missing} timestamps after horizon/index alignment.")
        spot_prices = spot_series.astype(float).values
    except Exception as e:
        raise RuntimeError(f"Failed to load spot prices from {PRICE_CSV_PATH}: {e}")
    forward_series = metadata.set_index("instrument_id").loc[subset_ids, "price"].astype(float)
    if forward_series is None or len(forward_series) == 0:
        raise RuntimeError("forward_prices is None or empty. Check your subset filter and metadata for matching instruments.")

    # Drop instruments without finite forward prices to avoid NaN objective values.
    valid_mask = np.isfinite(forward_series.values)
    if not valid_mask.all():
        invalid_ids = list(np.array(subset_ids)[~valid_mask])
        print(f"Warning: Dropping {len(invalid_ids)} instruments with invalid forward prices (NaN/inf).")
        print("Invalid instrument IDs:", invalid_ids)

        diagnostics_path = os.path.join(
            run_dir,
            f"filtered_metadata_{suffix}_forward_price_diagnostics.csv",
        )
        if os.path.exists(diagnostics_path):
            try:
                diagnostics_df = pd.read_csv(diagnostics_path)
                invalid_diag = diagnostics_df[diagnostics_df["instrument_id"].isin(invalid_ids)]
                if not invalid_diag.empty:
                    reason_counts = invalid_diag["reason"].value_counts(dropna=False)
                    print("Invalid forward-price reason breakdown:")
                    for reason, count in reason_counts.items():
                        print(f"  - {reason}: {int(count)}")
            except Exception as e:
                print(f"Warning: Could not read diagnostics file {diagnostics_path}: {e}")

        subset_ids = list(np.array(subset_ids)[valid_mask])
        forward_series = forward_series.iloc[valid_mask]
        if not subset_ids:
            raise RuntimeError("All selected instruments have invalid forward prices; cannot optimize.")

    # Keep all filtered instruments (no pruning of low-coverage or duplicate columns).
    subset_ids = [sid for sid in subset_ids if sid in coverage.columns and sid in forward_series.index]
    if not subset_ids:
        raise RuntimeError("No subset instruments remain after alignment with coverage and forward prices.")

    forward_series = forward_series.loc[subset_ids]
    forward_prices = forward_series.values
    instrument_coverage = coverage[subset_ids].to_numpy(dtype=float)
    n_hours, n_instruments = instrument_coverage.shape

    if SOLVE_MODE not in {"global", "monthly"}:
        raise ValueError(f"Invalid SOLVE_MODE='{SOLVE_MODE}'. Use 'global' or 'monthly'.")

    # --- SOLVE ---
    # Solve the optimization problem using scipy.optimize.minimize with all defined constraints and bounds.
    # Prints the optimal hedge ratios, objective value, and constraint violations at the solution.
    # If the optimization fails, prints the full result and traceback for debugging.
    if __name__ == "__main__":
        try:
            import time
            t0 = time.time()

            if SOLVE_MODE == "monthly":
                print("Running monthly chunk initialization...")
                a_init = monthly_chunk_initializer(
                    instrument_coverage,
                    loads,
                    spot_prices,
                    forward_prices,
                    coverage.index,
                )
                if POLISH_GLOBAL_AFTER_CHUNKS:
                    print("Running final global polish solve...")
                    result, mats, constraints, bounds, hedge_profile_func = solve_vectorized(
                        instrument_coverage,
                        loads,
                        spot_prices,
                        forward_prices,
                        a0=a_init,
                        label="global polish",
                    )
                else:
                    result, mats, constraints, bounds, hedge_profile_func = solve_vectorized(
                        instrument_coverage,
                        loads,
                        spot_prices,
                        forward_prices,
                        a0=a_init,
                        label="monthly-seeded global",
                    )
            else:
                result, mats, constraints, bounds, hedge_profile_func = solve_vectorized(
                    instrument_coverage,
                    loads,
                    spot_prices,
                    forward_prices,
                    a0=None,
                    label="global",
                )

            objective_coeff = mats["objective_coeff"]
            t1 = time.time()
            print(f"Optimizer finished. Elapsed: {t1-t0:.2f} seconds.")
            # Print the optimal hedge ratios found by the optimizer
            print("\nOptimal hedge ratios:", result.x)
            print("Success:", result.success)
            print("Message:", result.message)
            print("Objective value:", objective(result.x))
            print("\nConstraint values at solution:")
            for i, c in enumerate(constraints):
                val = c['fun'](result.x)
                if c['type'] == 'eq':
                    print(f"Constraint {i} (equality): value = {val}")
                else:
                    minval = np.min(val) if hasattr(val, '__len__') else val
                    print(f"Constraint {i} (inequality): min value = {minval}")
            if not result.success:
                print("Full optimization result:")
                print(result)

            print("Preparing to write results to Excel...")
            t2 = time.time()
            # --- WRITE RESULTS TO EXCEL IN CURRENT RUN FOLDER ---
            # Prepare result dict for writing
            # Also collect constraint values at solution for Excel output
            constraint_values = []
            for i, c in enumerate(constraints):
                val = c['fun'](result.x)
                if c['type'] == 'eq':
                    constraint_values.append((f"Constraint {i} (equality)", float(val) if np.isscalar(val) or (hasattr(val, 'shape') and val.shape == ()) else val))
                else:
                    minval = np.min(val) if hasattr(val, '__len__') else val
                    constraint_values.append((f"Constraint {i} (inequality)", float(minval)))

            result_dict = {
                "success": result.success,
                "message": result.message,
                "cost": objective(result.x),
                "a": result.x,
                "coverage": instrument_coverage,  # Pass coverage matrix for hedge volume calculation
                "constraint_values": constraint_values,
            }
            # Use instrument names from metadata if available
            instrument_names = list(metadata.set_index("instrument_id").loc[subset_ids].get("name", pd.Series(subset_ids)).values)
            # Use coverage index as hour_index if available
            hour_index = coverage.index if hasattr(coverage, "index") else None
            # Compute hedge profile using the local function
            hedge_profile = hedge_profile_func(result.x)
            output_path = write_cost_neutral_hedge_results(
                result=result_dict,
                instrument_names=instrument_names,
                loads=loads,
                spot_prices=spot_prices,
                hour_index=hour_index,
                hedge_profile=hedge_profile,
                forward_prices=forward_prices,
                output_dir=run_dir if run_dir is not None else None,  # Use run_dir if available
            )
            t3 = time.time()
            print(f"Results written to: {output_path}")
            print(f"Excel writing elapsed: {t3-t2:.2f} seconds.")
        except Exception as e:
            print(f"Optimization failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()


