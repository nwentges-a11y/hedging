"""
Cost-Neutral Hedge Optimization Model
Implements the model from formulas.tex with optional constraints (III, IV, V),
all parameters settable at the top, and subset filtering using utils.
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy.optimize import minimize, Bounds, LinearConstraint
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
constraints = []
bounds = None

# --- OBJECTIVE (I) ---
def objective(a):
    """
    Objective function: minimize total hedging cost for the selected instruments.
    a: array of hedge ratios (length n_instruments)
    Implements: sum_j sum_i P^j * a^j * indicator(i in I^j) * L_i
    """
    # instrument_coverage: shape (n_hours, n_instruments), loads: shape (n_hours,)
    # forward_prices: shape (n_instruments,)
    total_cost = 0.0
    for j in range(len(forward_prices)):
        total_cost += np.sum(forward_prices[j] * a[j] * instrument_coverage[:, j] * loads)
    return total_cost

def main():
    global forward_prices, instrument_coverage, loads
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

    forward_prices = forward_series.values
    instrument_coverage = coverage[subset_ids].values  # shape: (n_hours, n_instruments)

    n_hours, n_instruments = instrument_coverage.shape

    # --- CONSTRAINTS ---
    constraints = []

    # (II) Cost-neutrality constraint
    # Enforces that the total cost of the hedge matches the cost of the actual load profile,
    # using the spot price as the reference. This implements the LaTeX model constraint:
    #   sum_i [L_i - sum_j I_{i,j} a^j (sum_h I_{h,j} L_h / |I^j|)] * P_i^{spot} = 0
    # where:
    #   - L_i: actual load in hour i
    #   - I_{i,j}: indicator if instrument j covers hour i
    #   - a^j: hedge ratio for instrument j
    #   - (sum_h I_{h,j} L_h / |I^j|): average load covered by instrument j
    #   - P_i^{spot}: spot price in hour i
    # This ensures the hedge is cost-neutral with respect to the spot market.
    def cost_neutrality(a):
        # Implements: sum_i (L_i - sum_j I_{i,j} a^j (sum_h I_{h,j} L_h / |I^j|)) * P_i^{spot} = 0
        if spot_prices is None:
            return 0.0
        n_hours, n_instruments = instrument_coverage.shape
        # Precompute sum_h I_{h,j} L_h and |I^j| for each instrument j
        sum_h_Ihj_Lh = np.sum(instrument_coverage * loads[:, None], axis=0)  # shape: (n_instruments,)
        count_Ihj = np.sum(instrument_coverage, axis=0)  # shape: (n_instruments,)
        # Avoid division by zero
        avg_load_per_j = np.divide(
            sum_h_Ihj_Lh,
            count_Ihj,
            out=np.zeros_like(sum_h_Ihj_Lh, dtype=float),
            where=count_Ihj > 0,
        )
        # For each hour i, compute sum_j I_{i,j} a^j * avg_load_per_j[j]
        hedge_profile = np.zeros(n_hours)
        for j in range(n_instruments):
            hedge_profile += instrument_coverage[:, j] * a[j] * avg_load_per_j[j]
        return np.sum((loads - hedge_profile) * spot_prices)
    constraints.append({'type': 'eq', 'fun': cost_neutrality})

    def hedge_profile_latex(a):
        n_hours, n_instruments = instrument_coverage.shape
        sum_h_Ihj_Lh = np.sum(instrument_coverage * loads[:, None], axis=0)
        count_Ihj = np.sum(instrument_coverage, axis=0)
        avg_load_per_j = np.divide(
            sum_h_Ihj_Lh,
            count_Ihj,
            out=np.zeros_like(sum_h_Ihj_Lh, dtype=float),
            where=count_Ihj > 0,
        )
        profile = np.zeros(n_hours)
        for j in range(n_instruments):
            profile += instrument_coverage[:, j] * a[j] * avg_load_per_j[j]
        return profile

    # (III) Coverage/shape constraint (LaTeX model, optional)
    if ENFORCE_COVERAGE:
        # Implements: (1-epsilon_i)L_i <= sum_j I_{i,j} a^j (sum_h I_{h,j} L_h / |I^j|) <= (1+epsilon_i)L_i
        # This matches the LaTeX model for the coverage/shape constraint.
        def hedge_profile_latex(a):
            n_hours, n_instruments = instrument_coverage.shape
            # For each instrument j, compute the average load over all hours it covers
            sum_h_Ihj_Lh = np.sum(instrument_coverage * loads[:, None], axis=0)  # (n_instruments,)
            count_Ihj = np.sum(instrument_coverage, axis=0)  # (n_instruments,)
            avg_load_per_j = np.divide(
                sum_h_Ihj_Lh,
                count_Ihj,
                out=np.zeros_like(sum_h_Ihj_Lh, dtype=float),
                where=count_Ihj > 0,
            )
            # For each hour i, sum over all j: I_{i,j} * a^j * avg_load_per_j[j]
            profile = np.zeros(n_hours)
            for j in range(n_instruments):
                profile += instrument_coverage[:, j] * a[j] * avg_load_per_j[j]
            return profile
        def coverage_lower(a):
            # Lower bound: hedge profile >= (1-epsilon)*load
            profile = hedge_profile_latex(a)
            return profile - (1-EPSILON)*loads
        def coverage_upper(a):
            # Upper bound: hedge profile <= (1+epsilon)*load
            profile = hedge_profile_latex(a)
            return (1+EPSILON)*loads - profile
        constraints.append({'type': 'ineq', 'fun': coverage_lower})
        constraints.append({'type': 'ineq', 'fun': coverage_upper})

    # (IV) Hedge ratio bounds (optional)
    # Apply bounds only to hedge ratios a^j for instruments that actually cover at least one hour (|I^j| > 0).
    # For unused instruments (|I^j| == 0), set bounds to (0, 0) to fix their value at zero (or use (None, None) for unconstrained).
    bounds = None
    if HEDGE_RATIO_BOUNDS:
        count_Ihj = np.sum(instrument_coverage, axis=0)  # shape: (n_instruments,)
        lb = np.where(count_Ihj > 0, HEDGE_RATIO_LB, 0.0)  # Only bound active instruments, fix unused at 0
        ub = np.where(count_Ihj > 0, HEDGE_RATIO_UB, 0.0)
        bounds = Bounds(lb, ub)

    # (V) Minimum hedge ratio (optional)
    # Enforces that the total hedged volume (in MWh) is at least H_MIN times the total load volume.
    # Implements: sum_{i,j} I_{i,j} a^j L_i / sum_i L_i >= H_MIN
    if MIN_HEDGE_RATIO:
        def min_hedge_ratio(a):
            # Compute total hedged MWh: sum over all hours and instruments
            total_hedged = np.sum(instrument_coverage * a * loads[:, None])
            return total_hedged / np.sum(loads) - H_MIN
        constraints.append({'type': 'ineq', 'fun': min_hedge_ratio})

    # --- SOLVE ---
    # Solve the optimization problem using scipy.optimize.minimize with all defined constraints and bounds.
    # Prints the optimal hedge ratios, objective value, and constraint violations at the solution.
    # If the optimization fails, prints the full result and traceback for debugging.
    if __name__ == "__main__":
        try:
            # Set the initial guess for hedge ratios (a^j) to 0.5 for all instruments
            a0 = np.full(n_instruments, 0.5)
            # Print constraint values at initial guess
            print("\nConstraint values at initial guess (a0 = 0.5):")
            for i, c in enumerate(constraints):
                val = c['fun'](a0)
                if c['type'] == 'eq':
                    print(f"Constraint {i} (equality): value = {val}")
                else:
                    minval = np.min(val) if hasattr(val, '__len__') else val
                    print(f"Constraint {i} (inequality): min value = {minval}")

            import time
            print("Starting optimizer...")
            t0 = time.time()
            # Run the optimization using SLSQP with all constraints and bounds
            result = minimize(
                objective,           # Objective function to minimize
                a0,                  # Initial guess
                method='SLSQP',      # Sequential Least Squares Programming
                bounds=bounds,       # Bounds for hedge ratios
                constraints=constraints,  # List of constraints (dicts)
                options={'disp': True}    # Display solver output
            )
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
            hedge_profile = hedge_profile_latex(result.x)
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


