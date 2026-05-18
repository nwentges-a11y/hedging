"""
Cost-Neutral Hedge Optimization Model
Implements the model from formulas.tex with optional constraints (III, IV, V),
all parameters settable at the top, and subset filtering using utils.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import minimize, Bounds, LinearConstraint
# Add Excel writing utility
from utils.write_excel import write_cost_neutral_hedge_results

# --- PARAMETERS & FLAGS (set here) ---
# Subset filter (set to e.g. {"product_type": "year", "load_type": "base"} or None for all)
# SUBSET_FILTER = [
#     {"product_type": "day", "load_type": "base"},
#     {"product_type": "day", "load_type": "peak"},
#     {"product_type": "weekend", "load_type": "base"},
# ]

# Use this SUBSET_FILTER to run the optimization for year base and year peak products only
# SUBSET_FILTER = [
SUBSET_FILTER = [     
    {"product_type": "year", "load_type": "base"},
    {"product_type": "year", "load_type": "peak"},
]

# Use this SUBSET_FILTER to run the optimization for quarter base and year peak products only
# SUBSET_FILTER = [
#     {"product_type": "quarter", "load_type": "base"},
#     {"product_type": "quarter", "load_type": "peak"},
# ]

# Use this SUBSET_FILTER to run the optimization for month base and year peak products only
# SUBSET_FILTER = [
#     {"product_type": "month", "load_type": "base"},
#     {"product_type": "month", "load_type": "peak"},
# ]



# Flag to save the filtered subset coverage and metadata for reference/debugging
SAVE_FILTERED_SUBSET = True

# Model flags
ENFORCE_COVERAGE = False # Constraint III
HEDGE_RATIO_BOUNDS = True # Constraint IV
MIN_HEDGE_RATIO = True    # Constraint V

# Model parameters
EPSILON = 0.3            # Tolerance band for constraint III
HEDGE_RATIO_LB = 0.0     # Lower bound for a^j (IV)
HEDGE_RATIO_UB = 1    # Upper bound for a^j (IV)
H_MIN = 0.95               # Minimum hedge ratio (V)

# Data paths
COVERAGE_PATH = "utils/data/hedge_instruments_coverage.parquet"
METADATA_PATH = "utils/data/hedge_instruments_metadata.parquet"

# Automatically select the current loads and price CSVs from Data/current/
from pathlib import Path
current_data_dir = Path("Data/current")
def find_csv_with_substring(substring):
    files = list(current_data_dir.glob("*.csv"))
    for f in files:
        if substring in f.name:
            return str(f)
    raise FileNotFoundError(f"No CSV file with '{substring}' in name found in {current_data_dir}")

try:
    LOADS_PATH = find_csv_with_substring("con")  # e.g., consumption/load file
    PRICE_CSV_PATH = find_csv_with_substring("pri")  # e.g., price file
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
    from utils.calculate_forward_prices_as_mean_from_pfc import calculate_forward_prices_for_coverage

    # Filter subset and get coverage/metadata
    try:
        filtered_coverage, run_dir = filter_hedge_instruments(
            subset_filter=SUBSET_FILTER,
            coverage_path=COVERAGE_PATH,
            mapping_path=METADATA_PATH,
            save=SAVE_FILTERED_SUBSET
        )
    except Exception as e:
        raise RuntimeError(f"Failed to filter hedge instruments: {e}")
    try:
        metadata = pd.read_parquet(METADATA_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to read metadata parquet file {METADATA_PATH}: {e}")
    # Support SUBSET_FILTER as a list of dicts for precise selection
    if isinstance(SUBSET_FILTER, list):
        mask = pd.Series([False] * len(metadata))
        for filt in SUBSET_FILTER:
            pt = filt["product_type"]
            lt = filt["load_type"]
            pt_mask = metadata["product_type"] == pt
            lt_mask = metadata["load_type"] == lt
            mask = mask | (pt_mask & lt_mask)
        subset_ids = metadata[mask]["instrument_id"].tolist()
    else:
        if isinstance(SUBSET_FILTER["product_type"], (list, tuple, set)):
            product_type_filter = metadata["product_type"].isin(SUBSET_FILTER["product_type"])
        else:
            product_type_filter = metadata["product_type"] == SUBSET_FILTER["product_type"]
        if isinstance(SUBSET_FILTER["load_type"], (list, tuple, set)):
            load_type_filter = metadata["load_type"].isin(SUBSET_FILTER["load_type"])
        else:
            load_type_filter = metadata["load_type"] == SUBSET_FILTER["load_type"]
        subset_ids = metadata[product_type_filter & load_type_filter]["instrument_id"].tolist()

    # Calculate forward prices for the subset (updates metadata)
    try:
        calculate_forward_prices_for_coverage(
            coverage_path=COVERAGE_PATH,
            metadata_path=METADATA_PATH,
            price_csv_path=PRICE_CSV_PATH,
            output_metadata_path=METADATA_PATH
        )
        metadata = pd.read_parquet(METADATA_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed to calculate or reload forward prices: {e}")

    # --- MODEL SETUP ---
    # Extract data arrays
    coverage = filtered_coverage.set_index("datetime")
    # Load the load data from LOADS_PATH (required)
    if LOADS_PATH is None:
        raise ValueError("LOADS_PATH must be set to a CSV file containing the true load (L_i) per hour. No fallback is allowed.")
    try:
        load_df = pd.read_csv(LOADS_PATH, delimiter=';')
        # Assign to columns by name before setting index (avoids Arrow dtype issues)
        load_df[load_df.columns[0]] = pd.to_datetime(load_df[load_df.columns[0]], dayfirst=True)
        load_df[load_df.columns[1]] = pd.to_numeric(load_df[load_df.columns[1]].astype(str).str.replace(',', '.'), errors='raise')
        load_df = load_df.set_index(load_df.columns[0])
        loads = load_df.iloc[:, 0].values
    except Exception as e:
        raise RuntimeError(f"Failed to read or process load data from {LOADS_PATH}: {e}")
    # Load spot prices from the price CSV (first column = datetime, second column = price)
    try:
        price_df = pd.read_csv(PRICE_CSV_PATH, delimiter=';')
        price_df[price_df.columns[0]] = pd.to_datetime(price_df[price_df.columns[0]], dayfirst=True)
        price_df[price_df.columns[1]] = pd.to_numeric(price_df[price_df.columns[1]].astype(str).str.replace(',', '.'), errors='raise')
        price_df = price_df.set_index(price_df.columns[0])
        spot_prices = price_df.iloc[:, 0].reindex(coverage.index).astype(float).values
    except Exception as e:
        raise RuntimeError(f"Failed to load spot prices from {PRICE_CSV_PATH}: {e}")
    forward_prices = metadata.set_index("instrument_id").loc[subset_ids, "price"].values
    if forward_prices is None or len(forward_prices) == 0:
        raise RuntimeError("forward_prices is None or empty. Check your subset filter and metadata for matching instruments.")
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
        avg_load_per_j = np.where(count_Ihj > 0, sum_h_Ihj_Lh / count_Ihj, 0.0)
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
        avg_load_per_j = np.where(count_Ihj > 0, sum_h_Ihj_Lh / count_Ihj, 0.0)
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
            avg_load_per_j = np.where(count_Ihj > 0, sum_h_Ihj_Lh / count_Ihj, 0.0)
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


