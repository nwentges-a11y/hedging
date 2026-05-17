# =============================================================================
# build_hedge_instrument_matrix_model_input.py
#
# This script generates a matrix of hedge instruments (day, week, month, etc.)
# and their coverage over a specified time horizon. It outputs both a Parquet
# file and a multi-sheet Excel file:
#   - The first sheet contains the hour-by-instrument coverage matrix, with instrument_id as column headers.
#   - The second sheet contains a mapping table with instrument_id, name, and all instrument metadata.
# This structure supports robust downstream analysis and easy reference for both machines and humans.
# The script is designed for use in energy trading and risk management applications, supporting both base and peak load products.
# =============================================================================

# --- Time horizon configuration (set your desired period here) ---
# Set the start and end of the time horizon for which to generate the instrument matrix
# All times are in Europe/Berlin timezone (CET/CEST, local time)
TIME_HORIZON_START = "2026-01-01 00:00"
TIME_HORIZON_END = "2026-12-31 23:59"

# --- Required imports ---
import numpy as np  # For numerical operations and matrix handling
import pandas as pd  # For time series and DataFrame operations
from pathlib import Path  # For file and directory handling
from utils.hedge_instrument import HedgeInstrument
import openpyxl

# --- Peak hour constants ---
# Define the start and end hour for 'peak' load products (CET time)
PEAK_HOUR_START_CET = 8
PEAK_HOUR_END_CET = 20


# Generate a list of hedge instruments (day, week, month, etc.) for the given time horizon
def generate_instrument_list(hour_index: pd.DatetimeIndex) -> list[dict]:
    """
    Generate a complete list of hedge instruments for all product types and periods.
    """
    """
    Generate a complete list of instruments for all product types and periods.

    Args:
        hour_index: Europe/Berlin (CET/CEST) hourly timestamps, length n_hours.

    Returns:
        List of instrument dicts, each with keys:
            - 'product_type': str (day, saturday, sunday, weekend, week, month, quarter, year)
            - 'load_type': str ('base' or 'peak')
            - 'start': date object (first calendar day of period, inclusive)
            - 'end': date object (last calendar day of period, inclusive)
    """
    # Use the hourly index (already in Europe/Berlin) to get unique calendar dates
    dates = hour_index.normalize().tz_localize(None).unique().sort_values()
    min_date = dates.min().date()
    max_date = dates.max().date()

    instruments = []  # List to store all generated instruments

    # Helper function: add both base and peak variants for a given product type and period as HedgeInstrument objects
    def add_instrument(product_type: str, start: pd.Timestamp, end: pd.Timestamp):
        start_date = start.date() if isinstance(start, pd.Timestamp) else start
        end_date = end.date() if isinstance(end, pd.Timestamp) else end
        for load_type in ["base", "peak"]:
            # Prepare a dict for naming (instrument_name expects a dict)
            name_dict = {
                "product_type": product_type,
                "load_type": load_type,
                "start": start_date,
                "end": end_date,
            }
            name = instrument_name(name_dict)
            # Ensure unique name by appending a counter if needed
            base_name = name
            name_counts = getattr(add_instrument, "name_counts", None)
            if name_counts is None:
                name_counts = {}
                setattr(add_instrument, "name_counts", name_counts)
            if name in name_counts:
                name_counts[name] += 1
                name = f"{base_name}_{name_counts[base_name]}"
            else:
                name_counts[name] = 1
            # Create HedgeInstrument instance with sensible defaults
            instrument = HedgeInstrument(
                name=name,
                instrument_type="forward",
                product_type=product_type,
                load_type=load_type,
                start_date=start_date,
                end_date=end_date
                # price left as None (unknown)
            )
            instruments.append(instrument)

    # --- Generate instruments for each product type ---
    # Day products: each calendar day except Sat/Sun
    current_date = min_date
    while current_date <= max_date:
        if current_date.weekday() < 5:  # Mon-Fri
            add_instrument("day", current_date, current_date)
        current_date += pd.Timedelta(days=1)

    # Saturday products: each Saturday
    current_date = min_date
    while current_date <= max_date:
        if current_date.weekday() == 5:  # Saturday
            add_instrument("saturday", current_date, current_date)
        current_date += pd.Timedelta(days=1)

    # Sunday products: each Sunday
    current_date = min_date
    while current_date <= max_date:
        if current_date.weekday() == 6:  # Sunday
            add_instrument("sunday", current_date, current_date)
        current_date += pd.Timedelta(days=1)

    # Weekend products: (Sat + Sun) of each week
    current_date = min_date
    processed_weekends = set()
    while current_date <= max_date:
        if current_date.weekday() == 5:  # Saturday
            # Weekend starts on Saturday, ends on Sunday
            weekend_end = current_date + pd.Timedelta(days=1)
            if weekend_end <= max_date and (current_date, weekend_end) not in processed_weekends:
                add_instrument("weekend", current_date, weekend_end)
                processed_weekends.add((current_date, weekend_end))
        current_date += pd.Timedelta(days=1)

    # Week products: each ISO week (Mon-Sun)
    current_date = min_date
    processed_weeks = set()
    while current_date <= max_date:
        # Get ISO week start (Monday of that week)
        week_start = current_date - pd.Timedelta(days=current_date.weekday())
        week_end = week_start + pd.Timedelta(days=6)  # Sunday
        # Clamp to available data range
        week_start = max(week_start, min_date)
        week_end = min(week_end, max_date)
        week_key = (week_start, week_end)
        if week_key not in processed_weeks:
            add_instrument("week", week_start, week_end)
            processed_weeks.add(week_key)
        current_date += pd.Timedelta(days=1)

    # Month products: each calendar month
    current_date = min_date
    processed_months = set()
    while current_date <= max_date:
        month_start = current_date.replace(day=1)
        # Last day of month: add 1 month, subtract 1 day
        if current_date.month == 12:
            month_end = (current_date + pd.DateOffset(years=1)).replace(month=1, day=1) - pd.Timedelta(days=1)
        else:
            month_end = (current_date + pd.DateOffset(months=1)).replace(day=1) - pd.Timedelta(days=1)
        month_key = (month_start, month_end)
        if month_key not in processed_months:
            add_instrument("month", month_start, month_end)
            processed_months.add(month_key)
        current_date += pd.Timedelta(days=32)  # Skip to next month

    # Quarter products: Q1, Q2, Q3, Q4 for each year
    current_date = min_date
    processed_quarters = set()
    while current_date <= max_date:
        year = current_date.year
        quarter = (current_date.month - 1) // 3 + 1
        quarter_start_month = (quarter - 1) * 3 + 1
        quarter_start = pd.Timestamp(year=year, month=quarter_start_month, day=1).date()
        if quarter == 4:
            quarter_end_month = 12
        else:
            quarter_end_month = quarter_start_month + 2
        # Last day of quarter
        if quarter == 4:
            quarter_end = pd.Timestamp(year=year, month=12, day=31).date()
        else:
            quarter_end = (pd.Timestamp(year=year, month=quarter_end_month + 1, day=1) - pd.Timedelta(days=1)).date()
        quarter_key = (quarter_start, quarter_end)
        if quarter_key not in processed_quarters:
            add_instrument("quarter", quarter_start, quarter_end)
            processed_quarters.add(quarter_key)
        current_date += pd.Timedelta(days=90)  # Skip ~3 months

    # Year products: each calendar year
    current_date = min_date
    processed_years = set()
    while current_date <= max_date:
        year = current_date.year
        year_start = pd.Timestamp(year=year, month=1, day=1).date()
        year_end = pd.Timestamp(year=year, month=12, day=31).date()
        year_key = (year_start, year_end)
        if year_key not in processed_years:
            add_instrument("year", year_start, year_end)
            processed_years.add(year_key)
        current_date += pd.Timedelta(days=365)

    return instruments  # List of HedgeInstrument objects


 # Build a matrix indicating which hours are covered by which instruments
def build_instrument_coverage(
    hour_index: pd.DatetimeIndex,
    instruments: list,
) -> np.ndarray:
    """
    Build instrument coverage matrix I_{i,j}.

    Args:
        hour_index: Europe/Berlin (CET/CEST) hourly timestamps, length n_hours.
        instruments: List of HedgeInstrument objects, each with attributes like product_type, load_type, start_date, end_date, etc.

    Returns:
        coverage: np.ndarray of shape (n_hours, n_instruments), dtype float.
            coverage[i, j] = 1 if hour i is covered by instrument j, else 0.

        Notes:
                - All hour comparisons are done in Europe/Berlin (CET/CEST, local time).
                - Weekends never have peak hours per exchange definition, so a peak instrument
                    whose period falls entirely on weekends will have zero coverage.
    """
    # The hourly index is already in Europe/Berlin (CET/CEST)
    n_hours = len(hour_index)
    coverage = np.zeros((n_hours, len(instruments)), dtype=float)

    for j, instr in enumerate(instruments):
        # Get the start and end datetime for the instrument in Europe/Berlin
        start_cet = pd.Timestamp(instr.start_date).replace(hour=0, minute=0, second=0, tzinfo=hour_index.tz)
        end_cet = pd.Timestamp(instr.end_date).replace(hour=23, minute=0, second=0, tzinfo=hour_index.tz)
        mask = (hour_index >= start_cet) & (hour_index <= end_cet)

        if instr.load_type == "base":
            coverage[:, j] = mask.astype(float)
        else:
            peak_mask = (
                mask
                & (hour_index.weekday < 5)
                & (hour_index.hour >= PEAK_HOUR_START_CET)
                & (hour_index.hour < PEAK_HOUR_END_CET)
            )
            coverage[:, j] = peak_mask.astype(float)

    return coverage


# New naming convention for instruments
# Create a readable name for a hedge instrument based on its type and period
def instrument_name(instr):
    """
    Generate a human-readable name for a hedge instrument based on its type and period.
    """
    # Generate a human-readable name for a hedge instrument based on its type and period
    pt = instr['product_type']
    lt = instr['load_type']
    start = instr['start']
    end = instr['end']
    # Year product
    if pt == 'year':
        year = str(start)[-2:]
        return f"Cal{year}{'Base' if lt == 'base' else 'Peak'}"
    # Quarter product
    elif pt == 'quarter':
        year = str(start)[-2:]
        quarter = ((int(str(start)[5:7]) - 1) // 3) + 1
        return f"Q{quarter}{year}{'Base' if lt == 'base' else 'Peak'}"
    # Month product
    elif pt == 'month':
        dt = pd.Timestamp(start)
        month = dt.strftime('%b')
        year = dt.strftime('%y')
        return f"{month}{year}{'Base' if lt == 'base' else 'Peak'}"
    # Week product
    elif pt == 'week':
        dt = pd.Timestamp(start)
        week = dt.isocalendar().week
        year = dt.strftime('%y')
        return f"Week{week:02d}/{year}{'Base' if lt == 'base' else 'Peak'}"
    # Day, Saturday, Sunday products
    elif pt in ['day', 'saturday', 'sunday']:
        dt = pd.Timestamp(start)
        dow = dt.strftime('%a')
        day = dt.strftime('%d')
        month = dt.strftime('%m')
        year = dt.strftime('%y')
        return f"{dow}{day}/{month}/{year}{'Base' if lt == 'base' else 'Peak'}"
    # Weekend product
    elif pt == 'weekend':
        dt = pd.Timestamp(start)
        week = dt.isocalendar().week
        year = dt.strftime('%y')
        return f"WKND{week:02d}/{year}{'Base' if lt == 'base' else 'Peak'}"
    else:
        return f"{pt.capitalize()} {lt.capitalize()}"
    

# --- Main block for testing and Excel export ---
if __name__ == "__main__":
    # Generate a DatetimeIndex for the configured time horizon (hourly, Europe/Berlin local time)
    idx = pd.date_range(TIME_HORIZON_START, TIME_HORIZON_END, freq="h", tz="Europe/Berlin")

    # Generate the list of hedge instruments for the time horizon
    instruments = generate_instrument_list(idx)
    print("Number of instruments:", len(instruments))
    print("First instrument:", instruments[0])

    # Build the coverage matrix (hours x instruments)
    coverage = build_instrument_coverage(idx, instruments)
    print("Coverage shape:", coverage.shape)

    # Export both Parquet and Excel wide-format matrix (datetime as first column, one column per instrument)
    output_dir = Path("utils/data")
    output_dir.mkdir(exist_ok=True)
    # The coverage matrix columns are instrument_id for unique, machine-readable reference
    coverage_df = pd.DataFrame(coverage, columns=[instr.instrument_id for instr in instruments])
    coverage_df.insert(0, "datetime", idx)

    # Remove timezone info for both exports
    for col in coverage_df.select_dtypes(include=["datetimetz"]).columns:
        coverage_df[col] = coverage_df[col].dt.tz_localize(None)


    parquet_path = output_dir / "hedge_instruments_coverage.parquet"
    mapping_parquet_path = output_dir / "hedge_instruments_metadata.parquet"
    excel_path = output_dir / "hedge_instruments_coverage_and_metadata.xlsx"
    coverage_df.to_parquet(parquet_path, index=False)

    # Create mapping DataFrame for instrument metadata (for reference and joining)
    mapping_df = pd.DataFrame([
        {
            "instrument_id": instr.instrument_id,
            "name": instr.name,
            "instrument_type": instr.instrument_type,
            "product_type": instr.product_type,
            "load_type": instr.load_type,
            "start_date": instr.start_date,
            "end_date": instr.end_date,
            "region": instr.region,
            "market": instr.market,
            "currency": instr.currency,
            "underlying": instr.underlying,
            "volume": instr.volume,
            "price": instr.price
        }
        for instr in instruments
    ])

    mapping_df.to_parquet(mapping_parquet_path, index=False)

    # Write both DataFrames to Excel as separate sheets: coverage_matrix and instrument_metadata
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        coverage_df.to_excel(writer, index=False, sheet_name="coverage_matrix")
        mapping_df.to_excel(writer, index=False, sheet_name="instrument_metadata")

    print(f"Exported wide-format matrix to: {parquet_path}")
    print(f"Exported instrument metadata to: {mapping_parquet_path}")
    print(f"Exported wide-format matrix and mapping to: {excel_path}")

    # --- Automatically run only the hedge matrix test after building the matrix ---
    import subprocess
    print("\nRunning hedge instrument matrix tests (pytest)...")
    result = subprocess.run(["pytest", "tests/test_hedge_matrix.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Some tests failed. See output above.")
    else:
        print("All tests passed successfully.")