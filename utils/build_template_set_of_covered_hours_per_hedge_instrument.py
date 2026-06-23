# =============================================================================
# build_template_set_of_covered_hours_per_hedge_instrument.py
#
# Generates a coverage matrix for hedge instruments and their availability over a time horizon.
# Supports product types: day, week, month, quarter, year (with base and peak load variants).
#
# Outputs (to utils/data/):
#   - hedge_instruments_coverage.parquet: Hour-by-instrument coverage matrix (1=covered, 0=not covered)
#   - hedge_instruments_metadata.parquet: Instrument metadata (product_type, load_type, dates, etc.)
#   - hedge_instruments_coverage_and_metadata.xlsx: Excel workbook with coverage and metadata sheets
#
# Peak hours: Mon-Fri, 08:00-19:59 CET (instrument.load_type='peak')
# Base hours: All hours (instrument.load_type='base')
# =============================================================================

# --- Time horizon configuration (set your desired period here) ---
# Set the start and end of the time horizon for which to generate the instrument matrix
# All times are in Europe/Berlin timezone (CET/CEST, local time)
TIME_HORIZON_START = "2026-01-01 00:00"
TIME_HORIZON_END = "2031-12-31 23:59"

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
# Skip Excel export for very large matrices to avoid out-of-memory failures.
MAX_EXCEL_CELLS = 5_000_000


def generate_instrument_list(hour_index: pd.DatetimeIndex) -> list[dict]:
    """
    Generate a complete list of hedge instruments for all product types and periods.

    Args:
        hour_index: Europe/Berlin (CET/CEST) hourly timestamps (pd.DatetimeIndex).

    Returns:
        List of HedgeInstrument objects, each covering a specific period and load type:
            - product_type: day, saturday, sunday, weekend, week, month, quarter, year
            - load_type: 'base' (all hours) or 'peak' (Mon-Fri 08:00-19:59 CET)
            - start_date, end_date: Calendar date range (inclusive)
    """
    # Use the hourly index (already in Europe/Berlin) to get unique calendar dates
    dates = hour_index.normalize().tz_localize(None).unique().sort_values()
    min_date = dates.min().date()
    max_date = dates.max().date()

    instruments = []  # List to accumulate all generated HedgeInstrument objects

    def add_instrument(product_type: str, start: pd.Timestamp, end: pd.Timestamp):
        """
        Helper: Create both base and peak load variants of a hedge instrument.
        Assigns unique human-readable names and appends to instruments list.
        """
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


def build_instrument_coverage(
    hour_index: pd.DatetimeIndex,
    instruments: list,
) -> np.ndarray:
    """
    Build instrument coverage matrix: which hours are covered by which instruments.

    Args:
        hour_index: Europe/Berlin (CET/CEST) hourly timestamps.
        instruments: List of HedgeInstrument objects.

    Returns:
        coverage: np.ndarray of shape (n_hours, n_instruments), dtype float (0 or 1).
            coverage[i, j] = 1 if hour i is covered by instrument j, else 0.

    Notes:
        - All times are in Europe/Berlin (CET/CEST) local time.
        - Base instruments: coverage[i,j]=1 for all hours in [start_date, end_date].
        - Peak instruments: coverage[i,j]=1 only for Mon-Fri, 08:00-19:59 CET.
        - Peak instruments on weekends will have zero coverage.
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


def instrument_name(instr):
    """
    Generate a human-readable name for a hedge instrument based on product type and period.
    
    Examples:
        - Year: 'Cal26Base', 'Cal26Peak'
        - Quarter: 'Q1 26Base', 'Q2 26Peak'
        - Month: 'Jan26Base', 'Feb26Peak'
        - Week: 'Week01/26Base', 'Week52/26Peak'
        - Day: 'Mon05/06/26Base', 'Fri31/12/26Peak'
        - Weekend: 'WKND01/26Base', 'WKND52/26Peak'
    """
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
    

# =============================================================================
# MAIN: Generate instruments, build coverage matrix, and export results
# =============================================================================
if __name__ == "__main__":
    # Step 1: Create hourly timezone-aware DatetimeIndex for configured time horizon
    idx = pd.date_range(TIME_HORIZON_START, TIME_HORIZON_END, freq="h", tz="Europe/Berlin")
    print(f"Time horizon: {TIME_HORIZON_START} to {TIME_HORIZON_END}")
    print(f"Total hours: {len(idx)}")

    # Step 2: Generate all hedge instruments (day, week, month, quarter, year x base/peak)
    instruments = generate_instrument_list(idx)
    print(f"Total instruments: {len(instruments)}")
    print(f"First instrument: {instruments[0].name} ({instruments[0].product_type}/{instruments[0].load_type})")

    # Step 3: Build coverage matrix (n_hours x n_instruments)
    coverage = build_instrument_coverage(idx, instruments)
    print(f"Coverage matrix shape: {coverage.shape}")
    print(f"Total coverage cells: {coverage.size}, Non-zero cells: {np.count_nonzero(coverage)}")

    # Step 4: Prepare output directory
    output_dir = Path("utils/data")
    output_dir.mkdir(exist_ok=True)

    # Step 5: Create coverage DataFrame (columns=instrument_id, rows=hourly timestamps)
    coverage_df = pd.DataFrame(coverage, columns=[instr.instrument_id for instr in instruments])
    coverage_df.insert(0, "datetime", idx)

    parquet_path = output_dir / "hedge_instruments_coverage.parquet"
    mapping_parquet_path = output_dir / "hedge_instruments_metadata.parquet"
    excel_path = output_dir / "hedge_instruments_coverage_and_metadata.xlsx"

    # Step 6: Save coverage matrix to Parquet (preserves timezone for downstream workflows)
    coverage_df.to_parquet(parquet_path, index=False)
    print(f"✓ Coverage matrix saved: {parquet_path}")

    # Step 7: Create and save instrument metadata DataFrame
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
    print(f"✓ Instrument metadata saved: {mapping_parquet_path}")

    # Step 8: Save to Excel (if matrix size is manageable)
    # Note: Excel has memory limits; skip for very large matrices (e.g., 2020-2035 spanning 6+ years)
    excel_cells = int(coverage_df.shape[0]) * int(coverage_df.shape[1])
    if excel_cells <= MAX_EXCEL_CELLS:
        # Excel doesn't support timezone-aware datetimes, so create timezone-naive copy for export
        coverage_df_excel = coverage_df.copy()
        for col in coverage_df_excel.select_dtypes(include=["datetimetz"]).columns:
            coverage_df_excel[col] = coverage_df_excel[col].dt.tz_localize(None)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            coverage_df_excel.to_excel(writer, index=False, sheet_name="coverage_matrix")
            mapping_df.to_excel(writer, index=False, sheet_name="instrument_metadata")
        print(f"✓ Excel workbook saved: {excel_path}")
    else:
        print(
            f"⚠ Skipping Excel export: matrix too large ({excel_cells:,} cells > {MAX_EXCEL_CELLS:,} limit)"
        )

    print("\n✓ Build complete. Outputs in: utils/data/")
    
    # Run pytest on the hedge matrix tests
    print("\nRunning hedge instrument matrix tests (pytest)...")
    result = subprocess.run(["pytest", "tests/test_hedge_matrix.py"], capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("Some tests failed. See output above.")
    else:
        print("All tests passed successfully.")