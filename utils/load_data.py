"""Load the data from the Data/current directory into a dictionary of DataFrames 
for calculation of hedge profiles, with caching to avoid redundant loading. 
Includes utility functions for reading CSVs with encoding fallback and exporting results to Excel."""


from __future__ import annotations  # For forward type references (Python 3.7+ compatibility)
from pathlib import Path            # Pathlib for filesystem path operations
import pandas as pd                # Pandas for data manipulation
from pandas.api.types import is_datetime64_any_dtype


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Project root
DEFAULT_CURRENT_DATA_DIR = PROJECT_ROOT / "Data" / "current"  # Default data dir
# Business-local timezone used when user passes naive start/end boundaries.
DEFAULT_BOUNDARY_TIMEZONE = "Europe/Berlin"


# Module-level cache for loaded data
_data_cache = None


# --- Datetime Parsing and Horizon Filtering Utilities ---
# These functions provide flexible, timezone-aware datetime handling to support
# multiple input formats (ISO, dayfirst, timezone-aware) and optional time-window filtering.
# Use these helpers in all data ingestion paths to ensure consistent parsing across the pipeline.

def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse a datetime series while supporting both ISO and day-first input formats.
    
    For each row, tries ISO format first, then dayfirst if ISO fails.
    Preserves timezone information if present in input.
    Raises ValueError if any rows fail to parse with both formats.
    """
    text_values = series.astype(str).str.strip()
    # Try ISO format first, then dayfirst per row to handle mixed-format input.
    # Build result as a list to avoid pandas tz-aware/tz-naive Series assignment errors.
    # Use utc=True to handle mixed timezones by converting everything to UTC.
    parsed_iso = pd.to_datetime(text_values, errors="coerce", utc=True)
    parsed_dayfirst = pd.to_datetime(text_values, errors="coerce", dayfirst=True, utc=True)
    result = [
        iso if not pd.isna(iso) else df
        for iso, df in zip(parsed_iso, parsed_dayfirst)
    ]
    parsed = pd.Series(result, index=series.index, dtype="object")
    parsed = pd.to_datetime(parsed, errors="coerce", utc=True)

    invalid_count = int(parsed.isna().sum())
    if invalid_count > 0:
        raise ValueError(f"Datetime parsing failed for {invalid_count} rows.")
    return parsed


def detect_datetime_column(df: pd.DataFrame) -> str:
    """Detect a likely datetime column name, falling back to the first column."""
    preferred_names = {"datetime", "timestamp", "date", "time"}
    for col in df.columns:
        if str(col).strip().lower() in preferred_names:
            return col
    return df.columns[0]


def ensure_datetime_index(df: pd.DataFrame, datetime_col: str | None = None) -> pd.DataFrame:
    """Return a copy of df with a parsed DatetimeIndex.

    If datetime_col is None, a likely datetime column is detected automatically.
    """
    out = df.copy()
    if datetime_col is None:
        if isinstance(out.index, pd.DatetimeIndex):
            return out
        datetime_col = detect_datetime_column(out)

    out[datetime_col] = parse_datetime_series(out[datetime_col])
    out = out.set_index(datetime_col)
    return out


def _coerce_boundary_for_index(boundary: str | pd.Timestamp | None, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Coerce start/end boundary to be comparable with a DatetimeIndex."""
    if boundary is None:
        return None

    ts = pd.Timestamp(boundary)
    index_tz = index.tz

    if index_tz is not None:
        if ts.tzinfo is None:
            # Interpret naive boundaries as business-local (Europe/Berlin), then align to index tz.
            ts = ts.tz_localize(DEFAULT_BOUNDARY_TIMEZONE).tz_convert(index_tz)
        else:
            ts = ts.tz_convert(index_tz)
    elif ts.tzinfo is not None:
        ts = ts.tz_localize(None)

    return ts


def apply_time_horizon(
    df: pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter a DataFrame with DatetimeIndex to an optional [start_date, end_date] horizon.
    
    Both boundaries are INCLUSIVE. Timezone mismatches between boundaries and index
    are automatically handled by coercing boundaries to match the index timezone.
    If start_date or end_date is None, that side of the range is unbounded.
    """
    if start_date is None and end_date is None:
        return df
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("apply_time_horizon expects a DataFrame with DatetimeIndex.")

    start_ts = _coerce_boundary_for_index(start_date, df.index)
    end_ts = _coerce_boundary_for_index(end_date, df.index)

    mask = pd.Series(True, index=df.index)
    if start_ts is not None:
        mask &= df.index >= start_ts
    if end_ts is not None:
        mask &= df.index <= end_ts
    return df.loc[mask]


def read_csv_time_window(
    file_path: str | Path,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    sep: str = ";",
    decimal: str = ",",
    chunksize: int = 250_000,
) -> pd.DataFrame:
    """Read only the needed datetime slice from a CSV using chunked scanning.

    Returns a DataFrame indexed by parsed datetime (timezone-aware, UTC-normalized)
    filtered to the inclusive [start_date, end_date] window when provided.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    filtered_chunks: list[pd.DataFrame] = []
    first_columns: list[str] | None = None
    datetime_col_name: str | None = None

    for chunk in pd.read_csv(path, sep=sep, decimal=decimal, chunksize=chunksize):
        if first_columns is None:
            first_columns = [str(c) for c in chunk.columns]
        datetime_col = detect_datetime_column(chunk)
        if datetime_col_name is None:
            datetime_col_name = str(datetime_col)

        chunk[datetime_col] = parse_datetime_series(chunk[datetime_col])
        chunk = chunk.set_index(datetime_col)
        chunk = apply_time_horizon(chunk, start_date=start_date, end_date=end_date)
        if not chunk.empty:
            filtered_chunks.append(chunk)

    if filtered_chunks:
        return pd.concat(filtered_chunks, axis=0).sort_index()

    non_dt_cols: list[str] = []
    if first_columns is not None:
        non_dt_cols = [c for c in first_columns if c != datetime_col_name]
    empty = pd.DataFrame(columns=non_dt_cols)
    empty.index = pd.DatetimeIndex([], name=datetime_col_name, tz="UTC")
    return empty


def find_latest_csv_with_substring(
    substring: str,
    data_dir: str | Path | None = None,
) -> str:
    """Return newest CSV path in data_dir whose filename contains substring.

    Matching is case-insensitive. Raises FileNotFoundError when no match exists.
    """
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_CURRENT_DATA_DIR
    files = sorted(base_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    matches = [f for f in files if substring.lower() in f.name.lower()]
    if not matches:
        raise FileNotFoundError(f"No CSV file with '{substring}' in name found in {base_dir}")
    if len(matches) > 1:
        print(f"Multiple CSV files matched '{substring}'. Using newest: {matches[0].name}")
    return str(matches[0])

# Load and cache the current data as a dictionary of DataFrames
def load_my_data():
    """Load and return the current data as a dictionary of DataFrames, using a cache to avoid redundant loading."""
    global _data_cache
    if _data_cache is not None:
        return _data_cache
    _data_cache = load_current_data()
    return _data_cache

# Load all CSV files from a directory into a dictionary of DataFrames
def load_current_data(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    """Load all CSV files from Data/current (or a given directory) into a dictionary of DataFrames. The key is the file stem (name without extension), the value is the DataFrame."""
    # Use provided directory or default
    base_dir = Path(data_dir) if data_dir is not None else DEFAULT_CURRENT_DATA_DIR
    if not base_dir.exists():
        # Raise error if directory does not exist
        raise FileNotFoundError(f"Data directory not found: {base_dir}")
    # Find all CSV files in the directory
    csv_files = sorted(base_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {base_dir}")
    data: dict[str, pd.DataFrame] = {}
    # Load each CSV into a DataFrame and store in dict
    for csv_file in csv_files:
        data[csv_file.stem] = read_csv_general(csv_file)
    return data

# Read a CSV file with delimiter detection and encoding fallbacks
def read_csv_general(
    file_path: str | Path,
    encoding: str = "utf-8",
    delimiter: str | None = None,
    on_bad_lines: str = "error",
    decimal: str = ",",
    thousands: str | None = None,
    quotechar: str = '"',
    escapechar: str | None = None,
) -> pd.DataFrame:
    """Read a CSV file with delimiter detection and encoding fallbacks. Tries multiple encodings if the default fails."""
    path = Path(file_path)
    if not path.exists():
        # Raise error if file does not exist
        raise FileNotFoundError(f"File not found: {path}")
    # Use tab if delimiter is set to 'tab', else use provided delimiter
    sep_value = "\t" if delimiter == "tab" else delimiter
    # Try several encodings in order
    fallback_encodings = [encoding, "utf-8-sig", "latin1", "cp1252"]
    encodings_to_try: list[str] = []
    for enc in fallback_encodings:
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)
    last_decode_error: UnicodeDecodeError | None = None
    # Try reading the file with each encoding
    for current_encoding in encodings_to_try:
        try:
            df = pd.read_csv(
                path,
                sep=sep_value,
                engine="python",
                encoding=current_encoding,
                on_bad_lines=on_bad_lines,
                decimal=decimal,
                thousands=thousands,
                quotechar=quotechar,
                escapechar=escapechar,
            )
            return df
        except UnicodeDecodeError as err:
            # Save the last decode error to raise if all fail
            last_decode_error = err
    # If all encodings fail, raise the last error
    assert last_decode_error is not None
    raise last_decode_error


# Export a dictionary of DataFrames to an Excel file in the test_outputs folder
def export_data_to_excel(data: dict[str, pd.DataFrame], filename: str = "exported_data.xlsx") -> Path:
    """Export a dictionary of DataFrames to an Excel file in the test_outputs folder. Each key in the dictionary becomes a sheet in the Excel file."""
    # Ensure output directory exists
    output_dir = PROJECT_ROOT / "test_outputs"
    output_dir.mkdir(exist_ok=True)
    # Set output file path
    output_path = output_dir / filename
    # Write each DataFrame to a separate sheet
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in data.items():
            safe_sheet_name = sheet_name[:31]  # Excel sheet names have a 31 character limit
            df.to_excel(writer, sheet_name=safe_sheet_name)
    return output_path

if __name__ == "__main__":
    data = load_my_data()
    print("Loaded keys:", list(data.keys()))
    # Temporary: export to Excel for testing
    export_path = export_data_to_excel(data, filename="test_export.xlsx")
    print(f"Exported to: {export_path}")
