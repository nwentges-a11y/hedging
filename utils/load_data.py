"""Load the data from the Data/current directory into a dictionary of DataFrames 
for calculation of hedge profiles, with caching to avoid redundant loading. 
Includes utility functions for reading CSVs with encoding fallback and exporting results to Excel."""


from __future__ import annotations  # For forward type references (Python 3.7+ compatibility)
from pathlib import Path            # Pathlib for filesystem path operations
import pandas as pd                # Pandas for data manipulation


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Project root
DEFAULT_CURRENT_DATA_DIR = PROJECT_ROOT / "Data" / "current"  # Default data dir


# Module-level cache for loaded data
_data_cache = None

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
