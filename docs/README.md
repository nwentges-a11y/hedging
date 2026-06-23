# Hedging Project

Python tools for cost-neutral hedge optimization and analysis.

## Quick Start

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q -m "not slow"
python cost_neutral_hedge.py
```

## Setup

### Prerequisites
- Python 3.8 or newer

### Installation
1. Clone or copy the project folder to your computer.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On macOS/Linux:
   source .venv/bin/activate
   ```
3. Install all required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the main optimization script:
```bash
python cost_neutral_hedge.py
```

## Configuration

Most runtime settings are in `cost_neutral_hedge.py`:

- `SUBSET_FILTER`: instrument types/load types to optimize.
- `START_DATE` / `END_DATE`: optional inclusive optimization horizon.
- `SAVE_FILTERED_SUBSET`: whether to persist filtered coverage/metadata in `Data/runs/`.
- `ENFORCE_COVERAGE`, `HEDGE_RATIO_BOUNDS`, `MIN_HEDGE_RATIO`: model constraint toggles.
- `EPSILON`, `HEDGE_RATIO_LB`, `HEDGE_RATIO_UB`, `H_MIN`: model parameters.

Coverage file selection behavior:

- Base path: `utils/data/hedge_instruments_coverage.parquet`
- If `START_DATE` and `END_DATE` are in the same year and a yearly file exists
   (for example `hedge_instruments_coverage_2026.parquet`), that yearly file is used.
- Otherwise, it falls back to the base coverage parquet.

## Testing

Run all tests with pytest:
```bash
pytest
```

Or run a specific test file:
```bash
pytest tests/test_cost_neutral_hedge.py
```

### Fast vs Slow Tests

The default test run is optimized for fast feedback.

- Fast/default run (recommended during development):
   ```bash
   pytest -q
   ```

- Exclude slow tests explicitly:
   ```bash
   pytest -q -m "not slow"
   ```

- Run slow full-horizon tests (opt-in):
   - Windows PowerShell:
      ```powershell
      $env:RUN_SLOW_TESTS="1"
      pytest -q -m slow
      ```
   - macOS/Linux:
      ```bash
      RUN_SLOW_TESTS=1 pytest -q -m slow
      ```

## Data

The `Data/` directory contains market data used for hedging calculations.

Required runtime inputs:

- `Data/current/` must contain:
   - a load CSV whose filename contains `con`
   - a price CSV whose filename contains `pri`
   - When multiple files match, the newest file is selected automatically.
- `utils/data/` must contain:
   - `hedge_instruments_coverage.parquet` (or yearly variants)
   - `hedge_instruments_metadata.parquet`

CSV expectations:

- First column: datetime (ISO/day-first/timezone-aware formats supported).
- Value column: numeric (comma decimal separators are handled).

Generated outputs:

- `Data/runs/run_YYYYMMDD_HHMMSS/`
   - `filtered_coverage_*.parquet`
   - `filtered_metadata_*.parquet` (with forward prices)
   - optimization result Excel file (written by `write_cost_neutral_hedge_results`)

Optional utility script:

- `scripts/inspect_current_csv.py` to validate current CSV structure quickly.

## Troubleshooting

Common issues and fixes:

- `ModuleNotFoundError` (for example `scipy`, `openpyxl`):
   - ensure venv is active and run `pip install -r requirements.txt`
- Wrong Python interpreter / mixed environments:
   - run `python -c "import sys; print(sys.executable)"`
   - verify it points to your project venv (`...\Hedging\venv\Scripts\python.exe`)
- Missing input files:
   - ensure required parquet files exist in `utils/data/`
   - ensure `Data/current/` has `*con*.csv` and `*pri*.csv`
- Datetime alignment errors:
   - make sure requested `START_DATE`/`END_DATE` exists across coverage, load, and price data.

## Requirements

All required third-party packages are listed in `requirements.txt` (pandas, numpy, scipy, pytest, openpyxl).

## License

[Add license information here]
