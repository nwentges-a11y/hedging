# Hedging Project

# Hedging Project

Python tools for cost-neutral hedge optimization and analysis.

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

## Testing

Run all tests with pytest:
```bash
pytest
```

Or run a specific test file:
```bash
pytest tests/test_cost_neutral_hedge.py
```

## Data

The `Data/` directory contains market data files (CSV) used for hedging calculations. Place your input data in `Data/current/`.

## Requirements

All required third-party packages are listed in `requirements.txt` (pandas, numpy, scipy, pytest, openpyxl).

## License

[Add license information here]
