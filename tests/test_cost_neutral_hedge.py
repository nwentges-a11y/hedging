# Ensure the project root is in sys.path for import, so pytest works from any directory
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import pandas as pd
import cost_neutral_hedge
from utils.write_excel import write_cost_neutral_hedge_results

def test_cost_neutral_hedge_runs(monkeypatch):
    """
    Basic test: Runs the main optimization workflow with monkeypatched data and checks for success.
    This test does not require real data files, but patches the necessary variables/functions.
    """
    # Patch the required data and parameters
    import cost_neutral_hedge as cnh
    n_hours = 10
    n_instruments = 3
    cnh.loads = np.ones(n_hours)
    cnh.spot_prices = np.ones(n_hours)
    cnh.instrument_coverage = np.ones((n_hours, n_instruments))
    cnh.forward_prices = np.ones(n_instruments)
    cnh.n_instruments = n_instruments
    cnh.constraints = []
    cnh.bounds = None

    # Simple cost-neutrality constraint for test
    def dummy_constraint(a):
        return np.sum(a) - 1
    cnh.constraints.append({'type': 'eq', 'fun': dummy_constraint})

    # Run the optimization
    from scipy.optimize import minimize
    a0 = np.full(n_instruments, 0.5)
    result = minimize(
        cnh.objective,
        a0,
        method='SLSQP',
        bounds=None,
        constraints=cnh.constraints,
        options={'disp': False}
    )
    assert result.success, f"Optimization failed: {result.message}"
    # Check that the constraint is satisfied within tolerance
    assert abs(dummy_constraint(result.x)) < 1e-6, "Constraint not satisfied"
    # Check that the solution is reasonable (all hedge ratios positive)
    assert np.all(result.x >= 0), "Negative hedge ratio in solution"


def test_cost_neutral_hedge_basic_run(monkeypatch, tmp_path):
    """Test that the main optimization runs and writes an Excel file."""
    # Patch output directory to a temp path (if needed)
    # Patch data to minimal valid arrays
    n = 3
    monkeypatch.setattr(cost_neutral_hedge, "n_instruments", n)
    monkeypatch.setattr(cost_neutral_hedge, "loads", np.ones(24))
    monkeypatch.setattr(cost_neutral_hedge, "spot_prices", np.ones(24) * 50)
    monkeypatch.setattr(cost_neutral_hedge, "instrument_coverage", np.ones((24, n)))
    monkeypatch.setattr(cost_neutral_hedge, "forward_prices", np.ones(n) * 55)
    monkeypatch.setattr(cost_neutral_hedge, "constraints", [])
    monkeypatch.setattr(cost_neutral_hedge, "bounds", None)
    cost_neutral_hedge.subset_ids = [f"inst_{i}" for i in range(n)]
    cost_neutral_hedge.coverage = pd.DataFrame(np.ones((24, n)))

    # Run optimization
    result = cost_neutral_hedge.minimize(
        cost_neutral_hedge.objective,
        np.full(n, 0.5),
        method='SLSQP',
        bounds=None,
        constraints=[],
        options={'disp': False}
    )
    # Write results
    out_path = tmp_path / "test_output.xlsx"
    hedge_profile = np.ones(24) * result.x.mean()
    write_cost_neutral_hedge_results(
        result={
            "a": result.x,
            "success": result.success,
            "message": result.message,
            "cost": result.fun,
        },
        instrument_names=cost_neutral_hedge.subset_ids,
        loads=cost_neutral_hedge.loads,
        spot_prices=cost_neutral_hedge.spot_prices,
        hour_index=None,
        hedge_profile=hedge_profile,
        output_dir=tmp_path,
        filename="test_output.xlsx"
    )
    assert out_path.exists(), "Excel output file was not created."
    # Check sheets
    xls = pd.ExcelFile(out_path)
    assert set(xls.sheet_names) == {"summary", "hedge_ratios", "hourly_profile"}


def test_cost_neutrality_constraint_satisfied(monkeypatch, tmp_path):
    """Test that the cost-neutrality constraint is satisfied (close to zero)."""
    n = 2
    loads = np.ones(10) * 100
    spot_prices = np.ones(10) * 50
    instrument_coverage = np.ones((10, n))
    forward_prices = np.ones(n) * 55
    monkeypatch.setattr(cost_neutral_hedge, "n_instruments", n)
    monkeypatch.setattr(cost_neutral_hedge, "loads", loads)
    monkeypatch.setattr(cost_neutral_hedge, "spot_prices", spot_prices)
    monkeypatch.setattr(cost_neutral_hedge, "instrument_coverage", instrument_coverage)
    monkeypatch.setattr(cost_neutral_hedge, "forward_prices", forward_prices)
    monkeypatch.setattr(cost_neutral_hedge, "constraints", [])
    monkeypatch.setattr(cost_neutral_hedge, "bounds", None)
    # Define local cost_neutrality function for this test
    def cost_neutrality(a):
        hedge_profile = instrument_coverage @ a
        return np.sum((loads - hedge_profile) * spot_prices)
    a0 = np.full(n, 0.5)
    result = cost_neutral_hedge.minimize(
        cost_neutral_hedge.objective,
        a0,
        method='SLSQP',
        bounds=None,
        constraints=[{'type': 'eq', 'fun': cost_neutrality}],
        options={'disp': False}
    )
    assert abs(cost_neutrality(result.x)) < 1e-6, "Cost-neutrality constraint not satisfied."
