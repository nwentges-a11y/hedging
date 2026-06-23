import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from utils.build_template_set_of_covered_hours_per_hedge_instrument import (
    generate_instrument_list,
    build_instrument_coverage,
    TIME_HORIZON_START as FULL_TIME_HORIZON_START,
    TIME_HORIZON_END as FULL_TIME_HORIZON_END,
)


# Keep default unit tests fast by using a short synthetic horizon.
FAST_TIME_HORIZON_START = "2026-01-01 00:00"
FAST_TIME_HORIZON_END = "2026-01-31 23:00"

@pytest.fixture(scope="session")
def test_index():
    """
    Generate the hourly DatetimeIndex once for the full test session.
    """
    return pd.date_range(FAST_TIME_HORIZON_START, FAST_TIME_HORIZON_END, freq="h", tz="Europe/Berlin")


@pytest.fixture(scope="session")
def test_instruments(test_index):
    """
    Build the instrument list once for the full test session.
    """
    return generate_instrument_list(test_index)


@pytest.fixture(scope="session")
def test_coverage(test_index, test_instruments):
    """
    Build the coverage matrix once for the full test session.
    """
    return build_instrument_coverage(test_index, test_instruments)


def test_unique_instrument_names(test_instruments):
    """
    Test that all instrument names are unique in the generated instrument list.
    This prevents duplicate columns in the output matrix.
    """
    names = [instr.name for instr in test_instruments]
    assert len(names) == len(set(names)), "Instrument names are not unique!"

def test_base_coverage_sum(test_index, test_instruments, test_coverage):
    """
    Test that each base product covers exactly the expected number of hours.
    The sum of coverage for each base instrument should match the total hours in its period.
    """
    idx = test_index
    for j, instr in enumerate(test_instruments):
        if instr.load_type == "base":
            start = pd.Timestamp(instr.start_date).replace(hour=0, minute=0, second=0, tzinfo=idx.tz)
            end = pd.Timestamp(instr.end_date).replace(hour=23, minute=0, second=0, tzinfo=idx.tz)
            # In fast tests, idx may cover only part of the instrument period.
            # Expected coverage is the overlap between instrument window and idx.
            expected_hours = int(((idx >= start) & (idx <= end)).sum())
            actual_hours = int(test_coverage[:, j].sum())
            assert actual_hours == expected_hours, f"Base product {instr.name} covers {actual_hours} hours, expected {expected_hours}"

def test_peak_coverage_sum(test_index, test_instruments, test_coverage):
    """
    Test that each peak product covers the correct number of hours.
    Only Mon–Fri, 08:00–19:00 (inclusive) should be covered for each peak instrument.
    The sum of coverage should match the count of such hours in the product's period.
    """
    idx = test_index
    for j, instr in enumerate(test_instruments):
        if instr.load_type == "peak":
            start = pd.Timestamp(instr.start_date).replace(hour=0, minute=0, second=0, tzinfo=idx.tz)
            end = pd.Timestamp(instr.end_date).replace(hour=23, minute=0, second=0, tzinfo=idx.tz)
            hours = idx[(idx >= start) & (idx <= end)]
            expected_hours = sum((h.weekday() < 5 and 8 <= h.hour < 20) for h in hours)
            actual_hours = int(test_coverage[:, j].sum())
            assert actual_hours == expected_hours, f"Peak product {instr.name} covers {actual_hours} hours, expected {expected_hours}"

def test_coverage_start_end(test_index, test_instruments, test_coverage):
    """
    Test that the coverage for each instrument starts and ends at the correct datetime.
    The first and last covered hours should not be before/after the instrument's defined period.
    """
    idx = test_index
    for j, instr in enumerate(test_instruments):
        covered = np.where(test_coverage[:, j] == 1)[0]
        if len(covered) == 0:
            continue
        first = idx[covered[0]]
        last = idx[covered[-1]]
        start = pd.Timestamp(instr.start_date).replace(hour=0, minute=0, second=0, tzinfo=idx.tz)
        end = pd.Timestamp(instr.end_date).replace(hour=23, minute=0, second=0, tzinfo=idx.tz)
        assert first >= start, f"Coverage for {instr.name} starts at {first}, expected >= {start}"
        assert last <= end, f"Coverage for {instr.name} ends at {last}, expected <= {end}"


@pytest.mark.slow
def test_full_horizon_build_smoke():
    """
    Optional full-horizon smoke test.
    Run only when RUN_SLOW_TESTS=1 to avoid slowing normal unit-test workflows.
    """
    if os.environ.get("RUN_SLOW_TESTS") != "1":
        pytest.skip("Set RUN_SLOW_TESTS=1 to run full-horizon hedge matrix smoke test.")

    idx = pd.date_range(FULL_TIME_HORIZON_START, FULL_TIME_HORIZON_END, freq="h", tz="Europe/Berlin")
    instruments = generate_instrument_list(idx)
    coverage = build_instrument_coverage(idx, instruments)

    assert coverage.shape == (len(idx), len(instruments))
    assert coverage.size > 0
