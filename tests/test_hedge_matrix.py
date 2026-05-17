import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from utils.build_template_set_of_covered_hours_per_hedge_instrument import generate_instrument_list, build_instrument_coverage, TIME_HORIZON_START, TIME_HORIZON_END

def get_test_index():
    """
    Generate a test hourly DatetimeIndex in Europe/Berlin for the configured time horizon.
    """
    return pd.date_range(TIME_HORIZON_START, TIME_HORIZON_END, freq="h", tz="Europe/Berlin")

def test_unique_instrument_names():
    """
    Test that all instrument names are unique in the generated instrument list.
    This prevents duplicate columns in the output matrix.
    """
    idx = get_test_index()
    instruments = generate_instrument_list(idx)
    names = [instr.name for instr in instruments]
    assert len(names) == len(set(names)), "Instrument names are not unique!"

def test_base_coverage_sum():
    """
    Test that each base product covers exactly the expected number of hours.
    The sum of coverage for each base instrument should match the total hours in its period.
    """
    idx = get_test_index()
    instruments = generate_instrument_list(idx)
    coverage = build_instrument_coverage(idx, instruments)
    for j, instr in enumerate(instruments):
        if instr.load_type == "base":
            start = pd.Timestamp(instr.start_date).replace(hour=0, minute=0, second=0, tzinfo=idx.tz)
            end = pd.Timestamp(instr.end_date).replace(hour=23, minute=0, second=0, tzinfo=idx.tz)
            expected_hours = int((end - start).total_seconds() // 3600) + 1
            actual_hours = int(coverage[:, j].sum())
            assert actual_hours == expected_hours, f"Base product {instr.name} covers {actual_hours} hours, expected {expected_hours}"

def test_peak_coverage_sum():
    """
    Test that each peak product covers the correct number of hours.
    Only Mon–Fri, 08:00–19:00 (inclusive) should be covered for each peak instrument.
    The sum of coverage should match the count of such hours in the product's period.
    """
    idx = get_test_index()
    instruments = generate_instrument_list(idx)
    coverage = build_instrument_coverage(idx, instruments)
    for j, instr in enumerate(instruments):
        if instr.load_type == "peak":
            start = pd.Timestamp(instr.start_date).replace(hour=0, minute=0, second=0, tzinfo=idx.tz)
            end = pd.Timestamp(instr.end_date).replace(hour=23, minute=0, second=0, tzinfo=idx.tz)
            hours = idx[(idx >= start) & (idx <= end)]
            expected_hours = sum((h.weekday() < 5 and 8 <= h.hour < 20) for h in hours)
            actual_hours = int(coverage[:, j].sum())
            assert actual_hours == expected_hours, f"Peak product {instr.name} covers {actual_hours} hours, expected {expected_hours}"

def test_coverage_start_end():
    """
    Test that the coverage for each instrument starts and ends at the correct datetime.
    The first and last covered hours should not be before/after the instrument's defined period.
    """
    idx = get_test_index()
    instruments = generate_instrument_list(idx)
    coverage = build_instrument_coverage(idx, instruments)
    for j, instr in enumerate(instruments):
        covered = np.where(coverage[:, j] == 1)[0]
        if len(covered) == 0:
            continue
        first = idx[covered[0]]
        last = idx[covered[-1]]
        start = pd.Timestamp(instr.start_date).replace(hour=0, minute=0, second=0, tzinfo=idx.tz)
        end = pd.Timestamp(instr.end_date).replace(hour=23, minute=0, second=0, tzinfo=idx.tz)
        assert first >= start, f"Coverage for {instr.name} starts at {first}, expected >= {start}"
        assert last <= end, f"Coverage for {instr.name} ends at {last}, expected <= {end}"
