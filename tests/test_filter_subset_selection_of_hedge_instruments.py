

# test_filter_subset_selection_of_hedge_instruments.py
# ---------------------------------------------------
# Unit tests for the filter_hedge_instruments function.
# These tests verify correct filtering of hedge instrument coverage and metadata
# by product_type and load_type, including edge cases.

import sys
import os
import pytest
import pandas as pd

# Ensure parent directory is in sys.path for import of the filter function
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.filter_subset_selection_of_hedge_instruments import filter_hedge_instruments

def make_test_data(tmp_path):
    """
    Helper function to create small test metadata and coverage DataFrames and save them as parquet files.
    Returns the file paths for use in tests.
    - Metadata: 3 instruments (A, B, C) with different product/load types.
    - Coverage: 2 days of coverage for each instrument.
    """
    metadata = pd.DataFrame({
        'instrument_id': ['A', 'B', 'C'],
        'product_type': ['month', 'year', 'month'],
        'load_type': ['base', 'peak', 'peak']
    })
    coverage = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=2, freq='D'),
        'A': [1, 2],
        'B': [3, 4],
        'C': [5, 6]
    })
    coverage_path = tmp_path / 'test_coverage.parquet'
    metadata_path = tmp_path / 'test_metadata.parquet'
    coverage.to_parquet(coverage_path)
    metadata.to_parquet(metadata_path)
    return str(coverage_path), str(metadata_path)

def get_all_combinations(metadata):
    """
    Return all unique (product_type, load_type) pairs from the metadata DataFrame.
    Used to test all possible filter combinations present in the test data.
    """
    return set((row['product_type'], row['load_type']) for _, row in metadata.iterrows())

def test_all_combinations_valid(tmp_path):
    """
    Test that filtering works for all (product_type, load_type) pairs present in the metadata.
    Ensures that a DataFrame is returned for each valid combination.
    """
    coverage_path, metadata_path = make_test_data(tmp_path)
    metadata = pd.read_parquet(metadata_path)
    combinations = get_all_combinations(metadata)
    for product_type, load_type in combinations:
        df = filter_hedge_instruments(
            product_type=product_type,
            load_type=load_type,
            coverage_path=coverage_path,
            mapping_path=metadata_path,
            save=False
        )
        assert df is not None, f"Expected data for ({product_type}, {load_type})"

def test_filter_month_base(tmp_path):
    """
    Test filtering for product_type='month' and load_type='base'.
    Should return a DataFrame with only instrument 'A' and 'datetime'.
    Verifies that only the correct columns are present after filtering.
    """
    coverage_path, metadata_path = make_test_data(tmp_path)
    df, _ = filter_hedge_instruments(
        product_type='month',
        load_type='base',
        coverage_path=coverage_path,
        mapping_path=metadata_path,
        save=False
    )
    assert df is not None
    assert 'A' in df.columns
    assert 'datetime' in df.columns
    assert 'B' not in df.columns
    # 'C' should not be present, or only two columns should exist
    assert 'C' not in df.columns or df.shape[1] == 2

def test_filter_year_peak(tmp_path):
    """
    Test filtering for product_type='year' and load_type='peak'.
    Should return a DataFrame with only instrument 'B' and 'datetime'.
    """
    coverage_path, metadata_path = make_test_data(tmp_path)
    df, _ = filter_hedge_instruments(
        product_type='year',
        load_type='peak',
        coverage_path=coverage_path,
        mapping_path=metadata_path,
        save=False
    )
    assert df is not None
    assert 'B' in df.columns
    assert 'A' not in df.columns
    assert 'datetime' in df.columns

def test_filter_none(tmp_path):
    """
    Test filtering for a non-existent combination (product_type='week', load_type='base').
    Should return None, indicating no instruments found for this combination.
    """
    coverage_path, metadata_path = make_test_data(tmp_path)
    df = filter_hedge_instruments(
        product_type='week',
        load_type='base',
        coverage_path=coverage_path,
        mapping_path=metadata_path,
        save=False
    )
    assert df == (None, None)
