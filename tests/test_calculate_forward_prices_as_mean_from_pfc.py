import sys
import sys
import os
import pytest
import pandas as pd
import numpy as np
import tempfile
# Ensure the parent directory is in sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.calculate_forward_prices_as_mean_from_pfc import calculate_forward_prices_for_coverage
# This test suite verifies the correctness of the forward price calculation utility.
def make_test_files(tmp_path):
    """
    Create small test files for coverage, metadata, and price data.
    Returns file paths for use in tests.
    """
    # Create a small coverage DataFrame (2 instruments, 4 hours) with timezone-aware datetime
    # Must match timezone of price data for index alignment
    coverage = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=4, freq='h', tz='Europe/Berlin'),
        'A': [1, 0, 1, 0],
        'B': [0, 1, 0, 1]
    })
    # Create a small metadata DataFrame
    metadata = pd.DataFrame({
        'instrument_id': ['A', 'B'],
        'price': [np.nan, np.nan]
    })
    # Create a small price DataFrame with timezone-aware datetimes
    price = pd.DataFrame({
        'datetime': pd.date_range('2024-01-01', periods=4, freq='h', tz='Europe/Berlin'),
        'price close': [10, 20, 30, 40]
    })
    # Write to files
    coverage_path = tmp_path / 'coverage.parquet'
    metadata_path = tmp_path / 'metadata.parquet'
    price_path = tmp_path / 'price.csv'
    output_metadata_path = tmp_path / 'metadata_out.parquet'
    coverage.to_parquet(coverage_path)
    metadata.to_parquet(metadata_path)
    price.to_csv(price_path, sep=';', index=False, decimal=',')
    return str(coverage_path), str(metadata_path), str(price_path), str(output_metadata_path)

def test_forward_price_basic(tmp_path):
    """
    Test normal forward price calculation for two instruments with alternating coverage.
    Instrument A: covered at 0 and 2 (prices 10, 30) => mean 20
    Instrument B: covered at 1 and 3 (prices 20, 40) => mean 30
    """
    coverage_path, metadata_path, price_path, output_metadata_path = make_test_files(tmp_path)
    calculate_forward_prices_for_coverage(
        coverage_path=coverage_path,
        metadata_path=metadata_path,
        price_csv_path=price_path,
        output_metadata_path=output_metadata_path,
        price_column='price close'
    )
    result = pd.read_parquet(output_metadata_path)
    assert np.isclose(result.loc[result['instrument_id']=='A', 'price'].iloc[0], 20)
    assert np.isclose(result.loc[result['instrument_id']=='B', 'price'].iloc[0], 30)

def test_forward_price_missing_coverage(tmp_path):
    """
    Test case where one instrument (B) has no coverage hours. Should result in NaN price.
    """
    coverage_path, metadata_path, price_path, output_metadata_path = make_test_files(tmp_path)
    # Remove coverage for B
    coverage = pd.read_parquet(coverage_path)
    coverage['B'] = 0
    coverage.to_parquet(coverage_path)
    calculate_forward_prices_for_coverage(
        coverage_path=coverage_path,
        metadata_path=metadata_path,
        price_csv_path=price_path,
        output_metadata_path=output_metadata_path,
        price_column='price close'
    )
    result = pd.read_parquet(output_metadata_path)
    assert np.isnan(result.loc[result['instrument_id']=='B', 'price'].iloc[0])

def test_forward_price_missing_instrument(tmp_path):
    """
    Test case where an instrument (C) is present in metadata but not in coverage. Should result in NaN price.
    """
    coverage_path, metadata_path, price_path, output_metadata_path = make_test_files(tmp_path)
    # Add instrument C to metadata (not in coverage)
    metadata = pd.read_parquet(metadata_path)
    metadata = pd.concat([metadata, pd.DataFrame({'instrument_id': ['C'], 'price': [np.nan]})], ignore_index=True)
    metadata.to_parquet(metadata_path)
    calculate_forward_prices_for_coverage(
        coverage_path=coverage_path,
        metadata_path=metadata_path,
        price_csv_path=price_path,
        output_metadata_path=output_metadata_path,
        price_column='price close'
    )
    result = pd.read_parquet(output_metadata_path)
    assert np.isnan(result.loc[result['instrument_id']=='C', 'price'].iloc[0])


def test_forward_price_timezone_and_horizon(tmp_path):
    """Supports timezone-aware datetime CSVs and optional time-horizon filtering."""
    coverage = pd.DataFrame({
        'datetime': pd.to_datetime([
            '2026-01-01 00:00:00+01:00',
            '2026-01-01 01:00:00+01:00',
            '2026-01-01 02:00:00+01:00',
            '2026-01-01 03:00:00+01:00',
        ]),
        'A': [1, 1, 0, 0],
    })
    metadata = pd.DataFrame({'instrument_id': ['A'], 'price': [np.nan]})
    price_csv = tmp_path / 'price_tz.csv'
    price_csv.write_text(
        "datetime;price close\n"
        "2026-01-01 00:00:00+01:00;10,0\n"
        "2026-01-01 01:00:00+01:00;20,0\n"
        "2026-01-01 02:00:00+01:00;30,0\n"
        "2026-01-01 03:00:00+01:00;40,0\n",
        encoding='utf-8',
    )

    coverage_path = tmp_path / 'coverage.parquet'
    metadata_path = tmp_path / 'metadata.parquet'
    output_metadata_path = tmp_path / 'metadata_out.parquet'
    coverage.to_parquet(coverage_path)
    metadata.to_parquet(metadata_path)

    calculate_forward_prices_for_coverage(
        coverage_path=str(coverage_path),
        metadata_path=str(metadata_path),
        price_csv_path=str(price_csv),
        output_metadata_path=str(output_metadata_path),
        price_column='price close',
        start_date='2026-01-01 01:00:00+01:00',
        end_date='2026-01-01 02:00:00+01:00',
    )

    result = pd.read_parquet(output_metadata_path)
    # After horizon filter, instrument A only covers 01:00 with price 20.0.
    assert np.isclose(result.loc[result['instrument_id'] == 'A', 'price'].iloc[0], 20.0)
