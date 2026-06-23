import os
import shutil
import tempfile
import pandas as pd
import pytest
from utils import load_data
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Test that load_my_data loads and returns a dictionary of DataFrames
def test_load_my_data():
    data = load_data.load_my_data()
    assert isinstance(data, dict)
    assert all(isinstance(df, pd.DataFrame) for df in data.values())
    # At least one CSV should be present for a meaningful test
    assert len(data) > 0

# Test that read_csv_general reads a CSV and handles missing files
def test_read_csv_general(tmp_path):
    # Create a sample CSV
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b\n1,2\n3,4", encoding="utf-8")
    df = load_data.read_csv_general(csv_path)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)
    # Test missing file
    with pytest.raises(FileNotFoundError):
        load_data.read_csv_general(tmp_path / "missing.csv")

# Test that export_data_to_excel writes DataFrames to an Excel file and verifies sheet names
def test_export_data_to_excel(tmp_path):
    # Prepare sample data
    df1 = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    df2 = pd.DataFrame({"a": [5, 6], "b": [7, 8]})
    data = {"sheet1": df1, "sheet2": df2}
    # Patch PROJECT_ROOT to tmp_path for isolated test
    orig_root = load_data.PROJECT_ROOT
    load_data.PROJECT_ROOT = tmp_path
    try:
        excel_path = load_data.export_data_to_excel(data, filename="test.xlsx")
        assert excel_path.exists()
        # Read back and check sheet names
        xls = pd.ExcelFile(excel_path)
        assert set(xls.sheet_names) == {"sheet1", "sheet2"}
    finally:
        load_data.PROJECT_ROOT = orig_root

# Test that load_my_data uses caching and does not reload data unnecessarily
def test_cache_behavior(monkeypatch):
    # Clear cache
    load_data._data_cache = None
    called = {}
    def fake_loader():
        called["count"] = called.get("count", 0) + 1
        return {"dummy": pd.DataFrame({"a": [1]})}
    monkeypatch.setattr(load_data, "load_current_data", fake_loader)
    d1 = load_data.load_my_data()
    d2 = load_data.load_my_data()
    # Compare dictionary keys
    assert d1.keys() == d2.keys()
    # Compare DataFrame values using pandas testing utility
    for k in d1:
        pd.testing.assert_frame_equal(d1[k], d2[k])
    assert called["count"] == 1


def test_parse_datetime_series_timezone_and_dayfirst():
    # Test ISO+timezone format (new CSV format)
    series_iso = pd.Series([
        "2026-01-01 00:00:00+01:00",
        "2026-01-01 01:00:00+01:00",
    ])
    parsed_iso = load_data.parse_datetime_series(series_iso)
    assert isinstance(parsed_iso, pd.Series)
    assert parsed_iso.notna().all()

    # Test dayfirst format (old CSV format)
    series_dayfirst = pd.Series([
        "01.01.2026 00:00",
        "02.01.2026 01:00",
    ])
    parsed_dayfirst = load_data.parse_datetime_series(series_dayfirst)
    assert isinstance(parsed_dayfirst, pd.Series)
    assert parsed_dayfirst.notna().all()


def test_ensure_datetime_index_and_horizon():
    df = pd.DataFrame(
        {
            "Datetime": [
                "2026-01-01 00:00:00+01:00",
                "2026-01-01 01:00:00+01:00",
                "2026-01-01 02:00:00+01:00",
            ],
            "value": [1.0, 2.0, 3.0],
        }
    )
    indexed = load_data.ensure_datetime_index(df)
    filtered = load_data.apply_time_horizon(
        indexed,
        start_date="2026-01-01 01:00:00+01:00",
        end_date="2026-01-01 02:00:00+01:00",
    )
    assert list(filtered["value"]) == [2.0, 3.0]
