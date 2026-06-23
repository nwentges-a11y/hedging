
# ------------------------------------------------------------------------------
# Forward Price Calculation Utility
# ------------------------------------------------------------------------------
# This module provides a function to calculate the forward price for each hedge
# instrument as the mean of prices from a price CSV, but only for the hours where
# the instrument has coverage (coverage==1) in the coverage matrix.
#
# The result is written into the 'price' column of the instrument metadata file.
# ------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path
from utils.load_data import apply_time_horizon, ensure_datetime_index, read_csv_time_window

def calculate_forward_prices_for_coverage(
	coverage_path,
	metadata_path,
	price_csv_path,
	output_metadata_path,
	price_df=None,
	price_column=None,
	start_date=None,
	end_date=None,
):
	"""Calculate forward prices for each instrument within an optional time horizon.
	
	For each instrument, calculate mean price over covered hours (coverage==1)
	and write the result to metadata['price'] at output_metadata_path.

	Args:
		coverage_path (str): Path to the coverage matrix parquet file (datetime x instrument_id).
		metadata_path (str): Path to the instrument metadata parquet file.
		price_csv_path (str or None): Path to the price CSV file (used when price_df is None).
		output_metadata_path (str): Path to write the updated metadata parquet file.
		price_df (pd.DataFrame or None): Optional preloaded price data with DatetimeIndex.
		price_column (str or None): Name of the price column in the CSV (guessed if None).
		start_date (str or pd.Timestamp or None): Optional inclusive lower boundary.
		end_date (str or pd.Timestamp or None): Optional inclusive upper boundary.

	Returns:
		None. Writes the updated metadata parquet file with the 'price' column filled.
	"""
	try:
		# Load coverage matrix (datetime as index or column)
		coverage = pd.read_parquet(coverage_path)
		# Parse coverage datetime and apply optional time horizon.
		coverage = ensure_datetime_index(coverage, datetime_col="datetime")
		coverage = apply_time_horizon(coverage, start_date=start_date, end_date=end_date)
		# Load metadata
		metadata = pd.read_parquet(metadata_path)
		# Use preloaded price DataFrame when provided, otherwise read needed slice from CSV.
		if price_df is None:
			price_df = read_csv_time_window(
				price_csv_path,
				start_date=start_date,
				end_date=end_date,
				sep=';',
				decimal=',',
			)
		else:
			price_df = price_df.copy()
			if not isinstance(price_df.index, pd.DatetimeIndex):
				price_df = ensure_datetime_index(price_df)
			price_df = apply_time_horizon(price_df, start_date=start_date, end_date=end_date)
		if price_df.empty:
			raise ValueError("Price data is empty for the requested horizon.")
		# Always use the first column after 'datetime' as the price column
		if price_column is None:
			price_column = price_df.columns[0]
		# Calculate forward price for each instrument
		forward_prices = {}
		diagnostics = []
		for instrument_id in metadata['instrument_id']:
			# Skip if instrument not in coverage matrix
			if instrument_id not in coverage.columns:
				forward_prices[instrument_id] = np.nan
				diagnostics.append({
					"instrument_id": instrument_id,
					"reason": "missing_in_coverage",
					"covered_hours_count": 0,
					"matched_price_hours_count": 0,
					"non_null_price_count": 0,
					"forward_price": np.nan,
				})
				continue
			# Find all hours where this instrument has coverage (==1)
			covered_hours = coverage.index[coverage[instrument_id] == 1]
			if len(covered_hours) == 0:
				forward_prices[instrument_id] = np.nan
				diagnostics.append({
					"instrument_id": instrument_id,
					"reason": "no_coverage_hours",
					"covered_hours_count": 0,
					"matched_price_hours_count": 0,
					"non_null_price_count": 0,
					"forward_price": np.nan,
				})
				continue
			matched_hours_count = int(len(covered_hours.intersection(price_df.index)))
			if matched_hours_count == 0:
				forward_prices[instrument_id] = np.nan
				diagnostics.append({
					"instrument_id": instrument_id,
					"reason": "no_price_overlap",
					"covered_hours_count": int(len(covered_hours)),
					"matched_price_hours_count": 0,
					"non_null_price_count": 0,
					"forward_price": np.nan,
				})
				continue
			# Get prices for those hours
			reindexed_prices = price_df.reindex(covered_hours)[price_column].astype(float)
			non_null_price_count = int(reindexed_prices.notna().sum())
			prices = reindexed_prices.dropna()
			if prices.empty:
				forward_prices[instrument_id] = np.nan
				diagnostics.append({
					"instrument_id": instrument_id,
					"reason": "all_missing_prices",
					"covered_hours_count": int(len(covered_hours)),
					"matched_price_hours_count": matched_hours_count,
					"non_null_price_count": non_null_price_count,
					"forward_price": np.nan,
				})
				continue
			forward_price = float(prices.mean())
			forward_prices[instrument_id] = forward_price
			diagnostics.append({
				"instrument_id": instrument_id,
				"reason": "ok",
				"covered_hours_count": int(len(covered_hours)),
				"matched_price_hours_count": matched_hours_count,
				"non_null_price_count": non_null_price_count,
				"forward_price": forward_price,
			})
		# Write result into the existing 'price' column in metadata
		metadata['price'] = metadata['instrument_id'].map(forward_prices)
		metadata.to_parquet(output_metadata_path)

		# Persist diagnostics next to output metadata for traceability.
		diagnostics_df = pd.DataFrame(diagnostics)
		diagnostics_path = Path(output_metadata_path).with_name(
			f"{Path(output_metadata_path).stem}_forward_price_diagnostics.csv"
		)
		diagnostics_df.to_csv(diagnostics_path, index=False)
		print(f"Wrote metadata with updated price column to {output_metadata_path}")
		print(f"Wrote forward price diagnostics to {diagnostics_path}")
	except Exception as e:
		raise RuntimeError(f"Error in calculate_forward_prices_for_coverage: {e}")


# Example CLI usage (uncomment and update paths as needed):
# if __name__ == "__main__":
#     calculate_forward_prices_for_coverage(
#         coverage_path="utils/data/hedge_instruments_coverage.parquet",
#         metadata_path="utils/data/hedge_instruments_metadata.parquet",
#         price_csv_path="Data/current/26_pri_de_fut_front-year_clo_€-mwh_cet_h_f_202508010000.csv",
#         output_metadata_path="utils/data/hedge_instruments_metadata_with_forward.parquet"
#     )
