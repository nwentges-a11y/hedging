
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
import os

def calculate_forward_prices_for_coverage(coverage_path, metadata_path, price_csv_path, output_metadata_path, price_column=None):
	"""
	For each instrument, calculate the mean price from the price CSV for the hours where the instrument has coverage (coverage==1).
	The result is written into the 'price' column of the metadata and saved to output_metadata_path.

	Args:
		coverage_path (str): Path to the coverage matrix parquet file (datetime x instrument_id).
		metadata_path (str): Path to the instrument metadata parquet file.
		price_csv_path (str): Path to the price CSV file (must have 'datetime' column).
		output_metadata_path (str): Path to write the updated metadata parquet file.
		price_column (str or None): Name of the price column in the CSV (guessed if None).

	Returns:
		None. Writes the updated metadata parquet file with the 'price' column filled.
	"""
	try:
		# Load coverage matrix (datetime as index or column)
		coverage = pd.read_parquet(coverage_path)
		# Load metadata
		metadata = pd.read_parquet(metadata_path)
		# Load price CSV (parse datetime, handle decimal comma)
		price_df = pd.read_csv(price_csv_path, sep=';', decimal=',')
		price_df['datetime'] = pd.to_datetime(price_df['datetime'], format='%d.%m.%Y %H:%M')
		price_df = price_df.set_index('datetime')
		# Always use the first column after 'datetime' as the price column
		if price_column is None:
			price_column = price_df.columns[0]
		# Ensure coverage datetime is datetime type
		if not np.issubdtype(coverage['datetime'].dtype, np.datetime64):
			coverage['datetime'] = pd.to_datetime(coverage['datetime'])
		coverage = coverage.set_index('datetime')
		# Calculate forward price for each instrument
		forward_prices = {}
		for instrument_id in metadata['instrument_id']:
			# Skip if instrument not in coverage matrix
			if instrument_id not in coverage.columns:
				forward_prices[instrument_id] = np.nan
				continue
			# Find all hours where this instrument has coverage (==1)
			covered_hours = coverage.index[coverage[instrument_id] == 1]
			if len(covered_hours) == 0:
				forward_prices[instrument_id] = np.nan
				continue
			# Get prices for those hours
			prices = price_df.loc[covered_hours, price_column].astype(float)
			forward_prices[instrument_id] = prices.mean()
		# Write result into the existing 'price' column in metadata
		metadata['price'] = metadata['instrument_id'].map(forward_prices)
		metadata.to_parquet(output_metadata_path)
		print(f"Wrote metadata with updated price column to {output_metadata_path}")
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
