# Filtering and Subsetting Hedge Instruments
# ------------------------------------------
# This module provides functions to filter a hedge instrument coverage matrix and metadata
# by product_type and load_type, and to save the filtered results for further analysis or reporting.
#
# Features:
# - Supports both single and interactive batch filtering modes.
# - Saves filtered coverage and metadata as Parquet and Excel files in organized run directories.
# - Integrates forward price calculation for each subset, updating the price column in the filtered metadata.
#
# Typical usage: import and call filter_hedge_instruments() from a workflow script or CLI.



import pandas as pd
import argparse
import os
from datetime import datetime
# Import the forward price calculation utility
from utils.calculate_forward_prices_as_mean_from_pfc import calculate_forward_prices_for_coverage
from utils.load_data import find_latest_csv_with_substring

# Keep Excel export code available, but disable it by default for large runs.
SAVE_EXCEL = False

def load_coverage_and_metadata(coverage_path, mapping_path=None):
	"""
	Load the coverage matrix and metadata for hedge instruments from Parquet files.
	Returns:
		coverage_df (pd.DataFrame): The coverage matrix (datetime x instrument_id).
		metadata_df (pd.DataFrame): The instrument metadata.
	"""
	try:
		coverage_df = pd.read_parquet(coverage_path)
		if mapping_path and os.path.exists(mapping_path):
			metadata_df = pd.read_parquet(mapping_path)
		else:
			raise FileNotFoundError("No metadata file found. Please provide a metadata parquet file.")
		return coverage_df, metadata_df
	except Exception as e:
		raise RuntimeError(f"Failed to load coverage or metadata: {e}")

def filter_hedge_instruments(subset_filter, coverage_path='hedge_instruments_coverage.parquet', mapping_path=None, save=True, run_dir=None, price_csv_path=None, price_df=None):
	"""
	Filter hedge instruments by product_type and load_type, and save the filtered results.
	Steps:
	  1. Loads coverage and metadata.
	  2. Filters metadata for the given product_type and load_type.
	  3. Selects columns in coverage corresponding to the filtered instruments (plus 'datetime' if present).
	  4. Optionally saves filtered coverage and metadata to Parquet and Excel in a run directory.
	  5. Calculates and updates the price column in the filtered metadata using forward price calculation for the subset.

	Args:
		subset_filter (dict or list[dict]):
			Either a single dict like {"product_type": "month", "load_type": "base"}
			or a list of such dicts for multiple selections.
		coverage_path (str): Path to the coverage matrix file (Parquet or Excel).
		mapping_path (str): Path to the metadata file (Parquet or Excel).
		save (bool): Whether to save the filtered results to disk.
		run_dir (str or None): Directory to save the run output (auto-generated if None).
		price_csv_path (str or None): Optional explicit price CSV path. If None, auto-detect from Data/current by substring 'pri'.
		price_df (pd.DataFrame or None): Optional preloaded price data with DatetimeIndex.

	Returns:
		filtered_coverage (pd.DataFrame or None): The filtered coverage matrix, or None if no instruments found.
	"""
	try:
		# Load metadata first so parquet coverage can be read with column projection.
		# This avoids materializing the full wide coverage matrix when only a subset is needed.
		if mapping_path and os.path.exists(mapping_path):
			metadata_df = pd.read_parquet(mapping_path)
		else:
			raise FileNotFoundError("No metadata file found. Please provide a metadata parquet file.")
		# Support both a list of dicts (precise selection) or a dict (legacy)
		if isinstance(subset_filter, list):
			mask = pd.Series([False] * len(metadata_df))
			for filt in subset_filter:
				pt = filt["product_type"]
				lt = filt["load_type"]
				pt_mask = metadata_df["product_type"] == pt
				lt_mask = metadata_df["load_type"] == lt
				mask = mask | (pt_mask & lt_mask)
			filtered_meta = metadata_df[mask]
			suffix = "_and".join([f"{filt['product_type']}_{filt['load_type']}" for filt in subset_filter])
		else:
			if isinstance(subset_filter["product_type"], (list, tuple, set)):
				product_type_filter = metadata_df["product_type"].isin(subset_filter["product_type"])
			else:
				product_type_filter = metadata_df["product_type"] == subset_filter["product_type"]
			if isinstance(subset_filter["load_type"], (list, tuple, set)):
				load_type_filter = metadata_df["load_type"].isin(subset_filter["load_type"])
			else:
				load_type_filter = metadata_df["load_type"] == subset_filter["load_type"]
			filtered_meta = metadata_df[product_type_filter & load_type_filter]
			suffix = f"{subset_filter['product_type']}_{subset_filter['load_type']}"
		selected_ids = filtered_meta['instrument_id'].tolist()
		if not selected_ids:
			print(f"No instruments found for subset_filter={subset_filter}.")
			return None, None
		# Read only the required columns from parquet.
		parquet_cols = ['datetime'] + selected_ids
		filtered_coverage = pd.read_parquet(coverage_path, columns=parquet_cols)
		print(f"Filtered coverage matrix shape: {filtered_coverage.shape}")
		actual_run_dir = None
		if save:
			if run_dir is None:
				timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
				run_dir = os.path.join('Data', 'runs', f'run_{timestamp}')
			os.makedirs(run_dir, exist_ok=True)
			actual_run_dir = run_dir
			out_coverage_parquet = os.path.join(run_dir, f'filtered_coverage_{suffix}.parquet')
			out_metadata_parquet = os.path.join(run_dir, f'filtered_metadata_{suffix}.parquet')
			out_excel = os.path.join(run_dir, f'filtered_coverage_and_metadata_{suffix}.xlsx')
			filtered_coverage.to_parquet(out_coverage_parquet)
			filtered_meta.to_parquet(out_metadata_parquet)
			print(f"Filtered coverage saved to {out_coverage_parquet}")
			print(f"Filtered metadata saved to {out_metadata_parquet}")

			# --- Integrate forward price calculation for this subset ---
			resolved_price_csv_path = price_csv_path
			if price_df is None:
				resolved_price_csv_path = resolved_price_csv_path or find_latest_csv_with_substring("pri", data_dir="Data/current")
				print(f"Using price CSV: {resolved_price_csv_path}")
			else:
				print("Using preloaded price data for forward price calculation")
			calculate_forward_prices_for_coverage(
				coverage_path=out_coverage_parquet,
				metadata_path=out_metadata_parquet,
				price_csv_path=resolved_price_csv_path,
				price_df=price_df,
				output_metadata_path=out_metadata_parquet
			)

			if SAVE_EXCEL:
				# Reload the updated metadata (with prices) for Excel export
				updated_meta = pd.read_parquet(out_metadata_parquet)
				# Remove timezone from filtered_coverage for Excel export (Excel doesn't support timezones)
				filtered_coverage_for_excel = filtered_coverage.copy()
				if isinstance(filtered_coverage_for_excel.index, pd.DatetimeIndex) and filtered_coverage_for_excel.index.tz is not None:
					filtered_coverage_for_excel.index = filtered_coverage_for_excel.index.tz_localize(None)
				# Also remove timezones from any datetime columns
				for col in filtered_coverage_for_excel.columns:
					if isinstance(filtered_coverage_for_excel[col].dtype, pd.DatetimeTZDtype):
						filtered_coverage_for_excel[col] = filtered_coverage_for_excel[col].dt.tz_localize(None)
				with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
					filtered_coverage_for_excel.to_excel(writer, sheet_name='coverage', index=True)
					updated_meta.to_excel(writer, sheet_name='metadata', index=False)
				print(f"Filtered coverage and metadata saved to {out_excel}")
			else:
				print("Skipping Excel export (SAVE_EXCEL=False)")
		return filtered_coverage, actual_run_dir
	except Exception as e:
		raise RuntimeError(f"Error in filter_hedge_instruments: {e}")

def main():
	"""
	Command-line interface for filtering hedge instrument coverage and metadata.
	- Supports both single and interactive batch filtering modes.
	- Saves filtered results and updates forward prices for each subset.
	"""
	parser = argparse.ArgumentParser(description="Filter hedge_instruments_coverage.parquet by product_type and load_type.")
	parser.add_argument('--product_type', help="Product type to filter (e.g., month, week, year)")
	parser.add_argument('--load_type', help="Load type to filter (e.g., base, peak)")
	parser.add_argument('--coverage_path', default='hedge_instruments_coverage.parquet', help="Path to coverage parquet file")
	parser.add_argument('--mapping_path', default=None, help="Path to metadata parquet file")
	parser.add_argument('--run_dir', default=None, help="Directory to save the run output (default: auto-generated in Data/runs/)")
	parser.add_argument('--batch', action='store_true', help="Run interactively for all combinations of product_type and load_type")
	args = parser.parse_args()

	if args.batch:
		# Interactive batch mode: let user select product_types and load_types
		_, metadata_df = load_coverage_and_metadata(args.coverage_path, args.mapping_path)
		product_types = sorted(metadata_df['product_type'].unique())
		load_types = sorted(metadata_df['load_type'].unique())
		print("Available product_types:", product_types)
		print("Available load_types:", load_types)
		# Prompt user for selections
		selected_product_types = input(f"Enter product_types to filter (comma-separated, or 'all' for all): ").strip()
		selected_load_types = input(f"Enter load_types to filter (comma-separated, or 'all' for all): ").strip()
		if selected_product_types.lower() == 'all' or not selected_product_types:
			selected_product_types = product_types
		else:
			selected_product_types = [pt.strip() for pt in selected_product_types.split(',') if pt.strip() in product_types]
		if selected_load_types.lower() == 'all' or not selected_load_types:
			selected_load_types = load_types
		else:
			selected_load_types = [lt.strip() for lt in selected_load_types.split(',') if lt.strip() in load_types]
		# Run filter for each selected combination
		for pt in selected_product_types:
			for lt in selected_load_types:
				print(f"\n--- Running for product_type={pt}, load_type={lt} ---")
				filter_hedge_instruments(
					subset_filter={"product_type": pt, "load_type": lt},
					coverage_path=args.coverage_path,
					mapping_path=args.mapping_path,
					save=True,
					run_dir=args.run_dir
				)
	else:
		# Single run mode: require both product_type and load_type
		if not args.product_type or not args.load_type:
			parser.error('Either --batch or both --product_type and --load_type must be specified.')
		filter_hedge_instruments(
			subset_filter={"product_type": args.product_type, "load_type": args.load_type},
			coverage_path=args.coverage_path,
			mapping_path=args.mapping_path,
			save=True,
			run_dir=args.run_dir
		)

if __name__ == "__main__":
	# Run the main function if executed as a script
	main()



