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

def load_coverage_and_metadata(coverage_path, mapping_path=None):
	"""
	Load the coverage matrix and metadata for hedge instruments.
	- If coverage_path is an Excel file, loads both 'coverage' and 'metadata' sheets.
	- If coverage_path is a Parquet file, loads coverage and metadata from separate files.
	Raises FileNotFoundError if metadata is missing.
	Returns:
		coverage_df (pd.DataFrame): The coverage matrix (datetime x instrument_id).
		metadata_df (pd.DataFrame): The instrument metadata.
	"""
	"""
	Loads the coverage matrix and metadata.
	- If coverage_path is an Excel file, loads both 'coverage' and 'metadata' sheets.
	- If coverage_path is a Parquet file, loads coverage and metadata from separate files.
	Raises FileNotFoundError if metadata is missing.
	Returns: (coverage_df, metadata_df)
	"""
	try:
		if coverage_path.endswith('.xlsx'):
			# Excel file: try to read both sheets
			xls = pd.ExcelFile(coverage_path)
			coverage_df = pd.read_excel(xls, sheet_name='coverage', index_col=0)
			if 'metadata' in xls.sheet_names:
				metadata_df = pd.read_excel(xls, sheet_name='metadata')
			else:
				raise FileNotFoundError("No 'metadata' sheet found in the Excel file.")
			return coverage_df, metadata_df
		else:
			# Parquet file: try to read separate metadata file
			coverage_df = pd.read_parquet(coverage_path)
			if mapping_path and os.path.exists(mapping_path):
				metadata_df = pd.read_parquet(mapping_path)
			else:
				raise FileNotFoundError("No metadata file found. Please provide a metadata parquet or use an Excel file with a 'metadata' sheet.")
			return coverage_df, metadata_df
	except Exception as e:
		raise RuntimeError(f"Failed to load coverage or metadata: {e}")

def filter_hedge_instruments(product_type, load_type, coverage_path='hedge_instruments_coverage.parquet', mapping_path=None, save=True, run_dir=None):
	"""
	Filter hedge instruments by product_type and load_type, and save the filtered results.
	Steps:
	  1. Loads coverage and metadata.
	  2. Filters metadata for the given product_type and load_type.
	  3. Selects columns in coverage corresponding to the filtered instruments (plus 'datetime' if present).
	  4. Optionally saves filtered coverage and metadata to Parquet and Excel in a run directory.
	  5. Calculates and updates the price column in the filtered metadata using forward price calculation for the subset.

	Args:
		product_type (str or list): The product type to filter (e.g., 'month', 'year').
		load_type (str or list): The load type to filter (e.g., 'base', 'peak').
		coverage_path (str): Path to the coverage matrix file (Parquet or Excel).
		mapping_path (str): Path to the metadata file (Parquet or Excel).
		save (bool): Whether to save the filtered results to disk.
		run_dir (str or None): Directory to save the run output (auto-generated if None).

	Returns:
		filtered_coverage (pd.DataFrame or None): The filtered coverage matrix, or None if no instruments found.
	"""
	"""
	Filter hedge instruments by product_type and load_type.
	- Loads coverage and metadata.
	- Filters metadata for the given 	pytest tests/test_calculate_forward_prices_as_mean_from_pfc.pyproduct_type and load_type.
	- Selects columns in coverage corresponding to the filtered instruments (plus 'datetime' if present).
	- Optionally saves filtered coverage and metadata to Parquet and Excel in a run directory.
	Returns: filtered coverage DataFrame, or None if no instruments found.
	"""
	try:
		coverage_df, metadata_df = load_coverage_and_metadata(coverage_path, mapping_path)
		# Filter metadata for the selected product_type and load_type
		# Support both single values and lists for product_type/load_type
		if isinstance(product_type, (list, tuple, set)):
			product_type_filter = metadata_df['product_type'].isin(product_type)
		else:
			product_type_filter = metadata_df['product_type'] == product_type
		if isinstance(load_type, (list, tuple, set)):
			load_type_filter = metadata_df['load_type'].isin(load_type)
		else:
			load_type_filter = metadata_df['load_type'] == load_type
		filtered_meta = metadata_df[product_type_filter & load_type_filter]
		selected_ids = filtered_meta['instrument_id'].tolist()
		if not selected_ids:
			print(f"No instruments found for product_type='{product_type}' and load_type='{load_type}'.")
			return None, None
		# Always include the datetime column if present in coverage
		cols = ['datetime'] + selected_ids if 'datetime' in coverage_df.columns else selected_ids
		filtered_coverage = coverage_df[cols]
		print(f"Filtered coverage matrix shape: {filtered_coverage.shape}")
		actual_run_dir = None
		if save:
			# Create run directory if not provided
			if run_dir is None:
				timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
				run_dir = os.path.join('Data', 'runs', f'run_{timestamp}')
			os.makedirs(run_dir, exist_ok=True)
			actual_run_dir = run_dir
			# Build output file paths for this run
			out_coverage_parquet = os.path.join(run_dir, f'filtered_coverage_{product_type}_{load_type}.parquet')
			out_metadata_parquet = os.path.join(run_dir, f'filtered_metadata_{product_type}_{load_type}.parquet')
			out_excel = os.path.join(run_dir, f'filtered_coverage_and_metadata_{product_type}_{load_type}.xlsx')
			# Save filtered coverage and metadata
			filtered_coverage.to_parquet(out_coverage_parquet)
			filtered_meta.to_parquet(out_metadata_parquet)
			print(f"Filtered coverage saved to {out_coverage_parquet}")
			print(f"Filtered metadata saved to {out_metadata_parquet}")

			# --- Integrate forward price calculation for this subset ---
			price_csv_path = "Data/current/26_pri_de_fut_front-year_clo_€-mwh_cet_h_f_202508010000.csv"
			calculate_forward_prices_for_coverage(
				coverage_path=out_coverage_parquet,
				metadata_path=out_metadata_parquet,
				price_csv_path=price_csv_path,
				output_metadata_path=out_metadata_parquet
			)

			# Reload the updated metadata (with prices) for Excel export
			updated_meta = pd.read_parquet(out_metadata_parquet)
			# Save as Excel with metadata sheet (now including price column)
			with pd.ExcelWriter(out_excel, engine='openpyxl') as writer:
				filtered_coverage.to_excel(writer, sheet_name='coverage', index=True)
				updated_meta.to_excel(writer, sheet_name='metadata', index=False)
			print(f"Filtered coverage and metadata saved to {out_excel}")
		return filtered_coverage, actual_run_dir
	except Exception as e:
		raise RuntimeError(f"Error in filter_hedge_instruments: {e}")

def main():
	"""
	Command-line interface for filtering hedge instrument coverage and metadata.
	- Supports both single and interactive batch filtering modes.
	- Saves filtered results and updates forward prices for each subset.
	"""
	"""
	Main entry point for command-line usage.
	Parses arguments and runs filtering in either single or interactive batch mode.
	"""
	parser = argparse.ArgumentParser(description="Filter hedge_instruments_coverage.parquet by product_type and load_type.")
	parser.add_argument('--product_type', help="Product type to filter (e.g., month, week, year)")
	parser.add_argument('--load_type', help="Load type to filter (e.g., base, peak)")
	parser.add_argument('--coverage_path', default='hedge_instruments_coverage.parquet', help="Path to coverage parquet or Excel file")
	parser.add_argument('--mapping_path', default=None, help="Path to metadata parquet file (optional if using Excel with metadata sheet)")
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
					product_type=pt,
					load_type=lt,
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
			product_type=args.product_type,
			load_type=args.load_type,
			coverage_path=args.coverage_path,
			mapping_path=args.mapping_path,
			save=True,
			run_dir=args.run_dir
		)

if __name__ == "__main__":
	# Run the main function if executed as a script
	main()



