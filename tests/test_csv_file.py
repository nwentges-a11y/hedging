import pandas as pd
import os

# Automatically select the first CSV in Data/current/
csv_dir = os.path.join('Data', 'current')
files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
if not files:
    raise FileNotFoundError('No CSV files found in Data/current/')
csv_path = os.path.join(csv_dir, files[0])

def test_csv():
    print(f"Inspecting file: {csv_path}")
    delimiter = ';'
    # Try reading with semicolon delimiter first, then comma if it fails
    try:
        df = pd.read_csv(csv_path, delimiter=delimiter)
        print("Read with delimiter ';' (semicolon).")
    except Exception as e:
        print(f"Semicolon delimiter failed: {e}")
        print("Trying comma delimiter...")
        df = pd.read_csv(csv_path)
        delimiter = ','
        print("Read with default delimiter ',' (comma).")

    # Show first 5 rows and columns for inspection
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:")
    print(df.columns)
    print(f"\nShape: {df.shape}")

    # --- Suitability checks for main code ---
    suitable = True
    # 1. Check for exactly 2 columns
    if df.shape[1] != 2:
        print(f"[!] Expected 2 columns, found {df.shape[1]}.")
        print("[!] The file may have the wrong delimiter or column format.")
        suitable = False
    else:
        print("Found exactly 2 columns.")
    # Only proceed with further checks if there are at least 2 columns
    if df.shape[1] >= 2:
        # 2. Check first column is datetime-like with dayfirst=True
        try:
            pd.to_datetime(df.iloc[:, 0], dayfirst=True)
            print("First column can be parsed as datetime with dayfirst=True.")
        except Exception as e:
            print(f"[!] First column is not datetime-like (even with dayfirst=True): {e}")
            print("    Suggestion: Check your date format or use pd.to_datetime(..., dayfirst=True)")
            suitable = False
        # 3. Check second column is numeric (try conversion)
        try:
            pd.to_numeric(df.iloc[:, 1], errors='raise')
            print("Second column can be converted to numeric.")
        except Exception as e:
            print(f"[!] Second column is not numeric: {e}")
            print("    Trying to convert by replacing ',' with '.' (comma as decimal separator)...")
            try:
                col_as_float = pd.to_numeric(df.iloc[:, 1].str.replace(',', '.'), errors='raise')
                print("[OK] Second column can be converted to numeric after replacing ',' with '.' (comma as decimal separator).")
            except Exception as e2:
                print(f"[!] Still not numeric after replacement: {e2}")
                print("    Suggestion: Check for non-numeric values or extra spaces in the second column.")
                suitable = False
        # 4. Check for missing values
        if df.isnull().any().any():
            print("[!] Missing values detected in the CSV.")
            suitable = False
        else:
            print("No missing values detected.")
    # --- Final suitability message ---
    if suitable and df.shape[1] == 2:
        print("\n[OK] This CSV is suitable for the main code.")
        print("You can use the following code to load it in your main script:")
        print(f"""\nimport pandas as pd\ndf = pd.read_csv(r'{csv_path}', delimiter='{delimiter}')\ndf.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], dayfirst=True)\ndf.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors='raise')\n""")
    else:
        print("\n[FAIL] This CSV is NOT suitable for the main code.")
    print(f"Delimiter used: '{delimiter}'")

if __name__ == "__main__":
    # This script inspects the first CSV in Data/current/ and checks if it matches the requirements
    # for the main code: 2 columns, first column datetime, second numeric, no missing values.
    test_csv()