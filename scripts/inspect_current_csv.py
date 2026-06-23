import os

import pandas as pd


def inspect_csv() -> None:
    # Automatically select the first CSV in Data/current/
    csv_dir = os.path.join("Data", "current")
    files = [f for f in os.listdir(csv_dir) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError("No CSV files found in Data/current/")
    csv_path = os.path.join(csv_dir, files[0])

    print(f"Inspecting file: {csv_path}")
    delimiter = ";"

    # Try reading with semicolon delimiter first, then comma if it fails
    try:
        df = pd.read_csv(csv_path, delimiter=delimiter)
        print("Read with delimiter ';' (semicolon).")
    except Exception as e:
        print(f"Semicolon delimiter failed: {e}")
        print("Trying comma delimiter...")
        df = pd.read_csv(csv_path)
        delimiter = ","
        print("Read with default delimiter ',' (comma).")

    print("\nFirst 5 rows:")
    print(df.head())
    print("\nColumns:")
    print(df.columns)
    print(f"\nShape: {df.shape}")

    suitable = True

    # 1. Check for exactly 2 columns
    if df.shape[1] != 2:
        print(f"[!] Expected 2 columns, found {df.shape[1]}.")
        print("[!] The file may have the wrong delimiter or column format.")
        suitable = False
    else:
        print("Found exactly 2 columns.")

    # 2-4. Datetime, numeric, missing checks
    if df.shape[1] >= 2:
        try:
            pd.to_datetime(df.iloc[:, 0])
            print("First column can be parsed as datetime.")
        except Exception as e:
            print(f"[!] First column is not datetime-like: {e}")
            print("    Suggestion: Check your date format (ISO, dayfirst, or with timezone offset)")
            suitable = False

        try:
            pd.to_numeric(df.iloc[:, 1], errors="raise")
            print("Second column can be converted to numeric.")
        except Exception as e:
            print(f"[!] Second column is not numeric: {e}")
            print("    Trying to convert by replacing ',' with '.' (comma as decimal separator)...")
            try:
                pd.to_numeric(df.iloc[:, 1].astype(str).str.replace(",", "."), errors="raise")
                print("[OK] Second column can be converted after replacing ',' with '.'.")
            except Exception as e2:
                print(f"[!] Still not numeric after replacement: {e2}")
                print("    Suggestion: Check for non-numeric values or extra spaces in the second column.")
                suitable = False

        if df.isnull().any().any():
            print("[!] Missing values detected in the CSV.")
            suitable = False
        else:
            print("No missing values detected.")

    if suitable and df.shape[1] == 2:
        print("\n[OK] This CSV is suitable for the main code.")
    else:
        print("\n[FAIL] This CSV is NOT suitable for the main code.")
    print(f"Delimiter used: '{delimiter}'")


if __name__ == "__main__":
    inspect_csv()