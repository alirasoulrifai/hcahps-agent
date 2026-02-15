import pandas as pd

file_path = r"c:\Users\USER\Documents\streamlit_app\Arabic Poem Comprehensive Dataset (APCD).csv"
try:
    # Read first 5 rows to check structure
    df = pd.read_csv(file_path, nrows=5)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.to_string())
except Exception as e:
    print(f"Error reading with default encoding: {e}")
    try:
        # Try with utf-8-sig or cp1256 if failed
        df = pd.read_csv(file_path, nrows=5, encoding='utf-8-sig')
        print("Columns:", df.columns.tolist())
        print("\nFirst 5 rows (utf-8-sig):")
        print(df.to_string())
    except Exception as e2:
        print(f"Error reading with utf-8-sig: {e2}")
