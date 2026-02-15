import pandas as pd
import os

file_path = r"c:\Users\USER\Documents\streamlit_app\Arabic Poem Comprehensive Dataset (APCD).csv"

def analyze_csv():
    print(f"Analyzing {os.path.basename(file_path)} for embedding readiness...\n")
    
    # Read a sample to avoid memory issues (e.g., 50k rows)
    try:
        df = pd.read_csv(file_path, nrows=50000)
    except Exception as e:
        print(f"Error: {e}")
        return

    # 1. Missing Values
    print("--- Missing Values ---")
    print(df.isnull().sum())
    print("\n")

    # 2. Duplicates
    print("--- Duplicate Verses ---")
    # 'البيت' contains the full verse
    duplicates = df.duplicated(subset=['البيت']).sum()
    print(f"Duplicates in sample (50k): {duplicates} ({duplicates/50000*100:.2f}%)")
    print("\n")

    # 3. Column Variety
    print("--- Unique Values (Sample) ---")
    for col in ['العصر', 'البحر', 'القافية']:
        print(f"{col}: {df[col].nunique()} unique types")

    # 4. Verse Length (Tokens/Chars)
    print("\n--- Verse Length Stats (Characters) ---")
    lengths = df['البيت'].str.len()
    print(lengths.describe())

analyze_csv()
