import pandas as pd
from pathlib import Path


# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"


def main():
    print("=" * 60)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("Day 1 - Dataset Inspection")
    print("=" * 60)

    # Load dataset
    df = pd.read_csv(DATA_PATH, sep=";")

    print("\n1. Dataset loaded successfully")
    print(f"Rows    : {df.shape[0]}")
    print(f"Columns : {df.shape[1]}")

    # Dataset preview
    print("\n2. First five records")
    print(df.head())

    # Column names
    print("\n3. Column names")
    for column in df.columns:
        print(f"- {column}")

    # Data types
    print("\n4. Data types")
    print(df.dtypes)

    # Missing values
    print("\n5. Missing values")
    missing_values = df.isnull().sum()
    print(missing_values)

    # Duplicate records
    print("\n6. Duplicate records")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")

    # Statistical summary
    print("\n7. Statistical summary")
    print(df.describe())

    # Target variable
    print("\n8. Target variable")
    print("Target variable: G3")
    print(f"G3 minimum: {df['G3'].min()}")
    print(f"G3 maximum: {df['G3'].max()}")
    print(f"G3 average : {df['G3'].mean():.2f}")

    print("\n" + "=" * 60)
    print("Day 1 dataset inspection completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()