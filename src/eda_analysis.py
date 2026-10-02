import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
OUTPUT_DIR = BASE_DIR / "visualizations"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")


print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("Day 2 - Data Cleaning and Exploratory Data Analysis")
print("=" * 70)


# ---------------------------------------------------------
# 1. Dataset information
# ---------------------------------------------------------
print("\n1. DATASET INFORMATION")
print("-" * 50)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ---------------------------------------------------------
# 2. Missing value analysis
# ---------------------------------------------------------
print("\n2. MISSING VALUE ANALYSIS")
print("-" * 50)

missing_values = df.isnull().sum()
total_missing = missing_values.sum()

print(missing_values)

print(f"\nTotal missing values: {total_missing}")


# ---------------------------------------------------------
# 3. Duplicate analysis
# ---------------------------------------------------------
print("\n3. DUPLICATE ANALYSIS")
print("-" * 50)

duplicate_count = df.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# ---------------------------------------------------------
# 4. Data type analysis
# ---------------------------------------------------------
print("\n4. DATA TYPE ANALYSIS")
print("-" * 50)

print(df.dtypes.value_counts())


# ---------------------------------------------------------
# 5. Target variable analysis
# ---------------------------------------------------------
print("\n5. TARGET VARIABLE ANALYSIS - G3")
print("-" * 50)

print(f"Minimum G3 : {df['G3'].min()}")
print(f"Maximum G3 : {df['G3'].max()}")
print(f"Mean G3    : {df['G3'].mean():.2f}")
print(f"Median G3  : {df['G3'].median():.2f}")
print(f"Std G3     : {df['G3'].std():.2f}")


# ---------------------------------------------------------
# 6. G3 distribution
# ---------------------------------------------------------
plt.figure(figsize=(10, 6))

sns.histplot(
    df["G3"],
    bins=21,
    kde=True
)

plt.title("Distribution of Final Student Grades (G3)")
plt.xlabel("Final Grade (G3)")
plt.ylabel("Number of Students")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "g3_distribution.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 7. Study time vs G3
# ---------------------------------------------------------
study_time_performance = (
    df.groupby("studytime")["G3"]
    .mean()
    .reset_index()
)

print("\n6. STUDY TIME VS FINAL GRADE")
print("-" * 50)
print(study_time_performance)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=study_time_performance,
    x="studytime",
    y="G3"
)

plt.title("Average Final Grade by Study Time")
plt.xlabel("Study Time Category")
plt.ylabel("Average G3")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "studytime_vs_g3.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 8. Failures vs G3
# ---------------------------------------------------------
failure_performance = (
    df.groupby("failures")["G3"]
    .mean()
    .reset_index()
)

print("\n7. PREVIOUS FAILURES VS FINAL GRADE")
print("-" * 50)
print(failure_performance)


plt.figure(figsize=(8, 5))

sns.barplot(
    data=failure_performance,
    x="failures",
    y="G3"
)

plt.title("Average Final Grade by Number of Previous Failures")
plt.xlabel("Number of Previous Failures")
plt.ylabel("Average G3")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "failures_vs_g3.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 9. Absences vs G3
# ---------------------------------------------------------
print("\n8. ABSENCES VS FINAL GRADE")
print("-" * 50)

print(
    f"Correlation between absences and G3: "
    f"{df['absences'].corr(df['G3']):.4f}"
)


plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x="absences",
    y="G3"
)

plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade (G3)")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "absences_vs_g3.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 10. G1 vs G3
# ---------------------------------------------------------
print("\n9. G1 VS G3")
print("-" * 50)

print(
    f"Correlation between G1 and G3: "
    f"{df['G1'].corr(df['G3']):.4f}"
)


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="G1",
    y="G3"
)

plt.title("First Period Grade (G1) vs Final Grade (G3)")
plt.xlabel("G1")
plt.ylabel("G3")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "g1_vs_g3.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 11. G2 vs G3
# ---------------------------------------------------------
print("\n10. G2 VS G3")
print("-" * 50)

print(
    f"Correlation between G2 and G3: "
    f"{df['G2'].corr(df['G3']):.4f}"
)


plt.figure(figsize=(8, 6))

sns.scatterplot(
    data=df,
    x="G2",
    y="G3"
)

plt.title("Second Period Grade (G2) vs Final Grade (G3)")
plt.xlabel("G2")
plt.ylabel("G3")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "g2_vs_g3.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 12. Correlation analysis
# ---------------------------------------------------------
print("\n11. CORRELATION ANALYSIS")
print("-" * 50)

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

correlation_matrix = df[numeric_columns].corr()

print(
    correlation_matrix["G3"]
    .sort_values(ascending=False)
)


# ---------------------------------------------------------
# 13. Correlation heatmap
# ---------------------------------------------------------
plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0
)

plt.title("Correlation Heatmap of Numerical Features")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "correlation_heatmap.png",
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 14. Final summary
# ---------------------------------------------------------
print("\n12. DAY 2 SUMMARY")
print("-" * 50)

print("Data cleaning checks completed.")
print("Exploratory data analysis completed.")
print("Target variable G3 analyzed.")
print("Important feature relationships analyzed.")
print("Visualizations saved to the visualizations folder.")

print("\n" + "=" * 70)
print("Day 2 exploratory analysis completed successfully.")
print("=" * 70)