import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("Day 3 - Feature Selection and Preprocessing")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(DATA_PATH, sep=";")

print("\n1. DATASET LOADED")
print("-" * 50)

print(f"Rows    : {df.shape[0]}")
print(f"Columns : {df.shape[1]}")


# ---------------------------------------------------------
# 2. Define target and features
# ---------------------------------------------------------
target = "G3"

X = df.drop(columns=[target])
y = df[target]

print("\n2. TARGET AND FEATURES")
print("-" * 50)

print(f"Target variable: {target}")
print(f"Number of input features: {X.shape[1]}")


# ---------------------------------------------------------
# 3. Identify numerical and categorical columns
# ---------------------------------------------------------
categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\n3. FEATURE TYPES")
print("-" * 50)

print(f"Categorical features: {len(categorical_features)}")
print(f"Numerical features  : {len(numerical_features)}")

print("\nCategorical features:")
for feature in categorical_features:
    print(f"- {feature}")

print("\nNumerical features:")
for feature in numerical_features:
    print(f"- {feature}")


# ---------------------------------------------------------
# 4. Create preprocessing pipeline
# ---------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# ---------------------------------------------------------
# 5. Train-test split
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\n4. TRAIN-TEST SPLIT")
print("-" * 50)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")


# ---------------------------------------------------------
# 6. Fit preprocessing on training data
# ---------------------------------------------------------
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


print("\n5. PREPROCESSING")
print("-" * 50)

print(
    f"Original number of features : {X.shape[1]}"
)

print(
    f"Processed number of features: "
    f"{X_train_processed.shape[1]}"
)

print(
    f"Processed training shape    : "
    f"{X_train_processed.shape}"
)

print(
    f"Processed testing shape     : "
    f"{X_test_processed.shape}"
)


# ---------------------------------------------------------
# 7. Save preprocessing pipeline
# ---------------------------------------------------------
preprocessor_path = MODELS_DIR / "preprocessor.joblib"

joblib.dump(
    preprocessor,
    preprocessor_path
)


print("\n6. PREPROCESSOR SAVED")
print("-" * 50)

print(f"Saved to: {preprocessor_path}")


# ---------------------------------------------------------
# 8. Verify transformed data
# ---------------------------------------------------------
print("\n7. TRANSFORMED DATA VERIFICATION")
print("-" * 50)

print(
    f"Training data contains NaN: "
    f"{pd.isna(X_train_processed).any()}"
)

print(
    f"Testing data contains NaN: "
    f"{pd.isna(X_test_processed).any()}"
)


# ---------------------------------------------------------
# 9. Feature selection summary
# ---------------------------------------------------------
print("\n8. FEATURE SELECTION SUMMARY")
print("-" * 50)

print("Target: G3")
print("Input features: all available features except G3")
print("Categorical encoding: One-Hot Encoding")
print("Numerical features: passed through unchanged")
print("Unknown categories: ignored safely")
print("Test size: 20%")
print("Random state: 42")


print("\n" + "=" * 70)
print("Day 3 preprocessing completed successfully.")
print("=" * 70)