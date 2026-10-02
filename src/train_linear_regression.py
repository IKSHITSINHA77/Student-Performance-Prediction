import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "visualizations"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("Day 4 - Linear Regression Baseline")
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
# 2. Separate features and target
# ---------------------------------------------------------
target = "G3"

X = df.drop(columns=[target])
y = df[target]


print("\n2. TARGET AND FEATURES")
print("-" * 50)

print(f"Target variable: {target}")
print(f"Input features : {X.shape[1]}")


# ---------------------------------------------------------
# 3. Identify feature types
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


# ---------------------------------------------------------
# 4. Train-test split
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
# 5. Create preprocessing pipeline
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
# 6. Create complete ML pipeline
# ---------------------------------------------------------
model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression())
    ]
)


# ---------------------------------------------------------
# 7. Train model
# ---------------------------------------------------------
print("\n5. MODEL TRAINING")
print("-" * 50)

model_pipeline.fit(X_train, y_train)

print("Linear Regression model trained successfully.")


# ---------------------------------------------------------
# 8. Generate predictions
# ---------------------------------------------------------
y_pred = model_pipeline.predict(X_test)


print("\n6. PREDICTIONS GENERATED")
print("-" * 50)

print(f"Number of predictions: {len(y_pred)}")


# ---------------------------------------------------------
# 9. Calculate evaluation metrics
# ---------------------------------------------------------
mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


print("\n7. MODEL EVALUATION")
print("-" * 50)

print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")


# ---------------------------------------------------------
# 10. Display sample predictions
# ---------------------------------------------------------
comparison = pd.DataFrame({
    "Actual_G3": y_test.values,
    "Predicted_G3": y_pred
})

print("\n8. SAMPLE PREDICTIONS")
print("-" * 50)

print(comparison.head(10).to_string(index=False))


# ---------------------------------------------------------
# 11. Actual vs Predicted visualization
# ---------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.7
)

# Perfect prediction reference line
min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.title("Linear Regression - Actual vs Predicted G3")
plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")

plt.tight_layout()

plot_path = (
    OUTPUT_DIR /
    "linear_regression_actual_vs_predicted.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


# ---------------------------------------------------------
# 12. Save complete model pipeline
# ---------------------------------------------------------
model_path = (
    MODELS_DIR /
    "linear_regression_model.joblib"
)

joblib.dump(
    model_pipeline,
    model_path
)


print("\n9. MODEL SAVED")
print("-" * 50)

print(f"Model path: {model_path}")
print(f"Plot path : {plot_path}")


# ---------------------------------------------------------
# 13. Final summary
# ---------------------------------------------------------
print("\n10. DAY 4 SUMMARY")
print("-" * 50)

print("Linear Regression baseline completed.")
print("Model evaluation completed.")
print("Actual vs predicted visualization generated.")
print("Complete preprocessing + model pipeline saved.")

print("\n" + "=" * 70)
print("Day 4 Linear Regression completed successfully.")
print("=" * 70)