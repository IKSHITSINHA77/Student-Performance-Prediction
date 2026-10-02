import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

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
print("Day 5 - Tree-Based Regression Models")
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

print("\n2. FEATURES AND TARGET")
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
# 5. Preprocessing
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
# 6. Define models
# ---------------------------------------------------------
models = {

    "Decision Tree": DecisionTreeRegressor(
        random_state=42,
        max_depth=5,
        min_samples_split=5
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=10,
        min_samples_split=4,
        n_jobs=-1
    )
}


results = {}
trained_models = {}


# ---------------------------------------------------------
# 7. Train and evaluate models
# ---------------------------------------------------------
for model_name, model in models.items():

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = mse ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    results[model_name] = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    trained_models[model_name] = pipeline

    print(f"\n{model_name} Results")
    print("-" * 40)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


# ---------------------------------------------------------
# 8. Display comparison
# ---------------------------------------------------------
results_df = pd.DataFrame(results).T

print("\n\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(results_df.round(4).to_string())


# ---------------------------------------------------------
# 9. Save models
# ---------------------------------------------------------
decision_tree_path = (
    MODELS_DIR /
    "decision_tree_model.joblib"
)

random_forest_path = (
    MODELS_DIR /
    "random_forest_model.joblib"
)

joblib.dump(
    trained_models["Decision Tree"],
    decision_tree_path
)

joblib.dump(
    trained_models["Random Forest"],
    random_forest_path
)

print("\n8. MODELS SAVED")
print("-" * 50)

print(f"Decision Tree : {decision_tree_path}")
print(f"Random Forest : {random_forest_path}")


# ---------------------------------------------------------
# 10. Create model comparison visualization
# ---------------------------------------------------------
metrics_to_plot = [
    "MAE",
    "RMSE",
    "R2"
]

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5)
)

for index, metric in enumerate(metrics_to_plot):

    axes[index].bar(
        results_df.index,
        results_df[metric]
    )

    axes[index].set_title(
        f"{metric} Comparison"
    )

    axes[index].set_ylabel(metric)

    axes[index].tick_params(
        axis="x",
        rotation=20
    )


plt.tight_layout()

comparison_plot_path = (
    OUTPUT_DIR /
    "tree_models_comparison.png"
)

plt.savefig(
    comparison_plot_path,
    dpi=300
)

plt.close()


print(f"\nComparison plot saved: {comparison_plot_path}")


# ---------------------------------------------------------
# 11. Save comparison results
# ---------------------------------------------------------
results_path = (
    BASE_DIR /
    "reports" /
    "day5_model_comparison.csv"
)

results_path.parent.mkdir(
    parents=True,
    exist_ok=True
)

results_df.to_csv(
    results_path
)

print(f"Comparison results saved: {results_path}")


# ---------------------------------------------------------
# 12. Final summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("DAY 5 COMPLETED")
print("=" * 70)

print("Decision Tree model trained.")
print("Random Forest model trained.")
print("Models evaluated.")
print("Models saved.")
print("Comparison visualization generated.")
print("Comparison results saved.")

print("=" * 70)