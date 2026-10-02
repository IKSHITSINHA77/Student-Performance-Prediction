import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    KFold
)

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
REPORTS_DIR = BASE_DIR / "reports"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("Day 6 - Hyperparameter Tuning & Cross-Validation")
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
# 6. Cross-validation strategy
# ---------------------------------------------------------
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

print("\n5. CROSS-VALIDATION")
print("-" * 50)

print("Method : 5-Fold Cross-Validation")
print("Shuffle: True")
print("Random State: 42")


# ---------------------------------------------------------
# 7. Decision Tree pipeline
# ---------------------------------------------------------
decision_tree_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            DecisionTreeRegressor(
                random_state=42
            )
        )
    ]
)


# ---------------------------------------------------------
# 8. Decision Tree parameter grid
# ---------------------------------------------------------
decision_tree_params = {
    "model__max_depth": [
        3,
        5,
        7,
        10,
        None
    ],

    "model__min_samples_split": [
        2,
        5,
        10
    ],

    "model__min_samples_leaf": [
        1,
        2,
        4
    ]
}


# ---------------------------------------------------------
# 9. Tune Decision Tree
# ---------------------------------------------------------
print("\n6. TUNING DECISION TREE")
print("-" * 50)

decision_tree_search = GridSearchCV(
    estimator=decision_tree_pipeline,
    param_grid=decision_tree_params,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

decision_tree_search.fit(
    X_train,
    y_train
)

print("Decision Tree tuning completed.")

print("\nBest Decision Tree parameters:")
print(decision_tree_search.best_params_)

print(
    f"Best CV RMSE: "
    f"{-decision_tree_search.best_score_:.4f}"
)


# ---------------------------------------------------------
# 10. Random Forest pipeline
# ---------------------------------------------------------
random_forest_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)


# ---------------------------------------------------------
# 11. Random Forest parameter grid
# ---------------------------------------------------------
random_forest_params = {
    "model__n_estimators": [
        100,
        200
    ],

    "model__max_depth": [
        5,
        10,
        None
    ],

    "model__min_samples_split": [
        2,
        5
    ],

    "model__min_samples_leaf": [
        1,
        2
    ]
}


# ---------------------------------------------------------
# 12. Tune Random Forest
# ---------------------------------------------------------
print("\n7. TUNING RANDOM FOREST")
print("-" * 50)

random_forest_search = GridSearchCV(
    estimator=random_forest_pipeline,
    param_grid=random_forest_params,
    cv=cv,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

random_forest_search.fit(
    X_train,
    y_train
)

print("Random Forest tuning completed.")

print("\nBest Random Forest parameters:")
print(random_forest_search.best_params_)

print(
    f"Best CV RMSE: "
    f"{-random_forest_search.best_score_:.4f}"
)


# ---------------------------------------------------------
# 13. Evaluate tuned models on test set
# ---------------------------------------------------------
tuned_models = {
    "Tuned Decision Tree": decision_tree_search.best_estimator_,
    "Tuned Random Forest": random_forest_search.best_estimator_
}

results = {}


print("\n8. TEST SET EVALUATION")
print("-" * 50)


for model_name, model in tuned_models.items():

    y_pred = model.predict(X_test)

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

    print(f"\n{model_name}")
    print("-" * 40)

    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")


# ---------------------------------------------------------
# 14. Results table
# ---------------------------------------------------------
results_df = pd.DataFrame(results).T

print("\n\n" + "=" * 70)
print("TUNED MODEL COMPARISON")
print("=" * 70)

print(
    results_df.round(4).to_string()
)


# ---------------------------------------------------------
# 15. Save tuned models
# ---------------------------------------------------------
tuned_tree_path = (
    MODELS_DIR /
    "tuned_decision_tree_model.joblib"
)

tuned_forest_path = (
    MODELS_DIR /
    "tuned_random_forest_model.joblib"
)

joblib.dump(
    decision_tree_search.best_estimator_,
    tuned_tree_path
)

joblib.dump(
    random_forest_search.best_estimator_,
    tuned_forest_path
)


print("\n9. TUNED MODELS SAVED")
print("-" * 50)

print(f"Tuned Decision Tree : {tuned_tree_path}")
print(f"Tuned Random Forest : {tuned_forest_path}")


# ---------------------------------------------------------
# 16. Save results
# ---------------------------------------------------------
results_path = (
    REPORTS_DIR /
    "day6_tuned_model_results.csv"
)

results_df.to_csv(
    results_path
)

print(f"\nResults saved: {results_path}")


# ---------------------------------------------------------
# 17. Create comparison plot
# ---------------------------------------------------------
plt.figure(figsize=(9, 6))

results_df["RMSE"].plot(
    kind="bar"
)

plt.title(
    "Day 6 - Tuned Model RMSE Comparison"
)

plt.xlabel("Model")
plt.ylabel("RMSE")

plt.xticks(
    rotation=15
)

plt.tight_layout()

plot_path = (
    OUTPUT_DIR /
    "day6_tuned_model_rmse.png"
)

plt.savefig(
    plot_path,
    dpi=300
)

plt.close()


print(f"Plot saved: {plot_path}")


# ---------------------------------------------------------
# 18. Final summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("DAY 6 COMPLETED")
print("=" * 70)

print("5-fold cross-validation completed.")
print("Decision Tree hyperparameters tuned.")
print("Random Forest hyperparameters tuned.")
print("Tuned models evaluated on the test set.")
print("Tuned models saved.")
print("Results and visualization saved.")

print("=" * 70)