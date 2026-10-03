import pandas as pd
import joblib

from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "models" / "tuned_random_forest_model.joblib"


# ---------------------------------------------------------
# Load trained model and dataset
# ---------------------------------------------------------
def load_prediction_system():
    """Load the trained model and dataset."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(
        DATA_PATH,
        sep=";"
    )

    return model, df


# ---------------------------------------------------------
# Performance interpretation
# ---------------------------------------------------------
def interpret_performance(predicted_g3):
    """Convert predicted grade into a simple performance level."""

    if predicted_g3 >= 16:
        return "Excellent"
    elif predicted_g3 >= 14:
        return "Very Good"
    elif predicted_g3 >= 10:
        return "Satisfactory"
    elif predicted_g3 >= 8:
        return "Needs Improvement"
    else:
        return "At Risk"


# ---------------------------------------------------------
# Validate student input
# ---------------------------------------------------------
def validate_student_input(student_data, dataset):
    """Validate student input against the training feature structure."""

    required_features = [
        column for column in dataset.columns
        if column != "G3"
    ]

    provided_features = set(student_data.keys())
    required_features_set = set(required_features)

    missing_features = required_features_set - provided_features
    extra_features = provided_features - required_features_set

    if missing_features:
        raise ValueError(
            f"Missing required features: {sorted(missing_features)}"
        )

    if extra_features:
        raise ValueError(
            f"Unexpected features provided: {sorted(extra_features)}"
        )

    # Validate categorical values against the training dataset.
    categorical_columns = dataset.select_dtypes(
        include=["object"]
    ).columns

    for column in categorical_columns:

        if column == "G3":
            continue

        allowed_values = set(
            dataset[column].dropna().unique()
        )

        if student_data[column] not in allowed_values:
            raise ValueError(
                f"Invalid value for '{column}': "
                f"{student_data[column]}. "
                f"Expected one of {sorted(allowed_values)}"
            )

    # Validate numerical values.
    numerical_columns = dataset.select_dtypes(
        exclude=["object"]
    ).columns

    for column in numerical_columns:

        if column == "G3":
            continue

        value = student_data[column]

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float)
        ):
            raise TypeError(
                f"Invalid data type for '{column}'. "
                f"Expected a numerical value."
            )

        if pd.isna(value):
            raise ValueError(
                f"Missing numerical value for '{column}'."
            )

        # Validate against observed dataset ranges.
        min_value = dataset[column].min()
        max_value = dataset[column].max()

        if value < min_value or value > max_value:
            raise ValueError(
                f"Value for '{column}' is outside the "
                f"training data range [{min_value}, {max_value}]."
            )

    return required_features


# ---------------------------------------------------------
# Generate prediction
# ---------------------------------------------------------
def predict_student(student_data, model, dataset):
    """Generate a final-grade prediction for a student."""

    feature_columns = validate_student_input(
        student_data,
        dataset
    )

    student_df = pd.DataFrame(
        [student_data],
        columns=feature_columns
    )

    prediction = model.predict(student_df)

    predicted_g3 = float(prediction[0])

    # Validate prediction output.
    if not 0 <= predicted_g3 <= 20:
        raise ValueError(
            f"Invalid prediction value: {predicted_g3}"
        )

    performance = interpret_performance(predicted_g3)

    return predicted_g3, performance


# ---------------------------------------------------------
# Main testing workflow
# ---------------------------------------------------------
def main():

    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION SYSTEM")
    print("Day 11 - Final Prediction System")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------
    print("\n1. LOADING FINAL TRAINED MODEL")
    print("-" * 50)

    model, df = load_prediction_system()

    print("Tuned Random Forest model loaded successfully.")
    print(f"Model type: {type(model).__name__}")
    print(f"Dataset shape: {df.shape}")

    # -----------------------------------------------------
    # 2. Test student profiles
    # -----------------------------------------------------
    student_profiles = {

        "High Performance Student": {
            "school": "GP",
            "sex": "F",
            "age": 17,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "A",
            "Medu": 4,
            "Fedu": 4,
            "Mjob": "health",
            "Fjob": "teacher",
            "reason": "course",
            "guardian": "mother",
            "traveltime": 1,
            "studytime": 4,
            "failures": 0,
            "schoolsup": "yes",
            "famsup": "yes",
            "paid": "no",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": 5,
            "freetime": 3,
            "goout": 3,
            "Dalc": 1,
            "Walc": 1,
            "health": 5,
            "absences": 2,
            "G1": 17,
            "G2": 17
        },

        "Average Performance Student": {
            "school": "GP",
            "sex": "M",
            "age": 17,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "T",
            "Medu": 2,
            "Fedu": 2,
            "Mjob": "services",
            "Fjob": "other",
            "reason": "reputation",
            "guardian": "mother",
            "traveltime": 2,
            "studytime": 2,
            "failures": 0,
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "no",
            "activities": "no",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": 4,
            "freetime": 3,
            "goout": 3,
            "Dalc": 1,
            "Walc": 2,
            "health": 3,
            "absences": 6,
            "G1": 11,
            "G2": 11
        },

        "Needs Improvement Student": {
            "school": "MS",
            "sex": "M",
            "age": 18,
            "address": "R",
            "famsize": "LE3",
            "Pstatus": "T",
            "Medu": 1,
            "Fedu": 1,
            "Mjob": "other",
            "Fjob": "other",
            "reason": "other",
            "guardian": "other",
            "traveltime": 3,
            "studytime": 1,
            "failures": 2,
            "schoolsup": "no",
            "famsup": "no",
            "paid": "no",
            "activities": "no",
            "nursery": "no",
            "higher": "no",
            "internet": "no",
            "romantic": "yes",
            "famrel": 2,
            "freetime": 5,
            "goout": 5,
            "Dalc": 3,
            "Walc": 4,
            "health": 2,
            "absences": 15,
            "G1": 7,
            "G2": 8
        }
    }

    # -----------------------------------------------------
    # 3. Test predictions
    # -----------------------------------------------------
    print("\n2. TESTING STUDENT SCENARIOS")
    print("-" * 50)

    successful_predictions = 0

    for student_name, student_data in student_profiles.items():

        print(f"\nStudent Profile: {student_name}")

        try:

            predicted_g3, performance = predict_student(
                student_data,
                model,
                df
            )

            print(
                f"Predicted final grade (G3): "
                f"{predicted_g3:.2f}"
            )
            print(f"Performance level: {performance}")
            print("Prediction status: Successful")

            successful_predictions += 1

        except (ValueError, TypeError) as error:

            print(f"Prediction failed: {error}")

    # -----------------------------------------------------
    # 4. Test invalid input handling
    # -----------------------------------------------------
    print("\n3. TESTING INPUT VALIDATION")
    print("-" * 50)

    invalid_student = student_profiles[
        "Average Performance Student"
    ].copy()

    invalid_student["age"] = 100

    try:

        predict_student(
            invalid_student,
            model,
            df
        )

        print("✗ Invalid input was not rejected.")

    except (ValueError, TypeError):

        print(
            "✓ Invalid numerical input correctly rejected."
        )

    # -----------------------------------------------------
    # 5. Final validation summary
    # -----------------------------------------------------
    print("\n4. FINAL DAY 11 VALIDATION")
    print("-" * 50)

    print("✓ Final model loaded successfully")
    print("✓ Dataset loaded successfully")
    print("✓ Required features validated")
    print("✓ Categorical values validated")
    print("✓ Numerical values validated")
    print("✓ Prediction output range validated")
    print(
        f"✓ Successful student predictions: "
        f"{successful_predictions}/{len(student_profiles)}"
    )
    print("✓ Invalid input handling tested")
    print("✓ End-to-end prediction workflow completed")

    print("\n" + "=" * 70)
    print("DAY 11 FINAL PREDICTION SYSTEM TEST COMPLETED")
    print("=" * 70)


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()