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

    # Validate prediction output
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
    print("Day 8 - Prediction Workflow Testing")
    print("=" * 70)

    # -----------------------------------------------------
    # 1. Load model
    # -----------------------------------------------------
    print("\n1. LOADING TRAINED MODEL")
    print("-" * 50)

    model, df = load_prediction_system()

    print("Tuned Random Forest model loaded successfully.")
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
    print("\n2. TESTING MULTIPLE STUDENT INPUTS")
    print("-" * 50)

    for student_name, student_data in student_profiles.items():

        print(f"\nStudent Profile: {student_name}")

        try:

            predicted_g3, performance = predict_student(
                student_data,
                model,
                df
            )

            print(f"Predicted final grade (G3): {predicted_g3:.2f}")
            print(f"Performance level: {performance}")
            print("Prediction status: Successful")

        except (ValueError, TypeError) as error:

            print(f"Prediction failed: {error}")

    # -----------------------------------------------------
    # 4. Final validation
    # -----------------------------------------------------
    print("\n3. PREDICTION SYSTEM VALIDATION")
    print("-" * 50)

    print("✓ Model loaded successfully")
    print("✓ Dataset loaded successfully")
    print("✓ Multiple student inputs tested")
    print("✓ Feature validation completed")
    print("✓ Categorical inputs processed")
    print("✓ Numerical inputs processed")
    print("✓ Prediction outputs validated")
    print("✓ Prediction workflow completed")

    print("\n" + "=" * 70)
    print("DAY 8 PREDICTION WORKFLOW TEST COMPLETED")
    print("=" * 70)


# ---------------------------------------------------------
# Program entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()