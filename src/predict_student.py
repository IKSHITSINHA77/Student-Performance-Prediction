import pandas as pd
import joblib

from pathlib import Path


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "student-mat.csv"
MODEL_PATH = BASE_DIR / "models" / "tuned_random_forest_model.joblib"


print("=" * 70)
print("STUDENT PERFORMANCE PREDICTION SYSTEM")
print("Day 7 - Student Prediction Pipeline")
print("=" * 70)


# ---------------------------------------------------------
# 1. Load trained model
# ---------------------------------------------------------
print("\n1. LOADING TRAINED MODEL")
print("-" * 50)

model = joblib.load(MODEL_PATH)

print("Tuned Random Forest model loaded successfully.")


# ---------------------------------------------------------
# 2. Load dataset
# ---------------------------------------------------------
df = pd.read_csv(
    DATA_PATH,
    sep=";"
)

print("\n2. DATASET LOADED")
print("-" * 50)

print(f"Dataset shape: {df.shape}")


# ---------------------------------------------------------
# 3. Prepare sample student
# ---------------------------------------------------------
# This sample follows the same feature structure
# used during model training.

sample_student = {
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
    "studytime": 3,
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
    "absences": 4,
    "G1": 15,
    "G2": 15
}


student_df = pd.DataFrame(
    [sample_student]
)


print("\n3. SAMPLE STUDENT")
print("-" * 50)

print(student_df.to_string(index=False))


# ---------------------------------------------------------
# 4. Generate prediction
# ---------------------------------------------------------
prediction = model.predict(
    student_df
)


predicted_g3 = prediction[0]


# ---------------------------------------------------------
# 5. Display prediction
# ---------------------------------------------------------
print("\n4. PREDICTION")
print("-" * 50)

print(
    f"Predicted final grade (G3): "
    f"{predicted_g3:.2f}"
)


# ---------------------------------------------------------
# 6. Performance interpretation
# ---------------------------------------------------------
print("\n5. PERFORMANCE INTERPRETATION")
print("-" * 50)

if predicted_g3 >= 16:
    performance = "Excellent"
elif predicted_g3 >= 14:
    performance = "Very Good"
elif predicted_g3 >= 10:
    performance = "Satisfactory"
elif predicted_g3 >= 8:
    performance = "Needs Improvement"
else:
    performance = "At Risk"


print(f"Predicted performance level: {performance}")


# ---------------------------------------------------------
# 7. Final summary
# ---------------------------------------------------------
print("\n" + "=" * 70)
print("DAY 7 COMPLETED")
print("=" * 70)

print("Trained model loaded.")
print("Raw student data prepared.")
print("Prediction generated successfully.")
print("Performance level calculated.")

print("=" * 70)