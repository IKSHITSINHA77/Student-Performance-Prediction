# Student Performance Prediction System

A machine learning project that predicts a student's final mathematics grade (`G3`) using academic, demographic, social, and behavioral information from the UCI Student Performance dataset.

## Project Objective

The objective of this project is to build an end-to-end machine learning workflow for student performance prediction.

The implemented workflow is:

**Data -> Cleaning -> Analysis -> Feature Selection -> Preprocessing -> Training -> Testing -> Evaluation -> Prediction**

The project includes exploratory data analysis, preprocessing, multiple regression models, hyperparameter tuning, model evaluation, and a reusable prediction workflow.

---

## Problem Statement

Student academic performance can be influenced by several academic, demographic, social, and behavioral factors.

This project uses historical student information to estimate the final mathematics grade (`G3`) and demonstrates how machine learning regression techniques can be applied to student performance analysis.

---

## Dataset

The project uses the **UCI Student Performance Dataset**, specifically the mathematics dataset:

`student-mat.csv`

Dataset characteristics:

- **395 student records**
- **33 columns**
- Target variable: `G3`
- Delimiter: semicolon (`;`)
- No missing values
- No duplicate records

The dataset contains information related to:

- Student demographics
- Family background
- Previous academic performance
- Study habits
- Social activities
- Absences
- Previous failures
- School-related information

---

## Target Variable

The target variable is:

`G3`

`G3` represents the student's final mathematics grade on a scale from **0 to 20**.

---

## Important Features

Examples of features used by the model include:

- `school`
- `sex`
- `age`
- `address`
- `famsize`
- `Pstatus`
- `Medu`
- `Fedu`
- `Mjob`
- `Fjob`
- `studytime`
- `failures`
- `schoolsup`
- `famsup`
- `activities`
- `higher`
- `internet`
- `romantic`
- `famrel`
- `freetime`
- `goout`
- `Dalc`
- `Walc`
- `health`
- `absences`
- `G1`
- `G2`

---

## Exploratory Data Analysis

The project performs exploratory analysis to understand relationships between student characteristics and final grades.

Important observations from the analysis include:

- `G2` has a strong positive relationship with `G3`.
- `G1` also has a strong positive relationship with `G3`.
- Students with more previous failures generally have lower final grades.
- Study time shows a positive but comparatively weaker relationship with final grade.
- Absences show a relatively weak linear relationship with `G3`.

The project generates visualizations for:

- Final grade distribution
- Study time vs final grade
- Failures vs final grade
- Absences vs final grade
- G1 vs G3
- G2 vs G3
- Correlation heatmap
- Model comparison

---

## Data Preprocessing

The preprocessing pipeline separates features into:

### Numerical Features

Numerical features are passed through directly.

### Categorical Features

Categorical features are transformed using:

`OneHotEncoder(handle_unknown="ignore")`

A `ColumnTransformer` combines the numerical and categorical preprocessing steps.

The dataset is divided using:

- **80% training data**
- **20% testing data**
- `random_state=42`

---

## Machine Learning Models

The project evaluates the following regression approaches:

### 1. Linear Regression

A baseline linear regression model is implemented using a complete preprocessing and modeling pipeline.

### 2. Decision Tree Regression

The initial Decision Tree configuration uses:

- `max_depth=5`
- `min_samples_split=5`
- `random_state=42`

### 3. Random Forest Regression

The initial Random Forest configuration uses:

- `n_estimators=200`
- `max_depth=10`
- `min_samples_split=4`
- `random_state=42`
- `n_jobs=-1`

### 4. Tuned Decision Tree

Hyperparameters were selected using GridSearchCV with 5-fold cross-validation.

Best configuration:

- `max_depth=7`
- `min_samples_leaf=4`
- `min_samples_split=2`

### 5. Tuned Random Forest

Hyperparameters were selected using GridSearchCV with 5-fold cross-validation.

Best configuration:

- `n_estimators=200`
- `max_depth=10`
- `min_samples_leaf=2`
- `min_samples_split=2`

---

## Evaluation Metrics

The models are evaluated using:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted grades.

### MSE

Mean Squared Error calculates the average squared prediction error.

### RMSE

Root Mean Squared Error is the square root of MSE and represents prediction error on the same scale as the target.

### RÂ² Score

RÂ² measures how much of the variation in the target variable is explained by the model.

---

## Model Results

The final test-set comparison is:

| Model | MAE | MSE | RMSE | RÂ² |
|---|---:|---:|---:|---:|
| Decision Tree | 1.4558 | 7.0146 | 2.6485 | 0.6579 |
| Random Forest | 1.1646 | 3.8543 | 1.9632 | 0.8120 |
| Tuned Decision Tree | 1.3450 | 4.7375 | 2.1766 | 0.7690 |
| Tuned Random Forest | 1.1946 | 4.1370 | 2.0340 | 0.7982 |

The project records these results in:

`reports/final_model_results.csv`

### Model Selection Note

Hyperparameter tuning was performed using cross-validation to provide a more systematic model-selection process.

The tuned Random Forest did **not** produce a lower test-set RMSE than the original Random Forest on the fixed test split. Therefore, the project does not claim that tuning improved the final held-out test performance.

---

## Prediction Workflow

The project contains a reusable prediction script:

`src/predict_student.py`

The prediction workflow:

1. Loads the trained model.
2. Validates the required input features.
3. Accepts student profile information.
4. Generates a predicted final grade.
5. Checks that the prediction falls within the expected 0â€“20 range.
6. Handles invalid or incomplete input.

Example prediction profiles were tested during the project, including:

- High Performance
- Average Performance
- Needs Improvement

---

## Project Structure

```text
Student-Performance-Prediction/
â”‚
â”œâ”€â”€ data/
â”‚   â””â”€â”€ student-mat.csv
â”‚
â”œâ”€â”€ models/
â”‚   â”œâ”€â”€ preprocessor.joblib
â”‚   â”œâ”€â”€ linear_regression_model.joblib
â”‚   â”œâ”€â”€ decision_tree_model.joblib
â”‚   â”œâ”€â”€ random_forest_model.joblib
â”‚   â”œâ”€â”€ tuned_decision_tree_model.joblib
â”‚   â””â”€â”€ tuned_random_forest_model.joblib
â”‚
â”œâ”€â”€ notebooks/
â”‚   â””â”€â”€ student_performance_prediction.ipynb
â”‚
â”œâ”€â”€ reports/
â”‚   â”œâ”€â”€ day2_eda_findings.txt
â”‚   â”œâ”€â”€ day5_model_comparison.csv
â”‚   â”œâ”€â”€ day6_tuned_model_results.csv
â”‚   â””â”€â”€ final_model_results.csv
â”‚
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ data_inspection.py
â”‚   â”œâ”€â”€ eda_analysis.py
â”‚   â”œâ”€â”€ preprocess_data.py
â”‚   â”œâ”€â”€ train_linear_regression.py
â”‚   â”œâ”€â”€ train_tree_models.py
â”‚   â”œâ”€â”€ tune_tree_models.py
â”‚   â”œâ”€â”€ predict_student.py
â”‚   â””â”€â”€ final_model_comparison.py
â”‚
â”œâ”€â”€ visualizations/
â”‚   â”œâ”€â”€ absences_vs_g3.png
â”‚   â”œâ”€â”€ correlation_heatmap.png
â”‚   â”œâ”€â”€ failures_vs_g3.png
â”‚   â”œâ”€â”€ g1_vs_g3.png
â”‚   â”œâ”€â”€ g2_vs_g3.png
â”‚   â”œâ”€â”€ g3_distribution.png
â”‚   â”œâ”€â”€ studytime_vs_g3.png
â”‚   â”œâ”€â”€ linear_regression_actual_vs_predicted.png
â”‚   â”œâ”€â”€ tree_models_comparison.png
â”‚   â”œâ”€â”€ day6_tuned_model_rmse.png
â”‚   â”œâ”€â”€ final_model_rmse_comparison.png
â”‚   â””â”€â”€ final_model_r2_comparison.png
â”‚
â”œâ”€â”€ .gitignore
â”œâ”€â”€ requirements.txt
â””â”€â”€ README.md
Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Joblib
Jupyter Notebook
Git
GitHub
Installation

Clone the repository and enter the project directory:

git clone https://github.com/IKSHITSINHA77/Student-Performance-Prediction.git
cd Student-Performance-Prediction

Create and activate a virtual environment:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

Install dependencies:

pip install -r requirements.txt
Running the Project
Dataset Inspection
python .\src\data_inspection.py
Exploratory Data Analysis
python .\src\eda_analysis.py
Preprocessing
python .\src\preprocess_data.py
Linear Regression
python .\src\train_linear_regression.py
Tree Models
python .\src\train_tree_models.py
Hyperparameter Tuning
python .\src\tune_tree_models.py
Prediction Workflow
python .\src\predict_student.py
Final Model Comparison
python .\src\final_model_comparison.py
Jupyter Notebook

The complete project workflow is also documented in:

notebooks/student_performance_prediction.ipynb

The notebook covers:

Dataset loading
Dataset understanding
Data cleaning
Exploratory data analysis
Feature selection
Preprocessing
Train/test split
Model training
Model comparison
Evaluation
Prediction
Conclusion
Important Modeling Consideration

The dataset contains G1 and G2, which represent earlier-period grades.

Because these variables are strongly related to the final grade G3, including them makes the project a prediction of final performance using prior academic performance information.

Therefore, this system should not be interpreted as predicting a student's final grade before any previous grades are available.

A future version could evaluate a separate feature set that excludes G1 and G2 for an earlier-stage prediction scenario.

Future Improvements

Potential improvements include:

Testing additional regression algorithms.
Performing broader hyperparameter optimization.
Comparing models using repeated cross-validation.
Adding feature importance analysis.
Building an interactive web interface.
Adding a REST API for predictions.
Deploying the prediction system to the cloud.
Evaluating a prediction scenario that excludes previous-period grades.
Adding additional datasets for broader validation.
Project Status

The project includes:

Data analysis
Data preprocessing
Exploratory analysis
Multiple regression models
Hyperparameter tuning
Model evaluation
Prediction validation
Final model comparison
Visualizations
Jupyter notebook documentation

The project is being developed as part of an AI/ML internship project.
