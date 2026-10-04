# Student Performance Prediction System

A machine learning project that predicts a student's final mathematics grade (`G3`) using academic, demographic, social, and behavioral information from the UCI Student Performance dataset.

## Project Objective

The objective of this project is to build an end-to-end machine learning workflow for student performance prediction.

The implemented workflow is:

**Data -> Cleaning -> Analysis -> Feature Selection -> Preprocessing -> Training -> Testing -> Evaluation -> Prediction**

The project includes exploratory data analysis, preprocessing, multiple regression models, hyperparameter tuning, model evaluation, final model comparison, and a reusable prediction workflow.


## Problem Statement

Student academic performance can be influenced by several academic, demographic, social, and behavioral factors.

This project uses historical student information to estimate the final mathematics grade (`G3`) and demonstrates how machine learning regression techniques can be applied to student performance analysis.


## Dataset

The project uses the **UCI Student Performance Dataset**, specifically the mathematics dataset:

`student-mat.csv`

### Dataset Characteristics

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


## Target Variable

The target variable is:

`G3`

`G3` represents the student's final mathematics grade on a scale from **0 to 20**.


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


## Exploratory Data Analysis

The project performs exploratory analysis to understand relationships between student characteristics and final grades.

Important observations include:

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


## Data Preprocessing

The preprocessing pipeline separates features into numerical and categorical features.

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

The preprocessing and model stages are combined into scikit-learn pipelines to maintain a consistent transformation and prediction workflow.


## Machine Learning Models

The project evaluates the following regression approaches.

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


## Evaluation Metrics

The models are evaluated using:

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted grades.

### MSE

Mean Squared Error calculates the average squared prediction error.

### RMSE

Root Mean Squared Error is the square root of MSE and represents prediction error on the same scale as the target.

### R2 Score

R2 measures the proportion of variation in the target variable explained by the model.

---

## Model Results

The final held-out test-set comparison is:

| Model | MAE | MSE | RMSE | R2 |
|---|---:|---:|---:|---:|
| Decision Tree | 1.4558 | 7.0146 | 2.6485 | 0.6579 |
| Random Forest | 1.1646 | 3.8543 | 1.9632 | 0.8120 |
| Tuned Decision Tree | 1.3450 | 4.7375 | 2.1766 | 0.7690 |
| Tuned Random Forest | 1.1946 | 4.1370 | 2.0340 | 0.7982 |

The complete results are stored in:

`reports/final_model_results.csv`

### Model Selection Note

Hyperparameter tuning was performed using 5-fold cross-validation to provide a systematic model-selection process.

On the fixed held-out test split used in this project, the original Random Forest produced the lowest RMSE and highest R2 among the four reported models.

The tuned Random Forest did not produce a lower test-set RMSE than the original Random Forest. Therefore, the project does not claim that hyperparameter tuning improved the final held-out test performance.



## Prediction Workflow

The project contains a reusable prediction script:

`src/predict_student.py`

The prediction workflow:

1. Loads the trained prediction pipeline.
2. Loads the dataset used to define the expected feature structure.
3. Validates required input features.
4. Validates categorical values.
5. Validates numerical values and their observed training-data ranges.
6. Generates a predicted final grade.
7. Validates that the prediction is within the expected 0-20 range.
8. Converts the prediction into a simple performance category.
9. Handles invalid or incomplete input.

### Tested Student Scenarios

Three example profiles were tested:

| Student Profile | Predicted G3 | Performance Level |
|---|---:|---|
| High Performance Student | 17.97 | Excellent |
| Average Performance Student | 10.66 | Satisfactory |
| Needs Improvement Student | 7.80 | At Risk |

The validation workflow successfully produced predictions for **3/3 test profiles** and correctly rejected an invalid numerical input.



## Project Structure

```text
Student-Performance-Prediction/
|
|-- data/
|   `-- student-mat.csv
|
|-- models/
|   |-- preprocessor.joblib
|   |-- linear_regression_model.joblib
|   |-- decision_tree_model.joblib
|   |-- random_forest_model.joblib
|   |-- tuned_decision_tree_model.joblib
|   `-- tuned_random_forest_model.joblib
|
|-- notebooks/
|   `-- student_performance_prediction.ipynb
|
|-- reports/
|   |-- day2_eda_findings.txt
|   |-- day5_model_comparison.csv
|   |-- day6_tuned_model_results.csv
|   `-- final_model_results.csv
|
|-- src/
|   |-- data_inspection.py
|   |-- eda_analysis.py
|   |-- preprocess_data.py
|   |-- train_linear_regression.py
|   |-- train_tree_models.py
|   |-- tune_tree_models.py
|   |-- predict_student.py
|   `-- final_model_comparison.py
|
|-- visualizations/
|   |-- absences_vs_g3.png
|   |-- correlation_heatmap.png
|   |-- failures_vs_g3.png
|   |-- g1_vs_g3.png
|   |-- g2_vs_g3.png
|   |-- g3_distribution.png
|   |-- studytime_vs_g3.png
|   |-- linear_regression_actual_vs_predicted.png
|   |-- tree_models_comparison.png
|   |-- day6_tuned_model_rmse.png
|   |-- final_model_rmse_comparison.png
|   `-- final_model_r2_comparison.png
|
|-- .gitignore
|-- requirements.txt
`-- README.md
```

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Jupyter Notebook
- Git
- GitHub

---

## Installation

Clone the repository and enter the project directory:

```powershell
git clone https://github.com/IKSHITSINHA77/Student-Performance-Prediction.git
cd Student-Performance-Prediction
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

---

## Running the Project

### Dataset Inspection

```powershell
python .\src\data_inspection.py
```

### Exploratory Data Analysis

```powershell
python .\src\eda_analysis.py
```

### Preprocessing

```powershell
python .\src\preprocess_data.py
```

### Linear Regression

```powershell
python .\src\train_linear_regression.py
```

### Tree Models

```powershell
python .\src\train_tree_models.py
```

### Hyperparameter Tuning

```powershell
python .\src\tune_tree_models.py
```

### Prediction Workflow

```powershell
python .\src\predict_student.py
```

### Final Model Comparison

```powershell
python .\src\final_model_comparison.py
```

---

## Jupyter Notebook

The complete project workflow is also documented in:


`notebooks/student_performance_prediction.ipynb`

The notebook covers:

- Dataset loading
- Dataset understanding
- Data cleaning
- Exploratory data analysis
- Feature selection
- Preprocessing
- Train/test split
- Model training
- Model comparison
- Evaluation
- Prediction
- Conclusion

The notebook was validated successfully using nbformat.

---

## Important Modeling Consideration

The dataset contains G1 and G2, which represent earlier-period grades.

Because these variables are strongly related to the final grade G3, including them makes the project a prediction of final performance using prior academic performance information.

Therefore, this system should not be interpreted as predicting a student's final grade before previous grades are available.

A future version could evaluate a separate feature set that excludes G1 and G2 for an earlier-stage prediction scenario.

---

## Future Improvements

Potential improvements include:

- Testing additional regression algorithms.
- Performing broader hyperparameter optimization.
- Comparing models using repeated cross-validation.
- Adding feature importance analysis.
- Building an interactive web interface.
- Adding a REST API for predictions.
- Deploying the prediction system to the cloud.
- Evaluating a prediction scenario that excludes previous-period grades.
- Adding additional datasets for broader validation.

---

## Project Status

The project includes:

- Data analysis
- Data preprocessing
- Exploratory analysis
- Multiple regression models
- Hyperparameter tuning
- Model evaluation
- Prediction validation
- Final model comparison
- Visualizations
- Jupyter notebook documentation
- Repository cleanup and validation

The project was developed as part of an AI/ML internship project and has completed the implementation and validation stages through Day 14.
