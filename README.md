# Allegheny Pool Inspection ML Analysis README

Group Members:

- Ardak Baizhaxynova (abaizhax)
- Manraj Dhillon (mdhillon)
- Lucas Huynh (lqh)

## Project Proposal and Final Write-up

Please click on the following links to reference the policy proposal(s) associated with this analysis:

- Proposal: https://docs.google.com/document/d/1V986u67-z3tzk2liVbc4zPo8jxzgcRBDaoD69nFuwno/edit?usp=sharing
- Final Write-up: https://docs.google.com/document/d/1NVl9vSHRlnQ-cVoN3nf1d0w89aQBouSEMJuGn67lvDI/edit?usp=sharing

## Project Overview

This project analyzes public swimming pool, hot tub, and spa inspection records in Allegheny County. Our goal was to build a machine learning pipeline to predict whether a facility will pass inspection, identify key predictors, and support policy/inspection improvements through interpretable models.

## Files

- `inspection_data.csv`: Raw input data
- `ml_project_final.ipynb`: Final notebook including preprocessing, modeling, and interpretation with markdown comments.

## Note
More descriptive interpretations are included in the notebook. Descriptions here are merely meant to summarize and provide a broad
overview.

## Prerequisites

Before running the project, make sure you have:

- Anaconda or Python 3.8+
- Your working directory set to the location of the notebook

## Installation of Required Modules

This project requires the following Python modules:

- **pandas**
- **numpy**
- **matplotlib**
- **seaborn**
- **scikit-learn** (includes: `pipeline`, `model_selection`, `metrics`, `preprocessing`, `decomposition`, etc.)
- **xgboost**
- **lightgbm**
- **shap**

> **Note**: The following modules are part of the Python standard library and do **not** require separate installation:
- `csv`
- `time`
- `math`

### To install the necessary packages:

1. **Open Anaconda Prompt (or terminal)**  
2. **Activate your conda environment** (if applicable):

```bash
conda activate your_environment_name
```

Install required packages using the following commands:
- pip install pandas
- pip install numpy
- pip install matplotlib
- pip install seaborn
- pip install scikit-learn
- pip install xgboost
- pip install lightgbm
- pip install shap

## How to Run

After installing dependencies and navigating to the project directory, launch:

```bash
jupyter notebook ml_project_final.ipynb
```

Run each cell sequentially to load, clean, model, and interpret the inspection dataset.

## Column Descriptions

| Column Name                          | Description                                                                 |
|-------------------------------------|-----------------------------------------------------------------------------|
| Inspection ID                       | Unique identifier for each inspection                                      |
| Facility ID                         | Unique identifier for each facility                                        |
| Facility Name                       | Name of the facility (e.g., pool or spa)                                   |
| Facility Address                    | Full address of the facility                                               |
| Facility Municipality Name          | Name of the township/municipality                                          |
| Facility City                       | City name                                                                  |
| Facility Postal Code                | ZIP code                                                                   |
| Venue Type                          | Type of facility (e.g., Pool, Spa)                                         |
| Inspection Date                     | Date of the inspection                                                     |
| Inspection Purpose                  | Reason for inspection (e.g., Routine, Complaint)                           |
| Inspection Passed                   | Whether the inspection was passed (True/False)                             |
| Inspection Number                   | Internal inspection reference number                                       |
| Inspector Name                      | Name of the inspector                                                      |
| Free Chlorine, Shallow/Deep         | Chlorine levels at shallow and deep ends                                   |
| Combined Chlorine, Shallow/Deep     | Combined chlorine measurements                                             |
| Free Bromine, Shallow/Deep          | Bromine levels (if used instead of chlorine)                               |
| Turnover                            | Time (in minutes) required to filter all the water                         |
| Main Drain Visible                  | Whether the main drain was visible at inspection (True/False)              |
| Safety Equipment                    | Whether safety equipment was present                                       |
| pH Balance                          | Indicates if pH was in balance                                             |
| No Imminent Health Hazards          | True if no immediate risks were found                                      |
| Inspection Year/Month               | Extracted date parts for modeling                                          |
| pH Value, Shallow/Deep (Cleaned)    | Cleaned numeric values for pH                                              |
| pH Value, Shallow/Deep (Category)   | Categorized version (e.g., Acidic, Ideal, Basic)                           |
| Cyanuric Acid (Cleaned/Category)    | Chlorine stabilizer in cleaned numeric and categorical form                |
| Free Bromine (Exists)               | Boolean indicating if bromine level was present                            |

**Excluded to prevent data leakage**:
- `Main Drain Visible`
- `Safety Equipment`
- `pH Balance`
- `No Imminent Health Hazards`

## Preprocessing Pipeline

We implemented a data pipeline that includes:

- **Winsorization** of numeric outliers
- **Imputation**:
  - Median imputation for numeric columns
  - Mode imputation for categorical columns
- **Feature scaling** using `StandardScaler`
- Categorized chemical metrics:
  - pH: Ideal, Acidic, Basic
  - Cyanuric Acid levels
- **One-hot encoding** of categorical variables

The pipeline was wrapped with `Pipeline` and `ColumnTransformer` to avoid data leakage.

## Model Training

We tested the following classification models:

- Logistic Regression
- Random Forest
- Extra Trees
- Gradient Boosting
- XGBoost
- LightGBM
- K-Nearest Neighbors
- Naive Bayes
- SVM

Models were evaluated using:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- ROC AUC
- Confusion Matrix

## Model Comparison

| Model              | Accuracy | Precision | Recall | F1 Score |
|-------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.800  | 0.860     | 0.852  | 0.856    |
| Random Forest      | 0.802  | 0.861     | 0.857  | 0.859    |
| Extra Trees        | 0.802  | 0.857     | 0.861  | 0.859    |
| XGBoost            | 0.800  | 0.867     | 0.845  | 0.856    |
| Gradient Boosting  | 0.788  | 0.858     | 0.836  | 0.847    |
| SVM                | 0.784  | 0.855     | 0.834  | 0.845    |
| LightGBM           | 0.768  | 0.838     | 0.832  | 0.835    |
| KNN                | 0.756  | 0.834     | 0.814  | 0.824    |
| Naive Bayes        | 0.690  | 0.962     | 0.582  | 0.725    |

**Conclusion**: Logistic Regression was selected as the final model due to competitive performance and strong interpretability.

## Best Model

While several models performed similarly, **Logistic Regression** provided:

- **Accuracy**: ~80%
- **Recall**: ~85%
- **F1 Score**: ~0.86
- **Good interpretability** and SHAP support

We selected logistic regression as our baseline and tuned it using `GridSearchCV`.

## Logistic Regression Tuning

Used `GridSearchCV` with 5-fold stratified cross-validation to optimize:

- `C` (regularization strength)
- `penalty`
- `solver`

**Best Parameters**:
- `C=1`
- `penalty='l2'`
- `solver='lbfgs'`

**Best F1 Macro Score**: 0.7302

## Interpretation

### Logistic Coefficients
Showed that:
- Help identify globally impactful features
- `Inspection Purpose` and `Venue Type` were strong predictors
- Ideal chemical levels improved passing rates
- Regional variation was visible through `Facility Municipality`

### SHAP Values
- Provided row-level interpretability for individual inspections
- Highlight nonlinear impact of chlorine, pH, and turnover

## PCA & Variance

We applied PCA on scaled numeric features:
- Top 10 components explained **~70% of variance**
- Visualized explained variance using a scree plot

PCA did **not significantly improve model performance**, so we retained the full model.

## Summary

- Logistic regression provided competitive metrics across all evaluation dimensions
- Outperformed by ensembles only slightly, but offers much better transparency
- Final model supports interpretability + accuracy for public health and policy applications
