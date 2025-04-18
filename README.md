# Allegheny Pool Inspection ML Analysis README

Group Members:

- Ardak Baizhaxynova (abaizhax)
- Manraj Dhillon (mdhillon)
- Lucas Huynh (lqh)

## Project Proposal and Final Write-up

Please click on the following links to reference the policy proposal(s) associated with this analysis:

- Proposal: https://docs.google.com/document/d/1V986u67-z3tzk2liVbc4zPo8jxzgcRBDaoD69nFuwno/edit?usp=sharing
- Final Write-up:

## Project Overview

This project analyzes public swimming pool, hot tub, and spa inspection records in Allegheny County. Our goal was to build a machine learning pipeline to predict whether a facility will pass inspection, identify key predictors, and support policy/inspection improvements through interpretable models.

## Files

- `inspection_data.csv`: Raw input data
- `ml_project_final.ipynb`: Final notebook including preprocessing, modeling, and interpretation with markdown comments.

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
- `json`
- `re`
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

## Preprocessing Pipeline

We implemented a data pipeline that includes:

- **Winsorization** of numeric outliers
- **Imputation**:
  - Median imputation for numeric columns
  - Mode imputation for categorical columns
- **One-hot encoding** of categorical variables
- **Feature scaling** using `StandardScaler`
- pH, Cyanuric Acid, and other chemical metrics were categorized for modeling.

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

## Best Model

While several models performed similarly, **Logistic Regression** provided:

- **Accuracy**: ~80%
- **Recall**: ~85%
- **F1 Score**: ~0.86
- **Good interpretability** and SHAP support

We selected logistic regression as our baseline and tuned it using `GridSearchCV`.

## Interpretation

### Logistic Coefficients
Showed that:
- `Inspection Purpose` and `Venue Type` were strong predictors
- Ideal chemical levels improved passing rates
- Regional variation was visible through `Facility Municipality`

### SHAP Values
Provided row-level interpretability for individual inspections.

## PCA & Variance

We applied PCA on scaled numeric features:
- Top 10 components explained **~70% of variance**
- Visualized explained variance using a scree plot

PCA did **not significantly improve model performance**, so we retained the full model.
