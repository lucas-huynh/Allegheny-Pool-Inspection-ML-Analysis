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

## 📂 Files

- `inspection_data.csv`: Raw input data
- `ml_project_final.ipynb`: Final notebook including preprocessing, modeling, and interpretation

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

---

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

