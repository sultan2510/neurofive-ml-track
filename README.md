# Neurofive ML Track — Titanic Survival Prediction

This repository contains my work for the Neurofive Solutions ML internship track, covering exploratory data analysis, data cleaning, visualization, and a first machine learning model using the Titanic dataset.

## Approach

1. **Exploratory Data Analysis (EDA)** — Loaded the Titanic dataset with pandas and inspected its structure using `.info()`, `.describe()`, and `.head()` to understand row/column counts, data types, and missing values.

2. **Data Cleaning** — Handled missing values based on how much data was missing:
   - `Age` (177 missing) — filled with the median, chosen for its resistance to outliers
   - `Embarked` (2 missing) — filled with the mode (most frequent port)
   - `Cabin` (687 missing, ~77%) — dropped entirely, since filling that much missing data would introduce more noise than signal

3. **Visualization** — Used matplotlib and seaborn to build:
   - A histogram of passenger age distribution
   - A boxplot to detect outliers in `Fare`
   - A bar chart comparing survival rate across passenger classes
   - A correlation heatmap across numerical features

4. **Modeling** — Built a binary classification model to predict passenger survival:
   - Encoded categorical features (`Sex`, `Embarked`) using `pd.get_dummies()`
   - Split the data using `train_test_split` (80% train / 20% test)
   - Trained a **Logistic Regression** model using scikit-learn
   - Evaluated performance using `accuracy_score` and a confusion matrix

## Results

- **Model accuracy:** 81.01%
- **Confusion matrix:**

|  | Predicted: Did Not Survive | Predicted: Survived |
|---|---|---|
| **Actual: Did Not Survive** | 90 | 15 |
| **Actual: Survived** | 19 | 55 |

The model performs slightly better at identifying non-survivors than survivors, with a relatively balanced number of false positives (15) and false negatives (19).

## Tech Stack

- Python 3.13
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn

## Files

- `titanic_eda.ipynb` — Full notebook: EDA, cleaning, visualization, and modeling
- `data/` — Titanic dataset (train.csv, test.csv, gender_submission.csv)