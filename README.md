# Neurofive ML Track — Titanic Survival Prediction & Housing Price Regression

This repository contains my work for the Neurofive Solutions ML internship track, covering exploratory data analysis, data cleaning, visualization, classification, regression, and model evaluation/tuning.

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

4. **Classification Modeling** — Built a binary classification model to predict passenger survival:
   - Encoded categorical features (`Sex`, `Embarked`) using `pd.get_dummies()`
   - Split the data using `train_test_split` (80% train / 20% test)
   - Trained a **Logistic Regression** model using scikit-learn
   - Evaluated performance using `accuracy_score` and a confusion matrix

5. **Model Evaluation & Hyperparameter Tuning** — Went beyond accuracy to evaluate the classification model properly:
   - Calculated precision, recall, and F1-score using `classification_report`, since accuracy alone can be misleading on imbalanced datasets — a model could score high accuracy simply by always predicting the majority class, without correctly identifying the minority class (survivors)
   - Tuned two hyperparameters — `C` (regularization strength) and `solver` (optimization algorithm) — using `GridSearchCV` with 3-fold cross-validation across `C: [0.1, 1, 10]`

6. **Regression Modeling** (separate notebook, California Housing dataset) — Built a regression model to predict median house value:
   - Selected 4 features: `MedInc`, `AveRooms`, `HouseAge`, `AveOccup`
   - Split the data using `train_test_split` (80% train / 20% test)
   - Trained a **Linear Regression** model using scikit-learn
   - Evaluated performance using RMSE and R² score

## Results — Classification (Titanic)

- **Model accuracy:** 81.01%
- Correctly predicted 90 non-survivors and 55 survivors
- 15 false positives (predicted survived, but didn't)
- 19 false negatives (predicted did not survive, but did)
- The model performs slightly better at identifying non-survivors than survivors, with a relatively balanced number of false positives and false negatives

## Results — Hyperparameter Tuning (Titanic)

- **Best hyperparameters found:** `C=0.1`, `solver=liblinear`
- Original model accuracy: 81.01%
- Tuned model accuracy: 78.21%
- Tuned model precision (Survived): 0.76
- Tuned model recall (Survived): 0.69
- Tuned model F1-score (Survived): 0.72
- The tuned model performed slightly worse on the test set than the original default model. This highlights that hyperparameter tuning optimizes for average performance across cross-validation folds, which doesn't always guarantee improvement on a specific held-out test set — especially with a relatively small dataset (891 rows total). Tuned models should always be validated against a held-out test set rather than assumed to automatically improve results.

## Results — Regression (California Housing)

- **RMSE:** 0.8108 (≈ $81,000 average prediction error)
- **R² Score:** 0.4983 (model explains ~50% of price variation)
- The model captures a real relationship between income, home size, and price, but roughly half of the price variation remains unexplained by these 4 features alone — likely driven by unmodeled factors like exact location, school quality, and property condition

## Tech Stack

- Python 3.13
- pandas, NumPy
- matplotlib, seaborn
- scikit-learn

## Files

- `titanic_eda.ipynb` — EDA, cleaning, visualization, classification, and model tuning on the Titanic dataset
- `housing_regression.ipynb` — Regression model on the California Housing dataset
- `data/` — Titanic dataset (train.csv, test.csv, gender_submission.csv)
## Task 7: Handling Class Imbalance (Telco Churn Dataset)

Real-world classification problems like churn, fraud, or disease 
detection are rarely balanced — the "rare" outcome is usually far less 
common than the "normal" one. This task focused on identifying that 
imbalance and properly addressing it instead of letting it silently 
hurt the model's usefulness.

- Checked class balance of the target variable and visualized it with a bar chart
- Found the dataset was imbalanced: 5,174 customers did not churn vs 1,869 who did (~26.5% churn rate)
- Applied two techniques to address the imbalance: `class_weight='balanced'` and SMOTE (Synthetic Minority Oversampling)
- Retrained the model with each approach and compared precision, recall, and F1-score before and after

### Why Accuracy Would Be Misleading Here

With only ~27% of customers actually churning, a model that always 
predicted "No Churn" would still score around 73% accuracy — without 
ever correctly identifying a single customer who churned. This makes 
accuracy a poor measure of real performance, since it hides how well 
the model detects the minority class. Precision, recall, and F1-score 
reveal this gap because they specifically measure performance on each 
class individually, not just overall correctness.

### Before vs After: Handling Imbalance

- Precision (Churn): 0.66 → 0.51 (class_weight) → 0.50 (SMOTE)
- Recall (Churn): 0.57 → 0.79 (class_weight) → 0.80 (SMOTE)
- F1-score (Churn): 0.61 → 0.62 (class_weight) → 0.62 (SMOTE)

Both techniques significantly improved recall — the model went from 
catching only 57% of actual churners to catching around 80% of them. 
This came at the cost of precision, meaning the model now flags more 
false alarms. However, this trade-off is usually worthwhile in a churn 
prevention context: failing to identify an at-risk customer (a missed 
churner) is typically more costly to a business than occasionally 
offering a retention incentive to a customer who wasn't actually going 
to leave. F1-score stayed roughly the same, showing the overall balance 
was preserved while shifting the model's behavior toward catching more 
of the minority class — exactly the intended effect of addressing 
class imbalance.