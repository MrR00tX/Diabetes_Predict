import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# =====================
# 1. Data Loading 
# =====================
df = pd.read_csv("diabetes.csv")
df.head(5)
df.shape
# =====================
#  Data Preprocessing 
# =====================

print( df.isnull().sum())

cols_to_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_to_zero:
    df[col] = df[col].replace(0, df[col].median())

print(df[cols_to_zero].describe())


X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,stratify=y
)

numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
X = X[numeric_cols]

X = X.copy()
Q1 = pd.DataFrame(X_train).quantile(0.25)
Q3 = pd.DataFrame(X_train).quantile(0.75)
IQR = Q3 - Q1
outliers = ((pd.DataFrame(X_train) < (Q1 - 1.5 * IQR)) | (pd.DataFrame(X_train) > (Q3 + 1.5 * IQR))).sum()
print(outliers)

# =====================
#  Pipeline creation
# =====================



pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=1000,class_weight="balanced"))
])

# =====================
# Primary Model Selection 
# i am using Logistic Regression because it is suitable for binary classification and works well for numeric datasets.
# =====================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

# =====================
# Model Training 
# =====================
pipeline.fit(X_train, y_train)


# =====================
# Cross-Validation 
# =====================
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="accuracy")

print("CV Accuracy Scores:", cv_scores)
print("Avg Accuracy:", cv_scores.mean())
print("Standard Dev:", cv_scores.std())

# =====================
#  Hyperparameter Tuning 
# =====================


param_grid = {
    "model__C": [0.01, 0.1, 1, 10, 100],
    "model__solver": ["liblinear", "lbfgs"],
    "model__penalty": ["l2"]
}

grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("Parameters Tested:", param_grid)
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)
# =====================
# Best Model Selection 
# =====================
best_model = grid_search.best_estimator_
print(best_model)


y_pred = best_model.predict(X_test)



# =====================
# Model Performance Evaluation 
# =====================
# Accuracy, Precision, Recall, F1
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

y_pred = best_model.predict(X_test)


acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
# Full Classification Report
report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:\n", report)

# =====================
# save model
# =====================
with open ("diabeties_pred.pkl","wb") as f:
    pickle.dump(pipeline,f)
print("saved the pipeline")