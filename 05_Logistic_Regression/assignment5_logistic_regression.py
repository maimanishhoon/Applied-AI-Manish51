# Assignment 5: Logistic Regression (Diabetes Prediction)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, classification_report

# STEP 1: Load Dataset
# (Using breast cancer dataset as substitute for diabetes)
data = load_breast_cancer()

X = data.data
y = data.target

# STEP 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# STEP 3: Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 4: Logistic Regression Models

# No regularization (approx using very high C)
model_no_reg = LogisticRegression(C=1e6, max_iter=1000)
model_no_reg.fit(X_train, y_train)

# L1 Regularization
model_l1 = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)
model_l1.fit(X_train, y_train)

# L2 Regularization
model_l2 = LogisticRegression(penalty='l2', C=1.0)
model_l2.fit(X_train, y_train)

# STEP 5: Predictions
y_pred = model_l2.predict(X_test)
y_prob = model_l2.predict_proba(X_test)[:,1]

# STEP 6: ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# STEP 7: Evaluation Metrics
print("\nClassification Report (L2):\n")
print(classification_report(y_test, y_pred))

# STEP 8: Cross Validation for C
C_values = [0.01, 0.1, 1, 10, 100]
scores = []

for c in C_values:
    model = LogisticRegression(C=c, max_iter=1000)
    score = cross_val_score(model, X_train, y_train, cv=5).mean()
    scores.append(score)

best_C = C_values[np.argmax(scores)]

print("\nBest C value:", best_C)
