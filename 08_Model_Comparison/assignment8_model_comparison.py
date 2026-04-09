# Assignment 8: Model Comparison

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# STEP 1: Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# STEP 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# STEP 3: Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 4: Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "SVM (RBF)": SVC(probability=True),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}

# STEP 5: Training + Evaluation
for name, model in models.items():
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:,1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    results[name] = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "AUC": roc_auc,
        "FPR": fpr,
        "TPR": tpr
    }

# STEP 6: ROC Curve Plot
plt.figure(figsize=(6,5))

for name, res in results.items():
    plt.plot(res["FPR"], res["TPR"], label=f"{name} (AUC={res['AUC']:.2f})")

plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.show()

# STEP 7: Cross Validation
cv = StratifiedKFold(n_splits=5)

for name, model in models.items():
    scores = cross_val_score(model, X_train, y_train, cv=cv)
    print(f"{name} CV Accuracy: {scores.mean():.3f}")

# STEP 8: Final Results Table
df_results = pd.DataFrame(results).T
print("\nFinal Comparison:\n")
print(df_results[["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]])
