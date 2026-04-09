# Assignment 7: Random Forest Classifier

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier

# STEP 1: Create Dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# STEP 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 3: Train Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,
    random_state=42
)

rf.fit(X_train, y_train)

# STEP 4: Evaluate Model
train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print("Training Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)
print("OOB Score:", rf.oob_score_)

# STEP 5: Feature Importance
importances = rf.feature_importances_

plt.figure(figsize=(6,4))
plt.bar(range(len(importances)), importances)
plt.title("Feature Importances")
plt.xlabel("Feature Index")
plt.ylabel("Importance")
plt.show()

# STEP 6: Compare n_estimators
estimators = [10, 50, 100, 200]
scores = []

for n in estimators:
    model = RandomForestClassifier(n_estimators=n, random_state=42)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

plt.figure(figsize=(6,4))
plt.plot(estimators, scores, marker='o')
plt.title("Performance vs Number of Trees")
plt.xlabel("n_estimators")
plt.ylabel("Accuracy")
plt.show()

# STEP 7: Learning Curve
train_sizes, train_scores, test_scores = learning_curve(
    rf, X, y, cv=5
)

train_mean = train_scores.mean(axis=1)
test_mean = test_scores.mean(axis=1)

plt.figure(figsize=(6,4))
plt.plot(train_sizes, train_mean, label="Train Score")
plt.plot(train_sizes, test_mean, label="Test Score")
plt.title("Learning Curve")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend()
plt.show()
