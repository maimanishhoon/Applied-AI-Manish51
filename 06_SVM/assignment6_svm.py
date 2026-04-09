# Assignment 6: Support Vector Machine (SVM)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.decomposition import PCA

# STEP 1: Load Dataset
data = load_breast_cancer()
X = data.data
y = data.target

# STEP 2: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# STEP 3: Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# STEP 4: Train SVM with Different Kernels
kernels = ['linear', 'poly', 'rbf']

for kernel in kernels:
    model = SVC(kernel=kernel)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Accuracy ({kernel} kernel): {acc:.3f}")

# -----------------------------
# STEP 5: Hyperparameter Tuning
# -----------------------------
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X_train, y_train)

print("\nBest Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_)

# STEP 6: Decision Boundary (2D using PCA)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

model_2d = SVC(kernel='rbf')
model_2d.fit(X_pca, y_train)

# Create mesh grid
x_min, x_max = X_pca[:,0].min()-1, X_pca[:,0].max()+1
y_min, y_max = X_pca[:,1].min()-1, X_pca[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(6,5))
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_pca[:,0], X_pca[:,1], c=y_train)
plt.title("SVM Decision Boundary (2D PCA)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
