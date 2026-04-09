# Assignment 3: PCA for Dimensionality Reduction

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# STEP 1: Load Dataset (MNIST-like)
digits = load_digits()

X = digits.data   # 64 features (8x8 images)
y = digits.target

print("Original Shape:", X.shape)

# STEP 2: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 3: PCA to 2D (Visualization)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

# Plot 2D Projection
plt.figure(figsize=(6,5))
plt.scatter(X_2d[:,0], X_2d[:,1], c=y)
plt.title("PCA - 2D Projection")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar()
plt.show()

# STEP 4: PCA to 50D
pca_50 = PCA(n_components=50)
X_50 = pca_50.fit_transform(X_scaled)

print("Reduced Shape (50D):", X_50.shape)

# STEP 5: Explained Variance
pca_full = PCA()
pca_full.fit(X_scaled)

explained_variance = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot cumulative variance
plt.figure(figsize=(6,4))
plt.plot(cumulative_variance)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.title("Cumulative Explained Variance")
plt.xlabel("Number of Components")
plt.ylabel("Variance Explained")
plt.show()

# STEP 6: Minimum Components for 95% Variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1

print("\nNumber of components to retain 95% variance:", n_components_95)
