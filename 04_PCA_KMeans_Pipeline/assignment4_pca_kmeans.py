# Assignment 4: PCA + KMeans Pipeline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

# STEP 1: Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# STEP 2: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# STEP 3: KMeans WITHOUT PCA
kmeans_no_pca = KMeans(n_clusters=3, random_state=42)
labels_no_pca = kmeans_no_pca.fit_predict(X_scaled)

score_no_pca = adjusted_rand_score(y, labels_no_pca)
print("ARI without PCA:", score_no_pca)

# STEP 4: Pipeline (PCA + KMeans)
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

pipeline.fit(X)

labels_pca = pipeline.named_steps['kmeans'].labels_

score_pca = adjusted_rand_score(y, labels_pca)
print("ARI with PCA:", score_pca)

# STEP 5: Visualization

# PCA Transform for plotting
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(12,5))

# Without PCA (projected for visualization)
plt.subplot(1,2,1)
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_no_pca)
plt.title("KMeans without PCA")

# With PCA
plt.subplot(1,2,2)
plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_pca)
plt.title("PCA + KMeans")

plt.tight_layout()
plt.show()
