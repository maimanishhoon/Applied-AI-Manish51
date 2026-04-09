# Assignment 1: K-Means Clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# STEP 1: Create / Load Dataset
# You can replace this with real dataset later
np.random.seed(42)

data = pd.DataFrame({
    'Annual_Income': np.random.randint(20000, 100000, 200),
    'Spending_Score': np.random.randint(1, 100, 200),
    'Age': np.random.randint(18, 70, 200)
})

print("Sample Data:\n", data.head())


# STEP 2: Feature Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)


# STEP 3: Elbow Method
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    wcss.append(kmeans.inertia_)

# Plot Elbow Curve
plt.figure(figsize=(6,4))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


# STEP 4: Silhouette Score
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    score = silhouette_score(scaled_data, labels)
    print(f"Silhouette Score for k={k}: {score:.3f}")


# STEP 5: Final Model (choose k=4 for example)
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

data['Cluster'] = clusters


# STEP 6: Visualization (2D)
plt.figure(figsize=(6,4))
plt.scatter(data['Annual_Income'], data['Spending_Score'], c=clusters)
plt.title("Customer Segments")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()


# STEP 7: Interpretation
print("\nCluster Means:\n")
print(data.groupby('Cluster').mean())
