# Assignment 2: DBSCAN for Anomaly Detection

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler

# STEP 1: Create Synthetic Dataset

np.random.seed(42)

normal_data = np.random.randn(200, 3) * [50, 30, 10] + [500, 200, 50]
anomalies = np.random.uniform(low=[1000, 500, 200], high=[1500, 800, 400], size=(20, 3))

data = np.vstack([normal_data, anomalies])

df = pd.DataFrame(data, columns=['Packet_Size', 'Duration', 'Request_Frequency'])

print("Dataset Shape:", df.shape)


# STEP 2: Feature Scaling

scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)


# STEP 3: Apply DBSCAN

dbscan = DBSCAN(eps=0.8, min_samples=5)
labels_db = dbscan.fit_predict(scaled_data)

df['DBSCAN_Cluster'] = labels_db

# STEP 4: Identify Anomalies
anomalies = df[df['DBSCAN_Cluster'] == -1]

print("\nNumber of anomalies detected:", len(anomalies))

# STEP 5: Apply KMeans (for comparison)
kmeans = KMeans(n_clusters=3, random_state=42)
labels_km = kmeans.fit_predict(scaled_data)

df['KMeans_Cluster'] = labels_km

# STEP 6: Visualization
plt.figure(figsize=(12,5))

# DBSCAN Plot
plt.subplot(1,2,1)
plt.scatter(df['Packet_Size'], df['Duration'], c=labels_db)
plt.title("DBSCAN Clustering (Noise = -1)")
plt.xlabel("Packet Size")
plt.ylabel("Duration")

# KMeans Plot
plt.subplot(1,2,2)
plt.scatter(df['Packet_Size'], df['Duration'], c=labels_km)
plt.title("KMeans Clustering")
plt.xlabel("Packet Size")
plt.ylabel("Duration")

plt.tight_layout()
plt.show()

# STEP 7: Show Noise Points
print("\nAnomalous Points (first 5):\n")
print(anomalies.head())
