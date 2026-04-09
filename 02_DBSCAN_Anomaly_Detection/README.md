# Anomaly Detection using DBSCAN

## Objective
Detect anomalies in network traffic data using **DBSCAN** and compare results with **K-Means**.

---

## Methods Used
- DBSCAN Clustering  
- K-Means (for comparison)  
- Feature Scaling  

---

## Results

### Anomaly Detection (DBSCAN)
<img src="https://github.com/user-attachments/assets/10918681-64c0-41f2-b389-8b5aa10fecfc" width="350"/>

### DBSCAN Visualization
<img src="https://github.com/user-attachments/assets/0afccf2d-666c-4689-8a5d-8c5a81bba04c" width="700"/>

### K-Means Comparison
<img src="https://github.com/user-attachments/assets/31447f59-f595-48a8-a94c-b4647a5f4623" width="500"/>

---

## Key Findings
- DBSCAN identifies anomalies as noise points (`-1`)  
- K-Means assigns every point to a cluster (no true anomaly detection)  
- DBSCAN better captures irregular patterns and outliers  

---

## Insight
DBSCAN is more suitable for anomaly detection since it explicitly separates noise, unlike K-Means which forces all data into clusters.