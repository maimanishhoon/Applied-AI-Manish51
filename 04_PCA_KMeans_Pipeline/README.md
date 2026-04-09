# PCA + K-Means Pipeline

## Objective
Evaluate clustering performance before and after applying **PCA**.

---

## Methods Used
- PCA (Dimensionality Reduction)  
- K-Means Clustering  
- Scikit-learn Pipeline  
- Adjusted Rand Index (ARI)  

---

## Results

### Performance Comparison (ARI)
<img src="https://github.com/user-attachments/assets/2c74e1f2-1d02-4d7d-8887-d3f9b364f57d" width="350"/>

### Cluster Visualization
<img src="https://github.com/user-attachments/assets/3076f9ce-eb5c-4bc4-8bd9-0e4186b84e36" width="700"/>

---

## Key Findings
- Compared clustering performance with and without PCA using ARI  
- PCA helps reduce noise and improve cluster separation  
- Excessive reduction can lead to information loss  

---

## Insight
PCA can improve clustering by removing redundant features, but the number of components must be chosen carefully.