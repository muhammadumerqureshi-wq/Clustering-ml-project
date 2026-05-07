"""
ML Clustering Analysis Project
K-Means vs DBSCAN vs Isolation Forest Comparison
Author: Umer Qureshi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ML CLUSTERING ANALYSIS PROJECT")
print("="*60)

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================
print("\n✓ STEP 1: Loading Data...")
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(f"Dataset: Iris Flowers")
print(f"Samples: {len(df)}")
print(f"Features: {len(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# STEP 2: SCALE DATA
# ============================================================================
print("\n✓ STEP 2: Scaling Data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print("Data normalized (0 mean, 1 std)")

# ============================================================================
# STEP 3: ELBOW METHOD (Find optimal clusters)
# ============================================================================
print("\n✓ STEP 3: Finding Optimal Clusters (Elbow Method)...")
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method - Inertia')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs k')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_method.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ Elbow chart saved as 'elbow_method.png'")

# ============================================================================
# STEP 4: K-MEANS CLUSTERING
# ============================================================================
print("\n" + "="*60)
print("ALGORITHM 1: K-MEANS CLUSTERING")
print("="*60)

optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

print(f"\n✓ K-Means with k={optimal_k}")
print(f"Inertia: {kmeans.inertia_:.2f}")

silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
davies_bouldin_kmeans = davies_bouldin_score(X_scaled, kmeans_labels)

print(f"Silhouette Score: {silhouette_kmeans:.3f}", end="")
if silhouette_kmeans > 0.7:
    print(" → Excellent! ✓")
elif silhouette_kmeans > 0.5:
    print(" → Good ✓")
else:
    print(" → Fair")

print(f"Davies-Bouldin Index: {davies_bouldin_kmeans:.3f}", end="")
if davies_bouldin_kmeans < 0.5:
    print(" → Excellent ✓")
elif davies_bouldin_kmeans < 1.0:
    print(" → Good ✓")
else:
    print(" → Fair")

# K-Means cluster analysis
print("\nCluster Distribution:")
for i in range(optimal_k):
    count = np.sum(kmeans_labels == i)
    print(f"  Cluster {i}: {count} samples")

# ============================================================================
# STEP 5: DBSCAN CLUSTERING
# ============================================================================
print("\n" + "="*60)
print("ALGORITHM 2: DBSCAN (Density-Based)")
print("="*60)

dbscan = DBSCAN(eps=0.6, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)

n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"\n✓ DBSCAN (eps=0.6, min_samples=5)")
print(f"Clusters found: {n_clusters_dbscan}")
print(f"Noise points: {n_noise}")

if n_clusters_dbscan > 1:
    silhouette_dbscan = silhouette_score(X_scaled, dbscan_labels)
    davies_bouldin_dbscan = davies_bouldin_score(X_scaled, dbscan_labels)
    
    print(f"Silhouette Score: {silhouette_dbscan:.3f}")
    print(f"Davies-Bouldin Index: {davies_bouldin_dbscan:.3f}")
    
    print("\nCluster Distribution:")
    for i in set(dbscan_labels):
        count = np.sum(dbscan_labels == i)
        label = "Noise" if i == -1 else f"Cluster {i}"
        print(f"  {label}: {count} samples")

# ============================================================================
# STEP 6: ISOLATION FOREST (Anomaly Detection)
# ============================================================================
print("\n" + "="*60)
print("ALGORITHM 3: ISOLATION FOREST (Anomaly Detection)")
print("="*60)

iso_forest = IsolationForest(contamination=0.1, random_state=42)
iso_labels = iso_forest.fit_predict(X_scaled)

n_anomalies = np.sum(iso_labels == -1)
n_normal = np.sum(iso_labels == 1)

print(f"\n✓ Isolation Forest (contamination=0.1)")
print(f"Normal points: {n_normal}")
print(f"Anomalies: {n_anomalies}")
print(f"Anomaly percentage: {(n_anomalies/len(df)*100):.1f}%")

# ============================================================================
# STEP 7: VISUALIZATION
# ============================================================================
print("\n" + "="*60)
print("CREATING VISUALIZATIONS")
print("="*60)

# 2D Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# K-Means 2D
ax = axes[0, 0]
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
ax.scatter(scaler.transform(kmeans.cluster_centers_)[:, 0], 
           scaler.transform(kmeans.cluster_centers_)[:, 1], 
           c='red', marker='X', s=300, edgecolors='black', label='Centroids')
ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')
ax.set_title('K-Means Clustering (k=3)')
ax.legend()
ax.grid(True, alpha=0.3)

# DBSCAN 2D
ax = axes[0, 1]
scatter = ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='plasma', s=50, alpha=0.7)
ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')
ax.set_title('DBSCAN Clustering')
ax.grid(True, alpha=0.3)

# Isolation Forest 2D
ax = axes[1, 0]
colors = ['red' if label == -1 else 'blue' for label in iso_labels]
ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=colors, s=50, alpha=0.7)
ax.set_xlabel('Feature 1 (Scaled)')
ax.set_ylabel('Feature 2 (Scaled)')
ax.set_title('Isolation Forest (Anomalies in Red)')
ax.grid(True, alpha=0.3)

# Comparison metrics
ax = axes[1, 1]
ax.axis('off')
metrics_text = f"""
K-MEANS
Silhouette: {silhouette_kmeans:.3f}
Davies-Bouldin: {davies_bouldin_kmeans:.3f}

DBSCAN
Clusters: {n_clusters_dbscan}
Noise Points: {n_noise}

ISOLATION FOREST
Anomalies: {n_anomalies}
Normal: {n_normal}
"""
ax.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center', 
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('clustering_comparison_2d.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ 2D comparison saved as 'clustering_comparison_2d.png'")

# 3D Visualization
fig = plt.figure(figsize=(15, 5))

# K-Means 3D
ax1 = fig.add_subplot(131, projection='3d')
scatter = ax1.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                     c=kmeans_labels, cmap='viridis', s=50, alpha=0.7)
ax1.set_xlabel('Feature 1')
ax1.set_ylabel('Feature 2')
ax1.set_zlabel('Feature 3')
ax1.set_title('K-Means 3D')

# DBSCAN 3D
ax2 = fig.add_subplot(132, projection='3d')
scatter = ax2.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                     c=dbscan_labels, cmap='plasma', s=50, alpha=0.7)
ax2.set_xlabel('Feature 1')
ax2.set_ylabel('Feature 2')
ax2.set_zlabel('Feature 3')
ax2.set_title('DBSCAN 3D')

# Isolation Forest 3D
ax3 = fig.add_subplot(133, projection='3d')
colors = ['red' if label == -1 else 'blue' for label in iso_labels]
scatter = ax3.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2], 
                     c=colors, s=50, alpha=0.7)
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_zlabel('Feature 3')
ax3.set_title('Isolation Forest 3D')

plt.tight_layout()
plt.savefig('clustering_3d.png', dpi=100, bbox_inches='tight')
plt.show()
print("✓ 3D visualization saved as 'clustering_3d.png'")

# ============================================================================
# STEP 8: SAVE RESULTS
# ============================================================================
print("\n" + "="*60)
print("SAVING RESULTS")
print("="*60)

results_df = df.copy()
results_df['KMeans_Cluster'] = kmeans_labels
results_df['DBSCAN_Cluster'] = dbscan_labels
results_df['IsoForest_Anomaly'] = iso_labels

results_df.to_csv('clustering_results.csv', index=False)
print("✓ Results saved to 'clustering_results.csv'")

# Summary
summary = pd.DataFrame({
    'Algorithm': ['K-Means', 'DBSCAN', 'Isolation Forest'],
    'Type': ['Centroid-based', 'Density-based', 'Anomaly Detection'],
    'Clusters/Anomalies': [optimal_k, n_clusters_dbscan, n_anomalies],
    'Best For': ['Spherical clusters', 'Arbitrary shapes', 'Outlier detection']
})

print("\nAlgorithm Comparison:")
print(summary.to_string(index=False))

print("\n" + "="*60)
print("PROJECT COMPLETE! ✓")
print("="*60)
print("\nGenerated Files:")
print("  • elbow_method.png")
print("  • clustering_comparison_2d.png")
print("  • clustering_3d.png")
print("  • clustering_results.csv")
print("\nOutput saved successfully!")
