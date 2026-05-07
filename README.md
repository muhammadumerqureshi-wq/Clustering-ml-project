# ML Clustering Analysis Project

A comprehensive machine learning clustering analysis comparing **K-Means**, **DBSCAN**, and **Isolation Forest** algorithms on the Iris dataset.

## 📊 Project Overview

This project demonstrates three major clustering approaches:

1. **K-Means Clustering** - Centroid-based, partitions data into k clusters
2. **DBSCAN** - Density-based, finds arbitrary shaped clusters
3. **Isolation Forest** - Anomaly detection using random forests

## ✨ Features

- ✅ **Elbow Method** - Automatically finds optimal number of clusters
- ✅ **3D & 2D Visualizations** - Multiple perspectives of clustering results
- ✅ **Performance Metrics** - Silhouette Score, Davies-Bouldin Index, Inertia
- ✅ **Comparative Analysis** - Algorithm comparison with strengths/weaknesses
- ✅ **CSV Output** - Save clustering results for further analysis

## 📈 Algorithms Explained

### K-Means
- **Type**: Centroid-based clustering
- **When to use**: Spherical, evenly-sized clusters
- **Output**: Fixed number of clusters (k), cluster centers (centroids)
- **Metrics**: Silhouette Score, Davies-Bouldin Index, Inertia

### DBSCAN
- **Type**: Density-based clustering
- **When to use**: Arbitrary-shaped clusters, variable cluster sizes
- **Output**: Variable number of clusters + noise points
- **Advantage**: No need to specify k beforehand

### Isolation Forest
- **Type**: Anomaly detection
- **When to use**: Outlier/anomaly detection, fraud detection
- **Output**: Normal vs Anomalous points
- **Key Insight**: Isolates outliers instead of clustering

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- pip or conda

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/clustering-ml-project.git
cd clustering-ml-project

# Install dependencies
pip install -r requirements.txt
```

### Running the Project

```bash
# Run the complete analysis
python clustering_analysis.py

# Or use Jupyter Notebook
jupyter notebook notebooks/clustering_analysis.ipynb
```

## 📂 Project Structure

```
clustering-ml-project/
├── clustering_analysis.py          # Main analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── .gitignore                      # Git ignore rules
├── notebooks/
│   └── clustering_analysis.ipynb   # Jupyter notebook version
├── results/
│   ├── elbow_method.png           # Elbow curve visualization
│   ├── clustering_comparison_2d.png # 2D clustering results
│   ├── clustering_3d.png          # 3D clustering results
│   └── clustering_results.csv     # Detailed clustering results
└── data/
    └── iris.csv                   # Iris dataset (optional)
```

## 📊 Output

The script generates:

1. **elbow_method.png** - Optimal k selection chart
2. **clustering_comparison_2d.png** - 2D comparison of all three algorithms
3. **clustering_3d.png** - 3D visualization of results
4. **clustering_results.csv** - Full results with cluster assignments

## 🔍 Results Summary

### K-Means Results
- Clusters: 3 (optimized via Elbow method)
- Silhouette Score: ~0.55
- Davies-Bouldin Index: ~0.72

### DBSCAN Results
- Clusters found: ~3
- Noise points: ~15% of data
- Best for non-spherical clusters

### Isolation Forest Results
- Anomalies detected: ~10% of data
- Normal points: ~90%
- Useful for outlier detection

## 📚 Key Concepts

**Elbow Method**: Finding optimal k by plotting inertia vs number of clusters

**Silhouette Score**: Measures how similar an object is to its own cluster (-1 to 1, higher is better)

**Davies-Bouldin Index**: Ratio of within-cluster to between-cluster distances (lower is better)

**Contamination**: Expected proportion of anomalies (Isolation Forest parameter)

## 🎯 Use Cases

- **E-commerce**: Customer segmentation (K-Means)
- **Healthcare**: Disease subtype discovery (DBSCAN)
- **Finance**: Fraud/anomaly detection (Isolation Forest)
- **Retail**: Market segmentation

## 🔗 Technologies Used

- **pandas** - Data manipulation
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning algorithms
- **matplotlib** - Data visualization
- **mpl_toolkits** - 3D plotting

## 💡 Learning Outcomes

After running this project, you'll understand:

1. How different clustering algorithms work
2. When to use each algorithm
3. How to evaluate clustering quality
4. How to visualize high-dimensional data
5. Real-world machine learning workflow

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError: No module named 'sklearn'`
```bash
pip install scikit-learn
```

**Issue**: Plots not showing
```python
# Add this to clustering_analysis.py
import matplotlib
matplotlib.use('TkAgg')
```

## 📖 Further Reading

- [K-Means Clustering](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [DBSCAN Documentation](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [Isolation Forest](https://scikit-learn.org/stable/modules/ensemble.html#isolation-forest)

## 📝 Author

**Umer Qureshi**
- BS Computer Science, UMT (Expected July 2027)
- ML/AI Enthusiast | Web Developer
- Location: Lahore, Pakistan

## 📧 Contact

- GitHub: [@muhammadumerqureshi-wq](https://github.com/muhammadumerqureshi-wq)
- LinkedIn: [Umer Qureshi](https://www.linkedin.com/in/umer-qureshi-243b12387)
- Email: muhammadumerqureshi39@gmail.com

## 📄 License

This project is open source and available under the MIT License.

## ⭐ If you found this helpful, please star the repository!

---

**Last Updated**: 2026
**Status**: Complete & Production Ready ✓
