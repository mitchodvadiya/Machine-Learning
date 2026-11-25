# Experiment 8: K-Means Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=0.8, random_state=42)

model = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = model.fit_predict(X)

print("Cluster Centers:\n", model.cluster_centers_)
print("Inertia:", model.inertia_)
print("Silhouette Score:", silhouette_score(X, labels))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=30)
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1],
            color="red", marker="X", s=200, label="Centroids")
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.show()
