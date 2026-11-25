# Experiment 10: DBSCAN Clustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

X, _ = make_moons(n_samples=300, noise=0.06, random_state=42)
X = StandardScaler().fit_transform(X)

model = DBSCAN(eps=0.25, min_samples=5)
labels = model.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)
if n_clusters > 1 and n_clusters < len(X):
    mask = labels != -1
    if len(set(labels[mask])) > 1:
        print("Silhouette Score (excluding noise):", silhouette_score(X[mask], labels[mask]))

plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="viridis", s=25)
plt.title("DBSCAN Clustering (-1 represents noise)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.tight_layout()
plt.show()
