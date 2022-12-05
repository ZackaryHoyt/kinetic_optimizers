import numpy as np
import matplotlib.pyplot as plt

from knn_experiments import run_iko, run_kmeans, f_Δ, generate_clusters

np.random.seed(0)

k = 3
c = 3
d = 2
n_samples = 20

items = []
for _ in range(c):
	items.extend(np.random.normal(loc=np.random.normal(scale=4, size=d), scale=4, size=(n_samples,d)))
init_indices = np.random.choice(len(items), size=k, replace=False)

centers_kmeans, losses_kmeans = run_kmeans(init_indices=init_indices, items=items, k=k)
centers_iko, losses_iko = run_iko(init_indices=init_indices, items=items, k=k, f_Δ=f_Δ)

clusters_kmeans = generate_clusters(centres=centers_kmeans, items=items)[0]
clusters_iko = generate_clusters(centres=centers_iko, items=items)[0]


def plot_knn(centers, clusters, title, label):
	plt.gcf().set_size_inches(6.5, 6.5)
	plt.clf()
	plt.title(title, fontsize=12)
	for center, cluster in zip(centers, clusters):
		cluster = np.array(cluster)
		center = np.array(center)
		plt.scatter(x=cluster.T[0], y=cluster.T[1], alpha=0.5)
		plt.scatter(x=center[0], y=center[1], marker='^', color='black', edgecolors='black')
	plt.gca().tick_params(axis=u'both', which=u'both',length=0)
	plt.gca().set_xticks([]) 
	plt.gca().set_yticks([])
	plt.tight_layout()
	plt.savefig("{}.png".format(label))

print(losses_kmeans[-1])
print(losses_iko[-1])
plot_knn(centers=centers_kmeans, clusters=clusters_kmeans, title="K-Means Example", label="kmeans")
plot_knn(centers=centers_iko, clusters=clusters_iko, title="IKO Example", label="iko")