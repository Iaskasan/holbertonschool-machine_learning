#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris


# Load the iris dataset
iris = load_iris()
data = iris.data
labels = iris.target

# Save to .npz file
np.savez("pca.npz", **{
    "iris_dataset/data": data,
    "iris_dataset/labels": labels
})

lib = np.load("pca.npz")
print(lib.files)  # ['iris_dataset/data', 'iris_dataset/labels']

data = lib["iris_dataset/data"]
labels = lib["iris_dataset/labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

ax = Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
x = pca_data[:, 0]
y = pca_data[:, 1]
z = pca_data[:, 2]
ax.scatter(x, y, z, c=labels, cmap="plasma")
ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")
ax.set_title("PCA of Iris Dataset")
plt.show()
