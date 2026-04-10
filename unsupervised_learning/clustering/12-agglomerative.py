#!/usr/bin/env python3
"""Performs agglomerative clustering on a dataset."""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Performs agglomerative clustering on a dataset."""
    if getattr(X, "__class__", None).__module__ != "numpy":
        return None
    if len(getattr(X, "shape", ())) != 2:
        return None
    if isinstance(dist, bool) or not isinstance(dist, (int, float)) or dist < 0:
        return None

    try:
        linkage_matrix = scipy.cluster.hierarchy.linkage(X, method="ward")
        scipy.cluster.hierarchy.dendrogram(
            linkage_matrix,
            color_threshold=dist
        )
        plt.show()
        return scipy.cluster.hierarchy.fcluster(
            linkage_matrix,
            t=dist,
            criterion="distance"
        )
    except Exception:
        return None
