#!/usr/bin/env python3
"""Performs K-means on a dataset."""
import sklearn.cluster


def kmeans(X, k):
    """Performs K-means on a dataset."""
    if getattr(X, "__class__", None).__module__ != "numpy":
        return None, None
    if len(getattr(X, "shape", ())) != 2:
        return None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None

    try:
        model = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    except Exception:
        return None, None

    return model.cluster_centers_, model.labels_
