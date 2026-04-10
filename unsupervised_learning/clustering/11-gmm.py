#!/usr/bin/env python3
"""Calculates a GMM from a dataset."""
import sklearn.mixture


def gmm(X, k):
    """Calculates a GMM from a dataset."""
    if getattr(X, "__class__", None).__module__ != "numpy":
        return None, None, None, None, None
    if len(getattr(X, "shape", ())) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0 or k > X.shape[0]:
        return None, None, None, None, None

    try:
        model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    except Exception:
        return None, None, None, None, None

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
