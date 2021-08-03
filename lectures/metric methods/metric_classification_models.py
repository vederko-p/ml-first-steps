
import numpy as np
from scipy.spatial.distance import cdist


def gauss_core(r):
    '''
    :param r: numpy array
    '''
    return np.exp(-2*r**2)


# ---| kNN |---
def kNN(x, X, Y, k):
    # w(i, x):
    w = np.zeros(X.shape[0])
    t = sorted(enumerate(cdist(x.reshape(-1, 2), X)[0]), key=lambda x: x[1])[:k]  # sort by distance (+ indexes)
    u = np.array([i[0] for i in t])  # indexes of k nearest neighbors
    w[u] = 1
    # [y(i) = y]:
    a = np.meshgrid(np.unique(Y), w)[0] == Y.reshape(-1, 1)
    # sum([y(i) = y] w(i, x)):
    s = (a*w.reshape(-1, 1)).sum(axis=0)
    return np.argmax(s)


# ---| Weighted kNN |---
def weighted_kNN(x, X, Y, k, q):
    # w(i, x):
    w = np.zeros(X.shape[0])
    t = sorted(enumerate(cdist(x.reshape(-1, 2), X)[0]), key=lambda x: x[1])[:k]  # sort by distance (+ indexes)
    u = [i[0] for i in t]  # indexes of k nearest neighbors
    for local_indx,indx in enumerate(u):
        # local_indx - index of neighbor among k nearestneighbors
        w[indx] = q**(local_indx+1)
    # [y(i) = y]:
    a = np.meshgrid(np.unique(Y), w)[0] == Y.reshape(-1, 1)
    # sum([y(i) = y] w(i, x)):
    s = (a*w.reshape(-1, 1)).sum(axis=0)
    # return s
    return np.argmax(s)
