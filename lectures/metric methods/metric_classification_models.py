
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
    t = sorted(enumerate(cdist(x.reshape(-1, 2), X)[0]), key=lambda x: x[1])[:k]
    u = np.array([i[0] for i in t])
    w[u] = 1
    # [y(i) = y]:
    a = np.meshgrid(np.unique(Y), w)[0] == Y.reshape(-1, 1)
    # sum([y(i) = y] w(i, x)):
    s = (a*w.reshape(-1, 1)).sum(axis=0)
    return np.argmax(s)

