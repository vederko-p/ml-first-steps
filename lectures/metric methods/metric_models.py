
import numpy as np


def euclidean_distance(x, X):
    '''
    :param X: numpy array
    '''
    return ((X - x)**2)**0.5


def gauss_core(r):
    '''
    :param r: numpy array
    '''
    return np.exp(-2*r**2)


def quadr_core(r):
    '''
    :param r: numpy array
    '''
    return (1-r**2)**2 * ((r**2)**0.5 <= 1)


def nadaray_watson(x, X, y, K, h):
    '''
    :param x: one single digit
    :param X: numpy array
    :param y: numpy array
    :param K: core
    :param h: window size
    :return: a(h, x) due to Nadaray-Watson
    '''
    t = K(euclidean_distance(x, X) / h)
    s1 = (y*t).sum()
    s2 = t.sum()
    return s1 / s2


def lowess(X, y, K, Ks, h, eps):
    l = X.shape[0]
    gammas = np.ones(l)
    a = np.array([])
    while True:
        for i in range(l):
            t = gammas * K(euclidean_distance(X[i], X) / h)
            s1 = (y*t).sum() - y[i]*t[i]
            s2 = t.sum() - t[i]
            a = np.hstack([a, s1/s2])
        gammas_n = Ks(np.abs(a - y))
        if (gammas_n - gammas).dot(gammas_n - gammas) >= eps * (1 + gammas_n.dot(gammas_n)):
            gammas = gammas_n.copy()
            a = np.array([])
        else:
            break
    return gammas
