
import numpy as np


# ---| Стохастический градиент (Stochastic Gradient)|---
def stochastic_grad(X, Y, h, lam, eps, Q, L, a, lg):
    '''
    :param X: training sample - numpy array
    :param Y: training sample - numpy array
    :param h: gradient step (learning rate) - value in R
    :param lam: the rate of forgetting - value in (0, 1)
    :param eps: accuracy - value in (0, 1)
    :param Q: empirical risk function
    :param L: loss function
    :param a: model function
    :param lg: gradient evaluating function
    :return: weights - numpy array
    '''
    l, n = X.shape
    w0 = np.ones(n)
    q0 = (1/l * Q(L, a, X, w0, Y))
    while True:
        i = np.random.randint(0, l)
        xi, yi = X[i], Y[i, 0]
        ei = L(a, xi, w0, yi)
        w1 = w0 - h * lg(L, a, xi, w0, yi)
        q1 = lam*ei + (1-lam)*q0
        if (w1 - w0).dot(w1 - w0) >= eps*(1 + w1.dot(w1)):
            w0 = w1.copy()
            q0 = q1
        else:
            break
    return w1


# ---| Stochastic Average Gradient|---
def stochastic_average_grad(X, Y, h, lam, eps, Q, L, a, lg):
    '''
    :param X: training sample - numpy array
    :param Y: training sample - numpy array
    :param h: gradient step (learning rate) - value in R
    :param lam: the rate of forgetting - value in (0, 1)
    :param eps: accuracy - value in (0, 1)
    :param Q: empirical risk function
    :param L: loss function
    :param a: model function
    :param lg: gradient evaluating function
    :return: weights - numpy array
    '''
    l, n = X.shape
    w0 = np.ones(n)
    grads = [lg(L, a, xi, w0, yi[0]) for xi,yi in zip(X, Y)]
    s = sum(grads)
    q0 = (1 / l * Q(L, a, X, w0, Y))
    while True:
        i = np.random.randint(0, l)
        xi, yi = X[i], Y[i, 0]
        ei = L(a, xi, w0, yi)
        g = lg(L, a, xi, w0, yi)
        s = s - grads[i] + g
        grads[i] = g
        w1 = w0 - h * s/l
        q1 = lam*ei + (1-lam)*q0
        if (w1 - w0).dot(w1 - w0) >= eps*(1 + w1.dot(w1)):
            w0 = w1.copy()
            q0 = q1
        else:
            break
    return w1
