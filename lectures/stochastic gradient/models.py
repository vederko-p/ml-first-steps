
import numpy as np


# ---| Regression model |---
def linear_regression_model(x, w):  # Модель (a)
    '''
    :param x: x^(i) in X^(l) - numpy array
    :param w: weights - numpy array
    :return: value y =  w_1 * x^(i)_1 + w_2 * x^(i)_2 + ... + w_n * x^(i)_n
    '''
    return np.dot(x, w)


def regression_loss_function(a, x, w, y):  # Функция потерь (L)
    '''
    :param a: regression model function
    :param x: x^(i) in X^(l) - numpy array
    :param w: weights - numpy array
    :param y: y^(i) in y - value in R
    :return: error value: (a(x, w) - y)^2
    '''
    return (a(x, w) - y)**2


# ---| Classification model |---
def linear_classification_model(x, w):  # Модель (a)
    '''
    :param x: x^(i) in X^(l) - numpy array
    :param w: weights - numpy array
    :return: value y =  sign( w_1 * x^(i)_1 + w_2 * x^(i)_2 + ... + w_n * x^(i)_n )
    '''
    return np.sign(np.dot(x, w))


def classification_loss_function(a, x, w, y):  # Функция потерь (L)
    '''
    :param a: classification model function
    :param x: x^(i) in X^(l) - numpy array
    :param w: weights - numpy array
    :param y: y^(i) in y - value in R
    :return: error value: [a(x, w) * y < 0] ~ logistic 0/1
    '''
    '''
    if a(x, w) * y < 0:
        return 1
    else:
        return 0
    '''
    return np.log(1 + np.exp(-np.dot(w, x)*y))


# ---| Эмпирический риск (Q) |---
def empirical_risk(L, a, X, w, Y):
    '''
    :param L: loss function
    :param a: model function
    :param X: training sample - numpy array
    :param w: weighs - numpy array
    :param Y: training sample - numpy array
    :return: sum(L(x^(i), y^(i), w), i=1,l)
    '''
    return sum([L(a, xi, w, yi[0]) for xi,yi in zip(X, Y)])


def loss_function_gradient(L, a, x, w0, y, dw=10**-6):
    '''
    :param L: loss function
    :param a: model function
    :param x: x^(i) in X^(l) - numpy array
    :param w0: certain weight - numpy array
    :param y: y^(i) in y - value in R
    :param dw: increment
    :return: gradient of the loss function L in w0
    '''
    g = np.array([])
    for i in range(w0.shape[0]):
        w = w0.copy()
        w[i] = w[i] + dw
        g = np.append(g, (L(a, x, w, y) - L(a, x, w0, y)) / dw)
    return g
