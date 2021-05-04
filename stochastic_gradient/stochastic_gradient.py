
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---| Модель регрессии |---
def linear_regression_model(x, w):  # Модель (a)
    return np.dot(x, w)


def regression_loss_function(a, x, w, y):  # Функция потерь (L)
    return (a(x, w) - y)**2


# ---| Модель классификации |---
def linear_classification_model(x, w):  # Модель (a)
    return np.sign(np.dot(x, w))


def classification_loss_function(a, x, w, y):  # Функция потерь (L)
    if a(x, w) * y < 0:
        return 1
    else:
        return 0


# ---| Эмпирический риск (Q) |---
def empirical_risk(L, a, X, w, Y):
    return sum([L(a, xi, w, yi) for xi,yi in zip(X, Y)])


def loss_function_gradient(L, a, x, w0, y, dw=10**-6):
    g = np.array([])
    for i in range(w0.shape[0]):
        w = w0.copy()
        w[i] = w[i] + dw
        g = np.append(g, (L(a, x, w, y) - L(a, x, w0, y)) / dw)
    return g


# ---| Стохастический градиент (Stochastic Gradient)|
def stochastic_grad(X, Y, h, lam, eps, Q, L, a):
    l, n = X.shape
    w0 = np.ones(n)
    w1 = w0*100
    q0 = (1/l * Q(L, a, X, w0, Y))[0]
    q1 = q0*100
    counter = 100000
    while ((w1 - w0).dot(w1 - w0) >= eps*(1 + w1.dot(w1))) and (abs(q1 - q0) >= eps*(1 + abs(q1))) and (counter):
        i = np.random.randint(0, l)
        xi, yi = X[i], Y[i]
        ei = L(a, xi, w0, yi)
        w1 = w0 - h * loss_function_gradient(L, a, xi, w0, yi)  # FIXME: Проверь, все ли вподрядке с градиентом
        q1 = lam*ei + (1-lam)*q0
        counter -= 1
    if not counter:
        print('Алгоритм не сошелся')
    else:
        print('Алгоритм сошелся на шаге ', counter)
    return w1


regression_data = pd.read_csv('data/regression_simple_line.csv')
regr_val = regression_data.values
X, y = regr_val[:, 1].reshape(-1, 1), regr_val[:, 0].reshape(-1, 1)
X = np.append(np.ones(X.shape[0]).reshape(-1, 1), X, axis=1)

weights = stochastic_grad(X, y, 0.5, 0.5, 0.001, empirical_risk, regression_loss_function, linear_regression_model)
print(weights)
# print(empirical_risk(regression_loss_function, linear_regression_model, X, np.array([0.5]), y))
# print(linear_regression_model(X[0], np.array([0.5, 0.5])))
# print(regression_loss_function(linear_regression_model, 1, np.array([0.5]), 3))


approx_x = np.append(np.ones(100).reshape(-1, 1), np.linspace(0, 30, 100).reshape(-1, 1), axis=1)
approx_y = linear_regression_model(approx_x, weights)

fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(X[:, 1], y)
plt.plot(approx_x[:, 1], approx_y, c='r')
plt.show()
