
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stochastic_gradient import stochastic_grad, stochastic_average_grad, momentum, nestr_grad
from models import empirical_risk, regression_loss_function, linear_regression_model, loss_function_gradient


regression_data = pd.read_csv('data/regression_simple_line.csv')
regr_val = regression_data.values

X, y = regr_val[:, 1].reshape(-1, 1), regr_val[:, 0].reshape(-1, 1)
X = np.append(np.ones(X.shape[0]).reshape(-1, 1), X, axis=1)

methods = [stochastic_grad,
           stochastic_average_grad,
           momentum,
           nestr_grad]
names = ['Stochastic Gradient',
         'Stochastic Average Gradient',
         'Momentum',
         "Nesterov's accelerated gradient"]
params = [[0.0001, 0.5, 10**-10],
          [0.00001, 0.5, 10**-10],
          [0.001, 0.5, 10**-10],
          [0.001, 0.5, 10**-10]]

fig, axs = plt.subplots(4)
approx_x = np.append(np.ones(100).reshape(-1, 1), np.linspace(0, 30, 100).reshape(-1, 1), axis=1)
for m,n,p,ax in zip(methods, names, params, axs):
    weights = m(X, y, p[0], p[1], p[2],
                empirical_risk, regression_loss_function,
                linear_regression_model, loss_function_gradient)
    approx_y = linear_regression_model(approx_x, weights)
    ax.scatter(X[:, 1], y)
    ax.plot(approx_x[:, 1], approx_y, c='r')
    ax.set_title(n)
plt.show()
