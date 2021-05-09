
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stochastic_gradient import stochastic_grad
from models import empirical_risk, regression_loss_function, linear_regression_model, loss_function_gradient


regression_data = pd.read_csv('data/regression_simple_line.csv')
regr_val = regression_data.values

X, y = regr_val[:, 1].reshape(-1, 1), regr_val[:, 0].reshape(-1, 1)
X = np.append(np.ones(X.shape[0]).reshape(-1, 1), X, axis=1)

weights = stochastic_grad(X, y, 0.0001, 0.9, 10**-10,
                          empirical_risk, regression_loss_function,
                          linear_regression_model, loss_function_gradient)

approx_x = np.append(np.ones(100).reshape(-1, 1), np.linspace(0, 30, 100).reshape(-1, 1), axis=1)
approx_y = linear_regression_model(approx_x, weights)

fig, ax = plt.subplots(figsize=(7, 7))
plt.scatter(X[:, 1], y)
plt.plot(approx_x[:, 1], approx_y, c='r')
plt.show()
