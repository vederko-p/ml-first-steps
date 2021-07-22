
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import empirical_risk, loss_function, a, u, sigm
from back_prop import back_propagation


classification_data = pd.read_csv('data/classification_two_clusters.csv')
class_val = classification_data.values

X, Y = class_val[:, :2], class_val[:, -1:]
# x1, x2, y = class_val[:, 1].reshape(-1, 1), class_val[:, 0].reshape(-1, 1), class_val[:, 2]

# fig, ax = plt.subplots()
# ax.scatter(x1, x2)
# plt.show()

# whj = np.array([[0.7, -0.2], [-0.8, 0.4], [0.3, 0.6]])
# wmh = np.array([[0.4, -0.3, 0.8], [0.1, -0.3, -0.4]])

# print(empirical_risk(loss_function, wjh, whm, X, y))

'''
X0 = np.append(X, -np.ones(X.shape[0]).reshape(-1, 1), axis=1)
whj = np.array([[0.7, -0.2, 0.2], [-0.8, 0.4, -0.3], [0.3, 0.6, 0.4]])
wmh = np.array([[0.4, -0.3, 0.8, 0.7], [0.1, -0.3, -0.4, -0.3]])
# print(empirical_risk(loss_function, whj, wmh, X0, Y, 1))
# print(loss_function(whj, wmh, X0[0].reshape(-1, 1), Y[0].reshape(-1, 1), 1))
# print(u(X0[0].reshape(-1, 1), sigm, whj, 1))
# print(whj.dot(X0[0].reshape(-1, 1)))
'''

t = back_propagation(X, Y, 3, 1, 0.5, 0.01, 10**-10, empirical_risk, loss_function, a, u, sigm, 1)
print(t)
