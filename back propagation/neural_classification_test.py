
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import empirical_risk, loss_function, a, u, sigm, neural_model
from back_prop import back_propagation


classification_data = pd.read_csv('data/classification_two_clusters.csv')
class_val = classification_data.values

X, Y = class_val[:, :2], class_val[:, -1:]

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

one = 1
wh, wm = back_propagation(X, Y, 3, 1, 0.5, 0.1, 10**-10, empirical_risk, loss_function, a, u, sigm, one)

x1 = np.array([[25], [25]])
x2 = np.array([[4], [5]])
res1 = neural_model(x1, sigm, wh, wm, one)
res2 = neural_model(x2, sigm, wh, wm, one)

print('Результаты:', res1, res2)
print('Веса:')
print(wh)
print(wm)

x1, x2 = class_val[:, 1].reshape(-1, 1), class_val[:, 0].reshape(-1, 1)
fig, ax = plt.subplots()
ax.scatter(x1[:20], x2[:20])
ax.scatter(x1[20:], x2[20:], color='orange')
test_x = [np.array([[j], [j]]) for j in range(-5, 30, 2)]
for i in test_x:
    if neural_model(i, sigm, wh, wm, one) > 0.5:
        ax.scatter(i[0,0], i[1,0], color='red', edgecolors='black', linewidths=0.5, s=50)
    else:
        ax.scatter(i[0,0], i[1,0], color='blue', edgecolors='black', linewidths=0.5, s=50)
plt.show()
