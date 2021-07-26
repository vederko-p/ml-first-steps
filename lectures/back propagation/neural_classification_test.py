
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model import empirical_risk, loss_function, a, u, sigm, neural_model
from back_prop import back_propagation, back_propagation_average, rms_propagation
from back_prop import ada_delta, adam, nadam


classification_data = pd.read_csv('data/classification_two_clusters.csv')
class_val = classification_data.values

X, Y = class_val[:, :2], class_val[:, 2:]

methods = [[back_propagation, back_propagation_average],
           [rms_propagation, ada_delta],
           [adam, nadam]]

params = [[(X, Y, 3, 2, 0.5, 0.1, 10**-10, empirical_risk, loss_function, a, u, sigm, 1),
           (X, Y, 3, 2, 0.5, 0.1, 10**-10, empirical_risk, loss_function, a, u, sigm, 1)],
          [(X, Y, 3, 2, 0.5, 0.5, 0.1, 10**-10, 10**-3, empirical_risk, loss_function, a, u, sigm, 1),
           (X, Y, 3, 2, 0.5, 0.5, 1, 10**-10, 10**-3, empirical_risk, loss_function, a, u, sigm, 1)],
          [(X, Y, 3, 2, 0.9, 0.999, 10**-8, 0.5, 0.1, 10**-10, empirical_risk, loss_function, a, u, sigm, 1),
           (X, Y, 3, 2, 0.9, 0.999, 10**-8, 0.5, 0.1, 10**-10, empirical_risk, loss_function, a, u, sigm, 1)]]

labels = [['Back Propagation', 'SAGProp (Stochastic Average Gradient)'],
          ['RMSProp (Running Mean Square)', 'AdaDelta (Adaptive Learning Rate)'],
          ['Adam (Adaptive Momentum)', 'Nadam (Nesterov-accelerated adaptive momentum)']]


fig, axs = plt.subplots(3, 2)
# Обучающая выборка
x1, x2 = class_val[:, 1].reshape(-1, 1), class_val[:, 0].reshape(-1, 1)
for i in range(axs.shape[0]):
    for j in range(axs.shape[1]):
        axs[i, j].scatter(x1[:20], x2[:20], label='Обучающая выборка')
        axs[i, j].scatter(x1[20:], x2[20:], color='orange', label='Обучающая выборка')
        axs[i, j].set_title(labels[i][j])
# Тестовая выборка
test_x = [np.array([[j], [j]]) for j in range(-5, 30, 2)]

# Обучение
weights = []
for i in range(len(methods)):
    w = []
    for j in range(len(methods[0])):
        w.append(methods[i][j](*params[i][j]))
        print('Обучение методом ' + labels[i][j] + ' завершено')
    weights.append(w)

# Изображение
for i in range(len(methods)):
    for j in range(len(methods[0])):
        for x in test_x:
            res = neural_model(x, sigm, weights[i][j][0], weights[i][j][1], 1)
            if res[0, 0] > res[1, 0]:
                axs[i, j].scatter(x[0, 0], x[1, 0], color='red', edgecolors='black', linewidths=0.5, s=50)
            else:
                axs[i, j].scatter(x[0, 0], x[1, 0], color='blue', edgecolors='black', linewidths=0.5, s=50)
plt.legend()
plt.show()
