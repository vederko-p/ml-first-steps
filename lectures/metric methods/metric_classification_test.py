
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from metric_classification_models import gauss_core
from metric_classification_models import kNN, weighted_kNN, \
                                         parzen_window_consth, parzen_window_variateh


# ---| Выборка |---
# Обучающая:
np.random.seed(0)
l = 100
n = 2

X1 = np.array([[-0.5, -0.5]]) + 0.5*np.random.randn(l, n)
X2 = np.array([[0.5, 0.5]]) + 0.5*np.random.randn(l, n)

X = np.vstack([X1, X2])
y = np.hstack([[0]*l, [1]*l])

# Тестовая:
X1_test = np.array([[0, 0]]) + 0.5*np.random.randn(13, n)
X2_test = np.array([[0, 0]]) + 0.5*np.random.randn(13, n)
X_test = np.vstack([X1_test, X2_test])


# ---| Визуализация |---
methods = [kNN, weighted_kNN, parzen_window_consth, parzen_window_variateh]
arguments = [(X, y, 15), (X, y, 10, 0.5), (X, y, gauss_core, 1.2), (X, y, gauss_core, 7)]
titles = ['Метод k ближайших соседей',
          'Взвешенный метод k ближайших соседей',
          'Метод Парзеновского окна постоянной ширины',
          'Метод Парзеновского окна переменной ширины']
cols_test = [None, 'red']
cols_res = ['blue', 'pink']

fig, gs = plt.figure(), gridspec.GridSpec(2, 2)
ax = []
for i,method in enumerate(methods):
    ax.append(fig.add_subplot(gs[i]))
    # Тестовая выборка
    for k in np.unique(y):
        ax[i].plot(X[y == k, 0], X[y == k, 1],
                   'o', label='Класс {}'.format(k), color=cols_test[int(k)])
    # Результат классификации
    y_test = np.array([])
    for x in X_test:
        y_test = np.hstack([y_test, method(x, *arguments[i])])
    for xi, yi in zip(X_test, y_test):
        ax[i].scatter(xi[0], xi[1],
                    color=cols_res[int(yi)], edgecolors='black', s=100)
    ax[i].set_title(titles[i])
    ax[i].legend(loc='best')
plt.show()
