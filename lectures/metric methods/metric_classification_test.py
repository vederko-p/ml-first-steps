
import numpy as np
import matplotlib.pyplot as plt
from metric_classification_models import kNN


# Выборки
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

# Обучение и классификация
# kNN
y_test = np.array([])
for x in X_test:
    y_test = np.hstack([y_test, kNN(x, X, y, 10)])


# Визуализация
cols = [None, 'red']
for k in np.unique(y):
    plt.plot(X[y == k, 0], X[y == k, 1],
             'o', label='Класс {}'.format(k), color=cols[k])

cols = ['blue', 'pink']
for x,y in zip(X_test, y_test):
    plt.scatter(x[0], x[1],
                color=cols[int(y)], edgecolors='black', s=100)
plt.legend(loc='best')
plt.show()
