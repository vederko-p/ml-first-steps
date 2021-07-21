
import numpy as np
from math import exp


# ---| Сигмоидная функция |---
def sigm(z):
    '''
    :param z: numpy array
    :return: sigm-ed numpy array
    '''
    return 1 / (1 + np.exp(z))


def sigm_diff_numeric(sigm, z, dz=10**-6):
    return (sigm(z + dz) - sigm(z)) / dz


# ---| Выходные значения сети на xi (a(xi)) |---
def a(uh, sigm, wmh):
    '''
    :param uh: Значения на выходе нейронов скрытого слоя - numpy массив-столбец из h элементов
    :param x: Элемент обучающей выборки - numpy array
    :param sigm: Сигмоидная функция
    :param whm: Веса нейронов выходного слоя - numpy array
    :return: Возвращает выходные значения сети на объекте xi (numpy массив-столбец из m элементов)
    '''
    return sigm(wmh.dot(uh))


# ---| Выходные значения скрытого слоя сети на xi (u(xi)) |---
def u(x, sigm, whj):
    '''
    :param x: Элемент обучающей выборки - numpy массив-столбец
    :param sigm: Сигмоидная функция
    :param wjh: Веса нейронов скрытого слоя - numpy array; (h x j), h - кол-во нейронов скрытого слоя
    :return: Возвращает выходные значения скрытого слоя сети на объекте xi (numpy массив-столбец из h элементов)
    '''
    return sigm(whj.dot(x))


# ---| Функция потерь (L) |---
def loss_function(whj, wmh, x, y):
    '''
    :param wjh: Веса нейронов скрытого слоя - numpy array
    :param whm: Веса нейронов выходного слоя - numpy array
    :param x: Элемент обучающей выборки - numpy array
    :param y: Целевое значение соответствующего элемента обучающей выборки - numpy array
    :return: Взвращает значение ошибки на элементе обучающей выборки
    '''
    uh = u(x, sigm, whj)
    an = np.array(a(uh, sigm, wmh))
    return ((an - y)**2).sum() / 2


# ---| Эмпирический риск (Q) |---
def empirical_risk(L, whj, whm, X, Y):
    '''
    :param wjh: Веса нейронов скрытого слоя - numpy array
    :param whm: Веса нейронов выходного слоя - numpy array
    :param X: Обучающая выборка - numpy array
    :param Y: Целевые значения обучающей выборки - numpy array
    :return: Значение эмпирического риска на обучающей выборке
    '''
    return sum([L(whj, wmh, x, y) for x,y in zip(X,Y)]) / X.shape[0]


'''
x = np.array([[-0.39], [0.15]])
whj = np.array([[0.7, -0.2], [-0.8, 0.4], [0.3, 0.6]])
wmh = np.array([[0.4, -0.3, 0.8]])
uh = u(x, sigm, whj)
# print(a(uh, sigm, wmh))

# print(loss_function(whj, wmh, x, [[1]]))
'''
