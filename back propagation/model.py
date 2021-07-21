
import numpy as np
from math import exp


# ---| Сигмоидная функция |---
def sigm(z):
    return 1 / (1 + exp(z))


# ---| Выходные значения сети на xi (a(xi)) |---
def a(um, sigm, whm):
    '''
    :param um: Значения на выходе нейронов скрытого слоя
    :param x: Элемент обучающей выборки
    :param sigm: Сигмоидная функция
    :param whm: Веса нейронов выходного слоя
    :return: Возвращает выходные значения сети на объекте xi (массив из m элементов)
    '''
    am = []
    for m in whm:
        am.append(sigm(sum([w*u for w, u in zip(m, um)])))
    return am


# ---| Выходные значения скрытого слоя сети на xi (a(xi)) |---
def u(x, sigm, wjh):
    '''
    :param x: Элемент обучающей выборки
    :param sigm: Сигмоидная функция
    :param wjh: Веса нейронов скрытых слоев
    :return: Возвращает выходные значения скрытого слоя сети на объекте xi (массив из h элементов)
    '''
    uh = []
    for h in wjh:  # h-ый нейрон скрытого слоя
        uh.append(sigm(sum([w*x for w,x in zip(h, x)])))
    return uh

# ---| Эмпирический риск (Q) |---
def empirical_risk(L, a, X, w, Y):
    pass


# print(u([-0.39, 0.15], sigm, [[0.5, 0.7], [0.1, 0.3], [-0.4, 0.2]]))
um = u([-0.39, 0.15], sigm, [[0.5, 0.7], [0.1, 0.3], [-0.4, 0.2]])
print(a(um, sigm, [[0.7, -0.3, 0.6], [-0.5, 0.7, -0.8]]))
