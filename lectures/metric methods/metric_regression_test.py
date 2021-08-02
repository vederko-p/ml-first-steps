
import numpy as np
import matplotlib.pyplot as plt
from metric_models import nadaray_watson, lowess, nadaray_watson_lowess
from metric_models import gauss_core, quadr_core


# Выборка
def true_func(x):
    return 5*np.cos(1.5*np.sin(x/8 + 3)) - 0.5*x


def rand_x(a, b, n):
    return np.round(np.random.rand(100) * (b-a) + a, 2)


np.random.seed(12)
# Обучающая выборка
l = 100
x = rand_x(-13, 5, l)
y = true_func(x) + np.random.normal(0, 0.4, l)

# Выборка с шумами
x_noise = np.hstack([x, -7.6, 1])
y_noise = np.hstack([y, -25, 25])


# Обучение по формуле Надарая-Ватсона
x_ed = np.linspace(-13, 5, 500)
y_ed_nw = []
for i in x_ed:
    t = nadaray_watson(i, x_noise, y_noise, gauss_core, 1)
    y_ed_nw.append(t)
y_ed_nw = np.array(y_ed_nw)

# Обучение весов по LOWESS
gammas = lowess(x_noise, y_noise, gauss_core, quadr_core, 1, 0.01)
# Обучение по формуле Надарая-Ватсона + LOWESS
y_ed_nw_lowess = []
for i in x_ed:
    t = nadaray_watson_lowess(i, x_noise, y_noise, gauss_core, 1, gammas)
    y_ed_nw_lowess.append(t)
y_ed_nw_lowess = np.array(y_ed_nw_lowess)


# Визуализация
fig, ax = plt.subplots()
ax.scatter(x, y, label='Обучающая выборка')
ax.plot(x_ed, y_ed_nw, c='orange', label='Модель по формуле Надарая-Ватсона')
ax.plot(x_ed, y_ed_nw_lowess, c='red', label='Модель по формуле Н-В + LOWESS')
plt.legend(loc='best')
plt.show()
