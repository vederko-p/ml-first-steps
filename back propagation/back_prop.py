
import numpy as np
from model import sigm_diff_numeric as sd


def back_propagation(X0, Y, h, m, lam, eta, eps, Q, L, a, u, sigm, one=0):
    '''
    :param X: training sample - numpy array
    :param Y: training sample - numpy array
    :param h: Количество нейронов скрытого слоя
    :param m: Количество нейронов выходного слоя
    :param lam: the rate of forgetting - value in (0, 1)
    :param eta: gradient step (learning rate) - value in R
    :param eps: accuracy - value in (0, 1)
    :param Q: empirical risk function
    :param L: loss function
    :param a: outer layer function
    :param u: hidden layer function
    :param sigm: sigma function
    :param one: Включение в сеть минус единиц
    '''
    if one:
        X = np.append(X0, -np.ones(X0.shape[0]).reshape(-1, 1), axis=1)
    else:
        X = X0.copy()
    l, n = X.shape
    whj0 = np.ones(h*n).reshape(h, n)
    wmh0 = np.ones((h + one)*m).reshape(m, (h + one))
    q0 = Q(L, whj0, wmh0, X, Y, one)
    k = 10000
    while k:  # True
        # Прямой ход
        i = np.random.randint(0, l)
        xi, yi = X[i].reshape(-1, 1), Y[i].reshape(-1, 1)
        ui = u(xi, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - yi)
        q1 = (1-lam)*q0 + lam*L(whj0, wmh0, xi, yi, one)
        k -= 1
        if True:  # (q1 - q0) >= eps*(1 + q1)
            # Обратный ход
            q0 = q1
            sdf_m = sd(sigm, wmh0.dot(ui))
            sdf_h = sd(sigm, whj0.dot(xi))
            eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
            # Градиентный шаг
            mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim*sdf_m).flatten())
            wmh0 -= mesh_t*mesh_u*eta
            mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih*sdf_h).flatten())
            whj0 -= mesh_t*mesh_u*eta
        else:
            break
    print('--------------------------------------')
    return whj0, wmh0
