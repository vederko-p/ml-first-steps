
import numpy as np
from model import sigm_diff_numeric as sd


# ---| Back Propagation |---
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
            # Веса выходного слоя
            mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim*sdf_m).flatten())
            wmh0 -= mesh_t*mesh_u*eta
            # Веса скрытого слоя
            mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih*sdf_h).flatten())
            whj0 -= mesh_t*mesh_u*eta
        else:
            break
    return whj0, wmh0


# ---| SAGProp (Stochastic Average Gradient) |---
def back_propagation_average(X0, Y, h, m, lam, eta, eps, Q, L, a, u, sigm, one=0):
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
    # Расчет всех градиентов
    grads_wmh = []
    grads_whj = []
    for xi, yi in zip(X, Y):
        x, y = xi.reshape(-1, 1).copy(), yi.reshape(-1, 1).copy()
        ui = u(x, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - y)
        sdf_m = sd(sigm, wmh0.dot(ui))
        sdf_h = sd(sigm, whj0.dot(x))
        eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
        mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
        grads_wmh.append(mesh_t*mesh_u*eta)
        mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
        grads_whj.append(mesh_t*mesh_u*eta)
    s_wmh = sum(grads_wmh)
    s_whj = sum(grads_whj)
    q0 = Q(L, whj0, wmh0, X, Y, one)
    k = 100000
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
            # Веса выходного слоя
            mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim*sdf_m).flatten())
            s_wmh = s_wmh - grads_wmh[i] + mesh_t*mesh_u*eta
            grads_wmh[i] = mesh_t*mesh_u*eta
            wmh0 -= eta*s_wmh/l
            # Веса скрытого слоя
            mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih*sdf_h).flatten())
            s_whj = s_whj - grads_whj[i] + mesh_t*mesh_u*eta
            grads_whj[i] = mesh_t*mesh_u*eta
            whj0 -= eta*s_whj/l
        else:
            break
    return whj0, wmh0


# ---| RMSProp (Running Mean Square) |---
def rms_propagation(X0, Y, h, m, alpha, lam, eta, acc, eps, Q, L, a, u, sigm, one=0):
    '''
    :param X: training sample - numpy array
    :param Y: training sample - numpy array
    :param h: Количество нейронов скрытого слоя
    :param m: Количество нейронов выходного слоя
    :param lam: the rate of forgetting - value in (0, 1)
    :param eta: gradient step (learning rate) - value in R
    :param acc: accuracy - value in (0, 1)
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
    # Расчет всех градиентов
    grads_wmh = np.array([])
    grads_whj = np.array([])
    for xi, yi in zip(X, Y):
        x, y = xi.reshape(-1, 1).copy(), yi.reshape(-1, 1).copy()
        ui = u(x, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - y)
        sdf_m = sd(sigm, wmh0.dot(ui))
        sdf_h = sd(sigm, whj0.dot(x))
        eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
        mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
        grads_wmh = np.append(grads_wmh, mesh_t * mesh_u * eta)
        mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
        grads_whj = np.append(grads_whj, mesh_t * mesh_u * eta)
    s_wmh = (grads_wmh.reshape(l, m, h + one).sum(axis=0))**2
    s_whj = (grads_whj.reshape(l, h, n).sum(axis=0))**2
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
        if True:  # (q1 - q0) >= acc*(1 + q1)
            # Обратный ход
            q0 = q1
            sdf_m = sd(sigm, wmh0.dot(ui))
            sdf_h = sd(sigm, whj0.dot(xi))
            eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
            # Градиентный шаг
            # Веса выходного слоя
            mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim*sdf_m).flatten())
            lg = mesh_t * mesh_u
            s_wmh = alpha*s_wmh + (1-alpha)*(lg**2)
            wmh0 -= eta*lg / (s_wmh**0.5 + eps)
            # Веса скрытого слоя
            mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih*sdf_h).flatten())
            lg = mesh_t * mesh_u
            s_whj = alpha*s_whj + (1 - alpha) * (lg**2)
            whj0 -= eta*lg / (s_whj**0.5 + eps)
        else:
            break
    return whj0, wmh0


# ---| AdaDelta (Adaptive Learning Rate) |---
def ada_delta(X0, Y, h, m, alpha, lam, eta, acc, eps, Q, L, a, u, sigm, one=0):
    '''
    :param X: training sample - numpy array
    :param Y: training sample - numpy array
    :param h: Количество нейронов скрытого слоя
    :param m: Количество нейронов выходного слоя
    :param lam: the rate of forgetting - value in (0, 1)
    :param eta: gradient step (learning rate) - value in R
    :param acc: accuracy - value in (0, 1)
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
    # Расчет всех градиентов
    grads_wmh = np.array([])
    grads_whj = np.array([])
    for xi, yi in zip(X, Y):
        x, y = xi.reshape(-1, 1).copy(), yi.reshape(-1, 1).copy()
        ui = u(x, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - y)
        sdf_m = sd(sigm, wmh0.dot(ui))
        sdf_h = sd(sigm, whj0.dot(x))
        eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
        mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
        grads_wmh = np.append(grads_wmh, mesh_t * mesh_u * eta)
        mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
        grads_whj = np.append(grads_whj, mesh_t * mesh_u * eta)
    s_wmh = (grads_wmh.reshape(l, m, h + one).sum(axis=0))**2
    s_whj = (grads_whj.reshape(l, h, n).sum(axis=0))**2
    q0 = Q(L, whj0, wmh0, X, Y, one)
    d_wmh, d_whj = 0, 0
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
        if True:  # (q1 - q0) >= acc*(1 + q1)
            # Обратный ход
            q0 = q1
            sdf_m = sd(sigm, wmh0.dot(ui))
            sdf_h = sd(sigm, whj0.dot(xi))
            eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
            # Градиентный шаг
            # Веса выходного слоя
            mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
            lg = mesh_t * mesh_u
            s_wmh = alpha*s_wmh + (1-alpha)*(lg**2)
            delta_wmh = lg * ((d_wmh**0.5 + eps)/(s_wmh**0.5 + eps))
            d_wmh = alpha*d_wmh + (1-alpha)*(delta_wmh**2)
            wmh0 -= eta*delta_wmh
            # Веса скрытого слоя
            mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
            lg = mesh_t * mesh_u
            s_whj = alpha*s_whj + (1-alpha)*(lg**2)
            delta_whj = lg * ((d_whj**0.5 + eps) / (s_whj**0.5 + eps))
            d_whj = alpha*d_whj + (1-alpha)*(delta_whj**2)
            whj0 -= eta*delta_whj
        else:
            break
    return whj0, wmh0


# ---| Adam (Adaptive Momentum) |---
def adam(X0, Y, h, m, gamma, alpha, eps, lam, eta, acc, Q, L, a, u, sigm, one=0):
    '''
    :param X: training sample - numpy array
    :param Y: training sample - numpy array
    :param h: Количество нейронов скрытого слоя
    :param m: Количество нейронов выходного слоя
    :param lam: the rate of forgetting - value in (0, 1)
    :param eta: gradient step (learning rate) - value in R
    :param acc: accuracy - value in (0, 1)
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
    # Расчет всех градиентов
    grads_wmh = np.array([])
    grads_whj = np.array([])
    for xi, yi in zip(X, Y):
        x, y = xi.reshape(-1, 1).copy(), yi.reshape(-1, 1).copy()
        ui = u(x, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - y)
        sdf_m = sd(sigm, wmh0.dot(ui))
        sdf_h = sd(sigm, whj0.dot(x))
        eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
        mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
        grads_wmh = np.append(grads_wmh, mesh_t * mesh_u * eta)
        mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
        grads_whj = np.append(grads_whj, mesh_t * mesh_u * eta)
    s_wmh = (grads_wmh.reshape(l, m, h + one).sum(axis=0))**2
    s_whj = (grads_whj.reshape(l, h, n).sum(axis=0))**2
    q0 = Q(L, whj0, wmh0, X, Y, one)
    v_wmh, v_whj = 0, 0
    k = 1
    while k <= 10000:  # True
        # Прямой ход
        i = np.random.randint(0, l)
        xi, yi = X[i].reshape(-1, 1), Y[i].reshape(-1, 1)
        ui = u(xi, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - yi)
        q1 = (1-lam)*q0 + lam*L(whj0, wmh0, xi, yi, one)
        if True:  # (q1 - q0) >= acc*(1 + q1)
            # Обратный ход
            q0 = q1
            sdf_m = sd(sigm, wmh0.dot(ui))
            sdf_h = sd(sigm, whj0.dot(xi))
            eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
            # Градиентный шаг
            # Веса выходного слоя
            mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
            lg = mesh_t * mesh_u
            v_wmh = gamma*v_wmh + (1-gamma)*lg
            vs_wmh = v_wmh*((1-gamma**k)**-1)
            s_wmh = alpha*s_wmh + (1-alpha)*(lg**2)
            ss_wmh = s_wmh*((1-alpha**k)**-1)
            wmh0 -= eta*vs_wmh / (ss_wmh**0.5 + eps)
            # Веса скрытого слоя
            mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
            lg = mesh_t * mesh_u
            v_whj = gamma*v_whj + (1-gamma)*lg
            vs_whj = v_whj*((1-gamma**k)**-1)
            s_whj = alpha*s_whj + (1-alpha)*(lg**2)
            ss_whj = s_whj*((1-alpha**k)**-1)
            whj0 -= eta*vs_whj / (ss_whj**0.5 + eps)
        else:
            break
        k += 1
    return whj0, wmh0


# ---| Nadam (Nesterov-accelerated adaptive momentum) |---
def nadam(X0, Y, h, m, gamma, alpha, eps, lam, eta, acc, Q, L, a, u, sigm, one=0):
    '''
    :param X: training sample - numpy array
    :param Y: training sample - numpy array
    :param h: Количество нейронов скрытого слоя
    :param m: Количество нейронов выходного слоя
    :param lam: the rate of forgetting - value in (0, 1)
    :param eta: gradient step (learning rate) - value in R
    :param acc: accuracy - value in (0, 1)
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
    # Расчет всех градиентов
    grads_wmh = np.array([])
    grads_whj = np.array([])
    for xi, yi in zip(X, Y):
        x, y = xi.reshape(-1, 1).copy(), yi.reshape(-1, 1).copy()
        ui = u(x, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - y)
        sdf_m = sd(sigm, wmh0.dot(ui))
        sdf_h = sd(sigm, whj0.dot(x))
        eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
        mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
        grads_wmh = np.append(grads_wmh, mesh_t * mesh_u * eta)
        mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
        grads_whj = np.append(grads_whj, mesh_t * mesh_u * eta)
    s_wmh = (grads_wmh.reshape(l, m, h + one).sum(axis=0))**2
    s_whj = (grads_whj.reshape(l, h, n).sum(axis=0))**2
    q0 = Q(L, whj0, wmh0, X, Y, one)
    v_wmh, v_whj = 0, 0
    k = 1
    while k <= 10000:  # True
        # Прямой ход
        i = np.random.randint(0, l)
        xi, yi = X[i].reshape(-1, 1), Y[i].reshape(-1, 1)
        ui = u(xi, sigm, whj0, one)
        ai = a(ui, sigm, wmh0)
        eim = (ai - yi)
        q1 = (1-lam)*q0 + lam*L(whj0, wmh0, xi, yi, one)
        if True:  # (q1 - q0) >= acc*(1 + q1)
            # Обратный ход
            q0 = q1
            sdf_m = sd(sigm, wmh0.dot(ui))
            sdf_h = sd(sigm, whj0.dot(xi))
            eih = wmh0[:, :(-one or None)].transpose().dot(eim * sdf_m)
            # Градиентный шаг
            # Веса выходного слоя
            mesh_t, mesh_u = np.meshgrid(ui.flatten(), (eim * sdf_m).flatten())
            lg = mesh_t * mesh_u
            v_wmh = gamma*v_wmh + (1-gamma)*lg
            vs_wmh = v_wmh*((1-gamma**k)**-1)
            s_wmh = alpha*s_wmh + (1-alpha)*(lg**2)
            ss_wmh = s_wmh*((1-alpha**k)**-1)
            wmh0 -= eta*vs_wmh / (ss_wmh**0.5 + eps)
            # Веса скрытого слоя
            mesh_t, mesh_u = np.meshgrid(xi.flatten(), (eih * sdf_h).flatten())
            lg = mesh_t * mesh_u
            v_whj = gamma*v_whj + (1-gamma)*lg
            vs_whj = v_whj*((1-gamma**k)**-1)
            s_whj = alpha*s_whj + (1-alpha)*(lg**2)
            ss_whj = s_whj*((1-alpha**k)**-1)
            whj0 -= eta*vs_whj / (ss_whj**0.5 + eps)
        else:
            break
        k += 1
    return whj0, wmh0
