import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
import matplotlib.patches as patches
import jax.numpy as jnp # this is a thin wrapper to NumPy within JAX
from jax import grad, hessian
from scipy.optimize import minimize
from scipy.optimize import Bounds

def TransferCalc (delVi,phi) :
    rE = 149.95e6
    rM = 228e6
    muS = 1.327e11

    vE = (muS / rE)**0.5
    vM = (muS / rM) ** 0.5
    r0 = jnp.array([rE, 0])
    v0 = jnp.array([delVi*np.sin(phi), vE + delVi*jnp.cos(phi)])

    E = jnp.linalg.norm(v0)**2 / 2 - muS / rE
    a = -muS / (2 * E)

    hvec = jnp.cross([r0,0],[v0,0])
    hmag = jnp.linalg.norm(hvec)

    p = hmag**2 / muS
    e = (1 - p/a)**0.5

    thetai = jnp.arccos(p/(jnp.linalg.norm(r0)*e) - 1/e) * jnp.sign(jnp.dot(r0, v0))

    thetaf = jnp.arccos(p/(rM*e) - 1/e)

    dtheta = thetaf - thetai

    E1 = 2*jnp.arctan(((1-e)/(1+e))**0.5 * jnp.arctan(thetai/2))
    E2 = 2*jnp.arctan(((1-e)/(1+e))**0.5 * jnp.arctan(thetaf/2))

    dM = E2 - E1 + e*jnp.sin(E1) - e*jnp.sin(E2)
    dt = (a^3/muS)**0.5 * dM

    Ft = jnp.dot(r0, v0) / (p * rE) * (1 - jnp.cos(dtheta)) - 1 / rE * (muS / p)**0.5 * jnp.sin(dtheta)
    Gt = 1 - rE / p * (1 - jnp.cos(dtheta))

    v2 = Ft * r0 + Gt * v0

    vM = jnp.array([-vM*jnp.sin(dtheta), vM*jnp.cos(dtheta)])
    delV2 = jnp.linalg.norm(vM - v2)

    return dtheta, dt, delV2

def totalFun (delVi, phi) :
    mtotal = 6000
    vex = 4.5126
    mstruct = 1000

    [delth, delt, delV2] = TransferCalc(delVi, phi)

    # 11520 = Mar 12, 2035
    t0 = 11520 * 24 * 3600
    wE = 2*jnp.pi/(365 * 24 * 3600 + 6 * 3600 + 9 * 60)
    wM = 2*jnp.pi/(687 * 24 * 3600)

    dw = wM - wE

    theta0 = (dw * t0) % (2 * jnp.pi)

    tf = (delth - theta0) / dw - wM / dw * delt + delt
    ti = tf - delt

    delV = delVi + delV2

    mc = mtotal * jnp.e**(-delV / vex) - mstruct
    mf = mtotal - mstruct - mc

    return mf, tf

def Objective_Function (x) :
    [mf, tf] = totalFun (x[0], x[1])
    w1 = 0.5
    w2 = 1-w1
    return w1*mf + w2*tf


def Step_Length_Q (func, xk, pk, mu):
    k = 0

    c1 = 1e-3
    alpha = 5
    gfunc = grad(func)
    while True:
        k += 1
        sufDec = ((func(xk + alpha * pk, mu)) < (func(xk, mu) + c1 * alpha * jnp.dot(gfunc(xk, mu), pk)))
        if sufDec or (alpha < 1e-3):
            break

        C = func(xk, mu)
        B = -np.linalg.norm(gfunc(xk, mu))
        alpha_i = xk + alpha * pk
        A = (func(alpha_i, mu) - B * alpha - C) / (alpha ** 2)
        alpha = -B / (2 * A)

    return [alpha, k]


def Quad_Penalty (func, x0, mu, tau, eta, rho):
    gfunc = grad(func)

    maxiters = 250  # maximum number of iterations

    bounds = jnp.array([[-5., 0], [10., 15.]])  # optimization variable bounds

    g_inf = np.linalg.norm(gfunc(x0, mu), ord=np.inf)  # starting gradient infinity norm
    k = 0  # iteration counter
    xk = x0  # starting point
    fk = func(xk, mu)

    # empty lists to store optimization history
    gQPen = []  # first-order optimality
    gQPen.append(g_inf)
    xQPen = []  # iterate history
    xQPen.append(x0)  # include the starting point
    fQPen = []  # objective history
    fQPen.append(fk)
    cQPen = []
    cQPen.append(max(max(0, -c1(x0)), max(0, -c2(x0)), c3(x0) ** 2, max(0, -c4(x0)), max(0, -c5(x0)), max(0, -c6(x0)), max(0, -c7(x0))))

    np.set_printoptions(precision=3)
    print(f'starting point x0: {xk}, f0: {fk}')
    print('----------------------------------')

    while tau > 1e-4 or mu < 10:
        gfunc = grad(func)
        while g_inf >= tau and k < maxiters:
            gk = gfunc(xk, mu)
            pk = -gk / np.linalg.norm(gk)  # steepest-descent direction
            sl = Step_Length_Q(func, xk, pk, mu)  # calculate step length
            alpha = sl[0]
            xk = xk + alpha * pk  # new iterate

            fk = func(xk, mu)  # evaluate f at new iterate
            g_inf = np.linalg.norm(gk, ord=np.inf)  # check first-order optimality (gradient)

            k += 1
            gQPen.append(g_inf)
            xQPen.append(xk)
            fQPen.append(fk)
            cQPen.append(max(max(0, -c1(xk)), max(0, -c2(xk)), c3(xk) ** 2, max(0, -c4(xk)), max(0, -c5(xk)), max(0, -c6(xk)), max(0, -c7(xk))))

            print(
                f'iteration {k} ,  function calls: {sl[1]},  alpha: {alpha:1.7f}, xk: {xk.squeeze()}, fk: {fk.item():2.6f}, gradient norm: {g_inf:2.6f}')

        mu = mu * rho
        tau = tau * eta

        print(f'mu: {mu} , tau: {tau}')
    return xQPen, fQPen, gQPen, cQPen


def SCIPY_SLSQP (f,init) :
    gk = grad(f)
    def callback(x):
        xx.append(x)  # iterate xk
        fx.append(f(x))  # function value
        c1x.append(ineq_con1['fun'](x))  # constraint evaluation for c1 only
        c2x.append(ineq_con2['fun'](x))
        c3x.append(eq_con3['fun'](x))
        c4x.append(ineq_con4['fun'](x))
        c5x.append(ineq_con5['fun'](x))
        c6x.append(ineq_con6['fun'](x))
        c7x.append(ineq_con7['fun'](x))

        print(f'xk {x}, fk {f(x):1.7f}, c1 {c1(x):1.7f}, c2 {c2(x):1.7f}, c3 {c3(x):1.7f}')

    x0 = init

    # this will need tailored to our specific problem later
    xx = []
    xx.append(x0)
    fx = []
    fx.append(f(x0))
    c1x = []
    c1x.append(c1(x0))
    c2x = []
    c2x.append(c2(x0))
    c3x = []
    c3x.append(c3(x0))
    c4x = []
    c4x.append(c4(x0))
    c5x = []
    c5x.append(c5(x0))
    c6x = []
    c6x.append(c6(x0))
    c7x = []
    c7x.append(c7(x0))

    res = minimize(f, x0, method='SLSQP', jac=gk,
                   constraints=[ineq_con1, ineq_con2, eq_con3, ineq_con4, ineq_con5, ineq_con6, ineq_con7],
                   options={'disp': True},
                   bounds=bounds_branin, callback=callback)

    xSLSQP = np.zeros((len(fx), 2))
    fSLSQP = np.zeros((len(fx), 1))
    cSLSQP = np.zeros((len(fx), 1))
    gSLSQP = np.zeros((len(fx), 1))
    for i in range(len(fx)):
        xSLSQP[i, 0] = np.concatenate(xx)[i * 2]
        xSLSQP[i, 1] = np.concatenate(xx)[i * 2 + 1]
        gSLSQP[i] = np.linalg.norm(gk(xSLSQP[i, :]))
        fSLSQP[i] = fx[i]
        cSLSQP[i] = max(max(0, -c1x[i]), max(0, -c2x[i]), c3x[i] ** 2, max(0, -c4x[i]), max(0, -c5x[i]), max(0, -c6x[i]), max(0, -c7x[i]))
    return xSLSQP, fSLSQP, gSLSQP, cSLSQP

def SCIPY_COBYLA (f,init) :
    gk = grad(f)

    def cob_callback(x):
        xx.append(x)  # iterate xk
        fx.append(f(x))  # function value
        c1x.append(ineq_con1['fun'](x))
        c2x.append(ineq_con2['fun'](x))
        c3x.append(eq_con3['fun'](x))
        c3ax.append(ineq_con3a['fun'](x))
        c3bx.append(ineq_con3b['fun'](x))
        c4x.append(ineq_con4['fun'](x))
        c5x.append(ineq_con5['fun'](x))
        c6x.append(ineq_con6['fun'](x))
        c7x.append(ineq_con7['fun'](x))

        print(f'xk {x}, fk {f(x):1.7f}, c1 {c1(x):1.7f}, c2 {c2(x):1.7f}, c3 {c3(x):1.7f}')

    x0 = init

    # this will need tailored to our specific problem later
    xx = []
    xx.append(x0)
    fx = []
    fx.append(f(x0))
    c1x = []
    c1x.append(c1(x0))
    c2x = []
    c2x.append(c2(x0))
    c3x = []
    c3x.append(c3(x0))
    c3ax = []
    c3ax.append(c3a(x0))
    c3bx = []
    c3bx.append(c3b(x0))
    c4x = []
    c4x.append(c4(x0))
    c5x = []
    c5x.append(c5(x0))
    c6x = []
    c6x.append(c6(x0))
    c7x = []
    c7x.append(c7(x0))

    res = minimize(f, x0, method='COBYLA',
                   constraints=[ineq_con1, ineq_con2, ineq_con3a, ineq_con3b, ineq_con4, ineq_con5, ineq_con6, ineq_con7],
                   options={'disp': True},
                   callback=cob_callback)

    xCOB = np.zeros((len(fx), 2))
    fCOB = np.zeros((len(fx), 1))
    cCOB = np.zeros((len(fx), 1))
    gCOB = np.zeros((len(fx), 1))
    for i in range(len(fx)):
        xCOB[i, 0] = np.concatenate(xx)[i * 2]
        xCOB[i, 1] = np.concatenate(xx)[i * 2 + 1]
        gCOB[i] = np.linalg.norm(gk(xCOB[i, :]))
        fCOB[i] = fx[i]
        cCOB[i] = max(max(0, -c1x[i]), max(0, -c2x[i]), max(0, -c3x[i]), max(0, -c4x[i]), max(0, -c5x[i]), max(0, -c6x[i]), max(0, -c7x[i]))
    return xCOB, fCOB, gCOB, cCOB


[xSLSQP, fSLSQP, gSLSQP, cSLSQP] = SCIPY_SLSQP(branin_, np.array([6.0, 10.0]))
[xCOB, fCOB, gCOB, cCOB] = SCIPY_COBYLA(branin_, np.array([6.0, 10.0]))
[xQPen, fQPen, gQPen, cQPen] = Quad_Penalty(branin_pen, np.array([6.0, 10.0]), 0.001, 1, 0.5, 2)

plt.figure(figsize=(8,8))
plt.plot(cSLSQP, marker='o', label = 'SLSQP (SciPy)')
plt.plot(cCOB, marker='o', label = 'COBYLA (SciPy)')
plt.plot(cQPen, marker='o', label = 'Quadratic Penalty (Personal)')
plt.xlabel('optimization iteration (k)')
plt.ylabel('Maximum Constraint Violation (MCV)')
plt.title('MCV vs iterations')
plt.legend()

plt.figure(figsize=(8,8))
plt.plot(gSLSQP, marker='o', label = 'SLSQP (SciPy)')
plt.plot(gCOB, marker='o', label = 'COBYLA (SciPy)')
plt.plot(gQPen, marker='o', label = 'Quadratic Penalty (Personal)')
plt.xlabel('optimization iteration (k)')
plt.ylabel(r'$ ||\nabla $$f(x_k)||$')
plt.title(r'$ ||\nabla $$f(x_k)||$ vs iterations')
plt.legend()

plt.figure(figsize=(8,8))
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.title(r'Trajectory Plot')

xxSLSQP = np.zeros((len(xSLSQP), 2))
for i in range(len(xSLSQP)):
    xxSLSQP[i,0] = np.concatenate(xSLSQP)[i*2]
    xxSLSQP[i,1] = np.concatenate(xSLSQP)[i*2 + 1]

xxCOB = np.zeros((len(xCOB), 2))
for i in range(len(xCOB)):
    xxCOB[i, 0] = np.concatenate(xCOB)[i*2]
    xxCOB[i, 1] = np.concatenate(xCOB)[i*2 + 1]

xxQPen = np.zeros((len(xQPen), 2))
for i in range(len(xQPen)):
    xxQPen[i, 0] = np.concatenate(xQPen)[i*2]
    xxQPen[i, 1] = np.concatenate(xQPen)[i*2 + 1]

plt.plot(xxSLSQP[:, 0], xxSLSQP[:, 1], marker='o', label='SLSQP (SciPy)')
plt.plot(xxCOB[:, 0], xxCOB[:, 1], marker='o', label='COBYLA (SciPy)')
plt.plot(xxQPen[:, 0], xxQPen[:, 1], marker='o', label='Quadratic Penalty (Personal)')
plt.legend()

plt.show()