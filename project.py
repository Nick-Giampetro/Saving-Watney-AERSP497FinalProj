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

branin_ = lambda x : jnp.array(((x[1] - (5.1*x[0]**2)/(4*jnp.pi**2) + (5*x[0])/jnp.pi - 6)**2 + 10*(1 - 1/(8*jnp.pi))*jnp.cos(x[0]) + 10))
bounds_branin = np.array([[-5., 0], [10., 15.]])

c1 = lambda x: (x[0] - 3)**2 + (x[1] +6)**2 - 225
c2 = lambda x: (x[0] + 10)**2 + (x[1] +5)**2 - 300
c3 = lambda x: x[0] + 15/8 * x[1] - 85/4 # equality constraint
c4 = lambda x: -x[0] + 10
c5 = lambda x: x[0] + 5
c6 = lambda x: x[1]
c7 = lambda x: -x[1] + 15

bounds = Bounds([-5., 0], [10., 15.])

def Objective_Function (x) :

def Steepest_Decent_Loop (func) :

    gfunc = grad(func)

    x0    = np.array([6., 10.]) # starting point

    eps   = 1e-3          # tolerance for convergence
    maxiters = 100         # maximum number of iterations

    bounds= bounds_branin # optimization variable bounds
    g_inf = 10                                     # starting gradient infinity norm
    k = 0                                          # iteration counter
    xk = x0                                        # starting point
    fk    = func(xk)

    # empty lists to store optimization history
    ginf_sd_b = []        # first-order optimality
    ginf_sd_b.append(np.linalg.norm(gfunc(x0), ord=np.inf))
    xk_sd_b = []          # iterate history
    xk_sd_b.append(x0)    # include the starting point
    ncalls_sd_b = []      # number of function calls
    f_sd_b      = []      # objective history
    f_sd_b.append(fk)
    stepDist_sd_b = []
    stepDist_sd_b.append(2)

    np.set_printoptions(precision=3)
    print(f'starting point x0: {xk}, f0: {fk}')
    print('----------------------------------')

    while g_inf >= eps and k < maxiters:
        gk = gfunc(xk)
        pk = -gk/np.linalg.norm(gk)                # steepest-descent direction
        sl = Step_Length(func, xk, pk)        # calculate step length
        alpha = sl[0]
        xk = xk + alpha * pk                       # new iterate
        fk = func(xk)                              # evaluate f at new iterate
        g_inf = np.linalg.norm(gk, ord=np.inf)     # check first-order optimality (gradient)

        k += 1
        ncalls_sd_b.append(sl[1])
        ginf_sd_b.append(g_inf)
        xk_sd_b.append(xk)
        f_sd_b.append(fk)
        stepDist_sd_b.append(alpha)


        print(f'iteration {k} ,  function calls: {sl[1]},  alpha: {alpha:1.7f}, xk: {xk.squeeze()}, fk: {fk.item():2.6f}, gradient norm: {g_inf:2.6f}')

    return [xk_sd_b,ginf_sd_b,f_sd_b]


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
    ginf_sd_b = []  # first-order optimality
    ginf_sd_b.append(g_inf)
    xk_sd_b = []  # iterate history
    xk_sd_b.append(x0)  # include the starting point
    ncalls_sd_b = []  # number of function calls
    f_sd_b = []  # objective history
    f_sd_b.append(fk)

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
            ncalls_sd_b.append(sl[1])
            ginf_sd_b.append(g_inf)
            xk_sd_b.append(xk)
            f_sd_b.append(fk)

            print(
                f'iteration {k} ,  function calls: {sl[1]},  alpha: {alpha:1.7f}, xk: {xk.squeeze()}, fk: {fk.item():2.6f}, gradient norm: {g_inf:2.6f}')

        mu = mu * rho
        tau = tau * eta

        print(f'mu: {mu} , tau: {tau}')
    return (ginf_sd_b, xk_sd_b)


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


def cob_callback(x):
    xx.append(x)  # iterate xk
    fx.append(f(x))  # function value
    c1x.append(ineq_con1['fun'](x))  # constraint evaluation for c1 only
    c2x.append(ineq_con2['fun'](x))
    c3x.append(eq_con3['fun'](x))
    c3ax.append(ineq_con3a['fun'](x))
    c3bx.append(ineq_con3b['fun'](x))
    c4x.append(ineq_con4['fun'](x))
    c5x.append(ineq_con5['fun'](x))
    c6x.append(ineq_con6['fun'](x))
    c7x.append(ineq_con7['fun'](x))

    print(f'xk {x}, fk {f(x):1.7f}, c1 {c1(x):1.7f}, c2 {c2(x):1.7f}, c3 {c3(x):1.7f}')


def SCIPY_SLSQP (f,init,c1,c2,c3,c4,c5,c6,c7) :
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

    res = minimize(f, x0, method='SLSQP', jac=grad(f),
                   constraints=[ineq_con1, ineq_con2, eq_con3, ineq_con4, ineq_con5, ineq_con6, ineq_con7],
                   options={'disp': True},
                   bounds=bounds, callback=callback)

    xSLSQP = np.zeros((len(fx), 2))
    fSLSQP = np.zeros((len(fx), 1))
    cSLSQP = np.zeros((len(fx), 1))
    for i in range(len(fx)):
        xSLSQP[i, 0] = np.concatenate(xx)[i * 2]
        xSLSQP[i, 1] = np.concatenate(xx)[i * 2 + 1]
        fSLSQP[i] = fx[i]
        cSLSQP[i] = max(max(0, -c1x[i]), max(0, -c2x[i]), c3x[i] ** 2, max(0, -c4x[i]), max(0, -c5x[i]), max(0, -c6x[i]), max(0, -c7x[i]))
    return (xSLSQP,fSLSQP,cSLSQP)

def COBYLA_SLSQP (f,init,c1,c2,c3,c4,c5,c6,c7) :
    x0 = init

    c3a = lambda x: x[0] + 15 / 8 * x[1] - (85 / 4 - 1 / 25)
    c3b = lambda x: - x[0] - 15 / 8 * x[1] + (85 / 4 + 1 / 25)

    ineq_con3a = {'type': 'ineq',
                  'fun': c3a,
                  'jac': grad(c3a)}

    ineq_con3b = {'type': 'ineq',
                  'fun': c3b,
                  'jac': grad(c3b)}

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
                   bounds=bounds, callback=cob_callback)

    xCOB = np.zeros((len(fx), 2))
    fCOB = np.zeros((len(fx), 1))
    cCOB = np.zeros((len(fx), 1))
    for i in range(len(fx)):
        xCOB[i, 0] = np.concatenate(xx)[i * 2]
        xCOB[i, 1] = np.concatenate(xx)[i * 2 + 1]
        fCOB[i] = fx[i]
        cCOB[i] = max(max(0, -c1x[i]), max(0, -c2x[i]), max(0, -c3x[i]), max(0, -c3bx[i]), max(0, -c4x[i]), max(0, -c5x[i]), max(0, -c6x[i]), max(0, -c7x[i]))
    return (xCOB,fCOB,cCOB)


