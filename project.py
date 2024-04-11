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

def Objective_Function(x) :

def Steepest_Decent_Loop(func) :

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

def Step_Length(func, xk, pk):
    k = 0

    c1 = 1e-2
    alpha = 2
    gfunc = grad(func)
    while True:
        k += 1
        sufDec = ((func(xk + alpha * pk)) < (func(xk) + c1 * alpha * jnp.dot(gfunc(xk), pk)))
        if sufDec or (alpha < 1e-4):
            nalpha = alpha
            break

        C = func(xk)
        B = -np.linalg.norm(gfunc(xk))
        alpha_i = xk + alpha * pk
        A = (func(alpha_i) - B * alpha - C) / (alpha ** 2)
        nalpha = -B / (2 * A)

        # print(nalpha < 0.3*alpha)

        if nalpha < 0.25 * alpha:
            nalpha = alpha * .5
        alpha = nalpha

    return [nalpha, k]

