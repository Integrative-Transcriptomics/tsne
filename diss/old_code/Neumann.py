import gzip
import pickle

import sys, os
from os import path

import time
import numpy as onp

import jax
from jax.lax import scan, cond
from jax import random, flatten_util
import jax.numpy as np
from jax import vjp, jvp, custom_vjp, jacfwd, jacrev, vmap, grad
from functools import partial
import openTSNE
from sklearn.utils import check_random_state
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from tsne.diss.tsne_jax_old import x2p, y2q


def neumannApproximation_vis(f_vjp, v, Y, iterations, alpha):
    '''Neumann approximation of inverse-Hessian-vector product for visualization'''
    p = v
    for i in range(iterations):
        v -= alpha * f_vjp(v)[0]
        p += v
    return p

def neumannApproximation(f_vjp, v, iterations):
    '''Neumann approximation of inverse-Hessian-vector product'''
    p = v
    for i in range(iterations):
        v -= f_vjp(v)[0]
        p += v
    return p

def compute_cov(neumann_fun, Jacobian_vjp, Jacobian_jvp, A, B, i, N, D):
    v1 = np.ravel(jax.nn.one_hot(np.array([i]), 2*N))
    print(v1.shape)
    v2 = neumann_fun(v1)
    print(v2.shape)
    v3 = Jacobian_vjp(v2)[0]
    print(v3.shape)
    v4 = np.ravel(np.dot(np.dot(A, np.reshape(v3, (N, D), 'C')), np.transpose(B)), 'C')
    print(v4.shape)
    v5 = Jacobian_jvp(v4)[0]
    print(v5)
    v6 = neumann_fun(v5)
    return v6

def error_propagation_tsne(X_flat, X_unflattener, Y_flat, Y_unflattener, A, B, neumann_iterations=200):
    N, D = X_unflattener(X_flat).shape
    funY = lambda y: regularized_KL_divergence(X_flat, y, X_unflattener, Y_unflattener)
    funX = lambda x: regularized_KL_divergence(x, Y_flat, X_unflattener, Y_unflattener)
    funXY = partial(regularized_KL_divergence, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
    _, hessian_vjp = vjp(grad(funY), Y_flat)
    neumann_fun = lambda v: neumannApproximation(hessian_vjp, v, neumann_iterations)
    Jx_fun = lambda x: jax.grad(funXY, argnums=1)(x, Y_flat)
    _, Jacobian_vjp = vjp(Jx_fun, X_flat)
    Jx_fun2 = lambda y: jax.grad(funXY, argnums=0)(X_flat, y)
    _, Jacobian_jvp = vjp(Jx_fun2, Y_flat)
    compute_cov_fun = lambda i: compute_cov(neumann_fun, Jacobian_vjp, Jacobian_jvp, A, B, i, N, D)
    cov = vmap(compute_cov_fun)(np.array([i for i in range(2*N)]))
    return cov

def get_Neumann_approximation(X_flat, X_unflattener, Y_flat, Y_unflattener, reg_factor = 0.00001, neumann_iterations=200):
    fun = lambda x: regularized_KL_divergence(X_flat, x, X_unflattener, Y_unflattener, reg_factor)
    _, f_vjp = vjp(grad(fun), Y_flat)
    v_in = np.eye(len(Y_flat))
    neumann_fun = lambda x: neumannApproximation_vis(f_vjp, x, Y_flat, neumann_iterations)
    H_inv_appr = vmap(neumann_fun)(v_in)
    return H_inv_appr