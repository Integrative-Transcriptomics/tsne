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
from jax import vjp, custom_vjp, jacfwd, jacrev, vmap
from functools import partial
import openTSNE
from sklearn.utils import check_random_state
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from tsne_jax import x2p, y2q


def KL_divergence(X_flat, Y_flat, X_unflattener, Y_unflattener):
    """
    (R^nxp x R^nxp)--> R
    """
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)
    learning_rate, perplexity = (200, 30.0)
    P = x2p(X, tol=1e-5, perplexity=perplexity)
    P = (P + np.transpose(P))
    P = P / np.sum(P)      # Why don't we devide by 2N as described everywhere?
    P = np.maximum(P, 1e-12)
    Q = y2q(Y)
    return np.sum(P * (np.log(P+1e-10) - np.log(Q+1e-10)))


def regularized_KL_divergence(X_flat, Y_flat, X_unflattener, Y_unflattener):
    """
    (R^nxp x R^nxp)--> R
    """
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)
    learning_rate, perplexity = (200, 30.0)
    P = x2p(X, tol=1e-5, perplexity=perplexity)
    P = (P + np.transpose(P))
    P = P / np.sum(P)      # Why don't we devide by 2N as described everywhere?
    P = np.maximum(P, 1e-12)
    Q = y2q(Y)
    return np.sum(P * (np.log(P+1e-10) - np.log(Q+1e-10))) + 1/Y.shape[0] * 0.0005*np.sum(np.square(Y))

def KL_divergence_log(X_flat, Y_flat, X_unflattener, Y_unflattener):
    """
    (R^nxp x R^nxp)--> R
    """
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)
    learning_rate, perplexity = (200, 30.0)
    P = x2p(X, tol=1e-5, perplexity=perplexity)
    Q = y2q(Y)
    return np.log(np.sum(P * (np.log(P+1e-10) - np.log(Q+1e-10))))

def Hessian_y_y(f, X, Y):
    '''
    nxp, nx2 --> 2n x 2n
    '''
    X_flat, X_unflattener = flatten_util.ravel_pytree(X)   # row-wise
    Y_flat, Y_unflattener = flatten_util.ravel_pytree(Y)   # row-wise
    H = jax.hessian(f, argnums=1)(X_flat, Y_flat, X_unflattener, Y_unflattener)
    return H

def Mixed_Jacobian_x_y(f, X, Y):
    '''
    Symmetry of mixed partials (order of derivatives doesn't matter)
    nxp, nx2 --> 2n x np
    '''    
    X_flat, X_unflattener = flatten_util.ravel_pytree(X)   # row-wise
    Y_flat, Y_unflattener = flatten_util.ravel_pytree(Y)   # row-wise
    #J_X_Y = jacrev(jacfwd(f, argnums=0), argnums=1)(X_flat, Y_flat, X_unflattener, Y_unflattener)
    J_X_Y = jacfwd(jacfwd(f, argnums=1), argnums=0)(X_flat, Y_flat, X_unflattener, Y_unflattener)
    return J_X_Y

def d_y_star_d_x(H, J):
    lu, piv= jax.scipy.linalg.lu_factor(H+1e-5*np.eye(len(H)))
    return jax.scipy.linalg.lu_solve((lu, piv), -J), H+1e-5*np.eye(len(H)), J

def d_y_star_d_x_outer(f, X, Y_star):
    return d_y_star_d_x(Hessian_y_y(f, X, Y_star), Mixed_Jacobian_x_y(f, X, Y_star))

def tsne_fwd(x, y_guess):
    affinity = openTSNE.affinity.PerplexityBasedNN(
        x,
        perplexity=30.0,
        method="annoy",
        random_state=42,
        verbose=True,
    )

    init = openTSNE.initialization.random(
        x, n_components=2, random_state=42, verbose=True,
    )
    
    y_star = openTSNE.TSNEEmbedding(
        init,
        affinity,
        learning_rate=200,
        negative_gradient_method="fft",
        random_state=42,
        verbose=False
    )
    y_star.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
    y_star.optimize(750, momentum=0.5, inplace=True)
    return y_star









