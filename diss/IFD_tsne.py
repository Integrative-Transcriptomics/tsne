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

from tsne_jax import x2p, y2q

############################ Functions starting from KL_divergence #################################

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
    Q, _ = y2q(Y)
    return np.sum(P * (np.log(P+1e-10) - np.log(Q+1e-10)))
    
def regularized_KL_divergence(X_flat, Y_flat, X_unflattener, Y_unflattener, reg_factor=0.001):
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
    Q, _ = y2q(Y)
    return np.sum(P * (np.log(P+1e-10) - np.log(Q+1e-10))) + reg_factor * np.sum(np.square(Y))

def neumannApproximation_vis(f_vjp, v, Y, iterations):
    '''Neumann approximation of inverse-Hessian-vector product for visualization'''
    p = v
    for i in range(iterations):
        v -= f_vjp(v)[0]
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

def error_propagation_tsne(X_flat, X_unflattener, Y_flat, Y_unflattener, A, B, reg_factor, neumann_iterations=200):
    N, D = X_unflattener(X_flat).shape
    funY = lambda y: regularized_KL_divergence(X_flat, y, X_unflattener, Y_unflattener, reg_factor)
    funX = lambda x: regularized_KL_divergence(x, Y_flat, X_unflattener, Y_unflattener, reg_factor)
    funXY = partial(regularized_KL_divergence, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, reg_factor=reg_factor)
    _, hessian_vjp = vjp(grad(funY), Y_flat)
    neumann_fun = lambda v: neumannApproximation(hessian_vjp, v, neumann_iterations)
    Jx_fun = lambda x: jax.grad(funXY, argnums=1)(x, Y_flat)
    _, Jacobian_vjp = vjp(Jx_fun, X_flat)
    Jx_fun2 = lambda y: jax.grad(funXY, argnums=0)(X_flat, y)
    _, Jacobian_jvp = vjp(Jx_fun2, Y_flat)
    compute_cov_fun = lambda i: compute_cov(neumann_fun, Jacobian_vjp, Jacobian_jvp, A, B, i, N, D)
    cov = vmap(compute_cov_fun)(np.array([i for i in range(2*N)]))
    return cov

def get_exact_inverse(X, Y_star, reg_factor):
    f = partial(regularized_KL_divergence, reg_factor=reg_factor)
    dy, H, J = d_y_star_d_x_outer(f, X, Y_star)
    L, lower = jax.scipy.linalg.cho_factor(H, lower=True)
    u = jax.scipy.linalg.solve_triangular(L, np.eye(np.shape(L)[0]), lower=True)
    L_t, lower = jax.scipy.linalg.cho_factor(H, lower=False)
    H_inv = jax.scipy.linalg.solve_triangular(L_t, u, lower=False)
    return H_inv

def get_Neumann_approximation(X_flat, X_unflattener, Y_flat, Y_unflattener, reg_factor=0.001, neumann_iterations=200):
    fun = lambda x: regularized_KL_divergence(X_flat, x, X_unflattener, Y_unflattener, reg_factor)
    _, f_vjp = vjp(grad(fun), Y_flat)
    v_in = np.eye(len(Y_flat))
    neumann_fun = lambda x: neumannApproximation_vis(f_vjp, x, Y_flat, neumann_iterations)
    H_inv_appr = vmap(neumann_fun)(v_in)
    return H_inv_appr

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
        y_guess,
        affinity,
        learning_rate=200,
        negative_gradient_method="fft",
        random_state=42,
        verbose=False
    )
    y_star.optimize(250, exaggeration=12, momentum=0.8, inplace=True, verbose=True)
    y_star.optimize(750, momentum=0.5, inplace=True, verbose=True)
    return y_star

def tsne_fwd_for_MC(x, x_unflattener, y_guess):
    x = x_unflattener(x)
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
    y_star.optimize(250, exaggeration=12, momentum=0.8, inplace=True, verbose=True)
    y_star.optimize(750, momentum=0.5, inplace=True, verbose=True)
    return y_star

# first version
def mixed_Jacobian_vector_product_1(f, primals, v):
    X, Y = primals
    first = lambda X: jacfwd(f, argnums=1)(X, Y)
    second, second_t = jvp(first, (X,), (v,)) # --> first column of Jacobian!!!
    return second, second_t

# second version (probably faster due to less loops when traversing through np.eye when constructing full jacobian)
# but cov_X is in dimension that first version is better?
def mixed_Jacobian_vector_product_2(f, primals, v):
    print("Compute v3")
    X, Y = primals
    first = lambda Y: jacfwd(f, argnums=0)(X, Y)
    tangent = jvp(first, (Y,), (v,))[1] # --> first row of Jacobian!!!
    return tangent

def approxInverseHVP(f, primals, v, iterations):
    '''Neumann approximation of inverse-Hessian-vector product'''
    X, Y = primals
    p = v
    first = lambda Y: jacfwd(f, argnums=1)(X, Y)
    for i in range(iterations):
        v -= jvp(first, (Y,), (v,))[1]
        #print(dif)
        p += v
    return p

def VapproxInverseHP(f, primals, v, iterations):
    '''Neumann approximation of inverse-Hessian-vector product --> same result as approxInverseHVP'''
    print("Compute v2")
    X, Y = primals
    p = v
    first = lambda Y: jacfwd(f, argnums=1)(X, Y)
    for i in range(iterations):
        vjp_fun = vjp(first, Y)[1]
        v -= vjp_fun(v)[0]
        p += v
    return p

def approxInverseHVP_with_initialization(f, primals, v, iterations, M):
    '''Neumann approximation of inverse-Hessian-vector product (with initialization as described here
    https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-019-1631-x#Sec10'''
    X, Y = primals
    I = np.eye(len(M))
    D = np.linalg.inv(np.diag(np.diag(M)))
    Z = D @ (3 * I - M @ D @ (3 * I - M @ D))
    p = v
    first = lambda Y: jacfwd(f, argnums=1)(X, Y)
    for i in range(iterations):
        v -= jvp(first, (Y,), (Z@v,))[1]
        #print(dif)
        p += v
    return p @ Z

def approx_inv(M, iterations):
    # https://jwcn-eurasipjournals.springeropen.com/articles/10.1186/s13638-019-1631-x#Sec10
    I = np.eye(len(M))
    D = np.linalg.inv(np.diag(np.diag(M)))
    Z = D @ (3 * I - M @ D @ (3 * I - M @ D))
    s = np.zeros(M.shape)
    for i in range(1, iterations):
        s += (I - Z @ M)**i
    return s @ Z

def d_y_star_d_x_VP(f, primals, v, iterations):
    tangents_out = mixed_Jacobian_vector_product_1(f, primals, v)[1]
    print(tangents_out)
    return approxInverseHVP(f, primals, -tangents_out, iterations)

def d_y_star_d_x_MP(f, primals, M, iterations):
    _jvp = lambda s: d_y_star_d_x_VP(f, primals, s, iterations)
    return vmap(_jvp)(M)

def V_d_y_star_d_x_P(f, primals, v, iterations): 
    v2 = VapproxInverseHP(f, primals, v, 5)
    return mixed_Jacobian_vector_product_2(f, primals, -v2)

def M_d_y_star_d_x_P(f, primals, M, iterations):
    _jvp = lambda s: V_d_y_star_d_x_P(f, primals, s, iterations)
    return vmap(_jvp)(M)

num_vecs = 128


########################### Functions starting from explicit derivative of KL-divergence with respect to Y #########################

def KL_divergence_dy(X_flat, Y_flat, X_unflattener, Y_unflattener):
    """
    (R^nxp x R^nxp)--> R^nx2
    """
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)
    learning_rate, perplexity = (200, 30.0)
    P = x2p(X, tol=1e-5, perplexity=perplexity)
    P = (P + np.transpose(P))
    P = P / np.sum(P)      # Why don't we devide by 2N as described everywhere?
    P = np.maximum(P, 1e-12)
    Q, num = y2q(Y)
    # Compute gradient
    PQ = P - Q
    PQ_exp = np.expand_dims(PQ, 2)  # NxNx1
    Y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  # nx1x2 - 1xnx2= # NxNx2
    num_exp = np.expand_dims(num, 2)    # NxNx1
    Y_diffs_wt = Y_diffs * num_exp
    return np.ravel(4 * np.sum((PQ_exp * Y_diffs_wt), axis=1)) # Nx2

def mixed_Jacobian_vector_product_using_derivative(f, primals, v):
    print("Compute v3")
    X, Y = primals
    first = lambda X: f(X, Y)
    second, second_t = jvp(first, (X,), (v,)) # --> first column of Jacobian!!!
    return second, second_t

def vector_mixed_Jacobian_product_using_derivative(f, primals, v):
    print("Compute v3")
    X, Y = primals
    first = lambda X: f(X, Y)
    vjp_fun = vjp(first, X)[1] # --> first row of Jacobian!!!
    return vjp_fun(v)[0]

def VapproxInverseHP_using_derivative(f, primals, v, iterations):
    '''Neumann approximation of inverse-Hessian-vector product --> same result as approxInverseHVP'''
    print("Compute v2")
    X, Y = primals
    first = lambda Y: f(X, Y)
    p = v
    for i in range(iterations):
        vjp_fun = vjp(first, Y)[1]
        v -= vjp_fun(v)[0]
        p += v
    print(p.shape)
    return p

def approxInverseHVP_using_derivative(f, primals, v, iterations):
    '''Neumann approximation of inverse-Hessian-vector product'''
    X, Y = primals
    p = v
    first = lambda Y: f(X, Y)
    for i in range(iterations):
        v -= jvp(first, (Y,), (v,))[1]
        #print(dif)
        p += v
    return p

def d_y_star_d_x_VP_using_derivative(f, primals, v, iterations):
    tangents_out = mixed_Jacobian_vector_product_using_derivative(f, primals, v)[1]
    return approxInverseHVP_using_derivative(f, primals, -tangents_out, iterations)

def d_y_star_d_x_MP_using_derivative(f, primals, M, iterations):
    _jvp = lambda s: d_y_star_d_x_VP_using_derivative(f, primals, s, iterations)
    return vmap(_jvp)(M)

def V_d_y_star_d_x_P_using_derivative(f, primals, v, iterations): 
    v2 = VapproxInverseHP_using_derivative(f, primals, v, 5)
    return vector_mixed_Jacobian_product_using_derivative(f, primals, -v2)

def M_d_y_star_d_x_P_using_derivative(f, primals, M, iterations):
    _jvp = lambda s: V_d_y_star_d_x_P_using_derivative(f, primals, s, iterations)
    return vmap(_jvp)(M)