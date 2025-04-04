import jax.numpy as np
import numpy as onp
from functools import partial
from jax import vmap, flatten_util, vjp
from jax.lax import scan
from jax.lax import cond
from jax import random
from jax import jit
from jax import jacrev
from jax.lax import stop_gradient
import matplotlib.pylab as plt
import matplotlib as mpl
from sklearn import manifold, datasets
import seaborn as sns
import jax

#from jax.config import config
#config.update("jax_debug_nans", True)

import openTSNE

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

def logSoftmax(x):
    """Compute softmax for vector x."""
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    log_sum_exp_x = np.log(sum_exp_x)
    max_plus_log_sum_exp_x = max_x + log_sum_exp_x
    log_probs = x - max_plus_log_sum_exp_x

    # Recover probs
    exp_log_probs = np.exp(log_probs)
    sum_log_probs = np.sum(exp_log_probs)
    probs = exp_log_probs / sum_log_probs
    return probs

def Hbeta(D: np.ndarray, beta=1.0):
    """
    Compute the log2(perplexity)=Entropy and the P-row (P_i) for a specific value of the
        precision=1/(sigma**2) (beta) of a Gaussian distribution. D: vector of squared Euclidean distances (without i)
    :param D: vector of length d, squared Euclidean distances to all other datapoints (except itself)
    :param beta: precision = beta = 1/sigma**2
    :return: H: log2(Entropy), P: computed probabilites
    """
    # TODO: exchange by softmax as described by https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
    P = np.exp(-D * beta)     # numerator of p j|i
    sumP = np.sum(P, axis=None)    # denominator of p j|i --> normalization factor
    new_P = logSoftmax(-D * beta)
    sumP += 1e-8
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    return H, new_P

def Hbeta_final(D: np.ndarray, beta=1.0):
    """
    Compute the log2(perplexity)=Entropy and the P-row (P_i) for a specific value of the
        precision=1/(sigma**2) (beta) of a Gaussian distribution. D: vector of squared Euclidean distances (without i)
    :param D: vector of length d, squared Euclidean distances to all other datapoints (except itself)
    :param beta: precision = beta = 1/sigma**2
    :return: H: log2(Entropy), P: computed probabilites
    """
    # TODO: exchange by softmax as described by https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
    # P = np.exp(-D * beta)     # numerator of p j|i
    #sumP = np.sum(P, axis=None)    # denominator of p j|i --> normalization factor
    new_P = logSoftmax(-D * beta)
    #sumP += 1e-8
    #new_P = P/sumP
    return new_P

def HdiffGreaterTrue(*betas):
    beta, betamax = betas
    return beta*2

def HdiffGreaterFalse(*betas):
    beta, betamax = betas
    return (beta+betamax)/2

def HdiffSmallerTrue(*betas):
    beta, betamin = betas
    return beta/2

def HdiffSmallerFalse(*betas):
    beta, betamin = betas
    return (beta+betamin)/2

def HdiffGreater(*betas):
    beta, betamin, betamax = betas
    betamin = beta
    beta = cond((np.logical_or(betamax == np.inf, betamax == -np.inf)), HdiffGreaterTrue, HdiffGreaterFalse, *(beta, betamax))
    return beta, betamin, betamax

def HdiffSmaller(*betas):
    beta, betamin, betamax = betas
    betamax = beta
    beta = cond(np.logical_or(betamin == np.inf, betamin == -np.inf), HdiffSmallerTrue, HdiffSmallerFalse, *(beta, betamin))
    return beta, betamin, betamax

def HdiffGreaterTolerance(*betas):
    beta, betamin, betamax, Hdiff = betas
    beta, betamin, betamax = cond(Hdiff > 0, HdiffGreater, HdiffSmaller, *(beta, betamin, betamax))
    return beta, betamin, betamax, Hdiff

def binarySearch(res, el, Di, logU):
    Hdiff, thisP, beta, betamin, betamax = res
    beta, betamin, betamax, Hdiff = cond(np.abs(Hdiff) < 1e-5, lambda a, b, c, d: (a, b, c, d), HdiffGreaterTolerance, *(beta, betamin, betamax, Hdiff))
    (H, thisP) = Hbeta(Di, beta)
    Hdiff = H - logU
    return (Hdiff, thisP, beta, betamin, betamax), el

def x2beta_inner(Di: np.ndarray, iterator, beta, betamin, betamax, perplexity=30, tol=1e-5):
    """
    binary search for precision for Pi such that it matches the perplexity defined by the user
    :param Di: vector of length d-1, squared Euclidean distances to all other datapoints (except itself)
    :param beta: precision = beta = 1/sigma**2
    :return: final probabilites p j|i
    """
    # Compute the Gaussian kernel and entropy for the current precision
    logU = np.log(perplexity)
    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    binarySearch_func = partial(binarySearch, Di=Di, logU=logU)

    # Note: the following binary Search for suitable precisions (betas) will be repeated 50 times and does not include the threshold value
    (Hdiff, thisP, beta, betamin, betamax), el = scan(binarySearch_func, init=(Hdiff, thisP, beta, betamin, betamax), xs=None, length=1000)    # Set the final row of P
    #thisP = np.insert(thisP, iterator, 0)
    return beta

def x2beta(D: np.ndarray, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values (high-dim space) in such a way that each
        conditional Gaussian has the same perplexity.
    """
    # Initialize some variables
    n = D.shape[0]
    beta = np.ones(n)      # precisions (1/sigma**2)
    betamin = np.full(n, -np.inf)
    betamax = np.full(n, np.inf)
    betas_final = vmap(partial(x2beta_inner, perplexity=perplexity, tol=tol))(D, np.arange(n), beta=beta, betamin=betamin, betamax=betamax)
    return betas_final

def x2distance(X):
    sum_X = np.sum(np.square(X), 1)
    (n, d) = X.shape
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    D = np.reshape(np.delete(D, np.array([i for i in range(0, D.shape[0]**2, (D.shape[0]+1))])), (n , n - 1 ))
    return D

def distance2p(D, betas):
    P_final = vmap(Hbeta_final, in_axes=0)(D, betas)
    #print('P_final', P_final, P_final.shape)
    P_final = vmap(partial(np.insert, values=0))(P_final, np.arange(P_final.shape[0]))
    return P_final

def y2q(Y: np.ndarray):
    # Compute pairwise affinities
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)  # numerator
    num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    num = num.at[np.diag_indices_from(num)].set(0.)     # numerator
    Q = num / np.sum(num)
    Q = np.maximum(Q, 1e-12)
    return Q, num

def KL_divergence(X_flat, Y_flat, X_unflattener, Y_unflattener, perplexity):
    """
    (R^nxp x R^nxp)--> R
    """
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)
    learning_rate, perplexity = (200, perplexity)
    D = x2distance(X)
    #print('D', D.shape)
    # first compute betas without tracking the derivative
    betas = x2beta(jax.lax.stop_gradient(D), tol=1e-5, perplexity=perplexity)
    #print('betas', betas.shape, betas)
    # use final betas to compute the probability matrix from the distances.
    # here D and therefor X is tracked for derivative computation
    P = distance2p(D, betas)
    P = (P + np.transpose(P))
    P = P / np.sum(P)      # Why don't we devide by 2N as described everywhere?
    P = np.maximum(P, 1e-12)
    #print('P', P, P.shape)
    Q, _ = y2q(Y)
    #print('Q', Q)
    return np.sum(P * (np.log(P+1e-10) - np.log(Q+1e-10)))

def KL_divergence_dy(X_flat, Y_flat, X_unflattener, Y_unflattener, perplexity=30.0):
    """
    (R^nxp x R^nxp)--> R
    """
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)
    learning_rate, perplexity = (200, perplexity)
    D = x2distance(X)
    #print('D', D.shape)
    # first compute betas without tracking the derivative
    betas = x2beta(jax.lax.stop_gradient(D), tol=1e-5, perplexity=perplexity)
    #print('betas', betas.shape, betas)
    # use final betas to compute the probability matrix from the distances.
    # here D and therefor X is tracked for derivative computation
    P = distance2p(D, betas)
    P = (P + np.transpose(P))
    P = P / np.sum(P)      # Why don't we devide by 2N as described everywhere?
    P = np.maximum(P, 1e-12)
    #print('P', P, P.shape)
    Q, num = y2q(Y)
    #print('forward pass done')
    #print('Q', Q)

    PQ = P - Q
    PQ_exp = np.expand_dims(PQ, 2)  # NxNx1
    Y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  # nx1x2 - 1xnx2= # NxNx2
    num_exp = np.expand_dims(num, 2)    # NxNx1
    Y_diffs_wt = Y_diffs * num_exp
    return np.ravel(4 * np.sum((PQ_exp * Y_diffs_wt), axis=1))

def compute_cov_inner(vjp_fun, jvp_fun_lin, H_pinv_i, D, N, d, n, H_pinv):
  v1 = vjp_fun(-H_pinv_i)[0]
  v2 = np.ravel(np.dot(np.dot(D, np.reshape(v1, (d, n), 'C')), np.transpose(N)), 'C')
  v3 = jvp_fun_lin(v2)
  return np.dot(-H_pinv, v3)

def compute_cov(X_flat, Y_flat, X_unflattener, Y_unflattener, D, N, perplexity):
  f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
  H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
  H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)

  f = partial(KL_divergence_dy, Y_flat=Y_flat, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
  # jvp
  _, jvp_fun_lin = jax.linearize(f, X_flat)
  # vjp
  _, vjp_fun = vjp(f, X_flat)

  compute_cov_fun = lambda i: compute_cov_inner(vjp_fun=vjp_fun, jvp_fun_lin=jvp_fun_lin, 
                                        H_pinv_i=i, D=D, N=N, d=D.shape[0], n=N.shape[0], H_pinv=H_pinv)

  return vmap(compute_cov_fun)(H_pinv)


def compute_sensitivities_inner(vjp_fun, H_pinv_i):
    return vjp_fun(-H_pinv_i)[0]

def compute_sensitivities(X_flat, Y_flat, X_unflattener, Y_unflattener, perplexity):
    f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
    H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
    H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)  

    f = partial(KL_divergence_dy, Y_flat=Y_flat, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
    # vjp
    _, vjp_fun = vjp(f, X_flat)

    compute_sensitivities_fun = lambda i: compute_sensitivities_inner(vjp_fun=vjp_fun, H_pinv_i=i)

    return vmap(compute_sensitivities_fun)(H_pinv)

# fastes version for mixed Jacobian!!!
# J_X_Y = jacrev(jacfwd(KL_divergence, argnums=1), argnums=0)(X_flat, Y_flat, X_unflattener, Y_unflattener)
# print('J', J_X_Y)

# f = partial(KL_divergence, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
# H = jax.hessian(f, argnums=1)(X_flat, Y_flat)
# H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)

# or fast using derivative directly:
# f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
# J_X_Y = jacrev(f, argnums=0)(X_flat, Y_flat)

# f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
# H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
# H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)