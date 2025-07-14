import jax.numpy as np
from functools import partial
from jax import vmap, vjp
from jax.lax import scan
from jax.lax import cond
import jax

import openTSNE

def tsne_fwd(x, y_guess):
    """
    Performs a standard t-SNE forward pass using the openTSNE library.

    Args:
        x (np.ndarray): The high-dimensional input data of shape (n_samples, n_features).
        y_guess (np.ndarray): An initial guess for the low-dimensional embedding.

    Returns:
        openTSNE.TSNEEmbedding: The optimized t-SNE embedding object.
    """

    affinity = openTSNE.affinity.PerplexityBasedNN(
        x,
        perplexity=30.0,
        method="annoy",
        random_state=42,
        verbose=True,
    )
    
    y_star = openTSNE.TSNEEmbedding(
        y_guess,
        affinity,
        #learning_rate=200,
        negative_gradient_method="fft",
        random_state=42,
        verbose=False
    )
    y_star.optimize(250, exaggeration=12, momentum=0.8, inplace=True, verbose=True)
    y_star.optimize(750, momentum=0.5, inplace=True, verbose=True)
    return y_star


def softmax(x):
    """
    Computes the softmax function for a vector x in a numerically stable way.
    
    It uses the log-sum-exp trick internally
    for numerical stability to prevent overflow/underflow issues.

    Args:
        x (np.ndarray): An input vector.

    Returns:
        np.ndarray: The computed probabilities (softmax output).
    """

    # The log-sum-exp trick: subtract the max value before exponentiating
    # to prevent overflow with large input values.
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)

    # This part calculates log probabilities, which are then exponentiated.
    log_sum_exp_x = np.log(sum_exp_x)
    max_plus_log_sum_exp_x = max_x + log_sum_exp_x
    log_probs = x - max_plus_log_sum_exp_x

    # Recover probabilities from the log probabilities.
    exp_log_probs = np.exp(log_probs)
    sum_log_probs = np.sum(exp_log_probs)
    probs = exp_log_probs / sum_log_probs
    return probs


def Hbeta(D: np.ndarray, beta=1.0):
    """
    Computes the Shannon entropy (H) and conditional probabilities (P_j|i) for a
    given row of squared distances and a precision value (beta).

    This function is used during the binary search for the optimal beta.

    Args:
        D (np.ndarray): A vector of squared Euclidean distances from point i to all other points j.
        beta (float): The precision of the Gaussian kernel (beta = 1 / (2 * sigma^2)).

    Returns:
        tuple[float, np.ndarray]: A tuple containing:
            - H (float): The Shannon entropy of the resulting probability distribution.
            - new_P (np.ndarray): The computed conditional probabilities P_j|i.
    """

    # Compute the conditional probabilities P_j|i using a numerically stable softmax.
    P = np.exp(-D * beta)     # numerator of p j|i
    sumP = np.sum(P, axis=None)    # denominator of p j|i --> normalization factor
    new_P = softmax(-D * beta)
    sumP += 1e-8

    # Calculate the Shannon entropy H of the probability distribution P.
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    return H, new_P

def Hbeta_final(D: np.ndarray, beta=1.0):
    """
    Computes the final conditional probabilities (P_j|i) for a given row of
    squared distances and a final, optimized precision value (beta).

    Args:
        D (np.ndarray): A vector of squared Euclidean distances from point i to all other points j.
        beta (float): The final precision of the Gaussian kernel.

    Returns:
        np.ndarray: The computed final conditional probabilities P_j|i.
    """
    # Simply compute the conditional probabilities using the stable softmax.
    new_P = softmax(-D * beta)
    return new_P


# ----------------------------------------------------------------------------------
# Helper functions for the binary search over beta, designed for use with `jax.lax.cond`.
# These functions update the search range [betamin, betamax] for beta.
# ----------------------------------------------------------------------------------

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
    """
    Performs one step of the binary search for beta.
    This function serves as the body of the `jax.lax.scan` loop.

    Args:
        res (tuple): The loop state (carry): (Hdiff, thisP, beta, betamin, betamax).
        el: Unused element from the `scan`'s `xs` input.
        Di (np.ndarray): The vector of squared distances for the current point.
        logU (float): The target entropy (log of perplexity).

    Returns:
        tuple: The updated loop state `res` and an unused element.
    """

    Hdiff, thisP, beta, betamin, betamax = res

    # Update beta based on Hdiff, unless convergence is reached.
    # The lambda function `lambda a,b,c,d: (a,b,c,d)` is an identity function that does nothing if converged.
    beta, betamin, betamax, Hdiff = cond(np.abs(Hdiff) < 1e-5, lambda a, b, c, d: (a, b, c, d), HdiffGreaterTolerance, *(beta, betamin, betamax, Hdiff))

    # Re-compute entropy and probabilities with the new beta.
    (H, thisP) = Hbeta(Di, beta)

    # Update the difference between current and target entropy.
    Hdiff = H - logU

    return (Hdiff, thisP, beta, betamin, betamax), el

def x2beta_inner(Di: np.ndarray, iterator, beta, betamin, betamax, perplexity=30, tol=1e-5):
    """
    Performs a binary search to find the optimal beta for a single point `i`
    such that the perplexity of the conditional probability distribution P_j|i
    matches the user-defined value.

    Args:
        Di (np.ndarray): Vector of squared Euclidean distances to all other points.
        iterator: Index of the current point (unused in calculation, for vmap).
        beta (float): Initial guess for beta.
        betamin (float): Initial lower bound for beta search.
        betamax (float): Initial upper bound for beta search.
        perplexity (float): The target perplexity.
        tol (float): The tolerance for the binary search.

    Returns:
        float: The final, optimized beta value for the point.
    """
    # The target entropy is the log of the perplexity
    logU = np.log(perplexity)
    # Initial calculation of entropy and probability difference.
    H, thisP = Hbeta(Di, beta)
    Hdiff = H - logU

    # Create a partial function for the binary search step, fixing Di and logU
    binarySearch_func = partial(binarySearch, Di=Di, logU=logU)

    # Use `scan` to run the binary search for a fixed number of iterations (1000).
    # This is a common JAX pattern for loops.
    (Hdiff, thisP, beta, betamin, betamax), el = scan(binarySearch_func, init=(Hdiff, thisP, beta, betamin, betamax), xs=None, length=1000)    # Set the final row of P

    return beta

def x2beta(D: np.ndarray, tol=1e-5, perplexity=30.0):
    """
    Performs a vectorized binary search to find the optimal beta for each data point.

    This function uses `vmap` to apply `x2beta_inner` to all points in parallel,
    finding a unique `beta_i` for each point `i`.

    Args:
        D (np.ndarray): An (n, n-1) matrix of pairwise squared distances.
        tol (float): Tolerance for the binary search.
        perplexity (float): The target perplexity for all conditional distributions.

    Returns:
        np.ndarray: A vector of optimized beta values, one for each point.
    """
    # Initialize some variables
    n = D.shape[0]
    beta = np.ones(n)      # Initial guess for precisions
    betamin = np.full(n, -np.inf)   # Initial lower bounds
    betamax = np.full(n, np.inf)    # Initial upper bounds

    # Use vmap to apply the binary search to each row of D in parallel.
    betas_final = vmap(partial(x2beta_inner, perplexity=perplexity, tol=tol))(D, np.arange(n), beta=beta, betamin=betamin, betamax=betamax)
    
    return betas_final

def x2distance(X):
    """
    Computes the matrix of pairwise squared Euclidean distances between all points in X.
    It uses the efficient broadcast trick: ||a-b||^2 = ||a||^2 - 2a.b + ||b||^2.

    Args:
        X (np.ndarray): Input data of shape (n_samples, n_features).

    Returns:
        np.ndarray: An (n, n-1) matrix where D[i, j] is the squared distance
                    between point i and point j, with the diagonal (i=i) removed.
    """
    sum_X = np.sum(np.square(X), 1)
    (n, d) = X.shape

    # Compute the full distance matrix: D_ij = ||x_i||^2 + ||x_j||^2 - 2 * x_i^T * x_j
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # Remove the diagonal elements (distances from a point to itself)
    # This creates the (n, n-1) matrix required by x2beta.
    D = np.reshape(np.delete(D, np.array([i for i in range(0, D.shape[0]**2, (D.shape[0]+1))])), (n , n - 1 ))
    return D

def distance2p(D, betas):
    """
    Computes the conditional probability matrix P_j|i from the distance matrix
    and the optimized beta values.

    Args:
        D (np.ndarray): The (n, n-1) matrix of squared distances.
        betas (np.ndarray): The vector of optimized beta values for each point.

    Returns:
        np.ndarray: The (n, n) conditional probability matrix `P`, where P[i, j] = P_j|i.
    """
    # Compute P_j|i for each row i using its specific beta_i.
    P_final = vmap(Hbeta_final, in_axes=0)(D, betas)
    # Re-insert the zero diagonal into each row of the probability matrix.
    # vmap iterates over each row of P_final and the indices 0, 1, 2, ...
    P_final = vmap(partial(np.insert, values=0))(P_final, np.arange(P_final.shape[0]))
    return P_final

def y2q(Y: np.ndarray):
    """
    Computes the joint probabilities Q_ij in the low-dimensional embedding space.
    These probabilities are based on a Student's t-distribution with one degree of freedom.

    Args:
        Y (np.ndarray): The low-dimensional embedding of shape (n_samples, n_components).

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Q (np.ndarray): The normalized (n, n) joint probability matrix.
            - num (np.ndarray): The unnormalized numerators of Q.
    """
    # Compute pairwise squared Euclidean distances in the low-dimensional space.
    sum_Y = np.sum(np.square(Y), 1)
    num = -2. * np.dot(Y, Y.T)  # numerator
    num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
    # Set diagonal to zero, as q_ii is not defined.
    num = num.at[np.diag_indices_from(num)].set(0.)     # numerator
    # Normalize to get the final joint probabilities Q.
    Q = num / np.sum(num)
    # Add a small epsilon for numerical stability to avoid log(0).
    Q = np.maximum(Q, 1e-12)
    return Q, num

def KL_divergence(X_flat, Y_flat, X_unflattener, Y_unflattener, perplexity):
    """
    Computes the t-SNE objective function: the Kullback-Leibler (KL) divergence
    between the high-dimensional probabilities (P) and low-dimensional probabilities (Q).

    Args:
        X_flat (np.ndarray): A flattened representation of the high-dimensional data.
        Y_flat (np.ndarray): A flattened representation of the low-dimensional embedding.
        X_unflattener: A function to reshape X_flat to its original 2D shape.
        Y_unflattener: A function to reshape Y_flat to its original 2D shape.
        perplexity (float): The perplexity used to compute P.

    Returns:
        float: The KL divergence value.
    """
    # Reshape flattened inputs back to their matrix forms.
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)

    learning_rate, perplexity = (200, perplexity)

    # --- Compute High-Dimensional Probabilities P ---
    D = x2distance(X)
    # first compute betas without tracking the derivative
    # We stop gradients from flowing through it.
    betas = x2beta(jax.lax.stop_gradient(D), tol=1e-5, perplexity=perplexity)
    # Compute P_j|i using the final betas. Gradients can flow through this step.
    P = distance2p(D, betas)

    # Symmetrize to get the joint probability distribution P.
    P = (P + np.transpose(P))
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)

    # --- Compute Low-Dimensional Probabilities Q ---
    Q, _ = y2q(Y)
    
    # --- Compute KL Divergence ---
    return np.sum(P * (np.log(P+1e-10) - np.log(Q+1e-10)))

def KL_divergence_dy(X_flat, Y_flat, X_unflattener, Y_unflattener, perplexity=30.0):
    """
    Computes the analytical gradient of the KL divergence with respect to the
    low-dimensional embedding Y. This is much more efficient than using `jax.grad`.

    Args:
        X_flat (np.ndarray): Flattened high-dimensional data.
        Y_flat (np.ndarray): Flattened low-dimensional embedding.
        X_unflattener: Function to reshape X.
        Y_unflattener: Function to reshape Y.
        perplexity (float): Perplexity value.

    Returns:
        np.ndarray: The flattened gradient of the KL divergence w.r.t. Y.
    """
    # Reshape flattened inputs back to their matrix forms.
    X = X_unflattener(X_flat)
    Y = Y_unflattener(Y_flat)

    learning_rate, perplexity = (200, perplexity)

    # --- Compute High-Dimensional Probabilities P ---
    D = x2distance(X)
    # first compute betas without tracking the derivative
    # We stop gradients from flowing through it.
    betas = x2beta(jax.lax.stop_gradient(D), tol=1e-5, perplexity=perplexity)
    # Compute P_j|i using the final betas. Gradients can flow through this step.
    P = distance2p(D, betas)

    # Symmetrize to get the joint probability distribution P.
    P = (P + np.transpose(P))
    P = P / np.sum(P)
    P = np.maximum(P, 1e-12)

    # --- Compute Low-Dimensional Probabilities Q ---
    Q, num = y2q(Y)

    # --- Compute Analytical Gradient ---
    PQ = P - Q
    PQ_exp = np.expand_dims(PQ, 2)  # NxNx1
    Y_diffs = np.expand_dims(Y, 1) - np.expand_dims(Y, 0)  # nx1x2 - 1xnx2= # NxNx2
    num_exp = np.expand_dims(num, 2)    # NxNx1
    Y_diffs_wt = Y_diffs * num_exp
    return np.ravel(4 * np.sum((PQ_exp * Y_diffs_wt), axis=1))

def compute_cov_inner(vjp_fun, jvp_fun_lin, H_pinv_i, D, N, d, n, H_pinv):
    """Helper function for computing one row of the output covariance matrix."""
    # This function implements one step of the formula for covariance propagation
    # via the implicit function theorem, assuming a Kronecker product input covariance.
    v1 = vjp_fun(-H_pinv_i)[0]
    v2 = np.ravel(np.dot(np.dot(D, np.reshape(v1, (d, n), 'C')), np.transpose(N)), 'C')
    v3 = jvp_fun_lin(v2)
    return np.dot(-H_pinv, v3)

def compute_cov_inner_without_kronecker(vjp_fun, jvp_fun_lin, H_pinv_i, input_cov, H_pinv):
    """Helper function for computing one row of the output covariance matrix for a general input covariance."""
    # Step 1: Compute v^T * J_YX, where v is a row of -H_Y^-1 and J_YX is d(grad_Y)/dX.
    # This is done efficiently with a vector-Jacobian product (vjp).
    v1 = vjp_fun(-H_pinv_i)[0]
    # Step 2: Apply the input covariance matrix: (v^T * J_YX) * Sigma_X
    v2 = np.multiply(input_cov, v1)
    # Step 3: Apply the Jacobian again: J_YX * [(v^T * J_YX) * Sigma_X]^T
    # This is done efficiently with a Jacobian-vector product (jvp).
    v3 = jvp_fun_lin(v2)
    # Step 4: Final multiplication with -H_Y^-1.
    return np.dot(-H_pinv, v3)

def compute_cov(X_flat, Y_flat, X_unflattener, Y_unflattener, D, N, perplexity):
    """
    Computes the covariance of the output embedding Y with respect to noise in
    the input X, assuming the input covariance is a Kronecker product.
    This uses the implicit function theorem to propagate uncertainty.

    Args:
        (various): Inputs for the t-SNE process.
        D, N: Factors of the Kronecker product representing input covariance.

    Returns:
        np.ndarray: The computed output covariance matrix for Y.
    """
    # Define the gradient function whose derivatives we need.
    f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)

    # Compute the Hessian of the KL cost w.r.t Y: H_Y = d(grad_Y)/dY.
    H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
    # Regularized pseudo-inverse
    H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)

    # Define a new partial function for derivatives w.r.t. X.
    f = partial(KL_divergence_dy, Y_flat=Y_flat, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
    # Set up functions for Jacobian-vector and vector-Jacobian products for d(grad_Y)/dX.
    # jvp
    _, jvp_fun_lin = jax.linearize(f, X_flat)
    # vjp
    _, vjp_fun = vjp(f, X_flat)

    # Create a function for the inner loop, then vectorize it with vmap.
    compute_cov_fun = lambda i: compute_cov_inner(vjp_fun=vjp_fun, jvp_fun_lin=jvp_fun_lin, 
                                        H_pinv_i=i, D=D, N=N, d=D.shape[0], n=N.shape[0], H_pinv=H_pinv)

    return vmap(compute_cov_fun)(H_pinv)

def compute_cov_without_kronecker(X_flat, Y_flat, X_unflattener, Y_unflattener, input_cov, perplexity):
    """
    Computes the covariance of the output embedding Y for a general input covariance matrix.
    This is the more general version of `compute_cov`.

    Args:
        (various): Inputs for the t-SNE process.
        input_cov (np.ndarray): The full covariance matrix of the input data X.

    Returns:
        np.ndarray: The computed output covariance matrix for Y.
    """
    f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
    H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
    H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)

    f = partial(KL_divergence_dy, Y_flat=Y_flat, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
    # jvp
    _, jvp_fun_lin = jax.linearize(f, X_flat)
    # vjp
    _, vjp_fun = vjp(f, X_flat)

    compute_cov_fun = lambda i: compute_cov_inner_without_kronecker(vjp_fun=vjp_fun, jvp_fun_lin=jvp_fun_lin, 
                                        H_pinv_i=i, input_cov=input_cov, H_pinv=H_pinv)

    return vmap(compute_cov_fun)(H_pinv)


def compute_sensitivities_inner(vjp_fun, H_pinv_i):
    """Helper function to compute one row of the sensitivity matrix dY/dX."""
    # This computes v^T * J, where v is a row of -H_Y^-1 and J is the mixed partial d(grad_Y)/dX.
    return vjp_fun(-H_pinv_i)[0]

def compute_sensitivities(X_flat, Y_flat, X_unflattener, Y_unflattener, perplexity):
    """
    Computes the sensitivity matrix (Jacobian) dY/dX of the final embedding Y
    with respect to the input data X.

    The sensitivity is calculated as: dY/dX = -H_Y^-1 * J_YX, where
    H_Y is the Hessian of the cost w.r.t. Y, and J_YX is the mixed partial derivative.
    This computation is done efficiently using VJPs.

    Args:
        (various): Inputs for the t-SNE process.

    Returns:
        np.ndarray: The sensitivity matrix, representing how Y changes with X.
    """
    f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
    # Hessian w.r.t Y
    H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
    H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)  

    f = partial(KL_divergence_dy, Y_flat=Y_flat, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener, perplexity=perplexity)
    # vjp for the mixed partial derivative d(grad_Y)/dX
    _, vjp_fun = vjp(f, X_flat)

    # The function computes one row of (-H_pinv * J_YX)
    compute_sensitivities_fun = lambda i: compute_sensitivities_inner(vjp_fun=vjp_fun, H_pinv_i=i)
    
    # vmap to compute all rows of the sensitivity matrix in parallel.
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