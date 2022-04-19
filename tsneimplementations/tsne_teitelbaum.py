import jax.numpy as jnp
import numpy as np
from functools import partial
from jax import vmap
from jax.lax import scan
from jax.lax import cond
from jax import random
import matplotlib.pylab as plt
from sklearn import manifold, datasets


def pdiff(X):
    (n,d) = X.shape
    # np.square is the elementwise square; axis=1 means sum the rows.
    # so sum_X is the vector of the norms of the x_i
    sum_X = np.sum(np.square(X),1)
    # ||x_i-x_j||^2=||x_i||^2+||x_j||**2-2(x_i,x_j)
    # np.dot(X,X.T) has entries (x_i,x_j)
    # in position (i,j) you add ||x_i||^2 and ||x_j||^2 from the two sums with the transpose
    # the result is that D is a symmetric matrix with the pairwise distances.
    D = np.add(np.add(-2*np.dot(X,X.T),sum_X).T,sum_X)
    return D

def Hbeta(D=np.array([]), beta=1.0):
    """
    Compute the perplexity and the P-row for a specific value of the
    precision of a Gaussian distribution.  Note that D is the i_th row
    of the pairwise distance matrix with the i_th entry deleted.
    """

    # Compute P-row and corresponding perplexity
    # at this point, P is the numerator of the conditional probabilities
    P = np.exp(-D.copy() * beta)
    # sumP is the denominator, the normalizing factor
    sumP = sum(P)

    # the entropy is the sum of p \log p which is P/sumP
    # Checking with the formula above, sumP = S_i and np.sum(D*P/sumP) is the dot
    # product of the distances with the probabilities

    H = np.log(sumP) + beta * np.sum(D * P) / sumP

    # now normalize P to be the actual probabilities and return them
    P = P / sumP
    return H, P

def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
    Performs a binary search to get P-values in such a way that each
    conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    print('D', D)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        #if i % 500 == 0:
        print("Computing P-values for point %d of %d..." % (i, n))

        # prep for binary search on beta

        betamin = -np.inf
        betamax = np.inf

        # Compute the Gaussian kernel and entropy for the current precision
        # the first line drops the ith entry in row i from the pairwise distances
        # Hbeta in the second line expects this

        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision (via binary search)
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1


        # Set the final row of P, reinserting the missing spot as 0
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    # report mean value of sigma, but not the actual sigma values
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P

def pca(X=np.array([]), no_dims=50):
    """
    Runs PCA on the NxD array X in order to reduce its dimensionality to
    no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape

    # center the columns, so each column has mean zero
    X = X - np.tile(np.mean(X, 0), (n, 1))

    # project onto the first no_dims eigenspaces
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y

def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0):
    """
    Runs t-SNE on the dataset in the NxD array X to reduce its
    dimensionality to no_dims dimensions. The syntaxis of the function is
    `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

# Check inputs (first thing here looks wrong)

    # Initialize variables
    # use eigh and then you don't need the real
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 100
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01


    # initial Y placements are random points in (usually) 2-space
    random.PRNGKey(42)
    Y = random.normal(key, shape=(n, no_dims))
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))
    Ysave = Y.copy()
    Csave = 0.0



    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    # symmetrize the probabilities; see page 2584.  This makes sure that if a point is far from
    # all the other points, its position is still relevant.
    P = P + np.transpose(P)
    P = P / np.sum(P)

    # not clearly documented in the paper but seems to help with convergence
    P = P * 4.   # early exaggeration
    # avoid zeros
    P = np.maximum(P, 1e-12)

    # Run iterations
    for iter in range(250):
        # Compute pairwise affinities using the student t-kernel (equation 4)
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)
        #print(Q)
        # Compute gradient

        # PQ is a symmetric n x n matrix with entries p_ij-q_ij
        PQ = P - Q
        # This is a clever way to compute the gradient; dY is n x 2 matrix, each column is a dy_i
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        # this business with "gains" is the bar-delta-bar heuristic to accelerate gradient descent
        # code could be simplified by just omitting it

        #gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
        #(gains * 0.8) * ((dY > 0.) == (iY > 0.))
        #gains[gains < min_gain] = min_gain

        # this is the momentum update

        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        #print(np.tile(np.mean(Y, 0), (n, 1)))
        # recentering the data (doesn't affect the distances between points)
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))


        #Compute current value of cost function
        # if (iter + 1) % 100 == 0:
        #     Ysave = np.concatenate([Ysave, Y])
        #     C = np.sum(P * np.log(P / Q))
        #     print("Iteration %d: error is %f" % (iter + 1, C))
        # if np.abs(Csave - C)<.001:
        #     break
        # Csave = C



        # Stop lying about P-values
        # this is "early exaggeration" which is not really explained in the paper

        if iter == 100:
            P = P / 4.
    print(P, Y, dY, iY, gains, i)
    print('Y', Y)
    # Return solution
    return Y

if __name__ == '__main__':
    key = random.PRNGKey(42)
    X = random.uniform(key, shape=(100, 100))
    X, y = datasets.make_swiss_roll(n_samples=100, noise=0.0, random_state=0)
    #P = x2p(X, tol=1e-5, perplexity=30.0)
    #print('P', P)
    Y = tsne(X)
    f = plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=y)
    plt.savefig('swissroll_tsne_teitelbaum.pdf')
    #plt.show()
