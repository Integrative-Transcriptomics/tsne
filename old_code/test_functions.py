import numpy as np
from tsne_jax import pca
from sklearn.decomposition import PCA
import matplotlib.pylab as plt


def test_pca():
    X = np.random.random(size=(100, 100))
    pca_sklearn = PCA(n_components=2)
    Y_sklearn = pca_sklearn.fit_transform(X)
    Y_sklearn = Y_sklearn *(-1)

    Y_tsne = pca(X, no_dims=2)
    f = plt.figure()
    plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1])
    plt.scatter(Y_tsne[:, 0], Y_tsne[:, 1])
    plt.show()



if __name__ == '__main__':
    test_pca() # inspect via plotting as sign of eigenvectors not defined