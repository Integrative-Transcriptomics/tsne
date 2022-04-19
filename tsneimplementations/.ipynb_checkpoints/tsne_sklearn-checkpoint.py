from sklearn import manifold, datasets
import numpy as np
import matplotlib.pylab as plt

if __name__ == '__main__':
    X, y = datasets.make_swiss_roll(n_samples=100, noise=0.0, random_state=0)
    tsne = manifold.TSNE()
    Y = tsne.fit_transform(X)
    f = plt.figure()
    plt.scatter(Y[:, 0], Y[:, 1], c=y)
    plt.savefig('swissroll_tsne_sklearn.pdf')