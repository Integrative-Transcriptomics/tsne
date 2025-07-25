import matplotlib as mpl
import matplotlib.pyplot as plt
from tueplots import figsizes, fonts, bundles
from tueplots import cycler
from tueplots.constants import markers
from tueplots.constants.color import palettes
# Increase the resolution of all the plots below
plt.rcParams.update({"figure.dpi": 150})

import numpy as np
import gzip, pickle
import sys, os
from os import path
import seaborn as sns

cmap = mpl.colors.LinearSegmentedColormap.from_list("", [palettes.tue_plot[0], "white", palettes.tue_plot[3]])

class MidpointNormalize(mpl.colors.Normalize):
    def __init__(self, vmin, vmax, midpoint=0, clip=False):
        self.midpoint = midpoint
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        normalized_min = max(0, 1 / 2 * (1 - abs((self.midpoint - self.vmin) / (self.midpoint - self.vmax))))
        normalized_max = min(1, 1 / 2 * (1 + abs((self.vmax - self.midpoint) / (self.midpoint - self.vmin))))
        normalized_mid = 0.5
        x, y = [self.vmin, self.midpoint, self.vmax], [normalized_min, normalized_mid, normalized_max]
        return np.ma.masked_array(np.interp(value, x, y))
    
def load_data(n_samples=None):
    with gzip.open(path.join("examples/data/mnist", "mnist.pkl.gz"), "rb") as f:
        data = pickle.load(f)

    x, y = data["pca_50"], data["labels"]

    if n_samples is not None:
        indices = np.random.choice(
            list(range(x.shape[0])), n_samples, replace=False
        )
        x, y = x[indices], y[indices]

    return x, y

def compute_importances_per_sample(jacobian):
    jacobian_sum_per_sample = []
    jacobian_sum = np.sum(np.abs(jacobian), axis=0)
    for i in range(n):
        sum_for_sample = []
        for j in range(p):
            sum_for_sample.append(jacobian_sum[j*n+i])
        jacobian_sum_per_sample.append(sum(sum_for_sample))
    return jacobian_sum_per_sample

def equipotential_standard_normal(d, n):
    '''Draws n samples from standard normal multivariate gaussian distribution of dimension d which are equipotential
    and are lying on a grand circle (unit d-sphere) on a n-1 manifold which was randomly chosen.
    d: number of dimensions
    n: size of sample
    return: n samples of size d from the standard normal distribution which are equally likely'''
    x = np.random.standard_normal((d, 1))  # starting sample

    r = np.sqrt(np.sum(x ** 2))  # ||x||
    x = x / r  # project sample on d-1-dimensional UNIT sphere --> x just defines direction
    t = np.random.standard_normal((d, 1))  # draw tangent sample
    t = t - (np.dot(np.transpose(t), x) * x)  # Gram Schmidth orthogonalization --> determines which circle is traversed
    t = t / (np.sqrt(np.sum(t ** 2)))  # standardize ||t|| = 1
    s = np.linspace(0, 2 * np.pi, n)  # space to span --> once around the circle in n steps
    s = s[0:(len(s) - 1)]
    t = s * t #if you wrap this samples around the circle you get once around the circle
    X = r * exp_map(x, t)  # project onto sphere, re-scale
    return (X)


def exp_map(mu, E):
    '''starting from a point mu on the grand circle adding a tangent vector to mu will end at a position outside of the
    circle. Samples need to be maped back on the circle.
    mu: starting sample
    E: tangents of different length from 0 to 2 pi times 1
    returns samples lying onto the unit circle.'''
    D = np.shape(E)[0]
    theta = np.sqrt(np.sum(E ** 2, axis=0))
    M = np.dot(mu, np.expand_dims(np.cos(theta), axis=0)) + E * np.sin(theta) / theta
    if (any(np.abs(theta) <= 1e-7)):
        for a in (np.where(np.abs(theta) <= 1e-7)):
            M[:, a] = mu
    M[:, abs(theta) <= 1e-7] = mu
    return (M)

def plot_heatmaps(dy=None, H=None, J=None): 
    if H is not None:
        with plt.rc_context(bundles.beamer_moml()):
            fig, (ax1, ax2) = plt.subplots(1, 2)
            sns.heatmap(H, cmap=cmap, norm=(MidpointNormalize(midpoint=0, vmin=np.min(H), vmax=np.max(H))), ax=ax1)
            ax1.set_title('Hessian $\partial_Y^2 f(X, Y^*(X))$')
            ax1.tick_params(bottom=False, left=False)
            ax1.set(xticklabels=[], yticklabels=[])
            sns.heatmap(H-np.diag(np.diag(H)), cmap=cmap, norm=(MidpointNormalize(midpoint=0, vmin=np.min(H-np.diag(np.diag(H))), vmax=np.max(H-np.diag(np.diag(H))))), ax=ax2)
            ax2.set_title('Hessian $\partial_Y^2 f(X, Y^*(X))$ - diag')
            ax2.tick_params(bottom=False, left=False)
            ax2.set(xticklabels=[], yticklabels=[])
            #plt.savefig('ifd_results/Hessian.pdf')
        v, w = np.linalg.eigh(H)
        w_sorted = np.transpose(np.transpose(w)[np.flip(np.argsort(v))])
        with plt.rc_context(bundles.beamer_moml()):
            fig, ax1 = plt.subplots()
            ax1.scatter([i for i in range(1, len(v)+1)], np.flip(np.sort(np.abs(v))), c=[1 if i>0 else 0 for i in np.flip(np.sort(v))])
            ax1.set_xlabel('eigenvalue')
            ax1.set_yscale('log')
            print(np.flip(np.sort(v)))
            #plt.savefig('ifd_results/Eigenvalues_Hessian.pdf')
        with plt.rc_context(bundles.beamer_moml()):
            fig, ax1 = plt.subplots()
            sns.heatmap(w_sorted, cmap=cmap, norm=(MidpointNormalize(midpoint=0, vmin=np.min(w_sorted), vmax=np.max(w_sorted))), ax=ax1)
            ax1.set_title('Eigenvectors Hessian $\partial_Y^2 f(X, Y^*(X))$')
            ax1.tick_params(bottom=False, left=False)
            ax1.set(xticklabels=[], yticklabels=[])
            #plt.savefig('ifd_results/Eigenvectors_Hessian.pdf')
    if J is not None:
        with plt.rc_context(bundles.beamer_moml()):
            fig, ax1 = plt.subplots()
            sns.heatmap(J, cmap=cmap, norm=(MidpointNormalize(midpoint=0, vmin=np.min(J), vmax=np.max(J))), ax=ax1)
            ax1.set_title('Jacobian $\partial_Y \partial_X f(X, Y^*(X))}$')
            ax1.tick_params(bottom=False, left=False)
            ax1.set(xticklabels=[], yticklabels=[])
            #plt.savefig('ifd_results/Mixed_Jacobian.pdf')
    if dy is not None:
        with plt.rc_context(bundles.beamer_moml()):
            fig, ax1 = plt.subplots()
            sns.heatmap(dy, cmap=cmap, norm=(MidpointNormalize(midpoint=0, vmin=np.min(dy), vmax=np.max(dy))), ax=ax1)
            ax1.set_title('Jacobian $\partial Y^*(X)$')
            ax1.tick_params(bottom=False, left=False)
            ax1.set(xticklabels=[], yticklabels=[])
            #plt.savefig('ifd_results/Jacobian.pdf')