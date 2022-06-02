import gzip
import pickle

import sys, os
from os import path

import numpy as np
from sklearn.utils import check_random_state

from sklearn.manifold import TSNE as SKLTSNE
import openTSNE

import time

from tsne_jax import tsne


class TSNEBenchmark:
    perplexity = 30
    learning_rate = 200
    n_jobs = 1

    def run(self, n_samples=1000, random_state=None):
        raise NotImplementedError()

    def run_multiple(self, n=5, n_samples=1000):
        t = []
        for idx in range(n):
            t.append(self.run(n_samples=n_samples, random_state=idx))
        return t

    def load_data(self, n_samples=None):
        with gzip.open(path.join("examples/data/mnist", "mnist.pkl.gz"), "rb") as f:
            data = pickle.load(f)

        x, y = data["pca_50"], data["labels"]

        if n_samples is not None:
            indices = np.random.choice(
                list(range(x.shape[0])), n_samples, replace=False
            )
            x, y = x[indices], y[indices]

        return x, y

    
class sklearnBarnesHut(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state
        )

        start = time.time()
        SKLTSNE(
            early_exaggeration=12,
            learning_rate=self.learning_rate,
            angle=0.5,
            perplexity=self.perplexity,
            init=init,
            method='barnes_hut',
            verbose=True,
            random_state=random_state
        ).fit_transform(x)
        total_time = time.time() - start
        print("scikit-learn BarnesHut t-SNE:", total_time, flush=True)
        return total_time
        

        
class sklearnVanilla(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state
        )

        start = time.time()
        SKLTSNE(
            early_exaggeration=12,
            learning_rate=self.learning_rate,
            angle=0.5,
            perplexity=self.perplexity,
            init=init,
            method='exact',
            verbose=True,
            random_state=random_state
        ).fit_transform(x)
        total_time = time.time() - start
        print("scikit-learn Vanilla t-SNE:", total_time, flush=True)
        return total_time
        

class openTSNEFFT(TSNEBenchmark):
    def run(self, n_samples=1000, random_state=None):
        x, y = self.load_data(n_samples=n_samples)

        print("-" * 80)
        print("Random state", random_state)
        print("-" * 80, flush=True)

        random_state = check_random_state(random_state)

        start = time.time()
        start_aff = time.time()
        affinity = openTSNE.affinity.PerplexityBasedNN(
            x,
            perplexity=self.perplexity,
            method="annoy",
            random_state=random_state,
            verbose=True,
        )
        print("openTSNE: NN search", time.time() - start_aff, flush=True)

        init = openTSNE.initialization.random(
            x, n_components=2, random_state=random_state, verbose=True,
        )

        start_optim = time.time()
        embedding = openTSNE.TSNEEmbedding(
            init,
            affinity,
            learning_rate=self.learning_rate,
            n_jobs=self.n_jobs,
            negative_gradient_method="fft",
            random_state=random_state,
            verbose=True,
        )
        embedding.optimize(250, exaggeration=12, momentum=0.8, inplace=True)
        embedding.optimize(750, momentum=0.5, inplace=True)
        total_time = time.time() - start
        print("openTSNE: Optimization", time.time() - start_optim)
        print("openTSNE: Full", total_time, flush=True)
        return total_time

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

def run_multiple_jax_tsne(n=5, n_samples=1000):
    t = []
    for idx in range(n):
        t.append(runJaxTSNE(n_samples=n_samples, random_state=idx))
    return t

def runJaxTSNE(n_samples=1000, random_state=None):
    x, y = load_data(n_samples=n_samples)
    print("-" * 80)
    print("Random state", random_state)
    print("-" * 80, flush=True)

    start = time.time()
    tsne(X=x, no_dims=2, initial_dims=50, perplexity=30.0, learning_rate=200, max_iter = 1000, key=random_state)
    
    total_time = time.time() - start
    print("jax t-SNE:", total_time, flush=True)
    return total_time


if __name__ == "__main__":
    n = [100, 1000, 10000, 20000, 30000, 40000, 50000, 60000]
    n = [100, 1000]
    repetitions = 5
    
    tsnes = [run_multiple_jax_tsne, sklearnBarnesHut, openTSNEFFT]
    tsnes_str = ['JaxTSNE', 'sklearnBarnesHutTSNE', 'openTSNEFFT']
    times = {}
    for t_sne, t_sne_str in zip(tsnes, tsnes_str):
        times_s = []
        for s in n:
            if t_sne==run_multiple_jax_tsne:
                times_s.append(t_sne(n=repetitions, n_samples=s))
            else:
                t = t_sne()
                times_s.append(t.run_multiple(n=repetitions, n_samples=s))
        times[t_sne_str] = times_s
    times['repetitions'] = repetitions
    times['n_samples'] = n
    with gzip.open("benchmark_results/runtimes.pkl.gz", "wb") as f:
        pickle.dump(times, f)
