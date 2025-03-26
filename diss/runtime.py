from tsne_jax import *
import time
import pandas as pd
from sklearn.datasets import make_blobs, make_spd_matrix

if __name__ == "__main__":
  #n_iter = 2**11
  ns = [int(i) for i in np.logspace(1, 4, num=10, base=10)]
  ds = [int(i) for i in np.logspace(1, 4, num=10, base=10)]
  

  results = {'n':[],
             'd': [],
             'tsne_fwd [s]': [],
             'compute cov [s]': []}

  # init
  # Generate dataset
  X, y = make_blobs(n_samples=50, n_features=10, centers=10, random_state=0, shuffle=False, cluster_std=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

  N = np.array(onp.random.rand(50, 50))
  D = np.array(onp.random.rand(10, 10))

  # tsne fwd
  key = random.PRNGKey(41)
  y_guess = random.normal(key, shape=(X.shape[0], 2))
  Y_star = tsne_fwd(X, y_guess)
  X_flat, X_unflattener = flatten_util.ravel_pytree(X)   # row-wise
  Y_flat, Y_unflattener = flatten_util.ravel_pytree(Y_star) 

  # computation cov
  f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
  H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
  H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)

  f = partial(KL_divergence_dy, Y_flat=Y_flat, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
  # jvp
  _, jvp_fun_lin = jax.linearize(f, X_flat)
  # vjp
  _, vjp_fun = vjp(f, X_flat)

  compute_cov_fun = lambda i: compute_cov_inner(vjp_fun=vjp_fun, jvp_fun_lin=jvp_fun_lin, 
                                    H_pinv_i=i, D=D, N=N, d=D.shape[0], n=N.shape[0], H_pinv=H_pinv)
  final_cov = vmap(compute_cov_fun)(H_pinv)

  for n in ns:
    for d in ds:
      results['n'].append(n)
      results['d'].append(d)

      # Generate dataset
      X, y = make_blobs(n_samples=n, n_features=d, centers=10, random_state=0, shuffle=False, cluster_std=[2, 2, 2, 2, 2, 2, 2, 2, 2, 2])

      N = np.array(onp.random.rand(n, n))
      D = np.array(onp.random.rand(d, d))
      
      # tsne fwd
      start = time.time()
      key = random.PRNGKey(41)
      y_guess = random.normal(key, shape=(X.shape[0], 2))
      Y_star = tsne_fwd(X, y_guess)
      X_flat, X_unflattener = flatten_util.ravel_pytree(X)   # row-wise
      Y_flat, Y_unflattener = flatten_util.ravel_pytree(Y_star) 
      end = time.time()
      results['tsne_fwd [s]'].append(end-start)

      # computation cov
      start = time.time()
      f = partial(KL_divergence_dy, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
      H = jax.jacrev(f, argnums=1)(X_flat, Y_flat)
      H_pinv = np.linalg.pinv(H + 1e-3*np.eye(len(H)), hermitian=True)

      f = partial(KL_divergence_dy, Y_flat=Y_flat, X_unflattener=X_unflattener, Y_unflattener=Y_unflattener)
      # jvp
      _, jvp_fun_lin = jax.linearize(f, X_flat)
      # vjp
      _, vjp_fun = vjp(f, X_flat)

      compute_cov_fun = lambda i: compute_cov_inner(vjp_fun=vjp_fun, jvp_fun_lin=jvp_fun_lin, 
                                        H_pinv_i=i, D=D, N=N, d=D.shape[0], n=N.shape[0], H_pinv=H_pinv)
      final_cov = vmap(compute_cov_fun)(H_pinv)
      end = time.time()
      results['compute cov [s]'].append(end-start)

      results_datasets = pd.DataFrame(results)
      results_datasets.to_csv('runtime_results.csv', index=False)




            