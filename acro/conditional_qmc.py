"""Randomised Sobol/Brownian-bridge innovations for ACRO research validation.

The distribution of the finite-grid innovations is unchanged. RQMC is a
numerical integration method, not an alternative financial model. Independent
scrambles allow estimating *randomisation* error; they do not bound time-grid
bias or prove calibration quality.
"""
from __future__ import annotations
import heapq
import numpy as np
from scipy.stats import qmc, norm


def bridge_innovations(z):
    """Convert independent N(0,1) coordinates to independent Brownian increments.

    z: (paths, steps, independent_drivers) standard Gaussian innovations.
    Sobol dimensions enter in Brownian bridge priority order: endpoint first,
    then midpoints of the longest unfilled intervals. The output has shape
    (steps, paths, independent_drivers), suitable for simulate_phoenix.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 3 or z.shape[1] < 1 or z.shape[2] < 1:
        raise ValueError('Expected (n_paths, n_steps, n_drivers)')
    n, m, d = z.shape
    w = np.empty((n, m+1, d), dtype=np.float64)
    w[:, 0, :] = 0.
    w[:, m, :] = z[:, 0, :]
    # Heap priority: longest interval first, with reproducible tie-breaking.
    heap = [(-m, 0, m)]
    k = 1
    while heap:
        _negspan, l, r = heapq.heappop(heap)
        if r-l <= 1:
            continue
        mid = (l+r)//2
        w[:, mid, :] = ((r-mid)*w[:, l, :] + (mid-l)*w[:, r, :])/(r-l)
        w[:, mid, :] += np.sqrt((mid-l)*(r-mid)/(m*(r-l)))*z[:, k, :]
        k += 1
        heapq.heappush(heap, (-(mid-l), l, mid))
        heapq.heappush(heap, (-(r-mid), mid, r))
    if k != m:
        raise AssertionError(f'Expected {m} Brownian bridge coordinates, used {k}')
    return np.transpose(np.diff(w, axis=1)*np.sqrt(m), (1, 0, 2)).copy()


def innovation_bank(steps, npaths, nassets, seed, method='sobol_bridge'):
    """Generate eq and factor shocks with appropriate joint Gaussian law.

    'sobol_bridge': independent Owen-scrambled Sobol sequence, power-of-two n.
    'mc_bridge': ordinary independent pseudo-random normals mapped through the
    *same* bridge so the pricing code is identical across both benchmarks.
    """
    if steps < 1 or nassets < 1 or npaths < 2:
        raise ValueError('Invalid steps, nassets or npaths')
    drivers = nassets + 1
    if method == 'sobol_bridge':
        if npaths & (npaths-1):
            raise ValueError('Sobol requires a power-of-two path count')
        sampler = qmc.Sobol(d=steps*drivers, scramble=True, seed=int(seed))
        uniforms = sampler.random_base2(int(np.log2(npaths)))
        np.clip(uniforms, np.finfo(float).eps, 1-np.finfo(float).eps, out=uniforms)
        normals = norm.ppf(uniforms).reshape(npaths, steps, drivers)
    elif method == 'mc_bridge':
        normals = np.random.default_rng(seed).standard_normal((npaths, steps, drivers))
    else:
        raise ValueError(f'Unknown innovation method: {method}')
    bridge = bridge_innovations(normals)
    return bridge[:, :, :nassets], bridge[:, :, nassets]
