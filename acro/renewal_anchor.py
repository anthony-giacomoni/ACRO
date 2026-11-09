"""Experimental independent anchor + common-random-numbers renewal curve.

V(x) = independent GBM reference + E[SV(x0)-GBM]
       + E[SV(x)-SV(x0)].

Both expectations are estimated with independent Monte Carlo banks. This is
unbiased AT NODES for a fixed time grid and an unbiased GBM reference. In an
option positive part the noisy curve still has Jensen bias, and interpolation
between nodes has bias. The anchor uncertainty is shared by every curve node.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from .engine import simulate_phoenix
from .conditional_qmc import innovation_bank
from .gbm_control import aggregated_bank,gbm_observation_grid

@dataclass
class AnchoredCurve:
    nodes: np.ndarray
    prices: np.ndarray
    anchor: float
    anchor_delta_se: float
    anchor_reference_se: float
    relative_node_se: np.ndarray
    elapsed_seconds: float
    anchor_paths: int
    relative_paths: int


def grid_including_anchor(low:float, high:float, count:int, anchor:float) -> np.ndarray:
    """Exactly retain anchor as a node while preserving total grid size/endpoints.

    Replaces its nearest interior grid node. Nonuniform spacing is intentional.
    """
    if count < 3 or not (np.isfinite([low, high, anchor]).all() and low < anchor < high):
        raise ValueError('Anchor must lie strictly inside a finite grid of >=3 points')
    nodes = np.linspace(low, high, count)
    k = int(np.argmin(np.abs(nodes-anchor)))
    if k == 0 or k == count-1:
        raise ValueError('Anchor too close to grid boundary for fixed-size replacement')
    nodes[k] = anchor
    if np.any(np.diff(nodes) <= 0):
        raise ValueError('Cannot preserve monotonicity of anchored grid')
    return nodes


def renewal_curve_anchor(p,coupon,steps,nodes,reference_value,reference_se,
                         anchor_paths,relative_paths,seed):
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('Only stochastic_vol for anchored renewal curve')
    if p.renewal_phoenix.capital_barrier_monitoring!='european':
        raise ValueError('Only European barriers for sparse GBM reference')
    nodes=np.asarray(nodes,dtype=float)
    if nodes.ndim!=1 or not np.all(np.diff(nodes)>0):raise ValueError('Sorted nodes required')
    if not np.any(nodes == p.market.initial_logvol_factor):
        raise ValueError('Anchor x0 must be included as an exact grid node')
    if anchor_paths<2 or relative_paths<2:raise ValueError('At least 2 paths required')
    start=perf_counter()
    nsteps=int(round(steps*p.renewal_phoenix.maturity_years))
    nassets=p.market.n_assets
    x0=p.market.initial_logvol_factor
    b0=innovation_bank(nsteps,anchor_paths,nassets,seed,'mc_bridge')
    gbm_steps=gbm_observation_grid(p,steps)
    sparse_bank=aggregated_bank(b0,steps,gbm_steps)
    vg=simulate_phoenix(replace(p.market,dynamics='gbm'),p.renewal_phoenix,
                        anchor_paths,gbm_steps,0,innovations=sparse_bank).pv(coupon)
    v0=simulate_phoenix(p.market,p.renewal_phoenix,anchor_paths,steps,0,
                        initial_factor=x0,innovations=b0).pv(coupon)
    d0=v0-vg
    anchor=float(reference_value+np.mean(d0))
    se_anchor=float(np.std(d0,ddof=1)/np.sqrt(anchor_paths))
    # Different innovations from anchor: transparent additive error sources.
    b1=innovation_bank(nsteps,relative_paths,nassets,seed+100003,'mc_bridge')
    base=simulate_phoenix(p.market,p.renewal_phoenix,relative_paths,steps,0,
                          initial_factor=x0,innovations=b1).pv(coupon)
    values=[];ses=[]
    for x in nodes:
        if abs(float(x)-x0)<1e-14:
            values.append(anchor);ses.append(0.0)
        else:
            v=simulate_phoenix(p.market,p.renewal_phoenix,relative_paths,steps,0,
                               initial_factor=float(x),innovations=b1).pv(coupon)
            delta=v-base
            values.append(anchor+float(delta.mean()))
            ses.append(float(delta.std(ddof=1)/np.sqrt(relative_paths)))
    return AnchoredCurve(nodes,np.asarray(values),anchor,se_anchor,float(reference_se),
                         np.asarray(ses),float(perf_counter()-start),anchor_paths,
                         relative_paths)
