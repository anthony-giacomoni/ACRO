"""GBM-surrogate control variate for the 1-factor ACRO renewal pricer.

For European capital barriers, GBM log returns have exact normal increments.
Thus the GBM Phoenix payoff only depends on dates when coupons/calls/maturity
are observed. Its expectation can be estimated *cheaply* on that reduced time
grid. An SV payoff evaluated on the SAME underlying Brownian increments minus
the GBM payoff has a substantially smaller variance when the dynamics are close.

An independently randomized Sobol reference remains an estimate, not a closed-
form analytic price. Always propagate its statistical uncertainty.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t
from .config import Project
from .engine import simulate_phoenix
from .conditional_qmc import innovation_bank


@dataclass
class GBMReference:
    value: float
    std_error: float
    ci95: tuple[float, float]
    randomized_prices: np.ndarray
    runtime_seconds: float
    paths_per_scramble: int
    scrambles: int


def gbm_observation_grid(p:Project, fine_steps:int) -> int:
    """Sparse GBM simulation is exact at payoff observation dates (European barrier)."""
    t=p.renewal_phoenix
    if t.capital_barrier_monitoring != 'european':
        raise ValueError('Sparse GBM reference only supports European capital barrier')
    coarse=t.observations_per_year
    if fine_steps % coarse:
        raise ValueError('Fine time grid must be divisible by coupon observations/year')
    t.validate(coarse)  # Autocall dates must be a subset of coupon dates.
    return coarse


def aggregated_bank(bank, fine_steps, coarse_steps):
    if fine_steps % coarse_steps:
        raise ValueError('Incompatible observation grids')
    ratio=fine_steps//coarse_steps
    eq, fac=bank
    if eq.shape[0]%ratio:
        raise ValueError('Time grid does not align with issue tenor')
    nsteps=eq.shape[0]//ratio
    return (eq.reshape(nsteps,ratio,*eq.shape[1:]).sum(axis=1)/np.sqrt(ratio),
            fac.reshape(nsteps,ratio,fac.shape[1]).sum(axis=1)/np.sqrt(ratio))


def gbm_reference(p:Project, coupon:float, fine_steps:int,
                  paths_per_scramble=65536, scrambles=8, base_seed=506001):
    """Independent unbiased RQMC quote for the GBM surrogate.

    CI95 is Student-t on independent scrambled Sobol replicate quotes and only
    measures randomization error in the finite-grid GBM surrogate.
    """
    steps=gbm_observation_grid(p,fine_steps)
    market=replace(p.market,dynamics='gbm')
    t0=perf_counter()
    quotes=[]
    for r in range(scrambles):
        bank=innovation_bank(int(round(steps*p.renewal_phoenix.maturity_years)),
                             paths_per_scramble,p.market.n_assets,base_seed+101*r,
                             method='sobol_bridge')
        ph=simulate_phoenix(market,p.renewal_phoenix,paths_per_scramble,steps,seed=0,
                            innovations=bank)
        quotes.append(float(ph.pv(coupon).mean()))
    quotes=np.asarray(quotes)
    mean=float(quotes.mean()); se=float(quotes.std(ddof=1)/np.sqrt(scrambles))
    tcrit=float(student_t.ppf(.975,scrambles-1))
    return GBMReference(mean,se,(mean-tcrit*se,mean+tcrit*se),quotes,
                        perf_counter()-t0,paths_per_scramble,scrambles)


def renewal_curve_gbm_cv(p:Project,coupon:float,fine_steps:int,
                          nodes,innovations,gbm_fair_value:float):
    """Quote SV(x) via an independent GBM reference plus a correlated difference.

    The reference estimate must be independent of these simulation innovations.
    Returns (raw_node_prices, controlled_node_prices, raw_path_variance_at_x0,
    delta_path_variance_at_x0, extra_cpu_seconds). Pathwise Y_sv-Y_gbm uses
    common equity innovations; both expectations are under their respective Q
    model. Because the GBM reference is unbiased, E[CV_quote] = E[SV_quote].
    """
    t0=perf_counter()
    eq,fac=innovations;n=eq.shape[1]
    sparse=gbm_observation_grid(p,fine_steps)
    coarse_bank=aggregated_bank(innovations,fine_steps,sparse)
    gbm=simulate_phoenix(replace(p.market,dynamics='gbm'),p.renewal_phoenix,
                         n,sparse,seed=0,innovations=coarse_bank).pv(coupon)
    baseline_seconds=perf_counter()-t0
    raw=[];cv=[];raw_var=[];delta_var=[]; sv_seconds=0.
    for x in nodes:
        st=perf_counter()
        sv=simulate_phoenix(p.market,p.renewal_phoenix,n,fine_steps,seed=0,
                           initial_factor=float(x),innovations=innovations).pv(coupon)
        sv_seconds += perf_counter()-st
        delta=sv-gbm
        raw.append(float(sv.mean()))
        cv.append(float(gbm_fair_value+delta.mean()))
        raw_var.append(float(sv.var(ddof=1)))
        delta_var.append(float(delta.var(ddof=1)))
    return (np.asarray(raw),np.asarray(cv),np.asarray(raw_var),
            np.asarray(delta_var),sv_seconds,baseline_seconds)
