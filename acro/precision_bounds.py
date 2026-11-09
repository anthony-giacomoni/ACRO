"""Independent conditional-curve upper/lower ACRO pricing estimators.

Given two INDEPENDENT unbiased MC estimators V_A(x), V_B(x) for the true
renewal value at each interpolation node, and independent outer states:

upper = (E[(V_A-K)+] + E[(V_B-K)+])/2,
lower = (E[(V_A-K) 1_{V_B>K}] + E[(V_B-K) 1_{V_A>K}])/2.

For a FIXED interpolation rule and finite time grid, expected lower <= exact
price of that interpolated conditional-value function <= expected upper.
(Strictly, interpolation of the true nodal values occurs before the max.)
These are bounds IN EXPECTATION, not pathwise confidence intervals. Finite
reference randomizations, sampling uncertainty, interpolation bias and time
Euler bias still require separate treatment. A common GBM reference between
A and B is forbidden here, so each half uses a fresh GBM reference.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .config import Project
from .conditional_qmc import innovation_bank
from .gbm_control import gbm_reference,renewal_curve_gbm_cv,aggregated_bank
from .fast_engine import simulate_phoenix_fast
from dataclasses import replace

@dataclass
class CurveDraw:
    nodes: np.ndarray
    values: np.ndarray
    reference_se: float
    inner_paths: int
    delta_node_se: np.ndarray


def independent_curve_draw(p:Project,nodes,coupon,steps,n_inner,n_ref,scrambles,seed,use_fast=False):
    """GBM-CV curve using independent reference and inner innovations."""
    nodes=np.asarray(nodes,float)
    if nodes.ndim!=1 or nodes.size<2 or np.any(np.diff(nodes)<=0):
        raise ValueError('Strictly increasing nodes required')
    reference=gbm_reference(p,coupon,steps,n_ref,scrambles,seed+10001)
    bank=innovation_bank(int(round(p.renewal_phoenix.maturity_years*steps)),
                         n_inner,p.market.n_assets,seed+20007,'mc_bridge')
    if use_fast:
        sparse=aggregated_bank(bank,steps,p.renewal_phoenix.observations_per_year)
        gbm=simulate_phoenix_fast(replace(p.market,dynamics='gbm'),
                                  p.renewal_phoenix,n_inner,
                                  p.renewal_phoenix.observations_per_year,
                                  innovations=sparse).pv(coupon)
        values=[];delta_var=[]
        for x in nodes:
            sv=simulate_phoenix_fast(p.market,p.renewal_phoenix,n_inner,steps,
                                    innovations=bank,initial_factor=float(x)).pv(coupon)
            delta=sv-gbm
            values.append(float(reference.value+delta.mean()))
            delta_var.append(float(delta.var(ddof=1)))
        values=np.asarray(values);delta_var=np.asarray(delta_var)
    else:
        _,values,_,delta_var,_,_=renewal_curve_gbm_cv(p,coupon,steps,nodes,bank,reference.value)
    return CurveDraw(nodes,values,reference.std_error,n_inner,np.sqrt(delta_var/n_inner))


def crossfit_payoffs(p:Project,outer,curve_a:CurveDraw,curve_b:CurveDraw):
    """Pathwise upper and out-of-sample exercise-policy lower payoffs.

    A and B MUST use independent renewal inner simulations and GBM references.
    Caller is responsible for ensuring this independence and independent outer.
    Return arrays over all outer scenarios, including zero for no autocall.
    """
    if not np.array_equal(curve_a.nodes,curve_b.nodes):
        raise ValueError('Curves must have identical interpolation nodes')
    eligible=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= outer.first_call_year <= p.acro.last_eligible_call_year+1e-10
    y_upper=np.zeros(len(outer.first_call_year),float)
    y_lower=np.zeros(len(outer.first_call_year),float)
    if eligible.any():
        x=outer.factor_at_call[eligible]
        if x.min()<curve_a.nodes[0]-1e-10 or x.max()>curve_a.nodes[-1]+1e-10:
            raise ValueError('Outer states outside renewal grid')
        a=np.interp(x,curve_a.nodes,curve_a.values)-p.acro.strike
        b=np.interp(x,curve_b.nodes,curve_b.values)-p.acro.strike
        disc=p.acro.renewal_units*np.exp(-p.market.risk_free_rate*outer.first_call_year[eligible])
        y_upper[eligible]=disc*.5*(np.maximum(a,0.)+np.maximum(b,0.))
        y_lower[eligible]=disc*.5*(a*(b>0.)+b*(a>0.))
    return y_lower,y_upper
