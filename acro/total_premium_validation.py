"""V5.19: stratified *direct* nested premium audit, without renewal interpolation.

Pricing Q for a FIXED coupon. Inner simulators A and B have independent
reference estimators and Brownian innovations. They share innovations ACROSS
selected states within one half for efficiency (this is accounted for by
replicating the entire bank). Exercise is selected by A, evaluated by B, and
vice-versa. In expectation lower <= discrete-model value <= upper, conditional
on the sampled external paths, when the inner estimators are unbiased.

A pilot renewal curve, simulated independently of A/B, defines three strata
of eligible autocall states: OTM, near exercise, ITM. State selection is
uniform without replacement *within each stratum*, with finite-population
weights; it does NOT change the target Q expectation. The pilot is not a
pricing reference for the direct audit. Full statistical uncertainty includes
variation of the outer, pilot and inner banks across independent replications,
not discretization, market calibration or a rigorous interpolation bound.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t
from .config import Project
from .engine import simulate_phoenix
from .fast_engine import simulate_phoenix_fast
from .conditional_qmc import innovation_bank
from .gbm_control import gbm_reference,aggregated_bank
from .precision_bounds import independent_curve_draw


def stratified_indices(scores, wanted, threshold, rng):
    """Return indices and inverse-sampling weights N_h/m_h for 3 disjoint bins.

    Each nonempty bin receives >=1 draw. Importance of near-exercise set to 4.
    If all selected, returns a census with weights 1. All indices unique.
    """
    scores=np.asarray(scores,dtype=float)
    if scores.ndim!=1 or not np.all(np.isfinite(scores)):
        raise ValueError('finite one-dimensional moneyness scores required')
    if wanted<1 or threshold<=0:
        raise ValueError('positive wanted and threshold required')
    ids=np.arange(len(scores))
    groups=[ids[scores < -threshold], ids[np.abs(scores) <= threshold], ids[scores>threshold]]
    live=[i for i,g in enumerate(groups) if len(g)]
    if not live: return np.array([],int),np.array([],float),np.array([],int),np.zeros(3,int)
    if wanted<len(live): raise ValueError('wanted must cover every nonempty stratum')
    count=np.array([len(g) for g in groups],int)
    alloc=np.array([int(len(g)>0) for g in groups])
    wanted=min(int(wanted),len(scores))
    prop=np.array([1.,4.,1.])
    while alloc.sum()<wanted:
        capacity=count-alloc
        candidates=np.flatnonzero(capacity>0)
        k=max(candidates,key=lambda i:prop[i]*count[i]/(alloc[i]+.5))
        alloc[k]+=1
    selected=[]; weights=[]; labels=[]
    for j,g in enumerate(groups):
        if not len(g):continue
        chosen=np.asarray(rng.choice(g,size=alloc[j],replace=False),int)
        selected.extend(chosen)
        weights.extend([len(g)/alloc[j]]*len(chosen))
        labels.extend([j]*len(chosen))
    return np.array(selected,int),np.array(weights,float),np.array(labels,int),count


def direct_values(p:Project, x, coupon, *, inner_paths, ref_paths, scrambles, seed, fast=True):
    """SV value at EACH given x, without interpolating, with a GBM CV reference."""
    x=np.asarray(x,float)
    if x.ndim!=1 or not np.all(np.isfinite(x)):
        raise ValueError('x must be finite 1D array')
    if inner_paths<2 or scrambles<2:
        raise ValueError('two or more paths and scrambles required')
    s=p.simulation; terms=p.renewal_phoenix; steps=s.steps_per_year
    reference=gbm_reference(p,coupon,steps,ref_paths,scrambles,seed+10001)
    bank=innovation_bank(int(round(terms.maturity_years*steps)),inner_paths,
                         p.market.n_assets,seed+20007,method='mc_bridge')
    simulator=simulate_phoenix_fast if fast else simulate_phoenix
    gbm_market=replace(p.market,dynamics='gbm')
    gbm_bank=aggregated_bank(bank,steps,terms.observations_per_year)
    gbm=simulator(gbm_market,terms,inner_paths,terms.observations_per_year,0,
                  innovations=gbm_bank).pv(coupon)
    values=[];inner_ses=[]
    for z in x:
        sv=simulator(p.market,terms,inner_paths,steps,0,initial_factor=float(z),
                     innovations=bank).pv(coupon)
        diff=sv-gbm
        values.append(reference.value+diff.mean())
        inner_ses.append(diff.std(ddof=1)/np.sqrt(inner_paths))
    return np.asarray(values),np.asarray(inner_ses),float(reference.std_error)


def selected_bounds(val_a,val_b,discount,weights,*,total_paths,strike=100.,units=1.):
    """HT-weighted direct cross-fitting bounds over FULL initial outer paths."""
    a=np.asarray(val_a,float)-strike
    b=np.asarray(val_b,float)-strike
    d=np.asarray(discount,float); w=np.asarray(weights,float)
    if not (a.shape==b.shape==d.shape==w.shape):
        raise ValueError('Shapes must coincide')
    if total_paths<1 or np.any(w<0) or np.any(d<0):
        raise ValueError('Invalid weights or total_paths')
    upper=np.sum(w*d*units*(np.maximum(a,0)+np.maximum(b,0)))/(2*total_paths)
    lower=np.sum(w*d*units*(a*(b>0)+b*(a>0)))/(2*total_paths)
    return float(lower),float(upper)


def difference_corrected_bounds(pilot_full, selected_pilot, raw_lower, raw_upper):
    """Design-unbiased correction to direct crossfit bounds in expectation."""
    a=float(pilot_full)+float(raw_lower)-float(selected_pilot)
    b=float(pilot_full)+float(raw_upper)-float(selected_pilot)
    return a,b


def eligible_paths(p, outer):
    mask=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        mask &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    return mask


def ci_banks(data):
    x=np.asarray(data,float)
    if x.ndim!=1 or len(x)<2:
        raise ValueError('At least 2 independent banks required')
    mean=float(x.mean());se=float(x.std(ddof=1)/np.sqrt(len(x)))
    return mean,(mean-float(student_t.ppf(.975,len(x)-1))*se,
                 mean+float(student_t.ppf(.975,len(x)-1))*se),se


def run_bank(p:Project, *, seed, outer_paths, selected_states,
             inner_paths, ref_paths, scrambles,pilot_inner,pilot_nodes,margin, fast=True):
    """Independent direct bound experiment for one fresh outer+pilot+A+B bank."""
    p.validate()
    if p.initial_coupon_policy!='fixed':
        raise ValueError('Contractual coupon must be fixed ex ante')
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('This validation targets stochastic-vol dynamics')
    start=perf_counter(); coupon=p.acro.target_coupon_rate
    if coupon is None:coupon=p.initial_phoenix.coupon_rate
    steps=p.simulation.steps_per_year
    sim=simulate_phoenix_fast if fast else simulate_phoenix
    outer=sim(p.market,p.initial_phoenix,outer_paths,steps,seed+11)
    elig=eligible_paths(p,outer)
    factors=np.asarray(outer.factor_at_call[elig],float)
    years=np.asarray(outer.first_call_year[elig],float)
    if len(factors)<3:
        raise ValueError('Too few eligible autocalls')
    # A separate, independently simulated curve is used ONLY for sampling.
    low=float(min(factors.min()-.04,p.market.initial_logvol_factor-.02))
    high=float(max(factors.max()+.04,p.market.initial_logvol_factor+.02))
    nodes=np.unique(np.r_[np.linspace(low,high,pilot_nodes),p.market.initial_logvol_factor])
    pilot=independent_curve_draw(p,nodes,coupon,steps,pilot_inner,
                                 ref_paths,scrambles,seed+100000,fast)
    scores=np.interp(factors,pilot.nodes,pilot.values)-p.acro.strike
    indices,weights,classes,counts=stratified_indices(scores,selected_states,margin,
                                         np.random.default_rng(seed+120000))
    selected_factors=factors[indices]; discounted=np.exp(-p.market.risk_free_rate*years[indices])
    v_a,mcse_a,refse_a=direct_values(p,selected_factors,coupon,inner_paths=inner_paths,
                ref_paths=ref_paths,scrambles=scrambles,seed=seed+200000,fast=fast)
    v_b,mcse_b,refse_b=direct_values(p,selected_factors,coupon,inner_paths=inner_paths,
                ref_paths=ref_paths,scrambles=scrambles,seed=seed+400000,fast=fast)
    raw_lower,raw_upper=selected_bounds(v_a,v_b,discounted,weights,total_paths=outer_paths,
                strike=p.acro.strike,units=p.acro.renewal_units)
    # Generalized DIFFERENCE estimator: pilot integral over ALL outer states,
    # plus Horvitz-Thompson correction of direct-minus-pilot values on selected
    # states. Conditional on the independent pilot, the correction has the
    # same expected price as direct nested MC but smaller subsampling variance.
    # Crucially, interpolation of the pilot is NOT a bias of this estimator:
    # its total is canceled in expectation by the weighted correction.
    full_disc=np.exp(-p.market.risk_free_rate*years)
    pilot_value=float(np.sum(full_disc*p.acro.renewal_units*np.maximum(scores,0))/outer_paths)
    selected_pilot=float(np.sum(weights*discounted*p.acro.renewal_units*
                                      np.maximum(scores[indices],0))/outer_paths)
    lower,upper=difference_corrected_bounds(pilot_value,selected_pilot,raw_lower,raw_upper)
    rows=[]
    for k,j in enumerate(indices):
        rows.append({'factor_x':float(selected_factors[k]),'year':float(years[j]),
         'stratum':['otm','near','itm'][classes[k]],'weight':float(weights[k]),
         'pilot_moneyness':float(scores[j]),'va':float(v_a[k]),'vb':float(v_b[k]),
         'inner_se_a':float(mcse_a[k]),'inner_se_b':float(mcse_b[k]),
         'discount':float(discounted[k])})
    record={'bank_seed':int(seed),'n_outer':int(outer_paths),'n_eligible':int(len(factors)),
       'n_selected':int(len(indices)), 'strata_population':counts.tolist(),
       'strata_selected':[int(np.sum(classes==j)) for j in range(3)],
       'pilot_premium':pilot_value,'direct_lower':lower,'direct_upper':upper,
       'raw_direct_lower':raw_lower,'raw_direct_upper':raw_upper,
       'sampled_pilot_premium':selected_pilot,
       'direct_midpoint_diagnostic':(lower+upper)/2,
       'exercise_policy_gap':upper-lower,
       'reference_se_a':refse_a,'reference_se_b':refse_b,
       'max_inner_se':max(float(mcse_a.max()),float(mcse_b.max())),
       'time_seconds':perf_counter()-start}
    return record,rows
