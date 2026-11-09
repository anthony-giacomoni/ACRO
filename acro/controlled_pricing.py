"""Optional V5.7 GBM-control pricing route for the full ACRO demonstration.

Both Phoenix B (principal) and A (coupon annuity) are controlled separately.
GBM references use independent randomized Sobol scrambles, with coupon-only
observation dates when European capital protection is specified. The stochastic
volatility paths and the GBM surrogate share the same equity innovations.

This reduces INNER estimation noise; it does not remove temporal discretization,
interpolation, outer error, or the nonlinear Jensen bias. The legacy price_acro
route remains untouched and is the default for backward compatibility.
"""
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import replace
from time import perf_counter
import numpy as np
from .config import Project, Phoenix
from .engine import simulate_phoenix
from .conditional_qmc import innovation_bank
from .gbm_control import aggregated_bank
from .pricing import ConditionalCurve, Quote, same_contract


@dataclass
class ComponentReference:
    B: float
    A: float
    pv_se: float
    B_se: float
    A_se: float
    seconds: float
    scrambles: int
    paths_per_scramble: int
    replicate_components: np.ndarray
    def price_se(self, coupon):
        x=self.replicate_components[:,0]+coupon*self.replicate_components[:,1]
        return float(x.std(ddof=1)/np.sqrt(len(x)))


def _reference(p: Project, terms: Phoenix, coupon: float, scrambles: int,
               paths_per_scramble: int, seed: int) -> ComponentReference:
    if terms.capital_barrier_monitoring != 'european':
        raise ValueError('GBM coupon-only reference requires a European capital barrier')
    if scrambles < 2 or paths_per_scramble < 2 or paths_per_scramble & (paths_per_scramble-1):
        raise ValueError('Scrambles>=2; Sobol paths per scramble must be a power of two >=2')
    steps=terms.observations_per_year
    gbm=replace(p.market,dynamics='gbm')
    start=perf_counter(); vals=[]
    for k in range(scrambles):
        bank=innovation_bank(int(round(terms.maturity_years*steps)),paths_per_scramble,
                             p.market.n_assets,seed+271*k,'sobol_bridge')
        sim=simulate_phoenix(gbm,terms,paths_per_scramble,steps,0,innovations=bank)
        vals.append((float(sim.principal_pv.mean()),float(sim.coupon_annuity_pv.mean())))
    vals=np.asarray(vals); B,A=vals.mean(axis=0)
    se=np.std(vals,axis=0,ddof=1)/np.sqrt(scrambles)
    v=vals[:,0]+coupon*vals[:,1]
    return ComponentReference(float(B),float(A),float(v.std(ddof=1)/np.sqrt(scrambles)),
                              float(se[0]),float(se[1]),float(perf_counter()-start),
                              scrambles,paths_per_scramble,vals)


def _make_bank(p, terms, n, steps, seed):
    return innovation_bank(int(round(terms.maturity_years*steps)),n,p.market.n_assets,
                           seed,method='mc_bridge')


def _sv_gbm_pair(p, terms, steps, bank, x):
    n=bank[0].shape[1]
    gbm=simulate_phoenix(replace(p.market,dynamics='gbm'),terms,n,
                         terms.observations_per_year,0,
                         innovations=aggregated_bank(bank,steps,terms.observations_per_year))
    sv=simulate_phoenix(p.market,terms,n,steps,0,
                        initial_factor=x,innovations=bank)
    return sv,gbm


def controlled_quote(p: Project, *, ref_paths=4096, scrambles=4, seed_offset=0):
    """Return a GBM-controlled initial quote and reusable quote innovation bank."""
    p.validate()
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('Controlled route is for stochastic_vol; use legacy for GBM')
    s=p.simulation; steps=s.steps_per_year
    if steps%p.initial_phoenix.observations_per_year:
        raise ValueError('Grid incompatible with coupon dates')
    # The reference is independent of the inner SV sample.
    coupon_for_reference=(p.initial_phoenix.coupon_rate if p.initial_coupon_policy=='fixed'
                          else 0.18)
    ref=_reference(p,p.initial_phoenix,coupon_for_reference,scrambles,ref_paths,
                   int(s.seed)+671007+seed_offset)
    bank=_make_bank(p,p.initial_phoenix,s.quote_paths,steps,int(s.seed)+751007+seed_offset)
    sv,gbm=_sv_gbm_pair(p,p.initial_phoenix,steps,bank,p.market.initial_logvol_factor)
    db=sv.principal_pv-gbm.principal_pv
    da=sv.coupon_annuity_pv-gbm.coupon_annuity_pv
    B=ref.B+db.mean(); A=ref.A+da.mean()
    if A<=0: raise ValueError('Nonpositive coupon annuity')
    c=(p.initial_phoenix.nominal-B)/A if p.initial_coupon_policy=='fair' else p.initial_phoenix.coupon_rate
    if c<0: raise ValueError('Negative par coupon')
    note_pv=B+c*A
    # SE includes independent surrogate reference error and inner MC error,
    # but excludes any time-grid / calibration error. Delta-method for par c.
    se_inner=float(np.std(db+c*da,ddof=1)/np.sqrt(len(db)))
    se_value=float(np.hypot(se_inner,ref.price_se(c)))
    q=Quote(float(c),float(note_pv),float(se_value/A),se_value,
            float(B),float(A),sv)
    diagnostics={'gbm_reference_seconds':ref.seconds,
                 'gbm_reference_value_at_coupon_for_reference':ref.B+coupon_for_reference*ref.A,
                 'gbm_reference_se_at_coupon_for_reference':ref.pv_se,
                 'quote_inner_mc_se_at_selected_coupon':se_inner,
                 'quote_se_approx_at_selected_coupon':se_value,
                 'gbm_reference_se_at_selected_coupon':ref.price_se(c),
                 'gbm_scrambles':scrambles,'gbm_reference_paths_per_scramble':ref_paths}
    return q,bank,diagnostics


def controlled_curve(p: Project, q: Quote, quote_bank, eligible_factors,
                     *, ref_paths=4096, scrambles=4, seed_offset=0):
    """Curve for all configurations with a European capital barrier.

    Matching note: V_R(x0) anchored to the same GBM-controlled issue quote;
    changes in state are estimated using CRN increments. Non-matching terms:
    an independent GBM reference plus SV-minus-GBM control is used at each x.
    """
    s=p.simulation; steps=s.steps_per_year; terms=p.renewal_phoenix
    x0=p.market.initial_logvol_factor
    if terms.capital_barrier_monitoring!='european':
        raise ValueError('Controlled curve supports only European capital barriers')
    lower=min(float(np.min(eligible_factors))-.08,x0-.1) if len(eligible_factors) else x0-.5
    upper=max(float(np.max(eligible_factors))+.08,x0+.1) if len(eligible_factors) else x0+.5
    nodes=np.unique(np.r_[np.linspace(lower,upper,s.curve_nodes),x0])
    matching=same_contract(p.initial_phoenix,terms)
    if matching:
        bank=quote_bank; baseline=q.paths
        ref=None; gbm=None
    else:
        target=q.coupon_rate if p.acro.target_coupon_rate is None else p.acro.target_coupon_rate
        ref=_reference(p,terms,target,scrambles,ref_paths,int(s.seed)+951007+seed_offset)
        bank=_make_bank(p,terms,s.conditional_paths,steps,int(s.seed)+961007+seed_offset)
        gbm=simulate_phoenix(replace(p.market,dynamics='gbm'),terms,s.conditional_paths,
                             terms.observations_per_year,0,
                             innovations=aggregated_bank(bank,steps,terms.observations_per_year))
        baseline=None
    values_B=[];values_A=[]
    for x in nodes:
        if matching and abs(float(x)-x0)<1e-12:
            B,A=q.principal,q.annuity
        else:
            sv=simulate_phoenix(p.market,terms,bank[0].shape[1],steps,0,
                                initial_factor=float(x),innovations=bank)
            if matching:
                B=q.principal+float(np.mean(sv.principal_pv-baseline.principal_pv))
                A=q.annuity+float(np.mean(sv.coupon_annuity_pv-baseline.coupon_annuity_pv))
            else:
                B=ref.B+float(np.mean(sv.principal_pv-gbm.principal_pv))
                A=ref.A+float(np.mean(sv.coupon_annuity_pv-gbm.coupon_annuity_pv))
        values_B.append(B);values_A.append(A)
    return ConditionalCurve(nodes,np.asarray(values_B),np.asarray(values_A),
                            bank[0].shape[1])


def price_acro_controlled(p: Project, *, ref_paths=4096,scrambles=4,seed_offset=0):
    """Optional full pipeline: CV issue quote + controlled renewal curve + outer pricing."""
    from .pricing import price_acro
    from .engine import simulate_phoenix
    q,bank,diag=controlled_quote(p,ref_paths=ref_paths,scrambles=scrambles,seed_offset=seed_offset)
    s=p.simulation
    # The outer sample must match the one used by price_acro exactly.
    outer=simulate_phoenix(p.market,p.initial_phoenix,s.outer_paths,s.steps_per_year,s.seed+20)
    eligible=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    curve=controlled_curve(p,q,bank,outer.factor_at_call[eligible],
                           ref_paths=ref_paths,scrambles=scrambles,seed_offset=seed_offset)
    result=price_acro(p,quote=q,curve_override=curve)
    diag['conditional_curve_paths_per_node']=curve.paths_per_node
    diag['outer_se_ONLY']=result.outer_se
    diag['no_total_pricing_ci']=True
    return result,diag
