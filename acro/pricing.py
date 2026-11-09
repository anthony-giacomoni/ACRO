"""Issue quote, risk-neutral ACRO premium and at-call renewal values."""
from dataclasses import dataclass, replace
import numpy as np
from .config import Project, Phoenix
from .engine import Paths, simulate_phoenix

@dataclass
class Quote:
    coupon_rate: float
    price: float
    coupon_se: float
    price_se: float
    principal: float
    annuity: float
    paths: Paths

@dataclass
class ConditionalCurve:
    factors: np.ndarray
    principal: np.ndarray
    annuity: np.ndarray
    paths_per_node: int
    def at(self,x):
        if len(x) and (np.min(x)<self.factors[0]-1e-10 or np.max(x)>self.factors[-1]+1e-10):
            raise ValueError('Market factor outside conditional valuation interpolation grid.')
        return np.interp(x,self.factors,self.principal),np.interp(x,self.factors,self.annuity)

@dataclass
class ACROResult:
    quote: Quote
    target_coupon: float
    price: float
    outer_se: float
    call_probability: float
    discounted_call_probability: float
    same_price_fixed_payment: float
    call_years: np.ndarray
    eligible: np.ndarray
    fair_renewal_coupon: np.ndarray  # fair PAR coupon, independent of strike
    renewal_value: np.ndarray
    intrinsic: np.ndarray
    discounted_payoff: np.ndarray
    discounted_fixed: np.ndarray
    curve: ConditionalCurve
    zero_gbm_identity_applies: bool
    logvol_at_call: np.ndarray
    worst_at_call: np.ndarray
    exercise_coupon_threshold: np.ndarray  # (strike - B)/A, distinct from par coupon


def quote_initial(p:Project) -> Quote:
    s=p.simulation
    paths=simulate_phoenix(p.market,p.initial_phoenix,s.quote_paths,s.steps_per_year,s.seed+10)
    B=paths.principal_pv.mean(); A=paths.coupon_annuity_pv.mean()
    if A<=0: raise ValueError('Zero annuity: fair coupon cannot be determined.')
    if p.initial_coupon_policy=='fair':
        c=(p.initial_phoenix.nominal-B)/A
        if c<0: raise ValueError('Negative par coupon: change contract or use fixed policy.')
    else: c=p.initial_phoenix.coupon_rate
    price=B+c*A
    se=paths.pv(c).std(ddof=1)/np.sqrt(s.quote_paths)
    return Quote(c,price,se/A,se,B,A,paths)


def same_contract(a:Phoenix,b:Phoenix):
    return replace(a,coupon_rate=0)==replace(b,coupon_rate=0)


def conditional_curve(p:Project,eligible_factors,quote:Quote,seed_offset:int=0):
    s=p.simulation; market=p.market; renew=p.renewal_phoenix
    same=same_contract(p.initial_phoenix,renew)
    if market.dynamics=='gbm':
        if same: B,A=quote.principal,quote.annuity
        else:
            sm=simulate_phoenix(market,renew,s.conditional_paths,s.steps_per_year,s.seed+100+seed_offset)
            B,A=sm.principal_pv.mean(),sm.coupon_annuity_pv.mean()
        return ConditionalCurve(np.array([-1.,1.]),np.array([B,B]),np.array([A,A]),s.quote_paths if same else s.conditional_paths)
    mid=market.initial_logvol_factor
    lower=min(float(np.min(eligible_factors))-.04,mid-.05) if len(eligible_factors) else mid-.5
    upper=max(float(np.max(eligible_factors))+.04,mid+.05) if len(eligible_factors) else mid+.5
    nodes=np.unique(np.r_[np.linspace(lower,upper,s.curve_nodes),mid])
    # Identical terms: all grid nodes use *same* MC trajectories as the issue quote,
    # avoiding artificial renewal-value kinks and spurious optionality near t=0.
    n=s.quote_paths if same else s.conditional_paths
    seed=(s.seed+10 if same else s.seed+100)+seed_offset
    # The SAME-contract renewal is quoted relative to a common-risk-factor
    # baseline using identical per-path innovations (control variate). This
    # protects the positive-part payoff against a noisy unanchored inner quote.
    # It is essential even when seed_offset != 0: simply replacing the midpoint
    # by the issue quote while using independent node seeds produces artifacts.
    baseline=None
    if same:
        if seed_offset==0:
            baseline=quote.paths  # exact CRN baseline, same seed and number of paths
        else:
            baseline=simulate_phoenix(market,renew,n,s.steps_per_year,seed,
                                      initial_factor=float(mid))
    base=[]; ann=[]
    for x in nodes:
        if same and abs(x-mid)<1e-12:
            B,A=quote.principal,quote.annuity
        else:
            sim=simulate_phoenix(market,renew,n,s.steps_per_year,seed,
                                 initial_factor=float(x))
            if same:
                B=quote.principal + float(np.mean(sim.principal_pv-baseline.principal_pv))
                A=quote.annuity + float(np.mean(sim.coupon_annuity_pv-baseline.coupon_annuity_pv))
            else:
                B,A=sim.principal_pv.mean(),sim.coupon_annuity_pv.mean()
        base.append(B); ann.append(A)
    return ConditionalCurve(nodes,np.array(base),np.array(ann),n)


def price_acro(p:Project,quote:Quote|None=None,curve_override:ConditionalCurve|None=None) -> ACROResult:
    p.validate(); s=p.simulation; m=p.market; a=p.acro
    q=quote or quote_initial(p)
    c=q.coupon_rate if a.target_coupon_rate is None else a.target_coupon_rate
    outer=simulate_phoenix(m,p.initial_phoenix,s.outer_paths,s.steps_per_year,s.seed+20)
    years=outer.first_call_year
    eligible=np.isfinite(years)
    if a.last_eligible_call_year is not None:
        eligible &= years<=a.last_eligible_call_year+1e-10
    curve=curve_override if curve_override is not None else conditional_curve(p,outer.factor_at_call[eligible],q)
    fair=np.full(s.outer_paths,np.nan); exercise_threshold=np.full(s.outer_paths,np.nan); value=np.full(s.outer_paths,np.nan)
    intrinsic=np.zeros(s.outer_paths); discounted=np.zeros(s.outer_paths)
    fixed_weight=np.zeros(s.outer_paths)
    if eligible.any():
        B,A=curve.at(outer.factor_at_call[eligible])
        value[eligible]=B+c*A
        fair[eligible]=np.where(A>1e-10,(p.renewal_phoenix.nominal-B)/A,np.nan)
        exercise_threshold[eligible]=np.where(A>1e-10,(a.strike-B)/A,np.nan)
        intrinsic[eligible]=a.renewal_units*np.maximum(value[eligible]-a.strike,0.)
        factor=np.exp(-m.risk_free_rate*years[eligible])
        discounted[eligible]=factor*intrinsic[eligible]
        fixed_weight[eligible]=factor
    premium=float(discounted.mean())
    prob=float(fixed_weight.mean())
    fixed=premium/prob if prob else 0.
    identity=(m.dynamics=='gbm' and p.initial_coupon_policy=='fair' and
              same_contract(p.initial_phoenix,p.renewal_phoenix) and
              a.target_coupon_rate is None and a.strike==p.initial_phoenix.nominal)
    return ACROResult(q,c,premium,float(discounted.std(ddof=1)/np.sqrt(s.outer_paths)),
        float(eligible.mean()),prob,fixed,years,eligible,fair,value,intrinsic,
        discounted,fixed_weight*fixed,curve,identity,outer.factor_at_call,outer.first_call_worst,exercise_threshold)
