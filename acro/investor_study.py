"""V5.16 — Investor diagnostics. All pricing is under Q; P only changes path dynamics.

No P parameters are market-estimated. This is a scenario study. Credit stress
models *only* the failure of the option renewal promise, not issuer default on
Phoenix principal/coupons. It is deliberately NOT a bank credit-pricing model.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import t as student_t


@dataclass(frozen=True)
class InvestorScenario:
    name: str
    excess_equity_drift: float
    p_factor_mean: float
    option_delivery_failure_probability: float = 0.0
    # A financial commission on each new note purchase and an additional fee
    # only when the ACRO contractual right is exercised.
    renewal_fee_bps: float = 0.0
    acro_exercise_fee_bps: float = 0.0

    def validate(self):
        if not np.isfinite(self.excess_equity_drift) or not np.isfinite(self.p_factor_mean):
            raise ValueError('Scenario parameters must be finite')
        if not 0 <= self.option_delivery_failure_probability <= 1:
            raise ValueError('Failure probability must lie in [0,1]')
        if self.renewal_fee_bps < 0 or self.acro_exercise_fee_bps < 0:
            raise ValueError('Fees must not be negative')


def interval_of_mean(values, confidence=.95):
    x=np.asarray(values,float)
    if x.ndim!=1 or len(x)<2 or not np.all(np.isfinite(x)):
        raise ValueError('At least two independent finite estimates required')
    se=float(x.std(ddof=1)/np.sqrt(len(x)))
    half=float(student_t.ppf((1+confidence)/2,df=len(x)-1)*se)
    return float(x.mean()),se,(float(x.mean()-half),float(x.mean()+half))


def discounted_acro_cost_and_call(rec, rate):
    """ACRO theoretical at-call payoff, not realised coupon or final wealth."""
    z=np.asarray(rec['eligible'],bool)
    times=np.asarray(rec['call_year'],float)
    discount=np.exp(-rate*times)
    return np.where(z,np.asarray(rec['cost'])*discount,0.),np.where(z,discount,0.)


def calibrate_cash_fixed_on_independent_q(cost_pv,call_discount):
    """Repricing a paired fair-value indemnity using the *same* holdout bank."""
    a=np.asarray(cost_pv,float);d=np.asarray(call_discount,float)
    if a.shape!=d.shape or a.ndim!=1: raise ValueError('Unequal bank arrays')
    p=float(a.mean()); q=float(d.mean())
    if p<0 or q<=0: raise ValueError('Invalid Q option premium / call probability')
    return dict(premium=p,discounted_call_probability=q,fixed_amount=p/q)


def paired_present_value_diff(rec,*,rate):
    """Terminal wealth already contains cash reinvestment to common H."""
    H=rec['horizon'];W=rec['wealth']
    return (np.asarray(W['acro'])-np.asarray(W['fixed']))*np.exp(-rate*H)


def apply_investor_frictions(rec, *, rate, premium, scenario:InvestorScenario,
                             original_nominal, seed):
    """Explicit contract-specific, one-sided delivery stress (not full issuer credit).

    Returns modified terminal wealth and eligible-event option coverage. We model
    settlement failures occurring only at eligible calls (other note defaults
    are NOT simulated); the failed investor buys a fair-market renewal note.
    Fees are carried forward at the same risk-free rate as other cash flows.
    """
    scenario.validate()
    H=float(rec['horizon']); year=np.asarray(rec['call_year'],float)
    called=np.asarray(rec['called'],bool); eligible=np.asarray(rec['eligible'],bool)
    exercise=np.asarray(rec['exercise'],bool)
    rng=np.random.default_rng(seed)
    fails=eligible & (rng.random(len(year))<scenario.option_delivery_failure_probability)
    wealth={k:np.asarray(v,float).copy() for k,v in rec['wealth'].items()}
    # The baseline fair-coupon note, excluding the premium cash reserve.
    fair_renewal_wealth=wealth['reserve']-premium*np.exp(rate*H)
    extra=np.maximum(0.,wealth['acro']-fair_renewal_wealth)
    wealth['acro']-=extra*fails
    # Fee for subscribing to any new note after an autocall, regardless of
    # protection type (no-acro and fixed also buy the market coupon note).
    fee=original_nominal*scenario.renewal_fee_bps/1e4
    fees_at_H=np.where(called,fee*np.exp(rate*(H-year)),0.)
    for k in ['acro','fixed','reserve']:
        wealth[k]-=fees_at_H
    # Non-call renewal fee scales with actual maturity principal recovered.
    fractions=np.asarray(rec['no_call_renewal_fraction'],float)
    no_call_time=float(rec['no_call_renewal_time'])
    wealth['no_call']-=fee*fractions*np.exp(rate*(H-no_call_time))
    exercise_fee=original_nominal*scenario.acro_exercise_fee_bps/1e4
    wealth['acro']-=np.where(exercise & ~fails,exercise_fee*np.exp(rate*(H-year)),0.)
    cost=np.asarray(rec['cost'],float)
    acro_shortfall=np.where(fails,cost,0.)
    return dict(wealth=wealth,delivery_failed=fails,
                acro_shortfall=acro_shortfall,
                fixed_shortfall=np.asarray(rec['shortfall']['fixed'],float),
                covered_when_exercised=exercise & ~fails)


def wealth_metrics(wealth, *, budget, rate, horizon, gamma=2.):
    """Expected utility of positive terminal wealth under assumed P scenario."""
    w=np.asarray(wealth,float)
    if w.ndim!=1 or not len(w) or not np.all(np.isfinite(w)) or np.any(w<=0):
        raise ValueError('CRRA utility requires finite strictly positive wealth')
    if gamma<0:raise ValueError('Risk aversion must be nonnegative')
    # Scaling improves numerical stability for gamma != 1.
    scale=budget*np.exp(rate*horizon)
    if np.isclose(gamma,1): ce=float(scale*np.exp(np.mean(np.log(w/scale))))
    else:ce=float(scale*np.mean((w/scale)**(1-gamma))**(1/(1-gamma)))
    tail=float(np.quantile(w,.05))
    return dict(mean=float(w.mean()),median=float(np.median(w)),p05=tail,
                es05=float(np.mean(w[w<=tail])),
                p_below_start_growth=float(np.mean(w<scale)),
                crra_ce=ce)


def compare_investor_outcomes(rec, overlay, *, rate, gamma=2.):
    W=overlay['wealth'];H=rec['horizon'];budget=rec['budget']
    z={k:wealth_metrics(v,budget=budget,rate=rate,horizon=H,gamma=gamma)
       for k,v in W.items()}
    called=np.asarray(rec['eligible'],bool)
    d=np.asarray(W['acro'])-np.asarray(W['fixed'])
    def conditional_mean(arr):
        return float(np.mean(arr[called])) if np.any(called) else None
    return dict(strategies=z,eligible_calls=int(called.sum()),n=len(called),
        mean_paired_acro_minus_fixed=float(d.mean()),
        p_acro_above_fixed_given_call=conditional_mean((d>1e-9).astype(float)),
        p_acro_below_fixed_given_call=conditional_mean((d<-1e-9).astype(float)),
        mean_acro_minus_fixed_given_call=conditional_mean(d),
        p95_shortfall_fixed_given_call=(float(np.quantile(overlay['fixed_shortfall'][called],.95))
                                         if np.any(called) else None),
        mean_shortfall_acro_given_call=conditional_mean(overlay['acro_shortfall']),
        mean_shortfall_fixed_given_call=conditional_mean(overlay['fixed_shortfall']),
        p_failure_given_call=conditional_mean(overlay['delivery_failed'].astype(float)))


def apply_common_contingent_settlement_stress(rec, *, rate, premium,
                                              scenario:InvestorScenario,
                                              original_nominal, seed):
    """Limited *symmetric* settlement stress, NOT full issuer default.

    At a first eligible call, a common Bernoulli event prevents delivery of
    either contractual promise: the ACRO enhanced renewal OR the fixed
    indemnity. The initial Phoenix, other coupons and any market-renewal
    obligations are deliberately assumed to perform.

    The two contingent claims use the same realised default event, so this is
    a fairer *conditional-settlement* comparison than ACRO-only failure. It
    is NOT a calibrated bank credit / CVA model.
    """
    from dataclasses import replace
    scenario.validate()
    base=replace(scenario,option_delivery_failure_probability=0.)
    overlay=apply_investor_frictions(rec,rate=rate,premium=premium,
        scenario=base,original_nominal=original_nominal,seed=seed)
    H=float(rec['horizon']);year=np.asarray(rec['call_year'],float)
    eligible=np.asarray(rec['eligible'],bool)
    exercise=np.asarray(rec['exercise'],bool)
    failed=eligible & (np.random.default_rng(seed).random(len(eligible)) <
                       scenario.option_delivery_failure_probability)
    wealth=overlay['wealth']
    fair_renewal=np.asarray(rec['wealth']['reserve'])-premium*np.exp(rate*H)
    gain=np.maximum(0.,np.asarray(rec['wealth']['acro'])-fair_renewal)
    wealth['acro']-=np.where(failed,gain,0.)
    # Exercise fees are charged only if successful exercise takes place.
    fee=original_nominal*scenario.acro_exercise_fee_bps/1e4
    wealth['acro']+=np.where(failed & exercise,fee*np.exp(rate*(H-year)),0.)
    fixed_amount=float(rec.get('fixed_amount',np.nan))
    if not np.isfinite(fixed_amount):
        raise ValueError('A contractual fixed_amount must be given in rec')
    wealth['fixed']-=np.where(failed,fixed_amount*np.exp(rate*(H-year)),0.)
    # Shortfall is the contractual cost unfinanced by each commitment.
    cost=np.asarray(rec['cost'],float)
    shortfall_fixed=np.where(eligible,np.maximum(cost-np.where(failed,0.,fixed_amount),0.),0.)
    return dict(wealth=wealth, delivery_failed=failed,fixed_payment_failed=failed,
        acro_shortfall=np.where(failed,cost,0.),
        fixed_shortfall=shortfall_fixed,
        covered_when_exercised=exercise & ~failed,
        credit_model='shared_first_eligible_call_settlement_failure_only')


def investor_target_metrics(wealth, *, budget, rate, horizon, target_annual_growth=0.02):
    """Distribution diagnostics under the chosen hypothetical P measure.

    Target and expected shortfall use terminal wealth *discounted* to t=0.
    A missing target indicates a preference/goal, not guaranteed coupon income.
    """
    w=np.asarray(wealth,dtype=float)
    if w.ndim!=1 or not len(w) or not np.all(np.isfinite(w)):
        raise ValueError('Expected one-dimensional finite wealth samples')
    if budget<=0 or horizon<=0 or target_annual_growth<=-1:
        raise ValueError('Invalid target, budget or horizon')
    pv=w*np.exp(-rate*horizon)
    target=budget*(1+target_annual_growth)**horizon*np.exp(-rate*horizon)
    q05=float(np.quantile(pv,.05))
    return dict(mean_pv=float(pv.mean()),median_pv=float(np.median(pv)),
        es05_pv=float(np.mean(pv[pv<=q05])),p05_pv=q05,
        target_pv=float(target),
        p_missing_target=float(np.mean(pv<target)),
        mean_target_shortfall_pv=float(np.mean(np.maximum(target-pv,0.))))
