"""Transparent same-model benchmarks. Not a risk-adjusted investor-utility study."""
from dataclasses import replace
import numpy as np
from .pricing import quote_initial, price_acro
from .controlled_pricing import price_acro_controlled


def compare_shortfalls(r, rate):
    remaining=np.zeros(len(r.call_years))
    z=r.eligible
    remaining[z]=np.maximum(r.intrinsic[z]-r.same_price_fixed_payment,0)*np.exp(-rate*r.call_years[z])
    call_only=r.eligible
    fixed_amount=r.same_price_fixed_payment
    return dict(unhedged_mean=float(r.discounted_payoff.mean()),
                unhedged_p95=float(np.quantile(r.discounted_payoff,.95)),
                fixed_mean=float(remaining.mean()),
                fixed_p95=float(np.quantile(remaining,.95)),
                conditional_exercise_probability=float(np.mean(r.intrinsic[call_only]>0)) if call_only.any() else 0.,
                fixed_indemnity_paid_when_unnecessary_probability_Q=float(np.mean(call_only & (r.intrinsic==0))),
                fixed_conditional_overcompensation_mean=float(np.mean(np.maximum(fixed_amount-r.intrinsic[call_only],0))) if call_only.any() else 0.,
                fixed_conditional_uncovered_shortfall_mean=float(np.mean(np.maximum(r.intrinsic[call_only]-fixed_amount,0))) if call_only.any() else 0.,
                acro_contractual_residual=0.0,
                note='Same Q-cost benchmark, not real-world investment utility. Zero ACRO residual is a contractual identity; initial premium and issuer risk remain.')


def no_autocall_coupon(p):
    x=replace(p.initial_phoenix,autocall_enabled=False)
    return quote_initial(replace(p,initial_phoenix=x,initial_coupon_policy='fair'))


def factor_correlation_scenarios(p, initial_coupon, values=(-.85,-.55,-.2,.2),
                                 pricing_method='legacy', ref_paths=1024, scrambles=4,
                                 base_result=None):
    """Sensitivity at a FIXED initial coupon using the same estimator as the main price.

    A base-rho point is always included. A separately estimated fair coupon
    is NOT recomputed after the rho shock; re-pricing a fixed coupon note is
    the intended question. The renewed note keeps its contractual target
    coupon policy, even if its terms differ from the first Phoenix.
    """
    x=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=initial_coupon))
    baseline_rho=float(p.market.equity_factor_correlation)
    points=sorted(set(float(v) for v in values) | {baseline_rho})
    data=[]
    for rho in points:
        scenario=replace(x,market=replace(x.market,equity_factor_correlation=rho))
        if rho == baseline_rho and base_result is not None:
            res=base_result
        elif pricing_method=='gbm_cv':
            res,_=price_acro_controlled(scenario,ref_paths=ref_paths,scrambles=scrambles)
        elif pricing_method=='legacy':
            res=price_acro(scenario)
        else:
            raise ValueError(f'Unsupported sensitivity pricing method: {pricing_method}')
        data.append(dict(rho=rho,premium=res.price,outer_se=res.outer_se,
                         initial_note_mtm=res.quote.price,call_probability=res.call_probability,
                         method=pricing_method))
    return data
