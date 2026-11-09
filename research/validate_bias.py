"""ACRO numerical pricing diagnostics — no market calibration.

Use `python validate_bias.py --quick` (or full run). Four controls:
- independent GBM fair-coupon/renewal repricing, including uncertainty;
- paired independent inner curves, on identical outer scenarios;
- convexity noise diagnostic: mean(max(V1-K,0),max(V2-K,0)) vs max(mean(V1,V2)-K,0);
- selected high-precision independent continuation checks (see validate_nested.py).

The paired curve gap is NOT an unbiased estimate of total inner Monte Carlo bias.
Quote uncertainty, outer Monte Carlo and interpolation still matter. No market price claimed.
"""
from pathlib import Path
from dataclasses import replace
import argparse, csv, json
import numpy as np
from acro import load_project, quote_initial, price_acro, simulate_phoenix
from acro.pricing import conditional_curve, ConditionalCurve

ROOT=Path(__file__).resolve().parents[1]


def independent_gbm_control(p, n):
    gbm=replace(p,market=replace(p.market,dynamics='gbm'),
                initial_coupon_policy='fair')
    q=quote_initial(gbm)
    s=gbm.simulation
    fresh=simulate_phoenix(gbm.market,gbm.renewal_phoenix,n,
                           s.steps_per_year,s.seed+410117)
    values=fresh.pv(q.coupon_rate)
    renewal_val=float(values.mean())
    renewal_se=float(values.std(ddof=1)/np.sqrt(n))
    # For equal initial/renewal terms and quote set to par, estimate sampling
    # variance from independent issue coupon estimation (delta method).
    quote_induced_se=float(abs(fresh.coupon_annuity_pv.mean())*q.coupon_se)
    total_se=float(np.hypot(renewal_se,quote_induced_se))
    ident=(gbm.initial_phoenix==gbm.renewal_phoenix and
           gbm.acro.strike==gbm.initial_phoenix.nominal and
           gbm.acro.target_coupon_rate is None)
    return dict(theoretical_zero_identity_applies=ident,
                independent_renewal_price=renewal_val,
                independent_renewal_minus_par=renewal_val-gbm.initial_phoenix.nominal,
                independent_renewal_se=renewal_se,
                estimated_issue_coupon_uncertainty_se=quote_induced_se,
                combined_difference_se_approx=total_se,
                z_score_approx=(renewal_val-gbm.initial_phoenix.nominal)/total_se if total_se else None,
                hypothetical_spurious_exercise=max(renewal_val-gbm.acro.strike,0))


def inner_pair_diagnostic(p):
    s=p.simulation
    q=quote_initial(p)
    outer=simulate_phoenix(p.market,p.initial_phoenix,s.outer_paths,
                            s.steps_per_year,s.seed+20)
    eligible=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    factors=outer.factor_at_call[eligible]
    curve1=conditional_curve(p,factors,q,seed_offset=0)
    curve2=conditional_curve(p,factors,q,seed_offset=800013)
    # Both curves use the same grid, same coupon/quote, and same outer states.
    np.testing.assert_allclose(curve1.factors,curve2.factors,rtol=0,atol=0)
    avgcurve=ConditionalCurve(curve1.factors,
                              .5*(curve1.principal+curve2.principal),
                              .5*(curve1.annuity+curve2.annuity),
                              curve1.paths_per_node+curve2.paths_per_node)
    r1=price_acro(p,q,curve_override=curve1)
    r2=price_acro(p,q,curve_override=curve2)
    ra=price_acro(p,q,curve_override=avgcurve)
    convexity_gap=.5*(r1.price+r2.price)-ra.price
    # This gap compares inner N vs 2N, not N vs infinite inner resolution.
    return dict(price_inner_seed_1=r1.price,price_inner_seed_2=r2.price,
                price_doubled_inner_average=ra.price,
                convexity_gap_N_vs_2N=float(convexity_gap),
                inner_seed_spread_abs=float(abs(r1.price-r2.price)),
                outer_se_with_doubled_inner=ra.outer_se,
                quote_coupon=q.coupon_rate,
                quote_coupon_se=q.coupon_se,
                grid_nodes=len(avgcurve.factors),
                inner_paths_per_curve_node=curve1.paths_per_node,
                outer_paths=s.outer_paths)



def direct_nested_spotcheck(p, n_selected=120, n_per_half=1000):
    """Direct nested MC at random eligible call states, no interpolation.

    This is a low-budget *spot check*, not a high-precision benchmark for P0.
    Small vs pooled inner estimates reveal conditional Jensen bias in-sample.
    """
    s=p.simulation
    q=quote_initial(p)
    r=price_acro(p,q)
    outer=simulate_phoenix(p.market,p.initial_phoenix,s.outer_paths,
                            s.steps_per_year,s.seed+20)
    eligible=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    ids=np.flatnonzero(eligible)
    if len(ids)==0:
        return dict(eligible_count=0)
    k=min(n_selected,len(ids))
    rng=np.random.default_rng(s.seed+99151)
    chosen=rng.choice(ids,size=k,replace=False)
    c=r.target_coupon; K=p.acro.strike; units=p.acro.renewal_units
    curveB,curveA=r.curve.at(outer.factor_at_call[chosen])
    curve_payoff=units*np.maximum(curveB+c*curveA-K,0.)
    left=np.empty(k);right=np.empty(k)
    for j,i in enumerate(chosen):
        state=float(outer.factor_at_call[i])
        seed=s.seed+300000+11*j
        path1=simulate_phoenix(p.market,p.renewal_phoenix,n_per_half,
                              s.steps_per_year,seed,initial_factor=state)
        path2=simulate_phoenix(p.market,p.renewal_phoenix,n_per_half,
                              s.steps_per_year,seed+1,initial_factor=state)
        left[j]=path1.pv(c).mean()
        right[j]=path2.pv(c).mean()
    weight=np.exp(-p.market.risk_free_rate*outer.first_call_year[chosen])
    indiv=.5*(np.maximum(left-K,0.)+np.maximum(right-K,0.))*units
    pooled=np.maximum(.5*(left+right)-K,0.)*units
    multiplier=float(eligible.mean())
    px_small=multiplier*np.mean(weight*indiv)
    px_pooled=multiplier*np.mean(weight*pooled)
    px_curve=multiplier*np.mean(weight*curve_payoff)
    sample_se=multiplier*float(np.std(weight*pooled,ddof=1)/np.sqrt(k)) if k>1 else None
    return dict(selected_eligible_outer_states=int(k),
                independent_inner_paths_per_half=int(n_per_half),
                total_outer_paths=int(s.outer_paths),
                eligible_probability_outer=float(eligible.mean()),
                estimated_direct_price_individual_inner=float(px_small),
                estimated_direct_price_pooled_inner=float(px_pooled),
                estimated_interpolation_price_same_states=float(px_curve),
                pooled_minus_interpolation=float(px_pooled-px_curve),
                positive_part_gap_individual_minus_pooled=float(px_small-px_pooled),
                outer_state_sampling_se_ONLY=sample_se,
                full_curve_price=float(r.price),
                caution='No full MC CI; finite eligible-state sample, quote and inner noise remain. Gap is N versus 2N only.')

def main(config, quick, output):
    p=load_project(config)
    if quick:
        p=replace(p,simulation=replace(p.simulation,quote_paths=1800,
                  outer_paths=1100,conditional_paths=800,curve_nodes=7))
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    g=independent_gbm_control(p,n=2500 if quick else max(12000,p.simulation.quote_paths))
    d=inner_pair_diagnostic(p)
    v=direct_nested_spotcheck(p,n_selected=35 if quick else 140,
                              n_per_half=350 if quick else 1400)
    obj=dict(independent_gbm=g,inner_pair=d,direct_nested=v,notes=[
        'The GBM exact zero is an algebraic model identity; the renewal price here is a statistically independent MC estimate.',
        'The independent GBM renewal confidence check is illustrative; estimated coupon uncertainty uses a delta-method approximation.',
        'The paired inner curve convexity gap measures the effect of inner noise between N and 2N on the same outer states; it is NOT total bias and ignores interpolation error.',
        'The direct nested check simulates fresh renewal notes for a subset of eligible outer states; it compares two independent inner batches vs pooled estimates and model interpolation. Its small outer subset and finite inner sizes make it diagnostic, not a robust independent fair-price estimate.',
        'Quote, outer and inner errors and time-grid/model bias are not fully captured; no desk-grade total pricing confidence interval is reported.'
    ])
    (out/'pricing_bias_diagnostic.json').write_text(json.dumps(obj,indent=2),encoding='utf-8')
    rows=[dict(section='independent_gbm',metric=k,value=v) for k,v in g.items()]
    rows += [dict(section='inner_pair',metric=k,value=v) for k,v in d.items()]
    rows += [dict(section='direct_nested',metric=k,value=value) for k,value in v.items()]
    with open(out/'pricing_bias_diagnostic.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['section','metric','value']);w.writeheader();w.writerows(rows)
    print('Independent GBM renewal valuation minus 100:',round(g['independent_renewal_minus_par'],6),
          'approx z:',round(g['z_score_approx'],2) if g['z_score_approx'] is not None else 'NA')
    print('Inner N vs 2N convexity gap (NOT total bias):',round(d['convexity_gap_N_vs_2N'],6))
    print('Independent inner seed prices:',round(d['price_inner_seed_1'],5),
          round(d['price_inner_seed_2'],5),'averaged price:',round(d['price_doubled_inner_average'],5))
    print('Direct nested pooled vs interpolation (sample of call states):',
          round(v.get('estimated_direct_price_pooled_inner',0),5),
          round(v.get('estimated_interpolation_price_same_states',0),5))
    print('Direct nested inner Jensen gap N vs 2N:',round(v.get('positive_part_gap_individual_minus_pooled',0),6))
    print('Saved:',out/'pricing_bias_diagnostic.json')
    return obj

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--output',default=str(ROOT/'results'))
    a=parser.parse_args()
    main(a.config,a.quick,a.output)
