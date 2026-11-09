"""F5 in Spyder or `python run_demo.py --quick`. Edit config.json for parameters."""
from dataclasses import replace, asdict
from pathlib import Path
import argparse, csv, json, shutil, tempfile, uuid
import numpy as np
from acro import load_project, quote_initial, price_acro
from acro.controlled_pricing import price_acro_controlled
from acro.analysis import compare_shortfalls, no_autocall_coupon, factor_correlation_scenarios
from acro.structuring import solve_bundled_par_coupon
from acro.financial_diagnostics import phoenix_risk_diagnostics, draw_risk_decomposition
from plot_results import make_figures
from acro.portfolio_visuals import make_portfolio_figures, payoff_statistics

ROOT=Path(__file__).resolve().parent

def _run_unstaged(config=ROOT/'config.json',output=ROOT/'results',quick=False,sensitivity=True,pricing_method='auto',illustrative_paths=48):
    p=load_project(config)
    if quick:
        p=replace(p,simulation=replace(p.simulation,quote_paths=2500,outer_paths=1800,
                                       conditional_paths=1200,curve_nodes=7))
    out=Path(output); out.mkdir(parents=True,exist_ok=True)
    # Each execution writes to a fresh staging directory; do not delete prior results.
    if pricing_method == 'auto':
        supported = (p.market.dynamics=='stochastic_vol' and
                     p.initial_phoenix.capital_barrier_monitoring=='european' and
                     p.renewal_phoenix.capital_barrier_monitoring=='european')
        pricing_method = 'gbm_cv' if supported else 'legacy'
    if pricing_method=='gbm_cv':
        r, pricing_diag=price_acro_controlled(p,ref_paths=1024 if quick else 8192,
                                               scrambles=4 if quick else 6)
        q=r.quote
    elif pricing_method=='legacy':
        q=quote_initial(p)
        r=price_acro(p,q)
        pricing_diag={}
    else: raise ValueError('Unknown pricing method')
    no_call=no_autocall_coupon(p)
    bundled=solve_bundled_par_coupon(p,r)
    fixed_benchmark=compare_shortfalls(r,p.market.risk_free_rate)
    curve_scenarios=(factor_correlation_scenarios(
        p,q.coupon_rate,pricing_method=pricing_method,
        ref_paths=1024 if quick else 8192,scrambles=4 if quick else 6,
        base_result=r) if sensitivity else [])
    gbm=price_acro(replace(p,market=replace(p.market,dynamics='gbm'),initial_coupon_policy='fair'))
    phoenix_risk=phoenix_risk_diagnostics(q.paths,p.market,p.initial_phoenix,q.coupon_rate)
    metrics={
        'phoenix_risk':phoenix_risk,
        'pricing_method':pricing_method,'pricing_diagnostics':pricing_diag,
        'market_dynamics':p.market.dynamics,'assets':p.market.n_assets,
        'initial_nominal':p.initial_phoenix.nominal,'renewal_nominal':p.renewal_phoenix.nominal,
        'initial_coupon_policy':p.initial_coupon_policy,
        'initial_basket_rule':p.initial_phoenix.basket_rule,
        'renewal_basket_rule':p.renewal_phoenix.basket_rule,
        'annual_coupon':q.coupon_rate,'annual_coupon_se':q.coupon_se,
        'initial_phoenix_price':q.price,'initial_phoenix_price_se':q.price_se,
        'initial_principal_pv':q.principal,'initial_coupon_annuity_pv_per_unit_rate':q.annuity,
        'initial_coupon_leg_pv':q.coupon_rate*q.annuity,
        'acro_price':r.price,'acro_outer_se_ONLY':r.outer_se,
        'bundled_feasible':bundled.feasible,
        'bundled_fair_coupon':bundled.bundled_coupon if bundled.feasible else None,
        'bundled_coupon_reduction_bp':bundled.coupon_haircut_bp if bundled.feasible else None,
        'bundled_option_value':bundled.bundled_acro_premium if bundled.feasible else None,
        'bundled_note_value':bundled.bundled_initial_note_value if bundled.feasible else None,
        'bundled_total_value':bundled.combined_value if bundled.feasible else None,
        'acro_outer_only_95_ci':[max(0,r.price-1.96*r.outer_se),r.price+1.96*r.outer_se],
        'eligible_call_probability_Q':r.call_probability,
        'discounted_call_probability_Q':r.discounted_call_probability,
        'fixed_payment_same_initial_option_price':r.same_price_fixed_payment,
        'no_autocall_fair_coupon':no_call.coupon_rate,
        'gbm_zero_identity_applies':gbm.zero_gbm_identity_applies,
        'gbm_zero_premium_check':gbm.price,
        'shortfall_metrics':fixed_benchmark,'factor_correlation_scenarios':curve_scenarios,
        'n_quote_paths':p.simulation.quote_paths,'n_outer_paths':p.simulation.outer_paths,
        'n_inner_paths_per_node':r.curve.paths_per_node,'n_curve_nodes':len(r.curve.factors),
        **payoff_statistics(r),
    }
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    # Preserve the exact input term sheet for stand-alone plot regeneration.
    (out/'run_config.json').write_text(Path(config).read_text(encoding='utf-8'),encoding='utf-8')
    (out/'effective_config.json').write_text(json.dumps(asdict(p),indent=2,ensure_ascii=False),encoding='utf-8')
    with open(out/'scenario_paths.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['id','first_autocall_year','eligible','basket_reference_at_call','logvol_factor_at_call','fair_coupon','renewal_value','acro_intrinsic','discounted_acro','discounted_fixed','exercise_coupon_threshold'])
        for j in range(len(r.call_years)):
            w.writerow([j,r.call_years[j],int(r.eligible[j]),r.worst_at_call[j],r.logvol_at_call[j],r.fair_renewal_coupon[j],r.renewal_value[j],r.intrinsic[j],r.discounted_payoff[j],r.discounted_fixed[j],r.exercise_coupon_threshold[j]])
    with open(out/'conditional_curve.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['logvol_factor','principal_pv','coupon_annuity_pv','renewal_fair_coupon','renewal_value','exercise_coupon_threshold'])
        for x,B,A in zip(r.curve.factors,r.curve.principal,r.curve.annuity):
            w.writerow([x,B,A,(p.renewal_phoenix.nominal-B)/A,B+r.target_coupon*A,(p.acro.strike-B)/A])
    with open(out/'sensitivity.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['rho','premium','outer_se','initial_note_mtm','call_probability','method']);w.writeheader();w.writerows(curve_scenarios)
    pics=make_figures(out) if bool(r.eligible.any()) else []
    # Financial waterfall uses same raw initial-note paths for all three legs.
    pics.append(draw_risk_decomposition(phoenix_risk, out/'11_risque_capital_et_coupon.png'))
    # Visual-only paths are independent from pricing. Increasing the count
    # changes the 08/09 illustrations and CSV, never the price estimate.
    if illustrative_paths not in (48, 100, 500, 1000):
        raise ValueError('Illustrative paths must be one of 48, 100, 500, 1000.')
    extra_visuals=make_portfolio_figures(p,r,out,count=illustrative_paths)
    metrics.update(extra_visuals)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf-8')
    pics=list(pics)+list(extra_visuals['portfolio_figures'])
    gbm_label = ('Fair GBM identical-contract ACRO premium' if gbm.zero_gbm_identity_applies
                 else 'GBM reference premium (zero identity not applicable)')
    report=f'''# ACRO research — run report

**Scope:** synthetic market, risk-neutral Monte Carlo. Not executable market pricing.

## Term sheet of experiment

| Attribute | Assumption |
|---|---|
| Number of equity underlyings | {p.market.n_assets} |
| Initial / renewal reference rule | {p.initial_phoenix.basket_rule} / {p.renewal_phoenix.basket_rule} |
| Market model | {p.market.dynamics} |
| Initial / renewal maturity | {p.initial_phoenix.maturity_years} / {p.renewal_phoenix.maturity_years} years |
| Initial coupon | {q.coupon_rate:.3%} (policy: {p.initial_coupon_policy}) |
| Coupons and memory | {p.initial_phoenix.observations_per_year}/year; memory {p.initial_phoenix.memory} |
| Coupon barrier | {p.initial_phoenix.coupon_barrier:.0%} |
| Autocall first date, frequency, threshold | {p.initial_phoenix.first_autocall_year}y, every {p.initial_phoenix.autocall_every_years}y, {p.initial_phoenix.autocall_barrier:.0%} |
| Capital barrier | {p.initial_phoenix.capital_barrier:.0%}, {p.initial_phoenix.capital_barrier_monitoring} |
| ACRO strike, units, expiry | {p.acro.strike}, {p.acro.renewal_units}, {p.acro.last_eligible_call_year}y |
| Coupon guaranteed CONTRACTUALLY in renewed note | {r.target_coupon:.3%} (not guaranteed to be paid) |

## Pricing

**Pricing method:** {pricing_method}. GBM-CV intervals remain **partial**, not certified.

| Model result | Estimate |
|---|---:|
| First Phoenix theoretical initial price | {q.price:.4f} |
| Present value of expected principal repayment | {q.principal:.4f} |
| Present value of conditional coupon leg (= coupon rate × annuity) | {q.coupon_rate*q.annuity:.4f} |
| Exotic coupon annuity PV per unit of annual coupon rate | {q.annuity:.4f} |
| ACRO premium per initial contract (initial nominal {p.initial_phoenix.nominal:g} €) | **{r.price:.4f} €** |
| Alternative par-package coupon (model, if feasible) | {bundled.bundled_coupon:.3%} |
| Coupon haircut vs standalone fair coupon | {f'{bundled.coupon_haircut_bp:.1f} bp' if p.initial_coupon_policy=='fair' else 'not comparable: initial coupon contractually fixed'} |
| Bundled note plus right at issue | {bundled.combined_value:.4f} |
| Outer Monte Carlo standard error **only** | {r.outer_se:.4f} |
| Outer-only 95% interval | [{max(0,r.price-1.96*r.outer_se):.4f}, {r.price+1.96*r.outer_se:.4f}] |
| Q probability of eligible autocall | {r.call_probability:.2%} |
| Fixed payment at call with same Q option price | {r.same_price_fixed_payment:.4f} |
| No-autocall note re-quoted fair coupon | {no_call.coupon_rate:.3%} |
| {gbm_label} | {gbm.price:.8f} |

## Q capital risk — initial Phoenix (diagnostic, not investment forecast)

These figures are estimated from the **raw stochastic-volatility issue-quote paths**
({phoenix_risk['sample_paths']:,} scenarios). They do NOT use the GBM control variate and
therefore differ from the controlled `B+cA` figures above by Monte Carlo noise.
Never add raw components to controlled components.

| Q scenario diagnostic | Estimate |
|---|---:|
| Capital loss probability (nominal, not net of coupons) | {phoenix_risk['capital_loss_probability_Q']:.2%} |
| No early autocall | {phoenix_risk['no_autocall_probability_Q']:.2%} |
| Capital loss conditional on no autocall | {phoenix_risk['capital_loss_probability_given_no_autocall_Q'] if phoenix_risk['capital_loss_probability_given_no_autocall_Q'] is not None else float('nan'):.2%} |
| Mean nominal loss in a loss scenario | {phoenix_risk['mean_nominal_loss_if_loss'] if phoenix_risk['mean_nominal_loss_if_loss'] is not None else float('nan'):.2f} EUR |
| PV full face at actual exit date (raw) | {phoenix_risk['pv_redemption_before_loss']:.2f} EUR |
| PV contingent capital-loss leg (raw) | −{phoenix_risk['pv_capital_loss']:.2f} EUR |
| PV coupons on the same raw paths | +{phoenix_risk['pv_conditional_coupons_uncontrolled']:.2f} EUR |
| Note PV from the raw paths | {phoenix_risk['pv_note_uncontrolled']:.2f} EUR |

The raw components sum on the **same paths**; the controlled quote can be different.
The capital-loss leg is *not* a vanilla put quote. Under a European terminal
capital barrier, the loss is `N*(1-W_T)*1[W_T<barrier]*1[no early call]` paid
at maturity. No real-world probability P is inferred.

## Same-premium protection benchmark (model only)

Positive shortfall not covered by the fixed indemnity, discounted to time zero: average **{fixed_benchmark['fixed_mean']:.4f}**, 95th percentile **{fixed_benchmark['fixed_p95']:.4f}**. ACRO has zero *contractual* residual shortfall by payoff construction, **not** a free economic benefit. A fair-priced fixed payment also has the same risk-neutral expected present payout; unlike ACRO it can pay when no shortfall exists.

The no-autocall coupon is only a quote comparison, **not** a complete equal-horizon portfolio or risk utility comparison.

## Numerical and financial caveats

- At 12 steps/year, a separate AI-assisted coupled-grid study (script not included here) reported approximately +0.035 EUR per 100 of initial nominal on the fair-coupon ACRO premium relative to a fine-grid extrapolation in the reference configuration; this is a non-reproduced external-to-project estimate, NOT certified by this run. The grid bias changes with parameters.
- Inner conditional values are approximated by Monte Carlo on a log-vol factor grid, interpolated across autocall states. Outer confidence intervals **exclude conditional sampling error, interpolation error and the upward convexity bias** from applying a positive part to noisy inner values. Do not treat the intervals as total pricing uncertainty.
- GBM zero-value test applies **only** when renewed/initial terms match, coupon is set at par, and reset market parameters (including rates) are constant.
- Stochastic volatility is one common mean-reverting log-volatility factor, correlated with the equity basket. No stochastic correlation, skew, stochastic interest rates, implied-volatility surface calibration, physical risk premiums or issuer/funding/liquidity spreads.
- Continuous maturity barrier monitoring is approximated by checking the discrete simulation time steps.
- The physical option is represented by its frictionless cash-equivalent exercise value. A real obligation to issue another note requires legal terms, settlement arrangements, credit/counterparty exposure, and hedging assumptions not modeled here.
- Fair coupons are outputs of this hypothetical model, not current achievable bank coupons. Exercise preserves **nominal conditional coupon**, not net investor yield after option premium.

## Plots

'''+''.join(f'- `{pic}`\n' for pic in pics)
    (out/'summary.md').write_text(report,encoding='utf-8')
    print('ACRO (synthetic model, NOT market quote)')
    print(f'Initial coupon: {q.coupon_rate:.3%}; ACRO premium: {r.price:.4f} EUR per contract (initial nominal {p.initial_phoenix.nominal:g} EUR); outer-only SE: {r.outer_se:.4f}')
    print(f'P(eligible call): {r.call_probability:.2%}; GBM reference price: {gbm.price:.9f} ({"zero identity applicable" if gbm.zero_gbm_identity_applies else "zero identity not applicable"})')
    if p.initial_coupon_policy == 'fair':
        print(f'ACRO par-bundled fair coupon: {bundled.bundled_coupon:.3%}; coupon reduction: {bundled.coupon_haircut_bp:.1f} bp')
    else:
        print(f'Alternative fair bundled package coupon: {bundled.bundled_coupon:.3%} (NOT comparable to contractually fixed initial coupon)')
    print('Report generated in staged output directory.')
    return metrics

def run(config=ROOT/'config.json', output=ROOT/'results', quick=False,
        sensitivity=True, pricing_method='auto', illustrative_paths=48):
    """Publish a complete result set, preserving previous results on failure.

    First compute everything in a private staging directory on the same disk.
    Only after a successful pricing/plot/report run is the previous directory
    swapped out; if publishing fails, the old results are restored.
    """
    destination=Path(output).resolve()
    destination.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix=f'.{destination.name}.stage-',dir=destination.parent))
    backup=destination.with_name(destination.name+'.backup-'+uuid.uuid4().hex)
    try:
        # Preserve unrelated user files in the output folder, while replacing
        # previous generated data only after the new simulation succeeds.
        if destination.is_dir():
            shutil.copytree(destination,stage,dirs_exist_ok=True)
        for name in ('metrics.json','run_config.json','effective_config.json',
                     'scenario_paths.csv','conditional_curve.csv','illustrative_equity_trajectories.csv','sensitivity.csv',
                     'summary.md'):
            (stage/name).unlink(missing_ok=True)
        for figure in stage.glob('*.png'):
            if figure.name[:2] in ('01','02','03','04','05','06','07','08','09','10'):
                figure.unlink()
        result=_run_unstaged(config=config,output=stage,quick=quick,
                            sensitivity=sensitivity,pricing_method=pricing_method,
                            illustrative_paths=illustrative_paths)
        moved_old=False
        try:
            if destination.exists():
                destination.rename(backup)
                moved_old=True
            stage.rename(destination)
        except Exception:
            if moved_old and backup.exists():
                if destination.exists(): shutil.rmtree(destination)
                backup.rename(destination)
            raise
        if backup.exists(): shutil.rmtree(backup)
        print(f'Published complete simulation results: {destination}')
        return result
    finally:
        if stage.exists(): shutil.rmtree(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--output',default=str(ROOT/'results'))
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--no-sensitivity',action='store_true')
    parser.add_argument('--pricing-method',choices=('legacy','gbm_cv','auto'),default='auto')
    parser.add_argument('--illustrative-paths',type=int,choices=(48,100,500,1000),default=48,
                        help='Independent visualization paths for figure 08; does not alter pricing')
    args,_=parser.parse_known_args()  # Spyder sometimes passes its own arguments
    run(args.config,args.output,quick=args.quick,sensitivity=not args.no_sensitivity,
        pricing_method=args.pricing_method,illustrative_paths=args.illustrative_paths)
