"""Independent-seed numerical audit for ACRO research pricing.

This script does NOT calibrate the model or eliminate positive-part nested bias.
It reports sensitivity to MC budget/seeds, a nested Jensen diagnostic and
independent renewed-Phoenix repricing at selected factor states.

Usage:
  python research_validation.py --quick
  python research_validation.py           # more demanding
  python research_validation.py --config examples/four_assets_memory.json --quick
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import argparse, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from acro import load_project, quote_initial, price_acro, simulate_phoenix
from acro.pricing import conditional_curve, ConditionalCurve
from acro.structuring import solve_bundled_par_coupon

ROOT=Path(__file__).resolve().parents[1]


def _row(p, budget, repetition, seed):
    s=replace(p.simulation,quote_paths=budget['quote_paths'],
              outer_paths=budget['outer_paths'],conditional_paths=budget['conditional_paths'],
              curve_nodes=budget['curve_nodes'],seed=seed)
    p2=replace(p,simulation=s)
    q=quote_initial(p2)
    r=price_acro(p2,q)
    b=solve_bundled_par_coupon(p2,r)
    return dict(level=budget['name'],seed=seed,replicate=repetition,
                quote_paths=s.quote_paths,outer_paths=s.outer_paths,
                nodes=len(r.curve.factors),inner_paths_per_node=r.curve.paths_per_node,
                initial_coupon=q.coupon_rate,quote_coupon_se=q.coupon_se,
                acro_premium=r.price,outer_se_only=r.outer_se,
                bundle_coupon=b.bundled_coupon if b.feasible else float('nan'),
                coupon_haircut_bp=b.coupon_haircut_bp if b.feasible else float('nan'),
                call_probability=r.call_probability)


def independent_state_checks(p, num_states, paths_per_state):
    """Independent inner valuation checks, with uncertainty on DIFFERENCE.

    Checks selected conditional VALUE states, NOT global nested option price.
    """
    q=quote_initial(p)
    r=price_acro(p,q)
    outer=simulate_phoenix(p.market,p.initial_phoenix,p.simulation.outer_paths,
                           p.simulation.steps_per_year,p.simulation.seed+20)
    eligible=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    if not eligible.any():
        return []
    factors=np.quantile(outer.factor_at_call[eligible],np.linspace(.10,.90,num_states))
    rows=[]
    for j,x in enumerate(factors):
        B,A=r.curve.at(np.array([x]))
        interp_val=float(B[0]+r.target_coupon*A[0])
        inner=simulate_phoenix(p.market,p.renewal_phoenix,paths_per_state,
                               p.simulation.steps_per_year,p.simulation.seed+1000123+1000*j,
                               initial_factor=float(x))
        vals=inner.pv(r.target_coupon)
        direct=float(vals.mean())
        se=float(vals.std(ddof=1)/np.sqrt(len(vals)))
        rows.append(dict(state_quantile=float((j+1)/(num_states+1)),factor=float(x),
                         interpolated_value=interp_val,direct_value=direct,direct_mc_se=se,
                         difference=interp_val-direct,
                         direct_minus_strike=direct-p.acro.strike,
                         interpolated_minus_strike=interp_val-p.acro.strike,
                         difference_over_direct_se=(interp_val-direct)/se if se else None))
    return rows


def jensen_check(p):
    """Paired independent inner curves on same quote/outer states.

    Half-sample convexity gap is >= 0 algebraically. This is NOT a fully
    unbiased price bound or reliable estimate of total inner MC bias.
    """
    q=quote_initial(p)
    s=p.simulation
    outer=simulate_phoenix(p.market,p.initial_phoenix,s.outer_paths,s.steps_per_year,s.seed+20)
    eligible=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    f=outer.factor_at_call[eligible]
    c1=conditional_curve(p,f,q,seed_offset=0)
    c2=conditional_curve(p,f,q,seed_offset=1357911)
    avg=ConditionalCurve(c1.factors,.5*(c1.principal+c2.principal),
                         .5*(c1.annuity+c2.annuity),c1.paths_per_node+c2.paths_per_node)
    v1=price_acro(p,q,curve_override=c1).price
    v2=price_acro(p,q,curve_override=c2).price
    pool=price_acro(p,q,curve_override=avg).price
    return dict(first_inner_curve=v1,second_inner_curve=v2,
                pooled_curve_price=pool,
                finite_sample_jensen_gap=(v1+v2)/2-pool,
                inner_curve_spread=abs(v1-v2),
                interpretation='N vs 2N Jensen effect, not bound on true premium')


def visualize(rows,output):
    fig,ax=plt.subplots(figsize=(9.5,5.5))
    budgets=list(dict.fromkeys(x['level'] for x in rows))
    for i,name in enumerate(budgets):
        z=[r for r in rows if r['level']==name]
        ax.scatter(np.full(len(z),i,dtype=float)+np.linspace(-.08,.08,len(z)),
                   [r['acro_premium'] for r in z],s=54,label=name)
        mean=float(np.mean([r['acro_premium'] for r in z]))
        ax.hlines(mean,i-.17,i+.17,linewidth=2)
        ax.annotate(f'Moy. {mean:.3f}',(i+.19,mean),fontsize=10,va='center')
    ax.set_xticks(range(len(budgets)),budgets)
    ax.set_xlim(-.35,len(budgets)-.35)  # keep mean labels inside the image
    ax.set_ylabel('Prime théorique ACRO (EUR par 100)')
    ax.set_title('Stabilité de la prime — graines indépendantes et budgets de calcul')
    ax.grid(axis='y',alpha=.2)
    fig.text(.1,.015,'Chaque point = simulation indépendante (coupon, scénarios, courbe conditionnelle). Ne prouve pas l’absence de biais.',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output/'stabilite_prime_ACRO.png',dpi=170)
    plt.close(fig)


def run(config,output,quick=False):
    p=load_project(config)
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    if quick:
        levels=[dict(name='Rapide',quote_paths=5000,outer_paths=3000,conditional_paths=5000,curve_nodes=9),
                dict(name='Renforce',quote_paths=12000,outer_paths=6500,conditional_paths=12000,curve_nodes=13)]
        seeds=[p.simulation.seed+i*200019 for i in range(3)]
    else:
        levels=[dict(name='Base',quote_paths=12000,outer_paths=6500,conditional_paths=12000,curve_nodes=13),
                dict(name='Haute precision',quote_paths=24000,outer_paths=10000,conditional_paths=24000,curve_nodes=15)]
        seeds=[p.simulation.seed+i*200019 for i in range(3)]
    rows=[]
    for budget in levels:
        for k,seed in enumerate(seeds):
            r=_row(p,budget,k,seed)
            rows.append(r)
            print(f"[{r['level']}, seed {k+1}/{len(seeds)}] prime {r['acro_premium']:.4f}, coupon {r['initial_coupon']:.3%}, bundle {r['bundle_coupon']:.3%}",flush=True)
    with open(output/'replications.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={}
    for b in levels:
        v=np.array([r['acro_premium'] for r in rows if r['level']==b['name']])
        cp=np.array([r['initial_coupon'] for r in rows if r['level']==b['name']])
        hp=np.array([r['coupon_haircut_bp'] for r in rows if r['level']==b['name']])
        summary[b['name']]={
            'n_independent_runs':len(v),'mean_price':float(v.mean()),
            'run_to_run_sample_sd':float(v.std(ddof=1)),
            'min_price':float(v.min()),'max_price':float(v.max()),
            'mean_initial_coupon':float(cp.mean()),
            'mean_bundled_coupon_haircut_bp':float(hp.mean()),
            'mean_price_95_t_interval_across_seeds':(
                float(v.mean()-stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))),
                float(v.mean()+stats.t.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v))))}
    # High-fidelity model point checks and Jensen diagnostic use a *new* seed.
    best=levels[-1]
    verification=replace(p,simulation=replace(p.simulation,
                          quote_paths=best['quote_paths'],outer_paths=best['outer_paths'],
                          conditional_paths=best['conditional_paths'],curve_nodes=best['curve_nodes'],
                          seed=p.simulation.seed+988877))
    state_checks=independent_state_checks(verification,5,8000 if quick else 14000)
    print("Independent state checks completed",flush=True)
    jensen=jensen_check(verification)
    print("Paired inner-curve Jensen checks completed",flush=True)
    # Time-step convergence is a separate bias diagnostic. Seeds are matched
    # numerically, but the discretizations do NOT share identical Brownian
    # paths; differences include sampling noise and cannot certify convergence.
    time_grids=[12,24] if quick else [12,24,48]
    time_seeds=seeds[:1]  # exploratory: avoid repeated expensive time-grid MC
    time_grid_rows=[]
    grid_budget=dict(name="TimeGrid",quote_paths=4000,outer_paths=2400,curve_nodes=7)
    for step in time_grids:
        for k,seed in enumerate(time_seeds):
            ss=replace(p.simulation,steps_per_year=step,
                       quote_paths=grid_budget['quote_paths'],outer_paths=grid_budget['outer_paths'],
                       curve_nodes=grid_budget['curve_nodes'],seed=seed)
            pv=price_acro(replace(p,simulation=ss))
            time_grid_rows.append(dict(steps_per_year=step,seed=seed,
                                       premium=pv.price,outer_only_se=pv.outer_se))
    print('Time-grid checks completed',flush=True)
    with open(output/'time_grid_study.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(time_grid_rows[0]));w.writeheader();w.writerows(time_grid_rows)
    with open(output/'independent_state_checks.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(state_checks[0]) if state_checks else ['factor']);w.writeheader();w.writerows(state_checks)
    obj={'type':'numerical_stability_not_market_calibration','quick':quick,
         'levels':summary,'time_grid_study':time_grid_rows,'jensen_diagnostic':jensen,'independent_state_checks':state_checks,
         'limitations':['Empirical variation across independent seeds includes quote/outer/inner random noise but cannot certify zero bias.',
                        'Across-seed t intervals describe repeated estimator means, not a market-consistent price confidence interval.',
                        'Independent state spot-checks carry their own MC error and do not bound full interpolation error.',
                        'Nested positive-part Jensen bias, time-grid discretization, calibration and specification bias are not eliminated.']}
    (output/'numerical_audit.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')
    visualize(rows,output)
    lines=['# Audit numérique ACRO — simulations sous Q, marché hypothétique',
           '', 'Il ne s’agit **ni d’un prix de marché ni d’un intervalle de confiance total**.',
           '', '| Niveau | Réplications | Prime moyenne | Écart-type entre runs | Étendue observée | Coupon initial moyen | Baisse du coupon bundle |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for k,v in summary.items():
        lines.append(f"| {k} | {v['n_independent_runs']} | {v['mean_price']:.4f} | {v['run_to_run_sample_sd']:.4f} | [{v['min_price']:.4f}, {v['max_price']:.4f}] | {v['mean_initial_coupon']:.3%} | {v['mean_bundled_coupon_haircut_bp']:.1f} pb |")
    lines += ['', '## Sensibilité au pas temporel',
              'Cette étude comporte aussi `time_grid_study.csv` (12/24/48 pas par an selon mode). Les trajectoires Browniennes ne sont pas couplées entre grilles, et les écarts incluent du bruit Monte Carlo.']
    lines += ['', '## Biais numérique : test Jensen',
              f"Deux courbes MC indépendantes : {jensen['first_inner_curve']:.4f} et {jensen['second_inner_curve']:.4f}; courbe pooled : {jensen['pooled_curve_price']:.4f}.",
              f"Écart de convexité N vs 2N : {jensen['finite_sample_jensen_gap']:.5f}. Ce test ne borne PAS le biais total.",
              '', '## Repricing conditionnel indépendant',
              '| Facteur vol | Valeur interpolée | MC direct indépendant | Écart | SE MC direct |',
              '|---:|---:|---:|---:|---:|']
    for x in state_checks:
        lines.append(f"| {x['factor']:.3f} | {x['interpolated_value']:.3f} | {x['direct_value']:.3f} | {x['difference']:+.3f} | {x['direct_mc_se']:.3f} |")
    lines += ['', '## Ce qui manque pour qualifier un prix',
              'Des courbes de convergence beaucoup plus resserrées, des contrôles indépendants du biais de simulation interne et du pas temporel, puis une calibration de marché et des tests de risque/couverture. Les graphiques de cette étude n’impliquent ni viabilité commerciale ni performance investisseur.',
              '', '![Variabilité empirique des estimations](stabilite_prime_ACRO.png)']
    (output/'AUDIT_NUMERIQUE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Rapport:',output/'AUDIT_NUMERIQUE.md',flush=True)
    return obj


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--quick',action='store_true')
    cli.add_argument('--config',default=str(ROOT/'config.json'))
    cli.add_argument('--output',default=str(ROOT/'results_convergence'))
    args=cli.parse_args()
    run(args.config,args.output,args.quick)
