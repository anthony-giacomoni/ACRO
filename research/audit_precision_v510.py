"""ACRO V5.10: full-option independent out-of-sample exercise bounds.

python audit_precision_v510.py --quick
python audit_precision_v510.py

This does NOT change the original pricer or financial assumptions.  The
bounds are in expectation at the chosen time grid and interpolation nodes,
NOT guaranteed finite-sample numerical bounds on the continuous-time option.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse,csv,json
import numpy as np
from scipy.stats import t as student_t
from acro import load_project,simulate_phoenix
from acro.renewal_anchor import grid_including_anchor
from acro.precision_bounds import independent_curve_draw,crossfit_payoffs

ROOT=Path(__file__).resolve().parents[1]
COUPON=.18


def intervals(x):
    x=np.asarray(x,float)
    n=len(x)
    if n<2: raise ValueError('At least two independent repetitions')
    mean=float(x.mean());s=float(x.std(ddof=1));se=s/np.sqrt(n)
    crit=float(student_t.ppf(.975,n-1))
    return dict(mean=mean,sd=s,standard_error=float(se),ci95=[mean-crit*se,mean+crit*se],n=n)


def run(quick=False,output=None,high_precision=False,use_fast=True,start_rep=0):
    p=load_project(ROOT/'config.json')
    if p.market.dynamics!='stochastic_vol' or p.renewal_phoenix.capital_barrier_monitoring!='european':
        raise ValueError('This audit requires stochastic-volatility and European renewed-note barrier')
    p=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=COUPON),
              acro=replace(p.acro,target_coupon_rate=COUPON))
    cfg=(dict(reps=4,steps=48,inner_each=512,ref_each=1024,scrambles=4,outer=2000,nodes=17)
        if quick else dict(reps=5,steps=96,inner_each=4096,ref_each=8192,scrambles=6,outer=8000,nodes=23)
        if high_precision else dict(reps=10,steps=96,inner_each=1024,ref_each=2048,scrambles=4,outer=5000,nodes=17))
    out=Path(output or ROOT/('results_precision_v510_quick' if quick else 'results_precision_v510_high' if high_precision else 'results_precision_v510'))
    out.mkdir(exist_ok=True,parents=True)
    # Include the x0 node without perturbing its two neighbours; 1 extra node.
    nodes=np.unique(np.r_[np.linspace(-1.4,1.0,cfg['nodes']),p.market.initial_logvol_factor])
    rows=[]
    start=perf_counter()
    for k in range(cfg['reps']):
        seed=5100100+100009*(k+start_rep)
        outer=simulate_phoenix(p.market,p.initial_phoenix,cfg['outer'],cfg['steps'],seed)
        if np.any(np.isfinite(outer.first_call_year)):
            xf=outer.factor_at_call[np.isfinite(outer.first_call_year)]
            if xf.min()<nodes[0] or xf.max()>nodes[-1]:
                raise RuntimeError('Conditional factor beyond interpolation grid; extend nodes')
        a=independent_curve_draw(p,nodes,COUPON,cfg['steps'],cfg['inner_each'],cfg['ref_each'],cfg['scrambles'],seed+11001,use_fast=use_fast)
        b=independent_curve_draw(p,nodes,COUPON,cfg['steps'],cfg['inner_each'],cfg['ref_each'],cfg['scrambles'],seed+22001,use_fast=use_fast)
        lower,upper=crossfit_payoffs(p,outer,a,b)
        row=dict(rep=k+1+start_rep,lower=float(lower.mean()),upper=float(upper.mean()),
                 gap=float((upper-lower).mean()),outer_se_upper=float(upper.std(ddof=1)/np.sqrt(len(upper))),
                 outer_se_lower=float(lower.std(ddof=1)/np.sqrt(len(lower))),
                 ref_se_a=a.reference_se,ref_se_b=b.reference_se,
                 inner_node_se_a_max=float(a.delta_node_se.max()),
                 inner_node_se_b_max=float(b.delta_node_se.max()))
        rows.append(row)
        print(f"[v5.10] pair {k+1}/{cfg['reps']} lower={row['lower']:.5f} upper={row['upper']:.5f} gap={row['gap']:.5f}",flush=True)
    result=dict(version='5.10',coupon_contractual=COUPON,configuration=cfg,nodes=nodes.tolist(),
                lower_in_expectation=intervals([v['lower'] for v in rows]),
                upper_in_expectation=intervals([v['upper'] for v in rows]),
                jensen_exercise_gap=intervals([v['gap'] for v in rows]),
                runtime_s=perf_counter()-start,
                target_total_error=.02,
                certified=False,use_fast=use_fast,start_rep=start_rep,
                scope='Discrete-time model with linear interpolation of exact conditional node values',
                notes=['Independent inner curve and GBM reference on every side of cross fitting.',
                       'Finite-sample CI estimates computed from independent replication pairs, NOT rigorous confidence limits for the true price.',
                       'This is NOT a bound on time-discretization or interpolation bias.',
                       'Risk-neutral synthetic model; no market calibration.'])
    (out/'AUDIT_PRIX_COMPLET_V5_10.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    with (out/'replications.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# ACRO V5.10 — Audit de la prime complète par décision hors échantillon','',
           f'Coupon contractuel fixé à **18 %**, {cfg["steps"]} pas/an, {cfg["reps"]} paires indépendantes.',
           '', '| Estimateur | Moyenne €/100 | IC95 empirique entre paires |',
           '|---|---:|---|']
    for title,key in [('Inférieur en espérance','lower_in_expectation'),
                      ('Supérieur en espérance','upper_in_expectation'),
                      ('Écart supérieur–inférieur','jensen_exercise_gap')]:
        s=result[key]
        lines.append(f'| {title} | {s["mean"]:.5f} | [{s["ci95"][0]:.5f} ; {s["ci95"][1]:.5f}] |')
    lines +=['',f'Calcul total : **{result["runtime_s"]:.1f} s**.',
             '', '## Ce que les bornes démontrent','',
             '- Les courbes A et B utilisent des chocs et références GBM **indépendants**.',
             '- La décision d’exercice est prise sur une courbe, son résultat estimé sur l’autre.',
             '- Par convexité, l’estimateur supérieur est majorant **en espérance** ; la politique indépendante est minorante **en espérance**, sur la valeur vraie **interpolée** aux nœuds.',
             '', '## Ce qu’elles ne démontrent pas','',
             '- Les intervalles empiriques ne certifient pas les bornes numériques après un nombre fini de réplications.',
             '- Le biais temporel et celui d’interpolation ne sont pas bornés.',
             '- Le risque de marché, la calibration, le crédit et les coûts de couverture ne sont pas traités.',
             '- **La tolérance globale ±0,02 €/100 n’est pas certifiée.**', '']
    (out/'AUDIT_PRIX_COMPLET_V5_10_FR.md').write_text('\n'.join(lines),encoding='utf-8')
    print('[v5.10] gap=',result['jensen_exercise_gap']['mean'],'seconds=',result['runtime_s'],flush=True)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--high-precision',action='store_true');ap.add_argument('--slow',action='store_true');ap.add_argument('--start-rep',type=int,default=0);ap.add_argument('--output')
    z=ap.parse_args();run(z.quick,z.output,z.high_precision,not z.slow,z.start_rep)
