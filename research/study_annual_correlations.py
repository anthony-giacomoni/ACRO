"""Experimental annual correlation risk in ACRO, not a calibrated market quote.

Run in the ACRO project directory:
    python study_annual_correlations.py --quick
    python study_annual_correlations.py --full

Compares frozen correlation with uncertain yearly correlation using the EXACT
SAME 2D conditional pricing architecture, sample sizes, seeds and coupon.
Separate seeds give an indicative between-run standard error. This does NOT
establish full pricing accuracy nor remove 2D interpolation/Jensen bias.
"""
from __future__ import annotations
import argparse,csv,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro.config import load_project
from acro.annual_correlation import AnnualCorrelation,price_acro_annual_correlations


def run(out:Path,quick:bool,full:bool):
    project=load_project(Path(__file__).resolve().parents[1]/'config.json')
    if quick:
        seeds=(1801,1802); quote=2000;outer=2000; cond=1000; nodes=(5,5); strength=(0.,0.50)
    elif full:
        seeds=(1801,1802,1803,1804,1805);quote=100000;outer=22000;cond=10000;nodes=(9,7)
        strength=(0.,0.25,0.50,0.75)
    else:
        seeds=(1801,1802,1803,1804);quote=40000;outer=12000;cond=5000;nodes=(7,5)
        strength=(0.,0.50,0.75)
    out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for v in strength:
        for sd in seeds:
            p=replace(project,simulation=replace(project.simulation,
                        quote_paths=quote,outer_paths=outer,conditional_paths=cond,seed=sd))
            r=price_acro_annual_correlations(p,
                AnnualCorrelation(volatility=v),
                logvol_nodes=nodes[0],corr_nodes=nodes[1],fixed_coupon=.182)
            row=dict(seed=sd,corr_factor_annual_sigma=v,
                     acro_premium_eur=r.acro_premium,
                     outer_se_only_eur=r.outer_se_only,
                     original_phoenix_fixed_coupon_price_eur=r.initial_note_price,
                     original_coupon_pct=18.2,
                     fair_coupon_pct=100*r.fair_coupon_rate,
                     call_probability_pct=100*r.call_probability,
                     exercise_probability_pct=100*r.exercise_probability,
                     quote_paths=quote,outer_paths=outer,conditional_paths=cond,
                     n_logvol_nodes=nodes[0],n_correlation_nodes=nodes[1],
                     price_surface_sim_paths=r.surface.paths_per_node)
            rows.append(row)
            print(f'sigma_corr={v:.2f} seed={sd} premium={r.acro_premium:.4f} '
                  f'outSE={r.outer_se_only:.4f} note={r.initial_note_price:.3f}',flush=True)
    with open(out/'raw_runs.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary=[]
    for v in strength:
        a=np.array([r['acro_premium_eur'] for r in rows if r['corr_factor_annual_sigma']==v])
        summary.append(dict(corr_factor_annual_sigma=v,mean_premium_eur=float(a.mean()),
                            seed_to_seed_se_eur=float(a.std(ddof=1)/np.sqrt(len(a))),n_seeds=len(a)))
    control={r['seed']:r for r in rows if r['corr_factor_annual_sigma']==0.}
    paired=[]
    for v in strength[1:]:
        deltas=np.array([r['acro_premium_eur']-control[r['seed']]['acro_premium_eur']
                       for r in rows if r['corr_factor_annual_sigma']==v])
        paired.append(dict(annual_corr_sigma=v,mean_delta_eur=float(deltas.mean()),
                           se_between_seeds_eur=float(deltas.std(ddof=1)/np.sqrt(len(deltas))),
                           n_seeds=len(deltas)))
    with open(out/'comparison.json','w',encoding='utf-8') as f:
        json.dump(dict(parameters=dict(reversion=.8,amplitude=.8,starting_y=0.,
                      note_coupon_fixed=.182,market_calibrated=False),
                       summary=summary,paired=paired,
                       caveat='Sampling SE across seeds does not capture interpolation/Jensen bias or model miscalibration'),
                  f,ensure_ascii=False,indent=2)
    fig,ax=plt.subplots(figsize=(7,4.5))
    xx=[d['corr_factor_annual_sigma'] for d in summary]
    yy=[d['mean_premium_eur'] for d in summary]
    ee=[1.96*d['seed_to_seed_se_eur'] for d in summary]
    ax.errorbar(xx,yy,yerr=ee,fmt='o-',capsize=3)
    ax.set_xlabel('Volatilité annuelle du facteur de corrélation (hypothétique)')
    ax.set_ylabel('Prime ACRO (€/100) ; coupon contractuel 18,20 %')
    ax.set_title('ACRO : effet de la corrélation annuelle stochastique')
    ax.grid(alpha=.2)
    ax.text(.0,-.24,'Barres : ±1,96 erreurs standards entre graines ; hors biais de grille et risque de modèle.',
            transform=ax.transAxes,fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'13_prime_acro_correlation_annuelle.png',dpi=150,bbox_inches='tight')
    plt.close(fig)
    print('Comparaisons appariées:',json.dumps(paired,indent=2),flush=True)
    return summary,paired

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,default=Path('evidence/annual_correlation'))
    p.add_argument('--quick',action='store_true')
    p.add_argument('--full',action='store_true')
    args=p.parse_args()
    if args.quick and args.full: p.error('Choose only one study size')
    run(args.output,args.quick,args.full)
