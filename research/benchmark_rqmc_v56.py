"""V5.6: reproducible cost-aware research benchmark for *conditional* Phoenix values.

Usage:
  python benchmark_rqmc_v56.py --quick
  python benchmark_rqmc_v56.py --grid 96

The market model, Phoenix payoff, coupon and ACRO exercise rule are unchanged.
The primary estimate is the renewal note V_R(x; 18%) at multiple initial
log-volatility states x, using either pseudo-MC or *independent randomized Sobol*
with a Brownian bridge. The ACRO premium uses one FIXED outer path set to
isolate conditional-pricing uncertainty (it excludes outer-MC, time grid, model
and interpolation bias). Variance comparisons always mention runtime and
trajectory budget; RQMC intervals reflect randomization only.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse
import csv
import json
from time import perf_counter

import numpy as np
from scipy.stats import t as student_t

from acro import load_project, simulate_phoenix
from acro.conditional_qmc import innovation_bank

ROOT=Path(__file__).resolve().parents[1]
COUPON=0.18


def exercise_price(p, years, factors, nodes, values):
    m=np.isfinite(years)
    if p.acro.last_eligible_call_year is not None:
        m &= years <= p.acro.last_eligible_call_year + 1e-12
    if not np.any(m): return 0.
    if factors[m].min() < nodes[0]-1e-10 or factors[m].max() > nodes[-1]+1e-10:
        raise ValueError('Conditional grid does not cover all observed factors')
    intr=np.maximum(np.interp(factors[m], nodes, values) - p.acro.strike,0.)
    return float((np.exp(-p.market.risk_free_rate*years[m])*intr).sum()*p.acro.renewal_units / len(years))


def summary(vals):
    v=np.asarray(vals,float)
    if len(v)<2: raise ValueError('Need multiple independent repetitions')
    sd=v.std(ddof=1)
    hw=student_t.ppf(.975,len(v)-1)*sd/np.sqrt(len(v))
    return dict(mean=float(v.mean()),sd_across_replicates=float(sd),
                se_of_mean=float(sd/np.sqrt(len(v))),ci95=[float(v.mean()-hw),float(v.mean()+hw)])


def benchmark(p, out:Path, steps:int, n_qmc:int, reps:int,
              outer_paths:int, nodes_n:int, compare_mc_multiplier:int=2,
              base_seed:int=20261161):
    out.mkdir(parents=True,exist_ok=True)
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('Reference conditional QMC benchmark requires a stochastic factor')
    # The outer sample stays identical across both conditional pricing methods.
    # This is a comparison of inner-pricing efficiency, NOT a full premium CI.
    o=simulate_phoenix(p.market,p.initial_phoenix,outer_paths,steps,424241)
    valid=np.isfinite(o.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        valid &= o.first_call_year <= p.acro.last_eligible_call_year + 1e-12
    if not valid.any():raise RuntimeError('No covered autocalls in the outer benchmark sample')
    lo=min(float(o.factor_at_call[valid].min())-.15,p.market.initial_logvol_factor-.15)
    hi=max(float(o.factor_at_call[valid].max())+.15,p.market.initial_logvol_factor+.15)
    nodes=np.unique(np.r_[np.linspace(lo,hi,nodes_n),p.market.initial_logvol_factor])
    rows=[]; curves={}; all_anchors={}; all_premiums={}; runtimes={}
    for method in ('mc_bridge','sobol_bridge'):
        n=n_qmc*(compare_mc_multiplier if method=='mc_bridge' else 1)
        group=[]; anchors=[];premiums=[];times=[]
        for rep in range(reps):
            t0=perf_counter()
            innovations=innovation_bank(int(round(p.renewal_phoenix.maturity_years*steps)),
                                        n,p.market.n_assets,base_seed+rep*8191,
                                        method=method)
            points=[]
            for x in nodes:
                s=simulate_phoenix(p.market,p.renewal_phoenix,n,steps,seed=0,
                                   initial_factor=float(x),innovations=innovations)
                points.append(float(s.pv(COUPON).mean()))
            elapsed=perf_counter()-t0
            vals=np.asarray(points)
            anch=float(np.interp(p.market.initial_logvol_factor,nodes,vals))
            option=exercise_price(p,o.first_call_year,o.factor_at_call,nodes,vals)
            group.append(vals);anchors.append(anch);premiums.append(option);times.append(elapsed)
            rows.append(dict(method=method,rep=rep+1,paths=n,grid=steps,
                             elapsed_s=elapsed,anchor=anch,premium_fixed_outer=option))
            print(f'[{method}] {rep+1}/{reps} paths={n} '
                  f'anchor={anch:.4f}, ACRO={option:.4f}, elapsed={elapsed:.2f}s',flush=True)
        curves[method]=np.stack(group)
        all_anchors[method]=anchors;all_premiums[method]=premiums; runtimes[method]=times
    with (out/'per_replication.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'conditional_curve.csv').open('w',newline='',encoding='utf-8') as fh:
        w=csv.DictWriter(fh,fieldnames=['factor','method','mean_value','se_randomisation'],delimiter=',')
        w.writeheader()
        for method in curves:
            for k,x in enumerate(nodes):
                y=curves[method][:,k]
                w.writerow(dict(factor=float(x),method=method,mean_value=float(y.mean()),
                                se_randomisation=float(y.std(ddof=1)/np.sqrt(reps))))
    output={}
    for method in curves:
        times=np.array(runtimes[method])
        a=np.array(all_anchors[method]);prices=np.array(all_premiums[method])
        mean_curve=curves[method].mean(axis=0)
        output[method]=dict(paths_per_rep=n_qmc*(compare_mc_multiplier if method=='mc_bridge' else 1),
                            anchor=summary(a),premium_fixed_outer=summary(prices),
                            premium_of_mean_curve=exercise_price(p,o.first_call_year,o.factor_at_call,nodes,mean_curve),
                            total_runtime_s=float(times.sum()),time_per_rep_s=float(times.mean()),
                            efficiency_sd2_time=float(np.var(a,ddof=1)*times.mean()))
    output['relative_efficiency']={
      'anchor_var_time_sobol_over_mc':output['sobol_bridge']['efficiency_sd2_time']/output['mc_bridge']['efficiency_sd2_time'],
      'comment':'Lower is better; finite-sample empirical ratio, NOT an asymptotic guarantee.'}
    output['design']=dict(steps_per_year=steps, coupon_fixed=COUPON, reps=reps,
                          n_qmc=n_qmc,n_pseudo_mc=n_qmc*compare_mc_multiplier,
                          n_outer_fixed=outer_paths, nodes=len(nodes),
                          eligible_call_probability=float(valid.mean()),
                          states_of_market='One common fixed outer MC sample; no claim of full ACRO premium CI.',
                          no_change_to_financial_model=True)
    (out/'benchmark.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')
    mc=output['mc_bridge'];qmc=output['sobol_bridge']
    lines=[
      '# ACRO — Audit V5.6 : estimation conditionnelle du Phoenix renouvelé',
      '',f'- **Coupon contractuel fixé : {COUPON:.0%}** ; grille {steps} pas/an ; {reps} réplications indépendantes par méthode.',
      f'- Pseudo-MC : {n_qmc*compare_mc_multiplier} chemins par réplication ; Sobol randomisé + pont brownien : {n_qmc} chemins.',
      f'- **États de rappel identiques** pour chaque expérience, {outer_paths} trajectoires externes maintenues fixes.',
      '- Les IC reflètent uniquement l’aléa des échantillons internes. Ni biais temporel, ni interpolation, ni bruit externe, ni calibration ne sont couverts.',
      '', '| Mesure | Pseudo-MC | Sobol randomisé |','|---|---:|---:|',
      f'| Ancrage du Phoenix à x=0 (€/100) | {mc["anchor"]["mean"]:.4f} | {qmc["anchor"]["mean"]:.4f} |',
      f'| Demi-largeur IC 95 % de l’ancrage | {(mc["anchor"]["ci95"][1]-mc["anchor"]["mean"]):.4f} | {(qmc["anchor"]["ci95"][1]-qmc["anchor"]["mean"]):.4f} |',
      f'| Prime ACRO sur mêmes états (€/100) | {mc["premium_fixed_outer"]["mean"]:.4f} | {qmc["premium_fixed_outer"]["mean"]:.4f} |',
      f'| Demi-largeur IC 95 % de la prime conditionnelle | {(mc["premium_fixed_outer"]["ci95"][1]-mc["premium_fixed_outer"]["mean"]):.4f} | {(qmc["premium_fixed_outer"]["ci95"][1]-qmc["premium_fixed_outer"]["mean"]):.4f} |',
      f'| Temps total valorisations conditionnelles | {mc["total_runtime_s"]:.1f} s | {qmc["total_runtime_s"]:.1f} s |',
      '',f'- Rapport **variance de l’ancrage × temps** Sobol/MC : {output["relative_efficiency"]["anchor_var_time_sobol_over_mc"]:.3f} (plus faible = meilleur).',
      '', '## Interprétation et limites',
      '', '- Même diffusion risque-neutre, mêmes flux Phoenix : seule la génération des trajectoires change.',
      '- La prime moyenne de payoffs convexes calculés sur des courbes MC bruitées peut comporter un biais de Jensen, même avec Sobol.',
      '- Le prix de référence de la V5.5, la convergence 48/96/192 et le coupon au pair ne sont **pas validés** par cet audit.',
      '- Il faut vérifier la courbe sur des facteurs hors grille, et une convergence temporelle couplée avant de fixer un prix.',
      '', 'Les séries indépendantes sont dans `per_replication.csv` et `conditional_curve.csv`.' ]
    (out/'AUDIT_RQMC_V5_6_FR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return output


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true',help='6 scrambles, 512 Sobol paths, 1024 plain MC, 9 nodes')
    ap.add_argument('--grid',type=int,default=48,help='48, 96 or 192 recommended')
    ap.add_argument('--output',type=Path,default=None)
    args=ap.parse_args()
    p=load_project(ROOT/'config.json')
    b=dict(n_qmc=512,reps=6,outer_paths=1600,nodes_n=9) if args.quick else dict(n_qmc=2048,reps=8,outer_paths=4000,nodes_n=11)
    res=benchmark(p,args.output or ROOT/('results_rqmc_v56_quick' if args.quick else 'results_rqmc_v56'),steps=args.grid,**b)
    print('Sobol/MC anchor Var×time ratio:',round(res['relative_efficiency']['anchor_var_time_sobol_over_mc'],4))
