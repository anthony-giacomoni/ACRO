"""ACRO: pre-specified, independently seeded coupled time-grid audit.

The coupon, financial terms, conditional-volatility nodes and reference GBM
are fixed for all 96/192/384-step estimates.  Brownian shocks are identical
within each independent bank across time grids, including *both* the initial
note and the renewal note. No conclusion about continuous-time error is made.

Usage:
  python audit_temporal_robuste.py --quick
  python audit_temporal_robuste.py --banks 24 --outer 3000 --inner 1536 --nodes 17
  python audit_temporal_robuste.py --banks 40 --outer 6000 --inner 4096 --nodes 25
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t
from acro import load_project
from acro.conditional_qmc import innovation_bank
from acro.fast_engine import simulate_phoenix_fast
from acro.gbm_control import gbm_reference, aggregated_bank
from research.audit_variance_v58 import discounted_payoffs

ROOT=Path(__file__).resolve().parents[1]
GRIDS=(96,192,384)
COUPON=0.18


def paired_interval(data):
    arr=np.asarray(data,float)
    if arr.size<3 or not np.all(np.isfinite(arr)):
        raise ValueError('Require >=3 independent finite banks')
    mean=float(np.mean(arr))
    sd=float(np.std(arr,ddof=1))
    half=float(student_t.ppf(.975,arr.size-1)*sd/np.sqrt(arr.size))
    return dict(n=int(arr.size),mean=mean,sd=sd,half_width_95=half,
                ci95=[mean-half,mean+half])


def temporal_verdict(delta1,delta2,tolerance=0.02):
    """A strict grid-stability diagnostic, NOT a time-discretization bound."""
    def close_to_zero(d):
        return max(abs(v) for v in d['ci95']) <= tolerance
    good=close_to_zero(delta1) and close_to_zero(delta2)
    return {'grid_stability_criterion_passed':bool(good),
            'threshold':tolerance,
            'pricing_certified':False,
            'interpretation':('Two paired differences statistically confined to tolerance; '
                'does NOT bound the difference from the continuous-time price.'
                if good else 'At least one paired difference is not confined to tolerance.')}


def run(*,banks=16,outer_paths=1200,inner_paths=640,nodes_count=13,
        ref_paths=4096,scrambles=6,seed=583001,output=None,verbose=True):
    if banks<3 or outer_paths<10 or inner_paths<10 or nodes_count<5:
        raise ValueError('Invalid numerical budget')
    p=load_project(ROOT/'config.json')
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('Temporal audit uses the stochastic-volatility reference configuration')
    p=replace(p,initial_coupon_policy='fixed',
        initial_phoenix=replace(p.initial_phoenix,coupon_rate=COUPON),
        acro=replace(p.acro,target_coupon_rate=COUPON))
    nodes=np.unique(np.r_[np.linspace(-1.4,1.0,nodes_count),p.market.initial_logvol_factor])
    # Two independent reference estimations: the second is a common-reference
    # stress check. Their sampling uncertainty is kept OUTSIDE paired-bank CI.
    ref_a=gbm_reference(p,COUPON,384,ref_paths,scrambles,seed+1100001)
    ref_b=gbm_reference(p,COUPON,384,ref_paths,scrambles,seed+2200003)
    gbm=replace(p.market,dynamics='gbm')
    start=perf_counter()
    rows=[]
    for i in range(banks):
        bank_seed=seed+100003*i
        fine_outer=innovation_bank(384*int(p.initial_phoenix.maturity_years),
                                    outer_paths,p.market.n_assets,bank_seed+103,'mc_bridge')
        fine_inner=innovation_bank(384*int(p.renewal_phoenix.maturity_years),
                                    inner_paths,p.market.n_assets,bank_seed+701,'mc_bridge')
        gbm_paths=simulate_phoenix_fast(gbm,p.renewal_phoenix,inner_paths,
            p.renewal_phoenix.observations_per_year,
            innovations=aggregated_bank(fine_inner,384,p.renewal_phoenix.observations_per_year))
        gbm_payoff=gbm_paths.pv(COUPON)
        row={'bank':i+1,'seed':bank_seed}
        for steps in GRIDS:
            bank_outer=fine_outer if steps==384 else aggregated_bank(fine_outer,384,steps)
            bank_inner=fine_inner if steps==384 else aggregated_bank(fine_inner,384,steps)
            initial=simulate_phoenix_fast(p.market,p.initial_phoenix,outer_paths,steps,
                innovations=bank_outer)
            deltas=[]
            for x in nodes:
                renewable=simulate_phoenix_fast(p.market,p.renewal_phoenix,inner_paths,steps,
                    initial_factor=float(x),innovations=bank_inner)
                deltas.append(float(np.mean(renewable.pv(COUPON)-gbm_payoff)))
            deltas=np.asarray(deltas)
            outcome_a,labels=discounted_payoffs(p,initial,nodes,ref_a.value+deltas)
            outcome_b,_=discounted_payoffs(p,initial,nodes,ref_b.value+deltas)
            row[f'premium_{steps}_a']=float(outcome_a.mean())
            row[f'premium_{steps}_b']=float(outcome_b.mean())
            row[f'eligible_call_rate_{steps}']=float((labels>0).mean())
        for k,l in ((192,96),(384,192),(384,96)):
            for suffix in ('a','b'):
                row[f'delta_{k}_{l}_{suffix}']=row[f'premium_{k}_{suffix}']-row[f'premium_{l}_{suffix}']
        rows.append(row)
        if verbose:
            print(f"Bank {i+1}/{banks} 192-96={row['delta_192_96_a']:+.5f} "
                  f"384-192={row['delta_384_192_a']:+.5f}",flush=True)
    summary={'version':'5.10.1','coupon_fixed':COUPON,'grids':list(GRIDS),
        'budget':{'independent_banks':banks,'outer_paths_per_bank':outer_paths,
                  'renewal_paths_per_bank_and_node':inner_paths,'nodes':nodes.tolist(),
                  'reference_paths_per_scramble':ref_paths,'reference_scrambles':scrambles},
        'reference_a':{'value':float(ref_a.value),'se':float(ref_a.std_error)},
        'reference_b':{'value':float(ref_b.value),'se':float(ref_b.std_error)},
        'results':{},'elapsed_seconds':float(perf_counter()-start)}
    fields=[f'premium_{i}_a' for i in GRIDS]+[
        'delta_192_96_a','delta_384_192_a','delta_384_96_a',
        'delta_192_96_b','delta_384_192_b','delta_384_96_b',
        'eligible_call_rate_192','eligible_call_rate_384']
    for field in fields: summary['results'][field]=paired_interval([row[field] for row in rows])
    summary['verdict']=temporal_verdict(summary['results']['delta_192_96_a'],
                                         summary['results']['delta_384_192_a'])
    summary['common_reference_difference']=float(ref_b.value-ref_a.value)
    summary['limitations']=[
      'Independent bank CIs capture variation of paired inner and outer paths, not common reference error.',
      'Two GBM references probe common-reference sensitivity but do not prove a rigorous bound.',
      'Finite interpolation nodes induce a bias not bounded by paired time-grid comparisons.',
      'Euler discretization toward continuous-time price not certified even if successive differences are small.',
      'Coupon 18%, synthetic parameters, no market calibration or issuer credit.']
    output=Path(output or ROOT/'results_temporal_v5101')
    output.mkdir(parents=True,exist_ok=True)
    with (output/'banques_independantes.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output/'resultats.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    report=['# ACRO — Contre-audit temporel multi-banques (V5.10.1)','',
       f'Coupon **fixé ex ante à 18 %**. {banks} banques indépendantes; chocs couplés '
       'à l’intérieur de chaque banque, entre 96/192/384 pas/an.','',
       '| Mesure | Moyenne (€/100) | IC Student-t 95 % entre banques |',
       '|---|---:|---|']
    for label,key in [('Prime 96 pas','premium_96_a'),('Prime 192 pas','premium_192_a'),
                       ('Prime 384 pas','premium_384_a'),('192 − 96','delta_192_96_a'),
                       ('384 − 192','delta_384_192_a'),('384 − 96','delta_384_96_a')]:
        rr=summary['results'][key]
        report.append(f'| {label} | {rr["mean"]:+.5f} | [{rr["ci95"][0]:+.5f} ; {rr["ci95"][1]:+.5f}] |')
    report.extend(['',f'Référence GBM A : {ref_a.value:.5f} ± {ref_a.std_error:.5f} (erreur-type).',
         f'Référence GBM B : {ref_b.value:.5f} ± {ref_b.std_error:.5f} (erreur-type).',
         f'**Critère de stabilité sur les grilles** : {"atteint" if summary["verdict"]["grid_stability_criterion_passed"] else "NON atteint"}.',
         '**Prix théorique en temps continu : NON certifié.**','',
         'Les intervalles statistiques portent sur des banques indépendantes, avec une référence GBM commune.',
         'Les biais d’interpolation et de discrétisation absolue ne sont pas bornés.',
         'Les estimations ne sont ni des prix de marché ni des conclusions de couverture.'])
    (output/'AUDIT_TEMPOREL_V5_10_1_FR.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--banks',type=int,default=16)
    ap.add_argument('--outer',type=int,default=1200)
    ap.add_argument('--inner',type=int,default=640)
    ap.add_argument('--nodes',type=int,default=13)
    ap.add_argument('--reference-paths',type=int,default=4096)
    ap.add_argument('--scrambles',type=int,default=6)
    ap.add_argument('--seed',type=int,default=583001)
    ap.add_argument('--output')
    a=ap.parse_args()
    k={'banks':a.banks,'outer_paths':a.outer,'inner_paths':a.inner,
       'nodes_count':a.nodes,'ref_paths':a.reference_paths,
       'scrambles':a.scrambles,'seed':a.seed,'output':a.output}
    if a.quick:
        k.update(banks=4,outer_paths=256,inner_paths=192,nodes_count=9,
                 ref_paths=1024,scrambles=3)
    res=run(**k)
    print(json.dumps({'paired_differences':{j:res['results'][j] for j in (
        'delta_192_96_a','delta_384_192_a')},'verdict':res['verdict'],
        'seconds':res['elapsed_seconds']},indent=2))
