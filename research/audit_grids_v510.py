"""Coupled 48/96/192 steps/year full-prime experiment at fixed 18% coupon.

This compares finite-grid Euler model prices using the SAME fine Brownian bank
for initial-note autocall and for renewal conditional values. A shared GBM
reference cancels most of its error in *paired differences*.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import argparse,csv,json
import numpy as np
from scipy.stats import t as student_t
from acro import load_project
from acro.conditional_qmc import innovation_bank
from acro.fast_engine import simulate_phoenix_fast
from acro.gbm_control import gbm_reference,aggregated_bank
from research.audit_variance_v58 import discounted_payoffs

ROOT=Path(__file__).resolve().parents[1]

def run(quick=False,output=None,high_precision=False):
    p=load_project(ROOT/'config.json');c=.18
    p=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=c),
              acro=replace(p.acro,target_coupon_rate=c))
    budget=(dict(reps=3,inner=1024,outer=2500,nodes=19,ref=4096,scrambles=4)
            if quick else dict(reps=5,inner=3072,outer=5000,nodes=25,ref=8192,scrambles=6)
            if high_precision else dict(reps=4,inner=1536,outer=3000,nodes=19,ref=4096,scrambles=4))
    out=Path(output or ROOT/('results_grids_v510_quick' if quick else 'results_grids_v510_high' if high_precision else 'results_grids_v510'))
    out.mkdir(exist_ok=True,parents=True)
    nodes=np.linspace(-1.4,1.,budget['nodes'])
    ref=gbm_reference(p,c,192,budget['ref'],budget['scrambles'],5108100)
    gbm=replace(p.market,dynamics='gbm')
    rows=[];start=perf_counter()
    for rep in range(budget['reps']):
        seed=5108101+50003*rep
        outer_bank=innovation_bank(192*5,budget['outer'],p.market.n_assets,seed,'mc_bridge')
        inner_bank=innovation_bank(192*5,budget['inner'],p.market.n_assets,seed+9911,'mc_bridge')
        gbm_b=simulate_phoenix_fast(gbm,p.renewal_phoenix,budget['inner'],
                                    p.renewal_phoenix.observations_per_year,
                                    innovations=aggregated_bank(inner_bank,192,p.renewal_phoenix.observations_per_year)).pv(c)
        row={'rep':rep+1}
        for steps in (48,96,192):
            binner=inner_bank if steps==192 else aggregated_bank(inner_bank,192,steps)
            bouter=outer_bank if steps==192 else aggregated_bank(outer_bank,192,steps)
            outer=simulate_phoenix_fast(p.market,p.initial_phoenix,budget['outer'],steps,innovations=bouter)
            vals=[]
            for x in nodes:
                sv=simulate_phoenix_fast(p.market,p.renewal_phoenix,budget['inner'],steps,
                                         initial_factor=float(x),innovations=binner).pv(c)
                vals.append(ref.value+(sv-gbm_b).mean())
            y,_=discounted_payoffs(p,outer,nodes,np.asarray(vals))
            row[str(steps)]=float(y.mean())
        row['delta_192_96']=row['192']-row['96']
        row['delta_96_48']=row['96']-row['48']
        rows.append(row)
        print('grids',rep+1,row,flush=True)
    def summary(field):
        z=np.array([r[field] for r in rows]);m=float(z.mean());s=float(z.std(ddof=1))
        hw=float(student_t.ppf(.975,len(z)-1)*s/np.sqrt(len(z)))
        return {'mean':m,'sd':s,'ci95':[m-hw,m+hw]}
    result={'budget':budget,'ref_se':ref.std_error,'estimates':{str(k):summary(str(k)) for k in (48,96,192)},
            'paired_deltas':{k:summary(k) for k in ('delta_192_96','delta_96_48')},
            'seconds':perf_counter()-start,
            'precision_certified':False,
            'limitations':['Finite-grid paired convergence check; not a bound to continuous-time price.',
                           'Inner, GBM reference and interpolation uncertainties remain.',
                           'Few independent replications; wide Student-t confidence intervals.']}
    (out/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    with (out/'replications.csv').open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--high-precision',action='store_true');ap.add_argument('--output');a=ap.parse_args()
    run(a.quick,a.output,a.high_precision)
