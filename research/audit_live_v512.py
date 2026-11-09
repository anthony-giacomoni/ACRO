"""V5.12: independent controlled vs historical live-note renewal pricers.

A numerical experiment, not a confidence interval on the true market price.
CLI: python audit_live_v512.py --quick | python audit_live_v512.py
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import numpy as np
from acro import load_project
from acro.live_risk import LiveState, value_live, stock_delta_live

ROOT=Path(__file__).resolve().parents[1]

def audit(quick=False,output=None,reps=None):
    config=json.loads((ROOT/'risk_scenarios.json').read_text())
    p=load_project(ROOT/'config.json')
    coupon=float(config['coupon_fixed'])
    p=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=coupon))
    s=p.simulation
    p=replace(p,simulation=replace(s,steps_per_year=12 if quick else 24,
                 quote_paths=512 if quick else 1600,
                 conditional_paths=384 if quick else 1200,
                 outer_paths=500 if quick else 1200,
                 curve_nodes=7 if quick else 11))
    state=LiveState(float(config['live_time_years']),tuple(config['live_spot_ratios']),
                    float(config['live_logvol_factor']),int(config['missed_coupons']))
    external_seed=527001
    r=reps if reps is not None else (6 if quick else 10)
    ref_paths=256 if quick else 512
    results=[]
    for i in range(r):
        for method in ('legacy','gbm_cv'):
            # Change the inner samples, but hold the *outer* Brownian paths fixed.
            altered=replace(p,simulation=replace(p.simulation,seed=p.simulation.seed+1003*i))
            start=perf_counter()
            v=value_live(altered,state,n_paths=p.simulation.outer_paths,seed=external_seed,
                         pricing_method=method,ref_paths=ref_paths,scrambles=3,
                         curve_seed_offset=1003*i)
            results.append(dict(rep=i,method=method,price=v.option_price,
                                note=v.note_price,total=v.total_price,
                                seconds=perf_counter()-start))
    summary={}
    for method in ('legacy','gbm_cv'):
        a=np.asarray([x['price'] for x in results if x['method']==method])
        dt=np.asarray([x['seconds'] for x in results if x['method']==method])
        summary[method]=dict(n=len(a),mean=float(a.mean()),std=float(a.std(ddof=1)),
                             standard_error=float(a.std(ddof=1)/np.sqrt(len(a))),
                             avg_seconds=float(dt.mean()),
                             variance_time_proxy=float(a.var(ddof=1)*dt.mean()))
    # Spot Greek bump-size check, new independent simulation and pricing seeds.
    delta=[]
    for bump in (.005,.01,.02):
        for j in range(3 if quick else 4):
            pp=replace(p,simulation=replace(p.simulation,seed=p.simulation.seed+1891*j))
            _, rows=stock_delta_live(pp,state,relative_bump=bump,
                                    n_paths=300 if quick else 1000,
                                    pricing_method='gbm_cv',ref_paths=ref_paths,
                                    scrambles=3,seed=83100+97*j,
                                    curve_seed_offset=1201*j)
            for x in rows:
                delta.append(dict(bump_percent=100*bump,rep=j,asset=x['asset'],
                                  delta_total=x['delta_total'],delta_note=x['delta_note'],
                                  delta_acro=x['delta_acro'],
                                  paired_outer_se_only=x['delta_total_paired_outer_se']))
    output=Path(output) if output else ROOT/('results_live_v512_quick' if quick else 'results_live_v512')
    output.mkdir(exist_ok=True,parents=True)
    for name,items in [('live_curve_independent.csv',results),('live_delta_bumps.csv',delta)]:
        with open(output/name,'w',encoding='utf8',newline='') as fp:
            wr=csv.DictWriter(fp,fieldnames=list(items[0]));wr.writeheader();wr.writerows(items)
    (output/'audit_summary.json').write_text(json.dumps(dict(summary=summary,
        fixed_outer_seed=external_seed,conditional_only=True,
        warning='No global price/Greek confidence interval; common model bias not covered'),
        indent=2,ensure_ascii=False))
    print('V5.12 independent live curve comparisons, outer paths fixed:')
    for k,v in summary.items():
        print(k,'mean',round(v['mean'],4),'std',round(v['std'],4),
              'mean sec',round(v['avg_seconds'],2))
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--reps',type=int,default=None)
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    audit(quick=args.quick,output=args.output,reps=args.reps)
