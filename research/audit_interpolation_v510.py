"""Coupled interpolation diagnostic using one fine-node curve per replication.

All sparse curves are strict subsets of the dense 49-node grid. The same
GBM reference, SV Brownian innovations and outer paths are reused, so price
differences primarily diagnose linear interpolation (not the total bias).
"""
from pathlib import Path
from dataclasses import replace
from time import perf_counter
import argparse,csv,json
import numpy as np
from acro import load_project,simulate_phoenix
from acro.conditional_qmc import innovation_bank
from acro.gbm_control import gbm_reference,aggregated_bank
from acro.fast_engine import simulate_phoenix_fast
from research.audit_variance_v58 import discounted_payoffs

ROOT=Path(__file__).resolve().parents[1]

def run(quick=False,output=None):
    p=load_project(ROOT/'config.json');coupon=.18;steps=96
    p=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=coupon),
              acro=replace(p.acro,target_coupon_rate=coupon))
    b=(dict(inner=2048,outer=6000,reps=2,ref=4096,scrambles=4)
       if quick else dict(inner=4096,outer=12000,reps=4,ref=8192,scrambles=6))
    out=Path(output or ROOT/('results_interp_v510_quick' if quick else 'results_interp_v510'))
    out.mkdir(parents=True,exist_ok=True)
    fine_nodes=np.linspace(-1.4,1.0,49)
    ref=gbm_reference(p,coupon,steps,b['ref'],b['scrambles'],5107301)
    outer=simulate_phoenix(p.market,p.initial_phoenix,b['outer'],steps,5107302)
    if np.any(np.isfinite(outer.first_call_year)):
        states=outer.factor_at_call[np.isfinite(outer.first_call_year)]
        if min(states)<min(fine_nodes) or max(states)>max(fine_nodes):
            raise ValueError('Outer calls not covered by nodes')
    rows=[];start=perf_counter()
    gbm=replace(p.market,dynamics='gbm')
    for rep in range(b['reps']):
        bank=innovation_bank(steps*5,b['inner'],p.market.n_assets,5107303+10001*rep,'mc_bridge')
        sparse_bank=aggregated_bank(bank,steps,p.renewal_phoenix.observations_per_year)
        gbm_pv=simulate_phoenix_fast(gbm,p.renewal_phoenix,b['inner'],
                       p.renewal_phoenix.observations_per_year,innovations=sparse_bank).pv(coupon)
        values=[]
        for x in fine_nodes:
            sv=simulate_phoenix_fast(p.market,p.renewal_phoenix,b['inner'],steps,
                                     initial_factor=float(x),innovations=bank).pv(coupon)
            values.append(ref.value+(sv-gbm_pv).mean())
        values=np.asarray(values)
        yfine,_=discounted_payoffs(p,outer,fine_nodes,values)
        row=dict(rep=rep+1,dense49=float(yfine.mean()))
        for factor in (2,4):
            sparse_nodes=fine_nodes[::factor];sparse_value=values[::factor]
            yy,_=discounted_payoffs(p,outer,sparse_nodes,sparse_value)
            row[f'sparse{len(sparse_nodes)}']=float(yy.mean())
            row[f'delta_sparse{len(sparse_nodes)}_dense49']=float(yy.mean()-yfine.mean())
        rows.append(row)
        print('interp',rep+1,'delta25',round(row['delta_sparse25_dense49'],6),
              'delta13',round(row['delta_sparse13_dense49'],6),flush=True)
    result=dict(config=b,grid_steps_year=steps,coupon=coupon,ref_gbm_se=ref.std_error,
                runs=rows,time_s=perf_counter()-start,qualification='Coupled grid comparison, not rigorous interpolation bias bound')
    (out/'diagnostic.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    with (out/'replications.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--output');a=ap.parse_args()
    run(a.quick,a.output)
