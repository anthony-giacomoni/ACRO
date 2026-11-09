"""Optional convergence diagnostic: python check_convergence.py --quick

Changes N and random seeds; reports range / stability of model premiums.
This does NOT eliminate nested MC positive-part bias or model risk.
"""
import argparse, csv
from dataclasses import replace
from pathlib import Path
from acro import load_project, price_acro
ROOT=Path(__file__).resolve().parents[1]

def main(quick):
    p=load_project(ROOT/'config.json')
    runs=[(2000,1200,5),(5000,3000,9)] if quick else [(4000,2500,7),(9000,5000,11),(16000,9000,15)]
    results=[]
    for qn,on,gn in runs:
        for offset in (0,101):
            s=replace(p.simulation,quote_paths=qn,outer_paths=on,curve_nodes=gn,
                      seed=p.simulation.seed+offset)
            r=price_acro(replace(p,simulation=s))
            row=dict(quote_paths=qn,outer_paths=on,curve_nodes=gn,seed=s.seed,
                     premium=r.price,outer_se_only=r.outer_se)
            results.append(row)
            print(row)
    out=ROOT/'results';out.mkdir(exist_ok=True)
    with open(out/'convergence.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=results[0].keys());writer.writeheader();writer.writerows(results)
if __name__=='__main__':
    x=argparse.ArgumentParser();x.add_argument('--quick',action='store_true');a=x.parse_args();main(a.quick)
