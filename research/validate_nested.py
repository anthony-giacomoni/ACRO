"""Independent conditional-Phoenix valuation check.

Select autocall volatility-factor states, compare interpolation against a fresh
independent Monte Carlo pricing run for the NEW Phoenix at the SAME state.

python validate_nested.py --quick
python validate_nested.py

This checks interpolation / conditional bias on selected states, not global
nested option bias, hedge accuracy or calibration.
"""
from dataclasses import replace
from pathlib import Path
import argparse, csv
import numpy as np
from acro import load_project,quote_initial,price_acro,simulate_phoenix
ROOT=Path(__file__).resolve().parents[1]


def main(quick=False):
    p=load_project(ROOT/'config.json')
    if quick:
        p=replace(p,simulation=replace(p.simulation,quote_paths=2500,outer_paths=1800,
                                        conditional_paths=1000,curve_nodes=7))
    result=price_acro(p)
    states=simulate_phoenix(p.market,p.initial_phoenix,p.simulation.outer_paths,
                            p.simulation.steps_per_year,p.simulation.seed+20)
    eligible=np.isfinite(states.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= states.first_call_year <= p.acro.last_eligible_call_year+1e-9
    if not eligible.any():
        raise RuntimeError('No eligible autocalls: nothing to validate.')
    factors=np.quantile(states.factor_at_call[eligible],[.05,.25,.50,.75,.95])
    independent_inner=3000 if quick else 14000
    rows=[]
    for j,x in enumerate(factors):
        B,A=result.curve.at(np.array([x]))
        interpolated=float(B[0]+result.target_coupon*A[0])
        continuation=simulate_phoenix(p.market,p.renewal_phoenix,independent_inner,
                  p.simulation.steps_per_year,seed=p.simulation.seed+910+j,
                  initial_factor=float(x))
        sample=continuation.pv(result.target_coupon)
        ref=float(sample.mean())
        se=float(sample.std(ddof=1)/np.sqrt(len(sample)))
        rows.append(dict(quantile=['05','25','50','75','95'][j],logvol_factor=float(x),
                         curve_value=interpolated,direct_mc_value=ref,
                         direct_mc_se=se,difference=interpolated-ref,
                         diff_in_reference_95ci=abs(interpolated-ref) <= 1.96*se))
    folder=ROOT/'results';folder.mkdir(exist_ok=True)
    with open(folder/'nested_validation.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    print('Independent conditional valuation check (NOT complete pricing CI):')
    for row in rows:
        print(row)
    print('Output:',folder/'nested_validation.csv')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--quick',action='store_true')
    main(parser.parse_args().quick)
