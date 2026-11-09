"""Reproducible numerical counter-audit for experimental annual correlation ACRO.

Run from ACRO_GitHub_Final:
    python audit_annual_correlation_precision.py --quick
    python audit_annual_correlation_precision.py --full

Reports paired frozen-vs-stochastic-correlation premiums for fixed contractual
coupon, several inner simulation budgets, grids and independent seeds. Also
compares 2D interpolation against fresh, directly simulated conditional values
at selected autocall market states. This is an *accuracy diagnostic*, NOT a
calibration, a market quote or a proof of convergence.

The full profile can be computationally expensive. The quick profile is a
pipeline smoke check, not sufficient evidence of financial price precision.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from acro.config import load_project
from acro.annual_correlation import (
    AnnualCorrelation, price_acro_annual_correlations, simulate_phoenix_annual
)
from acro.pricing import same_contract

ROOT = Path(__file__).resolve().parents[1]


def _csv(path: Path, rows: list[dict]):
    if rows:
        with path.open('w', encoding='utf-8', newline='') as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


def _direct_conditional(project, regime, result, states, inner_paths, seed):
    """Compare surface-interpolated renewal values with CRN direct inner MC.

    The same fresh random innovation bank is applied to reference and state,
    reducing sampling noise of their price difference. *This does not remove
    anchor quote uncertainty or the positive-part / full outer bias*.
    """
    market = project.market
    same = same_contract(project.initial_phoenix, project.renewal_phoenix)
    if project.acro.target_coupon_rate is None:
        coupon = result.coupon_rate
    else:
        coupon = project.acro.target_coupon_rate
    x0 = market.initial_logvol_factor if market.dynamics == 'stochastic_vol' else 0.
    y0 = regime.initial_state
    base = None
    if same:
        base = simulate_phoenix_annual(market, project.renewal_phoenix, regime,
                    inner_paths, project.simulation.steps_per_year, seed,
                    initial_factor=x0, initial_corr_state=y0)
        # The original quote was computed at the contractual coupon, which may
        # differ from the renewal target coupon when an override is specified.
        if project.acro.target_coupon_rate is None:
            anchor = result.initial_note_price
        else:
            q = simulate_phoenix_annual(market, project.initial_phoenix, regime,
                    project.simulation.quote_paths, project.simulation.steps_per_year,
                    project.simulation.seed + 10)
            anchor = float(np.mean(q.pv(coupon)))
    rows = []
    for label, x, y in states:
        pred_b, pred_a = result.surface.at([x], [y])
        prediction = float(pred_b[0] + coupon * pred_a[0])
        sim = simulate_phoenix_annual(market, project.renewal_phoenix, regime,
                    inner_paths, project.simulation.steps_per_year, seed,
                    initial_factor=x, initial_corr_state=y)
        if same:
            diff = sim.pv(coupon) - base.pv(coupon)
            direct = float(anchor + np.mean(diff))
        else:
            diff = sim.pv(coupon)
            direct = float(np.mean(diff))
        se = float(np.std(diff, ddof=1) / np.sqrt(inner_paths))
        rows.append(dict(state=label, x_logvol=x, y_correlation=y,
                         grid_value_eur=prediction, direct_value_eur=direct,
                         difference_eur=prediction-direct, direct_mc_se_eur=se,
                         direct_paths=inner_paths,
                         near_option_boundary=abs(direct-project.acro.strike)<2.))
    return rows


def run(*, quick=False, full=False, output: Path | None = None,
        direct=True):
    project = load_project(ROOT / 'config.json')
    if quick:
        seeds = (3117, 1214)
        grids = ((5, 5), (7, 5))
        budgets = (350, 1100)
        quote, outer, direct_paths = 1300, 800, 2500
    elif full:
        seeds = (3117, 1214, 2119, 4111)
        grids = ((7, 5), (9, 7), (13, 9))
        budgets = (8500, 25000, 35000)
        quote, outer, direct_paths = 45000, 14000, 70000
    else:
        seeds = (3117, 1214, 2119)
        grids = ((7, 5), (9, 7))
        budgets = (2500, 8500, 25000)
        quote, outer, direct_paths = 25000, 9000, 25000
    out = output or ROOT / 'evidence' / 'annual_correlation_counteraudit'
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    direct_rows = []
    coupon = 0.182  # identical contractual coupon, same contract comparison
    for grid in grids:
        for budget in budgets:
            for seed in seeds:
                p = replace(project, simulation=replace(project.simulation,
                        quote_paths=quote, outer_paths=outer, conditional_paths=budget,
                        seed=seed))
                paired = {}
                for vol in (0., .5):
                    regime = AnnualCorrelation(volatility=vol)
                    start = time.perf_counter()
                    result = price_acro_annual_correlations(
                        p, regime, logvol_nodes=grid[0], corr_nodes=grid[1],
                        fixed_coupon=coupon)
                    elapsed = time.perf_counter()-start
                    row = dict(seed=seed, logvol_nodes=grid[0], corr_nodes=grid[1],
                               conditional_paths=budget, model='frozen' if vol==0 else 'annual_OU',
                               annual_corr_sigma=vol, coupon_fixed_pct=100*coupon,
                               initial_phoenix_price_eur=result.initial_note_price,
                               fair_coupon_pct=100*result.fair_coupon_rate,
                               premium_eur=result.acro_premium,
                               outer_only_se_eur=result.outer_se_only,
                               quote_paths=quote, outer_paths=outer,
                               elapsed_s=round(elapsed, 3))
                    rows.append(row)
                    paired[vol] = result.acro_premium
                    print(f"{grid=} {budget=} {seed=} corr_sigma={vol} "
                          f"ACRO={result.acro_premium:.5f} time={elapsed:.1f}s", flush=True)
                    if direct and vol==.5 and budget==max(budgets) and grid==grids[-1]:
                        outer_paths = result.outer_paths
                        sel = np.isfinite(outer_paths.first_call_year)
                        xv = outer_paths.factor_at_call[sel]
                        yv = outer_paths.correlation_at_call[sel]
                        if len(xv):
                            vv = (result.surface.at(xv,yv)[0]
                                +coupon*result.surface.at(xv,yv)[1])
                            k = int(np.argmin(np.abs(vv-p.acro.strike)))
                            hi = int(np.argmax(xv+yv))
                            lo = int(np.argmin(xv+yv))
                            positions = [('near_exercise_boundary',k),
                                         ('combined_high_tail',hi),
                                         ('combined_low_tail',lo)]
                            states = [(label,float(xv[j]),float(yv[j])) for label,j in positions]
                            results = _direct_conditional(p,regime,result,states,
                                      direct_paths,seed+902_100)
                            direct_rows.extend(dict(seed=seed, logvol_nodes=grid[0],
                                 corr_nodes=grid[1], conditional_paths=budget,**r)
                                 for r in results)
                rows.append(dict(seed=seed, logvol_nodes=grid[0],corr_nodes=grid[1],
                                 conditional_paths=budget, model='paired_delta',
                                 annual_corr_sigma=.5,coupon_fixed_pct=100*coupon,
                                 initial_phoenix_price_eur='',fair_coupon_pct='',
                                 premium_eur=paired[.5]-paired[0.],outer_only_se_eur='',
                                 quote_paths=quote,outer_paths=outer,elapsed_s=''))
    _csv(out/'raw_prices_and_paired_deltas.csv',rows)
    _csv(out/'direct_state_checks.csv',direct_rows)
    summary=[]
    for grid in grids:
        for budget in budgets:
            deltas = [r['premium_eur'] for r in rows if r['model']=='paired_delta'
                      and (r['logvol_nodes'],r['corr_nodes'])==grid
                      and r['conditional_paths']==budget]
            mean=float(np.mean(deltas))
            se=float(np.std(deltas,ddof=1)/np.sqrt(len(deltas)))
            summary.append(dict(logvol_nodes=grid[0],corr_nodes=grid[1],
                                conditional_paths=budget,n_seeds=len(deltas),
                                mean_paired_delta_eur=mean,seed_between_se_eur=se,
                                observed_min_eur=float(np.min(deltas)),
                                observed_max_eur=float(np.max(deltas))))
    _csv(out/'paired_delta_summary.csv',summary)
    meta = dict(profile='quick' if quick else 'full' if full else 'standard',
                fixed_contract_coupon=0.182,quote_paths=quote,outer_paths=outer,
                seeds=list(seeds),grids=[list(g) for g in grids],
                conditional_budgets=list(budgets),direct_paths=direct_paths,
                interpretation='Synthetic uncalibrated risk neutral model; outer SE excludes inner MC, 2D interpolation and Jensen positive-part biases. Seed-to-seed SE is NOT total error.')
    (out/'assumptions.json').write_text(json.dumps(meta,indent=2,ensure_ascii=False),
                                        encoding='utf-8')
    print(f"Saved counter-audit data: {out}",flush=True)
    return summary,direct_rows


if __name__=='__main__':
    p=argparse.ArgumentParser()
    group=p.add_mutually_exclusive_group()
    group.add_argument('--quick',action='store_true')
    group.add_argument('--full',action='store_true')
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--no-direct',action='store_true')
    a=p.parse_args()
    run(quick=a.quick,full=a.full,output=a.output,direct=not a.no_direct)
