"""Reproduce two SEPARATE ACRO premium experiments, not a market quote.

python -m research.reproduce_premium --quick
python -m research.reproduce_premium --standard

A re-estimates the fair coupon in every bank. B holds the contractual coupon
fixed at 18%; differences between A and B are NOT a convergence test.
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
from scipy.stats import t
from acro import load_project
from acro.controlled_pricing import price_acro_controlled
from acro.total_premium_validation import run_bank

ROOT = Path(__file__).resolve().parents[1]


def interval(values):
    """Approximate Student-t interval across independent whole-pipeline runs."""
    v = np.asarray(values, dtype=float)
    if len(v) < 3 or not np.all(np.isfinite(v)):
        raise ValueError("Need at least three finite independent banks")
    mean = float(v.mean())
    se = float(v.std(ddof=1) / np.sqrt(len(v)))
    margin = float(t.ppf(.975, len(v)-1) * se)
    return {"mean": mean, "bank_se": se, "ci95": [mean-margin, mean+margin]}


def save_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(*, quick=True, output=ROOT / "results_premium_validation"):
    p = load_project(ROOT / "config.json")
    if p.market.dynamics != "stochastic_vol" or p.initial_coupon_policy != "fair":
        raise ValueError("Reference configuration must use stochastic_vol and fair coupon")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    levels = ([("small", 1600, 1200, 7, 512), ("large", 4500, 3500, 11, 2048)]
              if quick else
              [("small", 5000, 4000, 11, 2048), ("large", 16000, 12000, 17, 8192)])
    n_banks = 4 if quick else 8

    fair_rows = []
    for level, quote, outer, nodes, ref in levels:
        for i in range(n_banks):
            sim = replace(p.simulation, quote_paths=quote, outer_paths=outer,
                          conditional_paths=quote, curve_nodes=nodes,
                          seed=610001 + (0 if level == "small" else 100000) + i*7919)
            result, _ = price_acro_controlled(replace(p, simulation=sim),
                                              ref_paths=ref, scrambles=4)
            fair_rows.append({"budget":level, "bank":i+1, "seed":sim.seed,
                "quote_paths":quote, "outer_paths":outer, "curve_nodes":nodes,
                "ref_paths":ref, "par_coupon":result.quote.coupon_rate,
                "premium":result.price, "outer_se_only":result.outer_se})
            print(f"Fair {level} {i+1}/{n_banks}: {result.price:.4f}", flush=True)
    save_rows(output / "fair_coupon_banks.csv", fair_rows)

    fixed = replace(p, initial_coupon_policy="fixed",
                    initial_phoenix=replace(p.initial_phoenix, coupon_rate=.18),
                    renewal_phoenix=replace(p.renewal_phoenix, coupon_rate=.18))
    grids = (12, 48) if quick else (12, 48, 96)
    direct_rows = []
    for steps in grids:
        trial = replace(fixed, simulation=replace(fixed.simulation, steps_per_year=steps))
        for i in range(n_banks):
            seed = 830017 + steps*100003 + i*7919
            result, _ = run_bank(trial, seed=seed,
                outer_paths=1600 if quick else 9000,
                selected_states=24 if quick else 110,
                inner_paths=384 if quick else 2048,
                pilot_inner=768 if quick else 4096,
                ref_paths=1024 if quick else 4096,
                scrambles=4, pilot_nodes=11 if quick else 21, margin=.6)
            direct_rows.append({"steps_per_year":steps, "bank":i+1, "seed":seed,
                "coupon":.18, "lower":result["direct_lower"],
                "upper":result["direct_upper"],
                "exercise_gap":result["exercise_policy_gap"],
                "elapsed_seconds":round(result["time_seconds"],3)})
            print(f"Fixed 18% {steps} steps {i+1}/{n_banks}: "
                  f"[{result['direct_lower']:.4f}, {result['direct_upper']:.4f}]",
                  flush=True)
    save_rows(output / "direct_fixed18_banks.csv", direct_rows)

    fair_summary = {}
    for level, *_ in levels:
        subset = [r for r in fair_rows if r["budget"] == level]
        fair_summary[level] = {
            "premium":interval([r["premium"] for r in subset]),
            "coupon":interval([r["par_coupon"] for r in subset])}
    direct_summary = {}
    for steps in grids:
        subset = [r for r in direct_rows if r["steps_per_year"] == steps]
        lower = interval([r["lower"] for r in subset])
        upper = interval([r["upper"] for r in subset])
        direct_summary[str(steps)] = {"lower":lower, "upper":upper,
            "midpoint":(lower["mean"]+upper["mean"])/2,
            "approximate_enclosure":[lower["ci95"][0],upper["ci95"][1]]}
    summary = {"scope":"Synthetic risk-neutral discrete model, NOT a market quote",
        "mode":"quick" if quick else "standard", "independent_banks":n_banks,
        "fair_coupon_controlled":fair_summary, "fixed_18pct_direct":direct_summary,
        "grid_comparison_brownian_coupled":False,
        "total_pricing_error_bounded":False,
        "inner_and_interpolation_error_fully_bounded":False,
        "continuous_time_limit_proved":False}
    (output / "summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    fair_table = "\n".join(
        f'| {k} | {v["coupon"]["mean"]:.2%} | {v["premium"]["mean"]:.4f} | '
        f'[{v["premium"]["ci95"][0]:.4f}, {v["premium"]["ci95"][1]:.4f}] |'
        for k,v in fair_summary.items())
    direct_table = "\n".join(
        f'| {k} | {v["lower"]["mean"]:.4f} | {v["upper"]["mean"]:.4f} | '
        f'[{v["approximate_enclosure"][0]:.4f}, {v["approximate_enclosure"][1]:.4f}] |'
        for k,v in direct_summary.items())
    report = f"""# ACRO premium validation — {summary['mode']} protocol

All amounts are EUR per EUR 100 initial nominal in a **synthetic Q model**.
The {n_banks} independent banks repeat the whole estimator with new seeds.
Confidence intervals use an approximate Student-t bank diagnostic; **quick**
mode is deliberately low-budget and not a precision certificate.

## A: coupon re-quoted at par in every bank

| Budget | Mean par coupon | Mean premium | Approx. 95% bank CI |
|---|---:|---:|---:|
{fair_table}

## B: contractual coupon fixed at 18%

Direct exercise-policy bounds evaluate renewal values at selected autocall
states without final interpolation. Independent inner valuations diagnose
positive-part and exercise-decision bias.

| Steps/year | Mean lower | Mean upper | Approx. bank enclosure |
|---|---:|---:|---:|
{direct_table}

**Do not compare A and B as a convergence test:** A re-quotes the coupon,
B keeps it fixed. B's time grids are **not Brownian-coupled** and their
differences are noisy. Approximate bank intervals are not rigorous bounds on
interpolation, inner sampling, continuous-time discretization, model risk,
market calibration, issuer credit, financing or hedging.

See `summary.json` and CSVs for seeds, simulation budgets and individual results.
"""
    (output / "README.md").write_text(report,encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="default: fast diagnostic")
    group.add_argument("--standard", action="store_true", help="larger simulation budgets")
    parser.add_argument("--output", type=Path, default=ROOT / "results_premium_validation")
    args = parser.parse_args()
    run(quick=not args.standard,output=args.output)
