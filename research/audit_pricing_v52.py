"""ACRO V5.2: independent full-premium repricing and fixed-coupon coupled-grid audit.

Run from the project folder:
    python audit_pricing_v52.py --quick
    python audit_pricing_v52.py

Research model and all product cash flows are unchanged from V5.1.
The direct benchmark performs a fresh conditional Phoenix Monte Carlo at each
sampled first-autocall market state; it never looks up/interpolates a grid.
A separate common-random-numbers (CRN) estimator improves precision but is
clearly labelled as an anchored control-variate estimate, NOT a fully independent
raw estimate. Finite inner samples introduce positive-part (Jensen) bias.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from statistics import mean
import argparse
import csv
import json
import time

import numpy as np
from scipy.stats import t as student_t

from acro import load_project, quote_initial, simulate_phoenix
from acro.pricing import conditional_curve
from research.audit_coupled import (innovations_for_grid, quote_from_bank, outer_from_bank,
                           renewal_differences, price_from_components)

ROOT = Path(__file__).resolve().parents[1]


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe_replicates(prices):
    """Across-replication Student interval; NOT a discretization/bias bound."""
    a = np.asarray(prices, dtype=float)
    if len(a) < 2:
        raise ValueError('At least two independent replications are necessary.')
    avg = float(a.mean())
    sd = float(a.std(ddof=1))
    se = sd / np.sqrt(len(a))
    tc = float(student_t.ppf(0.975, df=len(a) - 1))
    return dict(n=len(a), mean=avg, sd_replicates=sd, se_replicates=float(se),
                ci95_statistical=[avg - tc * se, avg + tc * se],
                min=float(a.min()), max=float(a.max()))


def conditional_direct_values(p, factor: float, coupon: float, n_inner: int,
                              steps_per_year: int, seed: int, anchor: float):
    """No grid, no interpolation, no use of the V5 curve.

    Independently estimates V_R(x) using fresh draw. The optional CRN estimate
    compares *matched* x and initial-factor paths with equal Gaussian shocks;
    its value is anchored to the separately estimated V_R(x0) = anchor.
    """
    renewed = simulate_phoenix(p.market, p.renewal_phoenix, n_inner,
                               steps_per_year, seed, initial_factor=factor)
    pv = renewed.pv(coupon)
    raw = float(pv.mean())
    half = n_inner // 2
    half1, half2 = float(pv[:half].mean()), float(pv[half:].mean())
    baseline = simulate_phoenix(p.market, p.renewal_phoenix, n_inner,
                                steps_per_year, seed,
                                initial_factor=p.market.initial_logvol_factor)
    anchored = float(anchor + (pv - baseline.pv(coupon)).mean())
    return raw, anchored, half1, half2


def fixed_coupon_grid(p, out: Path, quick: bool):
    if p.initial_phoenix != p.renewal_phoenix or p.acro.target_coupon_rate is not None:
        raise ValueError('Reference experiment needs same renewal contract, target coupon=None')
    n_assets = p.market.n_assets
    tenor = p.initial_phoenix.maturity_years
    finest = 48
    if int(tenor * finest) != tenor * finest:
        raise ValueError('Maturity does not fit 48-step grid')
    grids = (12, 24, 48)
    n_q, n_outer, n_inner, n_reference, n_nodes, reps = (
        (1700, 1600, 2000, 15000, 13, 3) if quick else
        (5500, 5000, 6000, 35000, 17, 6))
    # This reference coupon is sampled ONCE on the finest grid, independently
    # of the pricing banks. It stays identical across grids and replications.
    bank_ref = innovations_for_grid(int(tenor * finest), n_reference,
                                    n_assets, 757101)
    B_ref, A_ref, frozen_coupon = quote_from_bank(p, bank_ref, finest, finest)
    rows = []
    for rep in range(reps):
        seed = rep * 100003
        q_bank = innovations_for_grid(int(tenor * finest), n_q, n_assets, 11001 + seed)
        o_bank = innovations_for_grid(int(tenor * finest), n_outer, n_assets, 22001 + seed)
        i_bank = innovations_for_grid(int(tenor * finest), n_inner, n_assets, 33001 + seed)
        quotes, outers, factors = {}, {}, []
        for step in grids:
            quotes[step] = quote_from_bank(p, q_bank, step, finest)
            outers[step] = outer_from_bank(p, o_bank, step, finest)
            path, eligible = outers[step]
            factors.extend(path.factor_at_call[eligible].tolist())
        if not factors:
            raise ValueError('No eligible call states in grid experiment')
        nodes = np.unique(np.r_[np.linspace(min(factors) - .12,
                                             max(factors) + .12, n_nodes),
                                  p.market.initial_logvol_factor])
        for step in grids:
            B, A, c_fair = quotes[step]
            db, da = renewal_differences(p, i_bank, step, finest, nodes)
            path, eligible = outers[step]
            # Two estimates with identical paths, the *only* change is whether
            # coupon is recalculated at par (V5.1) or contractually frozen.
            dynamic = price_from_components(p, (B, A, c_fair), path,
                                            eligible, nodes, db, da)
            frozen = price_from_components(p, (B, A, frozen_coupon), path,
                                           eligible, nodes, db, da)
            rows.append(dict(rep=rep+1, steps_per_year=step,
                             recalculated_fair_coupon=c_fair,
                             reference_coupon=frozen_coupon,
                             price_recalculated_coupon=dynamic,
                             price_fixed_coupon=frozen,
                             eligible_call_fraction=float(eligible.mean())))
        print(f'[grid] paired repetition {rep+1}/{reps}', flush=True)
    save_csv(out/'paired_grids_fixed_coupon.csv', rows)
    deltas = []
    for rep in range(1, reps+1):
        d = {r['steps_per_year']: r for r in rows if r['rep'] == rep}
        deltas.append(dict(rep=rep,
                           fixed_48_minus_12=d[48]['price_fixed_coupon']-d[12]['price_fixed_coupon'],
                           fixed_48_minus_24=d[48]['price_fixed_coupon']-d[24]['price_fixed_coupon'],
                           dynamic_48_minus_12=d[48]['price_recalculated_coupon']-d[12]['price_recalculated_coupon'],
                           dynamic_48_minus_24=d[48]['price_recalculated_coupon']-d[24]['price_recalculated_coupon'],
                           coupon_48_minus_12=d[48]['recalculated_fair_coupon']-d[12]['recalculated_fair_coupon']))
    save_csv(out/'paired_grid_differences.csv', deltas)
    stats = {col: describe_replicates([r[col] for r in deltas])
             for col in ('fixed_48_minus_12', 'fixed_48_minus_24',
                         'dynamic_48_minus_12', 'dynamic_48_minus_24')}
    return dict(frozen_coupon=frozen_coupon, reference_quote_B=B_ref,
                reference_quote_A=A_ref, reference_steps=48,
                n_reference_paths=n_reference, repetitions=reps,
                paths_quote_per_rep=n_q, paths_outer_per_rep=n_outer,
                paths_inner_per_rep=n_inner, statistics=stats)


def direct_full_premium(p, out: Path, quick: bool):
    # Each repetition: fresh issue quote, fresh outer paths, a fresh gridded
    # conditional curve (V5 method), and *fresh direct nested paths per state*.
    # Selected eligible call states are a uniform random sample of the whole
    # eligible pool; the estimator includes observed call probability.
    if p.initial_phoenix != p.renewal_phoenix or p.acro.target_coupon_rate is not None:
        raise ValueError('Reference audit assumes same renewal contract and coupon')
    steps = 24
    n_quote, n_outer, n_curve, n_direct, n_selected, n_reps, n_nodes = (
        (4000, 3200, 3500, 1200, 85, 4, 13) if quick else
        (11000, 8500, 12000, 2400, 140, 7, 19))
    rows = []
    summary_rows = []
    for rep in range(n_reps):
        seed = 800009 + rep * 200071
        sp = replace(p.simulation, steps_per_year=steps,
                     quote_paths=n_quote, outer_paths=n_outer,
                     conditional_paths=n_curve, curve_nodes=n_nodes, seed=seed)
        pr = replace(p, simulation=sp)
        quote = quote_initial(pr)
        target = quote.coupon_rate
        outer = simulate_phoenix(pr.market, pr.initial_phoenix,
                                n_outer, steps, seed + 20)
        mask = np.isfinite(outer.first_call_year)
        if pr.acro.last_eligible_call_year is not None:
            mask &= outer.first_call_year <= pr.acro.last_eligible_call_year + 1e-12
        idx = np.flatnonzero(mask)
        if not len(idx):
            raise ValueError('No eligible autocalls: cannot estimate price')
        pick = np.random.default_rng(seed + 707).choice(
            idx, size=min(n_selected, len(idx)), replace=False)
        factors = outer.factor_at_call[mask]
        curve = conditional_curve(pr, factors, quote)
        B_curve, A_curve = curve.at(outer.factor_at_call[mask])
        total_call_discount = np.exp(-pr.market.risk_free_rate*outer.first_call_year[mask])
        full_curve_price = float(np.sum(
            total_call_discount * pr.acro.renewal_units *
            np.maximum(B_curve + target*A_curve-pr.acro.strike,0)) / n_outer)
        curve_price_selected = 0.0
        raw_price = 0.0
        cv_price = 0.0
        half_price = 0.0
        for j, k in enumerate(pick):
            x = float(outer.factor_at_call[k])
            discount = float(np.exp(-pr.market.risk_free_rate * outer.first_call_year[k]))
            direct_raw, direct_cv, h1, h2 = conditional_direct_values(
                pr, x, target, n_direct, steps, seed + 100000 + k*23, quote.price)
            b_g, a_g = curve.at(np.asarray([x]))
            val_curve = float(b_g[0] + target*a_g[0])
            # Eligible-state stratification: P(call) times conditional mean
            weight = (len(idx) / n_outer / len(pick)) * discount * pr.acro.renewal_units
            raw_price += weight * max(direct_raw - pr.acro.strike, 0)
            cv_price += weight * max(direct_cv - pr.acro.strike, 0)
            half_price += weight * .5 * (max(h1-pr.acro.strike,0)+max(h2-pr.acro.strike,0))
            curve_price_selected += weight * max(val_curve-pr.acro.strike,0)
            rows.append(dict(rep=rep+1, outer_scenario=int(k), call_year=float(outer.first_call_year[k]),
                             logvol_factor=x, weight=weight, initial_coupon=target,
                             raw_renewal_value=direct_raw, crn_renewal_value=direct_cv,
                             curve_renewal_value=val_curve, direct_raw_payoff=max(direct_raw-pr.acro.strike,0),
                             curve_payoff=max(val_curve-pr.acro.strike,0)))
        summary_rows.append(dict(rep=rep+1, coupon=target, eligible_call_probability=float(mask.mean()),
                                 n_eligible=int(mask.sum()), n_selected=len(pick),
                                 price_v51_curve_full_outer=full_curve_price,
                                 price_v51_curve_on_selected=curve_price_selected,
                                 price_direct_raw_nested=raw_price,
                                 price_direct_anchored_crn=cv_price,
                                 jensen_half_minus_full_raw=max(0.,half_price-raw_price),
                                 direct_raw_minus_curve_on_selected=raw_price-curve_price_selected,
                                 direct_anchored_minus_curve_on_selected=cv_price-curve_price_selected))
        print(f'[direct nested] independent repetition {rep+1}/{n_reps}: '
              f'raw={raw_price:.4f} anchored={cv_price:.4f} '
              f'curve(sample)={curve_price_selected:.4f}', flush=True)
    save_csv(out/'direct_selected_states.csv', rows)
    save_csv(out/'independent_premium_replications.csv', summary_rows)
    fields = ('price_direct_raw_nested', 'price_direct_anchored_crn',
              'price_v51_curve_on_selected', 'price_v51_curve_full_outer',
              'direct_raw_minus_curve_on_selected',
              'direct_anchored_minus_curve_on_selected', 'jensen_half_minus_full_raw')
    stats = {f: describe_replicates([row[f] for row in summary_rows]) for f in fields}
    return dict(repetitions=n_reps, steps_per_year=steps,
                paths_quote=n_quote, paths_outer=n_outer, paths_curve_per_node=n_curve,
                paths_direct_per_selected_state=n_direct,
                max_selected_eligible_states=n_selected,
                statistics=stats)


def run(config: str, output: str, quick: bool):
    p = load_project(config)
    if p.market.dynamics != 'stochastic_vol':
        raise ValueError('This audit targets the stochastic-vol reference configuration')
    output_dir = Path(output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fixed = fixed_coupon_grid(p, output_dir, quick)
    direct = direct_full_premium(p, output_dir, quick)
    report = dict(project='ACRO V5.2, synthetic Q dynamics unchanged',
                  config=str(Path(config).resolve()), quick=quick,
                  fixed_coupon_grid=fixed, direct_nested_full_premium=direct,
                  runtime_seconds=time.perf_counter()-started,
                  important_limitations=[
                      'Confidence intervals are statistical intervals across independent finite-budget estimators, not an interval for the bias-free continuous-time price.',
                      'The raw direct nested estimator evaluates max of a noisy conditional MC mean; it has positive Jensen bias even though no interpolation is used.',
                      'The anchored direct repricer uses CRN and the issue quote as a control variate; it is not the fully independent raw benchmark.',
                      'Outer eligible-state subsampling adds sampling variance and only estimates the full premium, not the result of evaluating every external scenario.',
                      'Fixed coupon grid test isolates contractual-coupon recalibration, not all sources of numerical discretization error.',
                      'No market calibration, stochastic stock-stock correlation, issuer credit, hedge costs, or empirical proof of economic value.'
                  ])
    (output_dir/'AUDIT_PRICING_V5_2.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    ds=direct['statistics']; gs=fixed['statistics']
    f=lambda v: f'{v:.4f}'
    def interval(z):
        return f"{f(z['mean'])} [{f(z['ci95_statistical'][0])} ; {f(z['ci95_statistical'][1])}]"
    lines=[
        '# ACRO V5.2 — Audit de la prime complète', '',
        '**Contrôles réalisés sur les hypothèses synthétiques de la V5.1. Aucun changement des payoffs ou de la dynamique de marché.**', '',
        '## 1. Second test de discrétisation, coupon contractuel fixé', '',
        f"Coupon gelé à la valeur estimée sur 48 pas/an, avec {fixed['n_reference_paths']} chemins de cotation indépendants : **{fixed['frozen_coupon']*100:.3f} %**.",
        'Ce coupon est utilisé **sans recalibrage** sur les grilles 12/24/48, avec chocs browniens couplés.', '',
        '| Différence appariée | Moyenne [IC statistique 95 %] en €/100 |', '|---|---:|',
        f"| Coupon fixé, 48−12 | {interval(gs['fixed_48_minus_12'])} |",
        f"| Coupon fixé, 48−24 | {interval(gs['fixed_48_minus_24'])} |",
        f"| Coupon recalibré, 48−12 | {interval(gs['dynamic_48_minus_12'])} |",
        f"| Coupon recalibré, 48−24 | {interval(gs['dynamic_48_minus_24'])} |",
        '', 'La comparaison évite de confondre mécaniquement la révision du coupon au pair avec le changement de grille. Les intervalles ne sont **pas** des bornes du biais temporel.',
        '', '## 2. Repricing de la prime entière sans interpolation', '',
        'Sur chaque réplication : cotation initiale indépendante, trajectoires externes nouvelles, sélection aléatoire uniforme de scénarios d’autocall admissible, puis **nouvelle simulation Monte Carlo du Phoenix renouvelé à chaque état sélectionné**.',
        'Les pondérations réintègrent la probabilité de rappel admissible et l’actualisation. Les deux variantes directes ne consultent aucune courbe interpolée.', '',
        '| Estimateur | Moyenne [IC statistique 95 %] en €/100 |', '|---|---:|',
        f"| Direct brut, référence indépendante | {interval(ds['price_direct_raw_nested'])} |",
        f"| Direct avec réduction de variance CRN | {interval(ds['price_direct_anchored_crn'])} |",
        f"| V5.1, courbe sur mêmes scénarios | {interval(ds['price_v51_curve_on_selected'])} |",
        f"| V5.1, courbe sur tous scénarios externes | {interval(ds['price_v51_curve_full_outer'])} |",
        f"| Écart direct brut − courbe sur mêmes scénarios | {interval(ds['direct_raw_minus_curve_on_selected'])} |",
        f"| Écart direct CRN − courbe sur mêmes scénarios | {interval(ds['direct_anchored_minus_curve_on_selected'])} |",
        '', f"Budget : {direct['repetitions']} réplications, {direct['paths_quote']} cotations, {direct['paths_outer']} chemins externes, "
           f"{direct['paths_direct_per_selected_state']} continuations internes par état, au plus {direct['max_selected_eligible_states']} états d’autocall par réplication.",
        '', '### Biais de Jensen',
        f"La différence moyenne (payoff des deux demi-échantillons − payoff plein échantillon) est {f(ds['jensen_half_minus_full_raw']['mean'])} €/100. Ce diagnostic est **indicatif**, pas une borne du biais pour une simulation infinie.",
        '', '## 3. Ce que les chiffres permettent et ne permettent pas de conclure', '',
        '- Une **différence statistiquement résolue** entre le repricing direct et la courbe sur les mêmes états signalerait un problème de biais de méthode ou de budget. Une différence non résolue ne prouve pas leur équivalence exacte.',
        '- Le repricing brut est indépendant de la courbe, mais peut avoir un biais positif de Jensen ; l’estimateur CRN améliore la variance sans constituer une vérité de marché.',
        '- Aucun intervalle ci-dessus ne contient par construction l’incertitude de calibration, de modèle ou la limite du pas de temps nul.',
        '- Ne pas annoncer une valeur unique stabilisée si les intervalles restent larges ou si les différences entre 24 et 48 pas ne sont pas résolues.',
        '', '## 4. Fichiers', '',
        '`paired_grids_fixed_coupon.csv`, `paired_grid_differences.csv`, `independent_premium_replications.csv`, `direct_selected_states.csv`, `AUDIT_PRICING_V5_2.json`.'
    ]
    (output_dir/'AUDIT_PRICING_V5_2_FR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Finished in {time.perf_counter()-started:.1f} s; report at {output_dir}', flush=True)
    return report


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(ROOT/'config.json'))
    ap.add_argument('--output', default=str(ROOT/'results_audit_v52'))
    ap.add_argument('--quick', action='store_true')
    a = ap.parse_args()
    run(a.config, a.output, a.quick)
