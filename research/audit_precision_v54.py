"""ACRO V5.4: fixed coupon quote, 1D CRN curve, 48/96/192 coupled grids.

This audit is *not* a market calibration or a proof of continuum convergence.
Main V5.3 pricing/payoff/dynamics modules are deliberately unmodified.

Run: python audit_precision_v54.py --quick
     python audit_precision_v54.py              # larger budget
     python audit_precision_v54.py --high        # expensive
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t
from acro import load_project, simulate_phoenix
from research.audit_coupled import (innovations_for_grid, quote_from_bank, outer_from_bank,
                           renewal_differences, price_from_components)

ROOT = Path(__file__).resolve().parents[1]
GRIDS = (48, 96, 192)
TOL = 0.02  # EUR per EUR 100 Phoenix nominal


def budgets(mode):
    if mode == 'quick':
        return dict(quote_batches=2, quote_per_batch=1400, reps=3, quote=550,
                    outer=850, inner=550, nodes=7, direct_states=12,
                    direct_inner=768, heldout=5)
    if mode == 'standard':
        return dict(quote_batches=4, quote_per_batch=4200, reps=4, quote=1400,
                    outer=2500, inner=1800, nodes=11, direct_states=28,
                    direct_inner=2048, heldout=9)
    if mode == 'high':
        return dict(quote_batches=8, quote_per_batch=13000, reps=8, quote=4800,
                    outer=7200, inner=8000, nodes=21, direct_states=90,
                    direct_inner=8192, heldout=19)
    raise ValueError(mode)


def ci(xs):
    x = np.asarray(xs, float)
    if x.ndim != 1 or len(x) < 2 or not np.isfinite(x).all():
        raise ValueError('Need at least two finite independent replications')
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    delta = float(student_t.ppf(.975, len(x)-1) * se)
    m = float(x.mean())
    return dict(mean=m, ci95=[m-delta,m+delta], halfwidth=delta,
                standard_error=se, n=len(x))


def save_csv(path, rows):
    if not rows: return
    with open(path, 'w', encoding='utf-8', newline='') as fp:
        wr = csv.DictWriter(fp, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)


def validate_scope(p):
    # The 1D CRN anchor below relies on identical contracts and no coupon overrides.
    if p.market.dynamics != 'stochastic_vol':
        raise ValueError('V5.4 numerical audit currently targets the 1-factor stochastic-volatility setup')
    if p.initial_phoenix != p.renewal_phoenix or p.acro.target_coupon_rate is not None:
        raise ValueError('V5.4 numerical audit requires identical initial/renewal notes and target_coupon_rate=null')
    if p.initial_coupon_policy != 'fair':
        raise ValueError('V5.4 quote audit requires a fair coupon')
    if p.initial_phoenix.maturity_years != p.renewal_phoenix.maturity_years:
        raise ValueError('Both notes must have the same tenor for coupled innovation banks')
    for g in GRIDS: p.initial_phoenix.validate(g)


def frozen_coupon(p, b, output):
    """One reference c=(N-mean B)/mean A at 192 steps/year, independent of audit replications.

    Delta-method SE from discounted per-path pricing residuals at the pooled c.
    That is a *sampling* SE for the coupon of the discrete 192-step model,
    NOT an estimate of its discretization or calibration error.
    """
    Bs = []; As = []; samples = []
    for k in range(b['quote_batches']):
        sim = simulate_phoenix(p.market, p.initial_phoenix, b['quote_per_batch'],
                               192, 541001 + k * 100019)
        Bs.append(sim.principal_pv)
        As.append(sim.coupon_annuity_pv)
        n = b['quote_per_batch'] * (k+1)
        cb = (p.initial_phoenix.nominal - np.concatenate(Bs).mean()) / np.concatenate(As).mean()
        samples.append(dict(batch=k+1, n_paths=n, coupon=cb))
    vB = np.concatenate(Bs); vA = np.concatenate(As)
    ann = float(vA.mean())
    if ann <= 0: raise ValueError('Cannot determine fair coupon: nonpositive annuity')
    coupon = float((p.initial_phoenix.nominal-vB.mean()) / ann)
    residual = vB + coupon*vA - p.initial_phoenix.nominal
    coupon_se = float(residual.std(ddof=1)/ann/np.sqrt(len(vB)))
    half = float(1.96 * coupon_se)
    save_csv(output/'coupon_reference_running.csv',samples)
    return coupon, float(vB.mean()), ann, dict(coupon=coupon, n_paths=len(vB), steps_per_year=192,
                        delta_method_se=coupon_se, approx_95_ci=[coupon-half,coupon+half],
                        approx_95_halfwidth=half,
                        disclaimer='Approximate asymptotic sampling CI at grid 192; not a bound on time discretization')


def factor_grid(arrays, n_nodes, center):
    valid = [np.asarray(x) for x in arrays if len(x)]
    if not valid: raise RuntimeError('No eligible first calls')
    allx = np.concatenate(valid)
    lo = float(min(allx.min()-.07,center-.05))
    hi = float(max(allx.max()+.07,center+.05))
    return np.unique(np.r_[np.linspace(lo,hi,n_nodes),center])


def jensen_bound(note_max:float, n:int) -> float:
    """Uniform bound sigma/sqrt(n) <= note_max/(2 sqrt(n)), via Popoviciu.

    Applies to unbiased i.i.d. direct continuation PVs in [0, note_max].
    Does NOT bound interpolation, coupon estimation, or finite outer MC.
    """
    if n < 1 or note_max < 0: raise ValueError('Invalid Jensen bound inputs')
    return note_max / (2 * np.sqrt(n))


def note_pv_max(p, coupon):
    # For memory/no-memory and any autocall, at most nominal is redeemed,
    # and at most nominal*c*T can be paid in total conditional coupons.
    T = p.renewal_phoenix.maturity_years
    return float(p.renewal_phoenix.nominal * (1 + max(0.,coupon)*T)
                 * np.exp(max(0.,-p.market.risk_free_rate*T)))


def crossfit_bounds(a, d, strike):
    '''Cross-fit policy lower and convex plug-in upper, in EXPECTATION.

    a and d are independent unbiased estimates of the SAME conditional renewal
    value, computed from disjoint i.i.d. inner continuation samples.
    E[L] <= (V-K)+ <= E[U]. This is NOT a sure pathwise price bound.
    '''
    low=0.5*((d-strike)*float(a>strike)+(a-strike)*float(d>strike))
    up=max((a+d)/2-strike,0.)
    if low > up + 1e-9: raise AssertionError('Crossfit pathwise order failed')
    return float(low),float(up)


def direct_states(p, c, outer, eligible, q, nodes, db, da, rep, b):
    """Reprice renewed Phoenix without interpolation using independent paths.

    Compare direct and interpolated *at identical, randomly selected outer states*.
    Paired-state comparison isolates inner/interpolation noise from selecting
    different outer scenarios. The 'raw' direct estimator retains Jensen bias.
    """
    choices = np.flatnonzero(eligible)
    nsel = min(b['direct_states'],len(choices))
    if not nsel: raise RuntimeError('No eligible states')
    picks = np.random.default_rng(544800+rep*11239).choice(choices,nsel,replace=False)
    qB,qA,_=q
    weighted_direct = 0.; weighted_interp = 0.; weighted_lower = 0.; weighted_gap = 0.
    weighted_jensen = 0.; weighted_sigma = 0.
    rows=[]
    inner_n=b['direct_inner']
    for j,k in enumerate(picks):
        factor=float(outer.factor_at_call[k]); tau=float(outer.first_call_year[k])
        bank = simulate_phoenix(p.market,p.renewal_phoenix,inner_n,192,
                                548000+rep*100000+j*433,initial_factor=factor)
        pv=bank.pv(c)
        direct=float(pv.mean()); half=int(inner_n//2)
        a=float(pv[:half].mean()); d=float(pv[half:2*half].mean())
        convex_gap=max(0.,0.5*(max(a-p.acro.strike,0.)+max(d-p.acro.strike,0.))
                            -max((a+d)/2-p.acro.strike,0.))
        low,up=crossfit_bounds(a,d,p.acro.strike)
        interp=float(qB+np.interp(factor,nodes,db) + c*(qA+np.interp(factor,nodes,da)))
        sd=float(pv.std(ddof=1))
        w=float(p.acro.renewal_units*np.exp(-p.market.risk_free_rate*tau)
                * len(choices)/(len(eligible)*nsel))
        weighted_direct+=w*up
        weighted_lower+=w*low
        weighted_gap+=w*(up-low)
        weighted_interp+=w*max(interp-p.acro.strike,0.)
        weighted_jensen+=w*convex_gap
        weighted_sigma+=w*sd/np.sqrt(inner_n)
        rows.append(dict(rep=rep+1,outer_id=int(k),tau=tau,factor=factor,
                         n_inner=inner_n,raw_renewal_value=direct,
                         interpolated_renewal_value=interp,
                         renewal_value_difference=direct-interp,
                         jensen_half_full=convex_gap, crossfit_lower=low, convex_upper=up,
                         crossfit_gap=up-low, conditional_sigma=sd,
                         weight=w))
    return dict(rep=rep+1,direct_raw=weighted_direct,
                crossfit_lower=weighted_lower, crossfit_upper_minus_lower=weighted_gap,
                interpolated_same_states=weighted_interp,
                direct_minus_interp=weighted_direct-weighted_interp,
                weighted_jensen_half_full=weighted_jensen,
                weighted_sigma_over_sqrt_n=weighted_sigma), rows


def perform(p, b, output):
    validate_scope(p)
    output.mkdir(parents=True,exist_ok=True)
    t0 = time.monotonic()
    coupon, refB, refA, qc = frozen_coupon(p,b,output)
    print(f'[V5.4] Coupon contractuel fige a 192 pas/an: {coupon:.6%}; '
          f'IC statistique approx +/- {10000*qc["approx_95_halfwidth"]:.1f} pb',flush=True)
    grid_rows=[]; deltas=[]; heldout_rows=[]; direct_rows=[]; state_rows=[]
    same_tenor = p.initial_phoenix.maturity_years
    m = int(round(192 * same_tenor)); a=p.market.n_assets
    for rep in range(b['reps']):
        base=549001 + rep*100307
        qbank=innovations_for_grid(m,b['quote'],a,base+11)
        obank=innovations_for_grid(m,b['outer'],a,base+21)
        ibank=innovations_for_grid(m,b['inner'],a,base+31)
        q={};outer={}; vals={}; allx=[]; masks={}
        baseB, baseA, _=quote_from_bank(p,qbank,192,192)
        for g in GRIDS:
            qB,qA,_=quote_from_bank(p,qbank,g,192)
            # High-budget reference quote + COUPLED coarse-minus-fine change.
            # Reduces sampling noise without erasing true time-grid differences.
            q[g]=(refB + qB-baseB, refA + qA-baseA, coupon)
            sim,mask=outer_from_bank(p,obank,g,192)
            outer[g]=sim; masks[g]=mask
            allx.append(sim.factor_at_call[mask])
        nodes=factor_grid(allx,b['nodes'],p.market.initial_logvol_factor)
        for g in GRIDS:
            # For each grid, reused Brownian bank across all conditional factor nodes.
            db,da=renewal_differences(p,ibank,g,192,nodes)
            val=price_from_components(p,q[g],outer[g],masks[g],nodes,db,da)
            vals[g]=val
            grid_rows.append(dict(rep=rep+1,grid=g,price=val,
                                  coupon_frozen=coupon, n_nodes=len(nodes),
                                  n_quote=b['quote'],n_outer=b['outer'],
                                  n_inner=b['inner'],
                                  call_probability=float(masks[g].mean())))
            if g==192:
                check=np.quantile(outer[g].factor_at_call[masks[g]],
                                  np.linspace(.05,.95,b['heldout']))
                # Use exactly same inner draws to measure *interpolation only*.
                exactB,exactA=renewal_differences(p,ibank,g,192,check)
                for x,refb,refa in zip(check,exactB,exactA):
                    approximate=float(q[g][0]+np.interp(x,nodes,db)
                                      +coupon*(q[g][1]+np.interp(x,nodes,da)))
                    direct=float(q[g][0]+refb+coupon*(q[g][1]+refa))
                    heldout_rows.append(dict(rep=rep+1,factor=float(x),
                                             same_path_interp_error=approximate-direct,
                                             absolute_error=abs(approximate-direct),
                                             n_inner=b['inner'],n_nodes=len(nodes)))
                row, rs=direct_states(p,coupon,outer[g],masks[g],q[g],nodes,db,da,rep,b)
                direct_rows.append(row);state_rows.extend(rs)
        deltas.append(dict(rep=rep+1,d96_48=vals[96]-vals[48],
                           d192_96=vals[192]-vals[96],d192_48=vals[192]-vals[48]))
        print(f'[V5.4] Rep {rep+1}/{b["reps"]}: '
              f'premium 48/96/192 = '+ '/'.join(f'{vals[x]:.4f}' for x in GRIDS)
              +f'; 192-96={vals[192]-vals[96]:+.4f}',flush=True)
    save_csv(output/'v54_frozen_coupon_grid.csv',grid_rows)
    save_csv(output/'v54_paired_grid_differences.csv',deltas)
    save_csv(output/'v54_interpolation_heldout.csv',heldout_rows)
    save_csv(output/'v54_independent_direct_states.csv',state_rows)
    save_csv(output/'v54_independent_direct_replications.csv',direct_rows)
    grid_ci={str(g):ci([r['price'] for r in grid_rows if r['grid']==g]) for g in GRIDS}
    delta_ci={k:ci([r[k] for r in deltas]) for k in ('d96_48','d192_96','d192_48')}
    direct_ci={k:ci([r[k] for r in direct_rows]) for k in ('direct_raw',
               'crossfit_lower','crossfit_upper_minus_lower',
               'interpolated_same_states','direct_minus_interp',
               'weighted_jensen_half_full','weighted_sigma_over_sqrt_n')}
    max_note=note_pv_max(p,coupon)
    # The Jensen bound below is for DIRECT iid inner sampling and ALL states,
    # but deliberately uses call probability<=1 and max possible discount factor.
    last=float(p.acro.last_eligible_call_year or p.initial_phoenix.maturity_years)
    worst_discount=float(np.exp(max(0.,-p.market.risk_free_rate*last)))
    upper=p.acro.renewal_units*worst_discount*jensen_bound(max_note,b['direct_inner'])
    held_abs=np.asarray([x['absolute_error'] for x in heldout_rows])
    # A conservative-in-sampling approximate 95% simultaneous confidence
    # bracket: two one-sided Student limits, Bonferroni adjustment.
    # Expectations, not realization-wise bounds. Small-rep approximation.
    bracket_lower=max(0., direct_ci['crossfit_lower']['mean']
                       -student_t.ppf(.975,b['reps']-1)*direct_ci['crossfit_lower']['standard_error'])
    bracket_upper=(direct_ci['direct_raw']['mean']
                   +student_t.ppf(.975,b['reps']-1)*direct_ci['direct_raw']['standard_error'])
    if bracket_lower>bracket_upper+1e-10: raise AssertionError('Invalid bracket')
    bracket=dict(approx_joint_95_percent=[bracket_lower,bracket_upper],
                 width=bracket_upper-bracket_lower,
                 interpretation='Approximate Student/Bonferroni lower and upper bound in expectation for fixed 192-step model and frozen coupon; no interpolation used in direct repricing; n independent outer/inner replications; not a bound on time discretization.')
    # A conservative gate requires the complete paired CI to fit the tolerance;
    # these criteria alone do NOT certify the true bias (global bound too loose).
    grid_ok=(max(abs(x) for x in delta_ci['d192_96']['ci95'])<=TOL)
    ci_interp=direct_ci['direct_minus_interp']['ci95']
    interp_ok=max(abs(v) for v in ci_interp)<=TOL
    coupon_ok=qc['approx_95_halfwidth']<0.0005  # 5 bps statistical, chosen QA criterion
    jensen_ok=(bracket['width'] <= 2*TOL)
    diagnostic=dict(coupon_precise_enough=bool(coupon_ok),paired_time_difference_enough=bool(grid_ok),
                    direct_vs_interp_difference_enough=bool(interp_ok),
                    jensen_crossfit_bracket_enough=bool(jensen_ok),
                    tolerance_met=bool(coupon_ok and grid_ok and interp_ok and jensen_ok))
    results=dict(version='5.4',scope='Synthetic one-factor stochastic-vol Q pricing; not calibrated',
                 coupon_reference=qc,grids=grid_ci,grid_pair_differences=delta_ci,
                 direct_repricing_same_states=direct_ci,
                 crossfit_direct_bracket=bracket,
                 interpolation_heldout=dict(n=len(held_abs),mean_abs=float(held_abs.mean()),
                                            max_abs=float(held_abs.max())),
                 jensen=dict(empirical_sigma_over_sqrt_n=direct_ci['weighted_sigma_over_sqrt_n'],
                             empirical_half_full=direct_ci['weighted_jensen_half_full'],
                             proven_bound_formula='PV note is in [0,M], so Jensen <= M/(2 sqrt(n)) before ACRO discount and exercise weighting',
                             deterministic_note_pv_upper_bound=max_note,
                             conservative_unconditional_acro_upper_bound=upper,
                             caveat='This rigorous global bound is conservative and does not cover outer, coupon or time-grid errors.'),
                 acceptance=diagnostic,settings=b,runtime_seconds=time.monotonic()-t0)
    (output/'AUDIT_PRECISION_V5_4.json').write_text(json.dumps(results,indent=2,ensure_ascii=False),encoding='utf-8')
    report=[
        '# ACRO V5.4 — Audit de precision (modele synthetique)', '',
        '**Verdict : '+('tolérance numerique atteinte dans les controles definis' if diagnostic['tolerance_met']
                       else 'tolerance ±0,02 €/100 NON certifiee')+'**', '',
        '## 1. Coupon contractuel unique',
        f'- Coupon reference fixe : **{coupon:.6%}** (coté sous Q a 192 pas/an, {qc["n_paths"]:,} chemins).',
        f'- Intervalle statistique approximatif a 95 % : [{qc["approx_95_ci"][0]:.6%} ; {qc["approx_95_ci"][1]:.6%}].',
        '- Il est conserve tel quel dans les grilles 48, 96 et 192 : pas de recalcul du coupon par grille.',
        '- Les cotations des grilles utilisent la reference commune 192 pas + (grille - 192) mesure sur les memes chemins : variate de controle couplee.', '',
        '## 2. Grilles temporelles couplees',
        '| Grille | Prime moyenne EUR/100 | IC statistique 95 % entre repetitions |',
        '|---:|---:|---|',
    ]
    for g in GRIDS:
        x=grid_ci[str(g)]
        report.append(f'| {g} | {x["mean"]:.4f} | [{x["ci95"][0]:.4f} ; {x["ci95"][1]:.4f}] |')
    report+=['', '| Difference | Moyenne | IC statistique 95 % |','|---|---:|---|']
    for k in ('d96_48','d192_96','d192_48'):
        x=delta_ci[k]
        report.append(f'| {k} | {x["mean"]:+.4f} | [{x["ci95"][0]:+.4f} ; {x["ci95"][1]:+.4f}] |')
    report+=['', 'Les IC ne prennent en compte que la dispersion entre repetitions; ils ne bornent pas le biais de temps.', '',
             '## 3. Repricing DIRECT sur les memes etats au rappel']
    for k in ('direct_raw','interpolated_same_states','direct_minus_interp'):
        x=direct_ci[k]
        report.append(f'- {k} : **{x["mean"]:.4f}** EUR/100 ; IC95 [{x["ci95"][0]:.4f}; {x["ci95"][1]:.4f}].')
    report+=['',f'- Erreur interpolation hors noeuds (memes trajectoires internes) : moyenne absolue {held_abs.mean():.4f} EUR **de valeur de note**, max {held_abs.max():.4f}.',
             '- Le repricing direct recalcule chaque nouveau Phoenix sans courbe de valeurs. Il conserve un biais de Jensen positif.', '',
             '### Encadrement par exercice teste hors echantillon',
             f'- Borne basse cross-fit (moyenne) : {direct_ci["crossfit_lower"]["mean"]:.4f} EUR.',
             f'- Borne haute plug-in convexe (moyenne) : {direct_ci["direct_raw"]["mean"]:.4f} EUR.',
             f'- Encadrement statistique **approximatif** simultane a 95 % sous Student/Bonferroni : [{bracket_lower:.4f}; {bracket_upper:.4f}] EUR.',
             '- Demonstration : l indicatrice d exercice fondee sur le premier demi-echantillon est testee sur le second, independant ; en esperance le gain de cette politique est inferieur au payoff optimal. La convexite donne la borne haute en esperance.',
             '- L encadrement porte sur le prix a coupon fige et grille 192 dans le modele, non sur le prix continu, la calibration ou le risque de marche.', '',
             '## 4. Biais de Jensen du Monte Carlo direct',
             f'- Diagnostic demi-echantillons / plein echantillon (prime) : {direct_ci["weighted_jensen_half_full"]["mean"]:.4f} EUR.',
             f'- Proxy de borne via ecart-type conditionnel estime : {direct_ci["weighted_sigma_over_sqrt_n"]["mean"]:.4f} EUR (non certifie).',
             f'- Borne theorique uniforme conservatrice sur le biais Jensen **du direct** : {upper:.4f} EUR/100 (n={b["direct_inner"]}, M={max_note:.2f} EUR).',
             '- La borne theorique est souvent trop large pour l objectif ±0,02. Elle n encadre ni interpolation ni autres erreurs.', '',
             '## 5. Criteres de validation',
             '| Critere | Reussi ? |','|---|---|']
    for k,v in diagnostic.items(): report.append(f'| {k} | {"oui" if v else "non"} |')
    report+=['', '## Limites scientifiques',
             '- Un test statistiquement compatible entre deux estimateurs n est pas une preuve de convergence vers le prix exact.',
             '- Les coefficients du modele, la dependance action-volatilite et les correlations ne sont pas calibres au marche.',
             '- Les paths coupes ou selectionnes aux rappels changent entre grilles, meme avec increments browniens couples.',
             '- Une estimation complete de l erreur demanderait un controle plus fort du biais de temps, du biais Jensen, des quotas d autocall et de l approximation de courbe.',
             '- Le prix obtenu est un prix de recherche theorique, pas un prix commercial ou une affirmation de profit investisseur.', '',
             'Fichiers sources : `v54_frozen_coupon_grid.csv`, `v54_paired_grid_differences.csv`, `v54_independent_direct_states.csv`, `v54_independent_direct_replications.csv`, `v54_interpolation_heldout.csv`, `coupon_reference_running.csv`, `AUDIT_PRECISION_V5_4.json`.']
    (output/'AUDIT_PRECISION_V5_4_FR.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print(f'[V5.4] Fin en {time.monotonic()-t0:.1f} s. '
          f'Tolérance ±{TOL:.2f}: {"OUI" if diagnostic["tolerance_met"] else "NON"}',flush=True)
    return results


def main():
    ap=argparse.ArgumentParser()
    grp=ap.add_mutually_exclusive_group()
    grp.add_argument('--quick',action='store_true')
    grp.add_argument('--high',action='store_true')
    ap.add_argument('--config',default=str(ROOT/'config.json'))
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    mode='quick' if args.quick else 'high' if args.high else 'standard'
    out=Path(args.output) if args.output else ROOT/('results_audit_v54_quick' if args.quick else
                                                   'results_audit_v54_high' if args.high else
                                                   'results_audit_v54')
    perform(load_project(args.config),budgets(mode),out)


if __name__=='__main__': main()
