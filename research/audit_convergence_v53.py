"""ACRO V5.3 — Jensen / conditional-MC budget / time-grid convergence.

Research audit. No modification to Phoenix cash flows or market dynamics.

Nested benchmark: independently resample initial quotations, outer call states,
and *uninterpolated* renewal-note paths. Inner sample sizes use nested prefixes
of the same i.i.d. path bank for a paired comparison; the benchmark is a RAW
nested estimator and still has positive-part Jensen bias.

Grid benchmark: fix a single contractual coupon quoted at 96 steps/year,
re-use coupled fine-grid Brownian innovations for 24 / 48 / 96, but do NOT
claim numerical convergence from noisy paired comparisons.

Run:
    python audit_convergence_v53.py --quick
    python audit_convergence_v53.py
    python audit_convergence_v53.py --high  # intentionally expensive
"""
from __future__ import annotations
import argparse
import csv
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from acro import load_project, simulate_phoenix, quote_initial
from research.audit_coupled import (innovations_for_grid, quote_from_bank, outer_from_bank,
                           renewal_differences, price_from_components)

ROOT=Path(__file__).resolve().parents[1]


def write_csv(path, rows):
    if not rows: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def mean_ci(data):
    """Student interval across independent Monte-Carlo replicates only."""
    a=np.asarray(data,dtype=float)
    if len(a)<2 or not np.isfinite(a).all():
        raise ValueError('Need >=2 finite independent replications')
    m=float(a.mean()); s=float(a.std(ddof=1)/np.sqrt(len(a)))
    t=float(student_t.ppf(.975,len(a)-1))
    return dict(n=len(a),mean=m,standard_error=s,ci95=[m-t*s,m+t*s],
                halfwidth=t*s)


def sample_hierarchy(pv, coupon, strike, levels):
    """Strictly nested MC prefixes and pathwise nonnegative Jensen gap.

    The half/full gap is NOT an estimator of the absolute remaining Jensen
    bias. It is a diagnostic of its decay at finite inner sample sizes.
    """
    v=np.asarray(pv,dtype=float)
    if len(v)<max(levels) or not np.isfinite(v).all():
        raise ValueError('Insufficient inner PV samples')
    results={}
    for n in levels:
        if n%2: raise ValueError('Inner levels must be even')
        a=float(v[:n//2].mean()); b=float(v[n//2:n].mean())
        m=(a+b)/2.
        exercise=max(m-strike,0.)
        half=(max(a-strike,0.)+max(b-strike,0.))/2.
        gap=half-exercise
        if gap < -1e-10: raise AssertionError('Positive part must be convex')
        sigma=float(v[:n].std(ddof=1))
        # Empirical plug-in estimate of a Cauchy-Schwarz BIAS upper bound;
        # not a rigorous confidence bound without a variance upper bound.
        error_proxy=sigma/np.sqrt(n)
        results[n]=dict(intrinsic=exercise,half_intrinsic=half,
                        jensen_half_full=max(0.,gap),note_value=m,
                        conditional_se=error_proxy)
    return results


def params(mode):
    if mode=='quick':
        return dict(reps=3,quote_paths=3200,outer_paths=2800,
                    selected_states=20,inner_levels=[128,256,512,1024],
                    grid_reps=3,grid_quote=900,grid_outer=1250,
                    grid_inner=850,grid_nodes=9,reference_quote=5000)
    if mode=='standard':
        return dict(reps=5,quote_paths=6500,outer_paths=5300,
                    selected_states=50,inner_levels=[256,512,1024,2048],
                    grid_reps=4,grid_quote=1900,grid_outer=2200,
                    grid_inner=1550,grid_nodes=13,reference_quote=8500)
    if mode=='high':
        return dict(reps=8,quote_paths=18000,outer_paths=12000,
                    selected_states=100,inner_levels=[512,1024,2048,4096,8192],
                    grid_reps=8,grid_quote=5000,grid_outer=8500,
                    grid_inner=5000,grid_nodes=21,reference_quote=30000)
    raise ValueError('Unknown mode')


def nested_replicates(p,cfg,out):
    """Full independent RAW nested repricing; no interpolation or CRN anchor.

    Eligible-call states are sampled uniformly from newly simulated outer
    paths. P(eligible call) times their conditional mean reconstructs the
    entire premium. Each inner state uses a fresh independent random stream.
    """
    steps=24; levels=cfg['inner_levels']; nmax=max(levels)
    by_state=[]; by_rep=[]
    for rep in range(cfg['reps']):
        seed=930001+rep*103081
        pr=replace(p,simulation=replace(p.simulation,seed=seed,
                   steps_per_year=steps,quote_paths=cfg['quote_paths'],
                   outer_paths=cfg['outer_paths']))
        quote=quote_initial(pr)
        coupon=(quote.coupon_rate if pr.acro.target_coupon_rate is None
                else pr.acro.target_coupon_rate)
        outer=simulate_phoenix(pr.market,pr.initial_phoenix,
                              cfg['outer_paths'],steps,seed+20)
        eligible=np.isfinite(outer.first_call_year)
        if pr.acro.last_eligible_call_year is not None:
            eligible &= (outer.first_call_year<=pr.acro.last_eligible_call_year+1e-10)
        choices=np.flatnonzero(eligible)
        if len(choices)==0: raise RuntimeError('No eligible autocalls')
        nsel=min(cfg['selected_states'],len(choices))
        picked=np.random.default_rng(seed+8808).choice(choices,size=nsel,replace=False)
        # Each sampled eligible state has the same conditional selection weight,
        # but a different timing discount factor.
        weight_base=pr.acro.renewal_units*len(choices)/len(eligible)/nsel
        totals={n:0. for n in levels}
        jensen={n:0. for n in levels}
        inner_err={n:0. for n in levels}
        for j,k in enumerate(picked):
            factor=float(outer.factor_at_call[k]); tau=float(outer.first_call_year[k])
            discount=float(np.exp(-pr.market.risk_free_rate*tau))
            weight=weight_base*discount
            # Single independent continuation bank: smaller levels are prefixes.
            paths=simulate_phoenix(pr.market,pr.renewal_phoenix,
                        nmax,steps,seed+500000+int(k)*37, initial_factor=factor)
            hierarchy=sample_hierarchy(paths.pv(coupon),coupon,pr.acro.strike,levels)
            for n in levels:
                state=hierarchy[n]
                totals[n]+=weight*state['intrinsic']
                jensen[n]+=weight*state['jensen_half_full']
                inner_err[n]+=weight*state['conditional_se']
                by_state.append(dict(rep=rep+1,outer_id=int(k),call_year=tau,
                    logvol_factor=factor,n_inner=n,discount_weight=weight,
                    renewal_price=state['note_value'],
                    intrinsic=state['intrinsic'],jensen_half_full=state['jensen_half_full'],
                    conditional_note_se=state['conditional_se']))
        row=dict(rep=rep+1,seed=seed,initial_coupon=coupon,
                 eligible_call_probability=float(eligible.mean()),
                 n_selected=nsel,n_quote=cfg['quote_paths'],n_outer=cfg['outer_paths'])
        for n in levels:
            row[f'raw_{n}']=totals[n]
            row[f'jensen_gap_{n}']=jensen[n]
            row[f'bias_proxy_{n}']=inner_err[n]
        by_rep.append(row)
        print(f'[V5.3 nested] {rep+1}/{cfg["reps"]}: '
              f'raw({levels[-1]})={totals[levels[-1]]:.4f}; '
              f'Jensen half/full={jensen[levels[-1]]:.4f}',flush=True)
    write_csv(out/'v53_direct_inner_states.csv',by_state)
    write_csv(out/'v53_nested_replications.csv',by_rep)
    results={}
    for n in levels:
        results[str(n)]=dict(price=mean_ci([r[f'raw_{n}'] for r in by_rep]),
                              jensen_half_full=mean_ci([r[f'jensen_gap_{n}'] for r in by_rep]),
                              bias_proxy=mean_ci([r[f'bias_proxy_{n}'] for r in by_rep]))
    # Exploratory bias extrapolations from last two budgets ONLY. They
    # assume a bias decay rate and are not used as quoted price estimates.
    small,big=levels[-2:]
    psmall=results[str(small)]['price']['mean']
    pbig=results[str(big)]['price']['mean']
    ratio=big/small
    extrapolations={
       'assumed_bias_order_1_over_n':(ratio*pbig-psmall)/(ratio-1),
       'assumed_bias_order_1_over_sqrt_n':(np.sqrt(ratio)*pbig-psmall)/(np.sqrt(ratio)-1),
       'last_budget_sample_mean':pbig,
       'warning':'Bias decay assumptions unproven; extrapolation may violate expected Jensen monotonicity in finite samples and is NOT a validated correction.'
    }
    pairs={}
    for small,big in zip(levels[:-1],levels[1:]):
        pairs[f'{big}_minus_{small}']=mean_ci([r[f'raw_{big}']-r[f'raw_{small}'] for r in by_rep])
    return dict(reps=cfg['reps'],steps_per_year=steps,quote_paths=cfg['quote_paths'],
                outer_paths=cfg['outer_paths'],selected_states=cfg['selected_states'],
                inner_levels=levels,prices_by_budget=results,paired_differences=pairs,
                exploratory_extrapolations=extrapolations)


def grid_replicates(p,cfg,out):
    """24/48/96 fixed-coupon grid, paired across Brownian increments."""
    grids=[24,48,96];fine=96;tenor=p.initial_phoenix.maturity_years
    na=p.market.n_assets
    nsteps=int(round(tenor*fine))
    refbank=innovations_for_grid(nsteps,cfg['reference_quote'],na,755290)
    B0,A0,frozen=quote_from_bank(p,refbank,fine,fine)
    del refbank
    rows=[];delta=[]
    for rep in range(cfg['grid_reps']):
        seed=160021+rep*131009
        qbank=innovations_for_grid(nsteps,cfg['grid_quote'],na,seed+10)
        obank=innovations_for_grid(nsteps,cfg['grid_outer'],na,seed+20)
        ibank=innovations_for_grid(nsteps,cfg['grid_inner'],na,seed+30)
        calls={}; quotes={}; factors=[]
        for g in grids:
            quotes[g]=quote_from_bank(p,qbank,g,fine)
            calls[g]=outer_from_bank(p,obank,g,fine)
            path,mask=calls[g]
            if mask.any(): factors.append(path.factor_at_call[mask])
        if not factors: raise RuntimeError('No eligible calls')
        allx=np.concatenate(factors)
        nodes=np.unique(np.r_[np.linspace(float(allx.min())-.12,
                              float(allx.max())+.12,cfg['grid_nodes']),
                              p.market.initial_logvol_factor])
        prices={}
        for g in grids:
            B,A,_=quotes[g]
            db,da=renewal_differences(p,ibank,g,fine,nodes)
            path,mask=calls[g]
            prices[g]=price_from_components(p,(B,A,frozen),path,mask,nodes,db,da)
            rows.append(dict(rep=rep+1,steps_per_year=g,
                             fixed_coupon=frozen,price=prices[g],
                             n_outer=cfg['grid_outer'],n_inner=cfg['grid_inner'],
                             n_quote=cfg['grid_quote'],call_probability=float(mask.mean())))
        delta.append(dict(rep=rep+1,delta_48_minus_24=prices[48]-prices[24],
                          delta_96_minus_48=prices[96]-prices[48],
                          delta_96_minus_24=prices[96]-prices[24]))
        print(f'[V5.3 grid] {rep+1}/{cfg["grid_reps"]}: '
              f'96-48={prices[96]-prices[48]:+.4f}',flush=True)
    write_csv(out/'v53_fixed_coupon_grids.csv',rows)
    write_csv(out/'v53_coupled_time_differences.csv',delta)
    return dict(frozen_coupon=frozen,reference_quote_price=B0+frozen*A0,
            ref_quote_paths=cfg['reference_quote'],reps=cfg['grid_reps'],
            grid_steps=grids,grid_paths_quote=cfg['grid_quote'],
            grid_paths_outer=cfg['grid_outer'],grid_paths_inner=cfg['grid_inner'],
            paired_differences={k:mean_ci([v[k] for v in delta])
               for k in ('delta_48_minus_24','delta_96_minus_48','delta_96_minus_24')})


def gate(nested,grid,tolerance=.02):
    """Conservative practical gate, NOT a proof of unbiasedness.

    All components must be below tolerance: 95% CI halfwidth of raw
    independent nested estimator, empirical Jensen gap/bias proxy,
    and absolute 96-48 coupled difference plus its CI halfwidth.
    """
    finest=str(max(nested['inner_levels']))
    stats=nested['prices_by_budget'][finest]
    gridstat=grid['paired_differences']['delta_96_minus_48']
    last_diff=list(nested['paired_differences'].values())[-1]
    checks={
       'raw_premium_ci95_halfwidth':stats['price']['halfwidth'],
       'conditional_mc_error_proxy_mean':stats['bias_proxy']['mean'],
       'jensen_half_full_mean':stats['jensen_half_full']['mean'],
       'last_inner_budget_delta_plus_ci95_halfwidth':abs(last_diff['mean'])+last_diff['halfwidth'],
       'time_96_minus_48_abs_mean_plus_ci95_halfwidth':abs(gridstat['mean'])+gridstat['halfwidth'],
    }
    meets=bool(all(v<=tolerance for v in checks.values()))
    return dict(target_eur_per_100=tolerance,meets_practical_gate=meets,
                diagnostics=checks,not_a_proof='Finite-budget tests cannot certify continuous-time / zero-Jensen bias')


def run(config,output,mode,tolerance=.02):
    p=load_project(config)
    if (p.market.dynamics!='stochastic_vol' or
        p.initial_phoenix !=p.renewal_phoenix or
        p.acro.target_coupon_rate is not None or p.initial_coupon_policy!='fair'):
        raise ValueError('V5.3 audit currently targets identical A/R terms, fair coupon, stochastic-vol factor')
    cfg=params(mode); out=Path(output).resolve();out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    nested=nested_replicates(p,cfg,out)
    grid=grid_replicates(p,cfg,out)
    verdict=gate(nested,grid,tolerance)
    res=dict(model='ACRO V5.3; existing Q factor model unchanged',mode=mode,
             config=str(Path(config).resolve()),nested=nested,grid=grid,gate=verdict,
             elapsed_seconds=time.perf_counter()-start,
             caveats=[
               'Confidence intervals reflect finite independent repetitions, not deterministic bounds on Jensen or time-discretization bias.',
               'Coupling: inner sizes share prefix paths; grids share aggregated 96-step Brownian innovations.',
               'Jensen half/full is nonnegative pathwise but is NOT the total finite-inner-MC bias.',
               'Empirical bias proxy is estimated from inner-PV sample standard deviations; not a certified bound.',
               'Raw nested repricing uses fresh quotations and outer scenarios per replicate, no interpolation or initial-quote anchor. It still has positive-part Jensen bias.',
               'Nested study samples a finite number of eligible outer states and weights by eligible-call probability. This adds sampling variance.',
               'Time grids use an initial coupon frozen from a single independent 96-step quotation. Their sampled B/A and conditional valuations retain MC noise.',
               'Numerical accuracy does not imply market calibration or issuer-credit/hedge implementation.'] )
    (out/'AUDIT_CONVERGENCE_V5_3.json').write_text(json.dumps(res,indent=2,ensure_ascii=False),encoding='utf-8')
    g=grid['paired_differences']; n=nested['prices_by_budget']
    def fmt(x):return f"{x['mean']:.4f} [{x['ci95'][0]:.4f} ; {x['ci95'][1]:.4f}]"
    lines=['# ACRO V5.3 — Audit Jensen et discrétisation', '',
        f'**Mode:** {mode}. **Tolérance pratique visée:** ±{tolerance:.2f} €/100. **Verdict : '+('tolérance pratique atteinte' if verdict['meets_practical_gate'] else 'tolérance NON démontrée')+'.**', '',
        '## 1. Nested Monte Carlo brut, indépendamment de la courbe interpolée','',
        f"{nested['reps']} réplications, {nested['quote_paths']} cotations, {nested['outer_paths']} trajectoires externes, jusqu’à {nested['selected_states']} états de rappel sélectionnés par réplication ; grilles internes préfixées à 24 pas/an.", '',
        '| Chemins internes par état | Prix [IC statistique 95 %] €/100 | Jensen demi−plein €/100 | Proxy erreur interne €/100 |',
        '|---:|---:|---:|---:|']
    for level in nested['inner_levels']:
        v=n[str(level)];lines.append(f"| {level} | {fmt(v['price'])} | {v['jensen_half_full']['mean']:.4f} | {v['bias_proxy']['mean']:.4f} |")
    extra=nested['exploratory_extrapolations']
    lines+=['', '**Attention :** la baisse du diagnostic demi-échantillons ne garantit pas la disparition du biais ; les budgets sont appariés sur les mêmes chemins internes, les IC portent uniquement sur les réplications finies.',
        '', '### Tentatives exploratoires de correction du biais',
        f"- Extrapolation supposant un biais en 1/n : **{extra['assumed_bias_order_1_over_n']:.4f} €/100**.",
        f"- Extrapolation supposant un biais en 1/√n : **{extra['assumed_bias_order_1_over_sqrt_n']:.4f} €/100**.",
        '- **Aucune de ces extrapolations n’est validée.** La décroissance attendue du biais n’est pas résolue face au bruit statistique, et aucune loi d’ordre 1/n ou 1/√n n’a été établie pour cette distribution d’états.',
        '', '## 2. Grilles 24 / 48 / 96 pas/an, coupon FIXÉ', '',
        f"Coupon contractualisé et gelé : **{grid['frozen_coupon']*100:.4f} %**, obtenu une seule fois sur {grid['ref_quote_paths']} simulations à 96 pas/an.", '',
        '| Différence appariée | Moyenne [IC statistique 95 %] €/100 |','|---|---:|',
        f"| 48−24 | {fmt(g['delta_48_minus_24'])} |",
        f"| 96−48 | {fmt(g['delta_96_minus_48'])} |",
        f"| 96−24 | {fmt(g['delta_96_minus_24'])} |", '',
        'La comparaison peut rester bruitée malgré le couplage ; l’IC n’est pas une borne de l’erreur de discrétisation.', '',
        '## 3. Contrôle de précision exigé', '',
        '| Diagnostic | Mesure €/100 |','|---|---:|']
    for label,val in verdict['diagnostics'].items(): lines.append(f'| {label} | {val:.4f} |')
    lines+=['', 'Ce contrôle conservateur est un critère de recherche, **pas** une certification du prix limite mathématique. Un échec ne signifie pas que le modèle est faux.', '',
        '## 4. Limitations restantes','',
        '- Absence de borne théorique établie sur le biais positif-part Jensen.','- Incertitude statistique due à la sélection des états de rappel et à la cotation initiale.',
        '- Dynamique à facteur de volatilité commun, corrélations constantes, taux fixes, pas de calibration.','- Aucun prix commercial ni stratégie de couverture validés.', '',
        'Données : `v53_direct_inner_states.csv`, `v53_nested_replications.csv`, `v53_fixed_coupon_grids.csv`, `v53_coupled_time_differences.csv`, `AUDIT_CONVERGENCE_V5_3.json`.']
    (out/'AUDIT_CONVERGENCE_V5_3_FR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('V5.3 completed in %.1f s; tolerance reached=%s'%(res['elapsed_seconds'],verdict['meets_practical_gate']),flush=True)
    return res

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--output',default=str(ROOT/'results_audit_v53'))
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--high',action='store_true')
    parser.add_argument('--tolerance',type=float,default=.02)
    args=parser.parse_args()
    if args.quick and args.high:parser.error('Choose either --quick or --high')
    run(args.config,args.output,'quick' if args.quick else 'high' if args.high else 'standard',args.tolerance)
