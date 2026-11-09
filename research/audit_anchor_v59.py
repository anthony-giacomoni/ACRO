"""V5.9 audit: exact anchor, equal-measured-runtime comparison, held-out repricing.

Independent repricing uses fresh Brownian shocks AND a fresh GBM reference.
Intervals are diagnostic only. They do not certify total ACRO pricing error.

Usage: python audit_anchor_v59.py --quick
       python audit_anchor_v59.py
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse, csv, json
import numpy as np
from acro import load_project
from acro.conditional_qmc import innovation_bank
from acro.gbm_control import gbm_reference, renewal_curve_gbm_cv
from acro.renewal_anchor import grid_including_anchor, renewal_curve_anchor
from research.audit_variance_v58 import COUPON, GRID, simulate_outer, discounted_payoffs, moments

ROOT=Path(__file__).resolve().parents[1]


def interpolate_anchored(nodes, prices, anchor, expected):
    nodes=np.asarray(nodes,float); prices=np.asarray(prices,float)
    if nodes.shape!=prices.shape or nodes.ndim!=1 or np.any(np.diff(nodes)<=0):
        raise ValueError('invalid anchored curve')
    if not np.any(nodes == anchor):
        raise ValueError('missing exact anchor node')
    got=float(np.interp(anchor,nodes,prices))
    if not np.isclose(got,expected,atol=1e-12,rtol=0):
        raise AssertionError('anchor not preserved by interpolation')
    return got


def independent_heldout(p, nodes, *, reference, reference_se, npaths, seed):
    bank=innovation_bank(int(round(p.renewal_phoenix.maturity_years*GRID)),
                         npaths,p.market.n_assets,seed,'mc_bridge')
    _,cv,_,delta_var,_,_=renewal_curve_gbm_cv(p,COUPON,GRID,nodes,bank,reference)
    # Conditional on independent GBM ref; combine independent variance sources
    se=np.sqrt(delta_var/npaths + reference_se**2)
    return cv, se


def run(*,quick=False, output=None):
    p=load_project(ROOT/'config.json')
    p=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=COUPON),
              acro=replace(p.acro,target_coupon_rate=COUPON))
    if p.market.dynamics!='stochastic_vol' or p.renewal_phoenix.capital_barrier_monitoring!='european':
        raise ValueError('V5.9 audit requires the reference SV model and European renewal barrier')
    b=(dict(reps=6, outer=1600, inner=256, anchor=1024, ref=4096,
            scrambles=4, heldout=2048) if quick else
       dict(reps=12, outer=3200, inner=512, anchor=2048, ref=8192,
            scrambles=6, heldout=8192))
    out=Path(output or ROOT/('results_audit_v59_quick' if quick else 'results_audit_v59'))
    out.mkdir(exist_ok=True,parents=True)
    x0=float(p.market.initial_logvol_factor)
    nodes=grid_including_anchor(-1.4,1.0,19,x0)
    assert np.any(nodes==x0)
    state=simulate_outer(p,595901,b['outer'])
    x=state.factor_at_call[np.isfinite(state.first_call_year)]
    if not x.size: raise RuntimeError('no autocall for audit')
    if x.min()<nodes[0] or x.max()>nodes[-1]: raise RuntimeError('out-of-grid calls: extend grid')
    # Independent reference for the experiment. Its uncertainty is common to all replicates.
    ref=gbm_reference(p,COUPON,GRID,b['ref'],b['scrambles'],595902)
    rows=[]; curves=[]
    for k in range(b['reps']):
        seed=595903 + 1009*k
        tic=perf_counter()
        bank=innovation_bank(int(round(p.renewal_phoenix.maturity_years*GRID)),
                             b['inner'],p.market.n_assets,seed,'mc_bridge')
        _,old,_,_,_,_=renewal_curve_gbm_cv(p,COUPON,GRID,nodes,bank,ref.value)
        direct_s=perf_counter()-tic
        tic=perf_counter()
        anchor=renewal_curve_anchor(p,COUPON,GRID,nodes,ref.value,ref.std_error,
                                    b['anchor'],b['inner'],seed)
        anchored_s=perf_counter()-tic
        interpolate_anchored(nodes,anchor.prices,x0,anchor.anchor)
        y_old,_=discounted_payoffs(p,state,nodes,old)
        y_new,_=discounted_payoffs(p,state,nodes,anchor.prices)
        row={'rep':k+1,'seed':seed,'direct_premium':float(y_old.mean()),
             'anchored_premium':float(y_new.mean()),
             'direct_seconds':direct_s,'anchored_seconds':anchored_s,
             'anchor':anchor.anchor,'anchor_interpolation_error':float(np.interp(x0,nodes,anchor.prices)-anchor.anchor)}
        rows.append(row);curves.append(anchor.prices)
    d=moments([r['direct_premium'] for r in rows]); a=moments([r['anchored_premium'] for r in rows])
    td=float(np.mean([r['direct_seconds'] for r in rows])); ta=float(np.mean([r['anchored_seconds'] for r in rows]))
    ratio=float(a['sd_across_reps']**2*ta/(d['sd_across_reps']**2*td)) if d['sd_across_reps']>0 else None
    # Independent state-level validation; no interpolation curve/GBM reference reuse.
    states=np.unique(np.r_[x0,np.quantile(x,[.10,.30,.50,.70,.90])])
    ref_hold=gbm_reference(p,COUPON,GRID,b['ref'],b['scrambles'],595991)
    hold, holdse=independent_heldout(p,states,reference=ref_hold.value,
                                    reference_se=ref_hold.std_error,
                                    npaths=b['heldout'],seed=595992)
    held=[]
    for j,xj in enumerate(states):
        v=np.array([np.interp(xj,nodes,cv) for cv in curves])
        vmean=float(v.mean()); vsd=float(v.std(ddof=1))
        # The CI on the difference is ONLY indicative: shared ref, inner biases, sampling limits.
        held.append({'factor':float(xj), 'anchored_mean':vmean,
                     'anchored_sd_across_reps':vsd, 'fresh_independent_gbm_cv':float(hold[j]),
                     'independent_pointwise_se':float(holdse[j]),
                     'difference':vmean-float(hold[j]),
                     'combined_diagnostic_se':float(np.hypot(vsd/np.sqrt(b['reps']),holdse[j]))})
    result={'version':'5.9','coupon_fixed':COUPON,'grid_steps_year':GRID,
            'anchor_exact':True,'anchor_factor':x0,'grid_nodes':nodes.tolist(),
            'max_anchor_interpolation_error':float(max(abs(r['anchor_interpolation_error']) for r in rows)),
            'budget':b,'reference_gbm':{'value':ref.value,'se':ref.std_error},
            'heldout_gbm_reference':{'value':ref_hold.value,'se':ref_hold.std_error},
            'direct_cv':d,'anchored_cv':a,'mean_seconds_direct':td,'mean_seconds_anchored':ta,
            'variance_times_runtime_ratio':ratio,
            'heldout':held,'global_tolerance_0p02_certified':False,
            'notes':['External autocall paths are held fixed during estimator comparisons.',
                     'Shared GBM reference error is not included in estimator replicate intervals.',
                     'Held-out quotes use independent innovations and independently estimated GBM reference.',
                     'Pointwise checks are diagnostics, not accuracy bounds for the full option premium.',
                     'No rigorous Jensen, interpolation or time-discretization bias bound; pricing not certified.']}
    (out/'AUDIT_ANCRAGE_V5_9.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    for name,data in [('estimator_replicates.csv',rows),('independent_heldout.csv',held)]:
        with (out/name).open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    lines=['# ACRO V5.9 — Audit indépendant de l’ancrage','',
           f'Coupon contractuel fixé à **18 %**, **{GRID} pas/an**, {b["reps"]} réplications par méthode.','',
           f'- Ancrage x0 = {x0:g} exactement présent dans la grille : **oui**.',
           f'- Erreur maximale de restitution de l’ancrage par interpolation : **{result["max_anchor_interpolation_error"]:.2e} €**.',
           f'- Pricer GBM-CV direct : dispersion **{d["sd_across_reps"]:.4f} €** ; {td:.3f} s/courbe.',
           f'- Courbe ancrée : dispersion **{a["sd_across_reps"]:.4f} €** ; {ta:.3f} s/courbe.',
           f'- Ratio variance × temps (ancré/direct) : **{ratio:.3f}** (< 1 favorable uniquement dans cette expérience).','',
           '## Prix de renouvellement : vérifications hors grille réellement indépendantes','',
           '| Facteur de volatilité | Courbe ancrée (moyenne) | Nouveau repricing | Écart | SE diagnostic |',
           '|---:|---:|---:|---:|---:|']
    for z in held:
        lines.append(f'| {z["factor"]:+.3f} | {z["anchored_mean"]:.3f} | {z["fresh_independent_gbm_cv"]:.3f} | {z["difference"]:+.3f} | {z["combined_diagnostic_se"]:.3f} |')
    lines += ['', '## Limites', '',
              'Les erreurs indiquées ne bornent pas le biais de Jensen, de discrétisation, ni celui de l’interpolation sur la prime finale.',
              'Une référence GBM commune subsiste entre les courbes d’une même expérience ; elle est indépendante du repricing hors grille.',
              'Les résultats sont conditionnels au modèle synthétique et à ses hypothèses ; il ne s’agit pas de prix de marché.',
              '**Tolérance globale ±0,02 €/100 : non démontrée.**','']
    (out/'AUDIT_ANCRAGE_V5_9_FR.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'[V5.9] exact anchor x0={x0:g} ; max error={result["max_anchor_interpolation_error"]:.3g}')
    print(f'[V5.9] direct sd={d["sd_across_reps"]:.4f}, anchored sd={a["sd_across_reps"]:.4f}, efficiency ratio={ratio:.3f}')
    print('[V5.9] result provisional; tolerance NOT certified')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args();run(quick=args.quick,output=args.output)
