"""ACRO V5.6 — GBM-surrogate CV with explicit reference and runtime budgets.

  python benchmark_gbm_cv_v56.py --quick
  python benchmark_gbm_cv_v56.py

Conditional pricing only: fixed initial coupon 18%, fixed outer market states,
unchanged V5.5 payoff. Contrasts plain MC on exactly the same stock/factor
paths with a control-variate estimator. The no-credit, no-calibration model
and the valuation-time grid of SV are unchanged. Not a certified full price.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t
from acro import load_project,simulate_phoenix
from acro.conditional_qmc import innovation_bank
from acro.gbm_control import gbm_reference,renewal_curve_gbm_cv
from research.benchmark_rqmc_v56 import exercise_price, summary

ROOT=Path(__file__).resolve().parents[1]
COUPON=.18

def run(quick=False, steps=48, output=None):
    p=load_project(ROOT/'config.json')
    if p.market.dynamics != 'stochastic_vol':
        raise ValueError('Reference configuration requires stochastic_vol')
    out=Path(output or ROOT/('results_gbm_cv_v56_quick' if quick else 'results_gbm_cv_v56'))
    out.mkdir(exist_ok=True,parents=True)
    b=(dict(inner=1024,reps=6,reference=32768,scrambles=6,nodes=9,outer=1800)
       if quick else dict(inner=4096,reps=10,reference=262144,scrambles=8,nodes=13,outer=6000))
    base=gbm_reference(p,COUPON,steps,b['reference'],b['scrambles'])
    print(f'[GBM] Independent reference {base.value:.5f}, SE={base.std_error:.5f}, '
          f'{base.runtime_seconds:.1f}s',flush=True)
    o=simulate_phoenix(p.market,p.initial_phoenix,b['outer'],steps,seed=0)
    # Explicit seed passed as positional integer below, do not use internal RNG defaults.
    eligible=np.isfinite(o.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= o.first_call_year <=p.acro.last_eligible_call_year + 1e-10
    if not eligible.any(): raise ValueError('No autocalls in benchmark outer market sample')
    xs=o.factor_at_call[eligible]
    lo=min(float(xs.min())-.12,p.market.initial_logvol_factor-.1)
    hi=max(float(xs.max())+.12,p.market.initial_logvol_factor+.1)
    nodes=np.unique(np.r_[np.linspace(lo,hi,b['nodes']),p.market.initial_logvol_factor])
    rec=[];total_sv=0.;total_gbm=0.;total_gen=0.
    curves={'plain':[],'gbm_cv':[]};premiums={'plain':[],'gbm_cv':[]}
    for k in range(b['reps']):
        st=perf_counter()
        bank=innovation_bank(int(round(p.renewal_phoenix.maturity_years*steps)),b['inner'],
                             p.market.n_assets,161016+k*1177,'mc_bridge')
        total_gen+=perf_counter()-st
        raw,cv,var_raw,var_diff,svtime,gbmtime=renewal_curve_gbm_cv(
            p,COUPON,steps,nodes,bank,base.value)
        total_sv+=svtime; total_gbm+=gbmtime
        idx=int(np.argmin(abs(nodes-p.market.initial_logvol_factor)))
        for name,v in [('plain',raw),('gbm_cv',cv)]:
            curves[name].append(v)
            premiums[name].append(exercise_price(p,o.first_call_year,o.factor_at_call,nodes,v))
        rec.append(dict(rep=k+1,raw_anchor=float(raw[idx]),cv_anchor=float(cv[idx]),
                        raw_premium=premiums['plain'][-1],cv_premium=premiums['gbm_cv'][-1],
                        path_variance_ratio_anchor=var_diff[idx]/var_raw[idx],
                        stock_path_cost_s=svtime,gbm_control_cost_s=gbmtime))
        print(f'[V5.6 GBM-CV] {k+1}/{b["reps"]} raw ACRO={premiums["plain"][-1]:.4f} '
              f'controlled={premiums["gbm_cv"][-1]:.4f} '
              f'variance pathwise ratio={rec[-1]["path_variance_ratio_anchor"]:.3f}',flush=True)
    with (out/'per_replication.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rec[0]));w.writeheader();w.writerows(rec)
    with (out/'conditional_curve.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['factor','plain','gbm_cv','plain_se','gbm_cv_se_cond_on_ref']);w.writeheader()
        for i,x in enumerate(nodes):
            raw=np.array(curves['plain'])[:,i];adj=np.array(curves['gbm_cv'])[:,i]
            w.writerow(dict(factor=float(x),plain=float(raw.mean()),gbm_cv=float(adj.mean()),
                            plain_se=float(raw.std(ddof=1)/np.sqrt(b['reps'])),
                            gbm_cv_se_cond_on_ref=float(adj.std(ddof=1)/np.sqrt(b['reps']))))
    data={}
    for name in curves:
        anchor=np.array(curves[name])[:,np.argmin(abs(nodes-p.market.initial_logvol_factor))]
        data[name]=dict(anchor=summary(anchor), premium_fixed_outer=summary(premiums[name]),
                        premium_of_avg_curve=exercise_price(p,o.first_call_year,o.factor_at_call,nodes,
                            np.mean(curves[name],axis=0)))
    discount_call_weight=np.mean(np.where(eligible,np.exp(-p.market.risk_free_rate*
                              np.where(eligible,o.first_call_year,0.)),0.)) * p.acro.renewal_units
    cv_se=data['gbm_cv']['anchor']['se_of_mean']
    raw_se=data['plain']['anchor']['se_of_mean']
    # This includes the statistically independent reference. Error in the option
    # premium due to reference is at most discounted_call_weight*|reference error|.
    data['gbm_cv']['anchor_se_combined_independent']=float(np.hypot(cv_se,base.std_error))
    data['gbm_cv']['reference_premium_se_bound']=float(discount_call_weight*base.std_error)
    data['costs']=dict(gbm_reference_s=base.runtime_seconds, sv_paths_s=total_sv,
                       coarse_gbm_paths_s=total_gbm, innovation_generation_s=total_gen,
                       full_cv_cost_s=base.runtime_seconds+total_sv+total_gbm+total_gen,
                       plain_mc_cost_s=total_sv+total_gen)
    cost=data['costs']
    # Same-time hypothetical MC sample-size scaling, based on iid path-mean CLT.
    # Not an observed MC experiment with extra paths; label accordingly.
    data['costs']['plain_anchor_se_scaled_to_cv_runtime']=float(
       raw_se*np.sqrt(cost['plain_mc_cost_s']/cost['full_cv_cost_s']))
    data['costs']['cv_to_mc_se_at_equal_cost_modelled']=float(
       data['gbm_cv']['anchor_se_combined_independent']/data['costs']['plain_anchor_se_scaled_to_cv_runtime'])
    data['gbm_reference']=dict(value=base.value,se=base.std_error,ci95=base.ci95,
        scrambles=base.scrambles,paths_per_scramble=base.paths_per_scramble,
        independent_from_inner_simulations=True)
    data['design']=dict(steps_per_year=steps,coupon_fixed=COUPON, inner_paths_per_rep=b['inner'],
       inner_reps=b['reps'], outer_paths_fixed=b['outer'], conditional_factor_nodes=len(nodes),
       model_and_payoffs_unchanged=True, converged_full_price=False)
    (out/'audit_gbm_cv.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    raw=data['plain'];cv=data['gbm_cv'];cost=data['costs']
    report=[
      '# ACRO — V5.6 : contrôle de variance par Phoenix Black-Scholes', '',
      f'**Coupon fixé à 18 %**, grille à {steps} pas/an. {b["reps"]} échantillons internes indépendants de {b["inner"]} trajectoires ;',
      f'{b["outer"]} scénarios externes **fixés** et {len(nodes)} états de volatilité conditionnels.',
      '', '## Baseline Black-Scholes indépendant', '',
      f'- Valeur du Phoenix GBM à coupon 18 % : **{base.value:.4f} €/100** ; SE randomisée **{base.std_error:.4f} €**.',
      f'- Intervalle t(95 %) sur scrambles : **[{base.ci95[0]:.4f} ; {base.ci95[1]:.4f}]**.',
      f'- {b["scrambles"]} séquences Sobol randomisées indépendantes de {b["reference"]} chemins chacune, avec simulation **aux seules dates de coupon/autocall**.',
      '', '## Deux méthodes sur exactement les mêmes trajectoires SV', '',
      '| Indicateur | Monte Carlo brut | Contrôle GBM |','|---|---:|---:|',
      f'| Ancrage du Phoenix renouvelé (€/100) | {raw["anchor"]["mean"]:.4f} | {cv["anchor"]["mean"]:.4f} |',
      f'| SE des réplications internes (hors baseline) | {raw["anchor"]["se_of_mean"]:.4f} | {cv["anchor"]["se_of_mean"]:.4f} |',
      f'| SE ancrage incluant baseline indépendante | {raw["anchor"]["se_of_mean"]:.4f} | {cv["anchor_se_combined_independent"]:.4f} |',
      f'| Prime ACRO sur échantillon externe fixé (€/100) | {raw["premium_fixed_outer"]["mean"]:.4f} | {cv["premium_fixed_outer"]["mean"]:.4f} |',
      f'| SE des primes entre réplications (baseline fixée) | {raw["premium_fixed_outer"]["se_of_mean"]:.4f} | {cv["premium_fixed_outer"]["se_of_mean"]:.4f} |',
      f'| Coût total interne (temps observé) | {cost["plain_mc_cost_s"]:.1f} s | {cost["full_cv_cost_s"]:.1f} s |',
      '',f'- Rapport moyen des variances **pathwise** (SV−GBM)/SV à x=0 : **{np.mean([v["path_variance_ratio_anchor"] for v in rec]):.3f}**.',
      f'- SE Monte Carlo brut projetée au **même temps total** que le CV, selon la loi en 1/√N : **{cost["plain_anchor_se_scaled_to_cv_runtime"]:.4f} €** (projection, pas mesure directe).',
      f'- Ratio SE ancrage du contrôle / SE brut projetée à temps égal : **{cost["cv_to_mc_se_at_equal_cost_modelled"]:.3f}**.',
      f'- Contribution maximale de la SE de baseline à la prime : **{cv["reference_premium_se_bound"]:.4f} €** (borne Lipschitz sur les variations).',
      '', '## Ce que ce test ne prouve pas', '',
      '- Le contrôle GBM réduit l’aléa de la cotation conditionnelle. Il ne fixe ni l’erreur de discrétisation de la dynamique à volatilité stochastique, ni le biais de Jensen, ni l’interpolation.',
      '- La prime calculée à partir des courbes moyennes est non linéaire. Une faible dispersion entre courbes ne garantit pas l’absence de biais.',
      '- L’échantillon externe est **fixé** : son erreur de Monte Carlo est absente des intervalles ci-dessus.',
      '- Les intervalles ne sont pas des prix de marché et aucune calibration implicite, crédit, frais ni couverture ne sont inclus.',
      '', 'Données : `per_replication.csv`, `conditional_curve.csv`, `audit_gbm_cv.json`.' ]
    (out/'AUDIT_GBM_CV_V5_6_FR.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    return data

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--grid',type=int,default=48)
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    run(args.quick,args.grid,args.output)
