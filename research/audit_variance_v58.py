"""V5.8 — variance of the FULL ACRO option premium (fixed contractual coupon).

Experiments at a single discretization (96 steps/year by default):
A. FIX renewal pricing curve; use independent outer market path batches.
B. FIX outer paths; use independent renewal pricing curves.
C. Date-of-call controls with INDEPENDENT pilot/train/eval samples, compared to
   plain MC using *all* the same paths, i.e. equal simulation budget.
D. Hold inner random numbers fixed to isolate linear interpolation error
   on the option premium from the conditionally estimated renewal curve.

Control variate V_SV - V_GBM uses same shocks; the reference V_GBM comes from
independent randomized Sobol scrambles. Reported errors are conditional on the
shared reference and model. Neither Jensen nor time bias is certified.

python audit_variance_v58.py --quick
python audit_variance_v58.py
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse, csv, json
import numpy as np
from scipy.stats import t as student_t
from acro import load_project, simulate_phoenix
from acro.conditional_qmc import innovation_bank
from acro.gbm_control import gbm_reference, renewal_curve_gbm_cv
from acro.renewal_anchor import renewal_curve_anchor, grid_including_anchor
from research.audit_outer_v55 import labels_for_years, years_of_eligible_calls

ROOT=Path(__file__).resolve().parents[1]
COUPON=0.18
GRID=96


def moments(x):
    x=np.asarray(x,dtype=float)
    if x.ndim!=1 or x.size<2 or not np.isfinite(x).all():
        raise ValueError('Finite independent replications required')
    mu=float(x.mean()); sd=float(x.std(ddof=1)); se=sd/np.sqrt(x.size)
    t=float(student_t.ppf(.975,x.size-1))
    return {'mean':mu,'sd_across_reps':sd,'se_of_mean':float(se),
            'ci95':[mu-t*se,mu+t*se],'replications':int(x.size)}


def discounted_payoffs(p, state, nodes, values):
    """Per-path discounted ACRO payoff and call-date label; no clipping.

    Label 0 is no eligible autocall. Numerically identical to positive-part
    option at eligible first autocall for the fixed 18% coupon experiment.
    """
    nodes=np.asarray(nodes,float);values=np.asarray(values,float)
    if nodes.ndim!=1 or values.shape!=nodes.shape or np.any(np.diff(nodes)<=0):
        raise ValueError('Bad conditional grid')
    dates=years_of_eligible_calls(p)
    years=state.first_call_year; x=state.factor_at_call
    labels=labels_for_years(years,dates)
    y=np.zeros(len(years),dtype=float)
    idx=labels>0
    if idx.any():
        if (x[idx]<nodes[0]-1e-10).any() or (x[idx]>nodes[-1]+1e-10).any():
            raise ValueError('Call factor not covered by conditional grid')
        y[idx]=np.exp(-p.market.risk_free_rate*years[idx])*p.acro.renewal_units * \
               np.maximum(np.interp(x[idx],nodes,values)-p.acro.strike,0.0)
    return y,labels


def date_control_equal_budget(ys, labels, n_train, n_pilot):
    """Unbiased date-control estimator with split pilot/train/eval.

    Fixed training sample chooses beta[k]=Ehat[Y|call_date=k]. Pilot estimates
    E[beta[call_date]] independently; evaluation applies the residual. The
    comparable naive estimator uses ALL paths, including train/pilot/eval.
    Splitting may make control worse; no hidden free paths or known p(call).
    """
    y=np.asarray(ys,float);l=np.asarray(labels,int)
    if y.ndim!=1 or l.shape!=y.shape or not np.isfinite(y).all() or (l<0).any():
        raise ValueError('Bad paths')
    if n_train<2 or n_pilot<2 or len(y)-n_train-n_pilot<2:
        raise ValueError('All three partitions need >=2 paths')
    beta=np.zeros(int(l.max())+1)
    for k in range(1,len(beta)):
        subset=y[:n_train][l[:n_train]==k]
        beta[k]=float(subset.mean()) if len(subset) else 0.
    yp=beta[l[n_train:n_train+n_pilot]]
    yr=y[n_train+n_pilot:]-beta[l[n_train+n_pilot:]]
    controlled=float(yp.mean()+yr.mean())
    controlled_conditional_se=float(np.hypot(yp.std(ddof=1)/np.sqrt(len(yp)),
                                               yr.std(ddof=1)/np.sqrt(len(yr))))
    return {'controlled':controlled,'plain_all_paths':float(y.mean()),
            'controlled_conditional_se':controlled_conditional_se,
            'plain_all_paths_se':float(y.std(ddof=1)/np.sqrt(len(y))),
            'train':int(n_train),'pilot':int(n_pilot),'eval':int(len(yr)),
            'total_paths':int(len(y))}


def simulate_outer(p,seed,n):
    return simulate_phoenix(p.market,p.initial_phoenix,n,GRID,seed)


def grid_curve(p,nodes,n,seed,reference):
    """Independent renewal curve at one grid with a common independent GBM anchor."""
    bank=innovation_bank(int(round(p.renewal_phoenix.maturity_years*GRID)),n,
                         p.market.n_assets,seed,'mc_bridge')
    raw,cv,_,_,_,_=renewal_curve_gbm_cv(p,COUPON,GRID,nodes,bank,reference)
    return raw,cv


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(*,quick=False, output=None):
    p=load_project(ROOT/'config.json')
    if p.market.dynamics!='stochastic_vol' or p.market.n_assets!=3:
        raise ValueError('V5.8 validation targets the standard three-equity SV example')
    if p.initial_phoenix.capital_barrier_monitoring!='european' or \
       p.renewal_phoenix.capital_barrier_monitoring!='european':
        raise ValueError('GBM CV requires European capital barrier')
    p=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=COUPON),
              acro=replace(p.acro,target_coupon_rate=COUPON))
    budgets= (dict(reps=5,outer=1800,inner=384,nodes=21,ref=4096,scrambles=4,
                   date_all=3000,heldout=8) if quick else
              dict(reps=6,outer=2200,inner=512,nodes=19,ref=8192,scrambles=4,
                   date_all=3600,heldout=8))
    out=Path(output or ROOT/('results_audit_v58_quick' if quick else 'results_audit_v58'))
    out.mkdir(exist_ok=True,parents=True)
    start=perf_counter()
    # Predeclared common nodes eliminate randomness in interpolation domain.
    # Wide enough for the reference configuration. Fail instead of clipping.
    nodes=grid_including_anchor(-1.4,1.0,budgets['nodes'],p.market.initial_logvol_factor)
    ref=gbm_reference(p,COUPON,GRID,budgets['ref'],budgets['scrambles'],5858001)
    curve_raw,curve_fixed=grid_curve(p,nodes,budgets['inner'],budgets['inner']*91+11,
                                    ref.value)
    print(f'[V5.8] GBM independent reference={ref.value:.5f}; CV nodes={len(nodes)}',flush=True)

    # A — fixed INNER curve, re-draw the OUTER sample.
    outer_rows=[];states=[]
    for k in range(budgets['reps']):
        state=simulate_outer(p,5858100+1009*k,budgets['outer'])
        y,l=discounted_payoffs(p,state,nodes,curve_fixed)
        outer_rows.append({'rep':k+1,'premium':float(y.mean()),
                           'conditional_outer_se':float(y.std(ddof=1)/np.sqrt(len(y))),
                           'autocall_probability':float((l>0).mean())})
        states.append(state)
    outer_stats=moments([r['premium'] for r in outer_rows])
    print(f'[V5.8] A — OUTER-only mean={outer_stats["mean"]:.5f} +/- '
          f'{outer_stats["se_of_mean"]:.5f} (1 standard error)',flush=True)

    # B — fix OUTER scenario states; use independent renewal valuation curves.
    inner_rows=[];curves=[]
    fixed_state=states[0]
    for k in range(budgets['reps']):
        t_curve=perf_counter()
        raw,cv=grid_curve(p,nodes,budgets['inner'],5868100+1709*k,ref.value)
        curve_seconds=perf_counter()-t_curve
        y,_=discounted_payoffs(p,fixed_state,nodes,cv)
        yr,_=discounted_payoffs(p,fixed_state,nodes,raw)
        curves.append(cv)
        inner_rows.append({'rep':k+1,'seconds':curve_seconds,'premium_cv':float(y.mean()),
                           'premium_plain':float(yr.mean()),
                           'max_cv_node_difference_from_first':float(np.max(np.abs(cv-curves[0])))})
    inner_stats=moments([r['premium_cv'] for r in inner_rows])
    print(f'[V5.8] B — INNER-only mean={inner_stats["mean"]:.5f} +/- '
          f'{inner_stats["se_of_mean"]:.5f}',flush=True)

    # B2 — compare new independent anchor + factor-relative curve estimator.
    # Same renewal terms, same fixed outer sample, similar inner node budget;
    # the anchor sample is counted and measured in elapsed time. This does
    # NOT remove Jensen bias or provide a globally validated option premium.
    anchor_rows=[]
    for k in range(budgets['reps']):
        seed=5868100+1709*k
        baseline_elapsed=inner_rows[k]['seconds']
        new=renewal_curve_anchor(p,COUPON,GRID,nodes,ref.value,ref.std_error,
                                 anchor_paths=4*budgets['inner'],
                                 relative_paths=budgets['inner'],seed=seed)
        y_new,_=discounted_payoffs(p,fixed_state,nodes,new.prices)
        anchor_rows.append({'rep':k+1,'old_premium':inner_rows[k]['premium_cv'],
                            'anchored_premium':float(y_new.mean()),
                            'old_seconds':baseline_elapsed,
                            'anchored_seconds':new.elapsed_seconds,
                            'anchor_value':new.anchor,'anchor_delta_se':new.anchor_delta_se,
                            'anchor_reference_se':new.anchor_reference_se,
                            'anchor_interp_error':float(np.interp(p.market.initial_logvol_factor,nodes,new.prices)-new.anchor)})
    anchor_grid_check = {
        'anchor_node_exact': bool(np.any(nodes == p.market.initial_logvol_factor)),
        'anchor_node': float(p.market.initial_logvol_factor),
        'max_abs_anchor_interp_error':float(max(abs(r['anchor_interp_error']) for r in anchor_rows))
    }
    anchor_stats={
        'direct_node_cv':moments([r['old_premium'] for r in anchor_rows]),
        'anchored_cv':moments([r['anchored_premium'] for r in anchor_rows]),
        'mean_runtime_direct_node_cv':float(np.mean([r['old_seconds'] for r in anchor_rows])),
        'mean_runtime_anchored_cv':float(np.mean([r['anchored_seconds'] for r in anchor_rows])),
        'description':'Independent anchor per replication; a common GBM reference remains fixed.'}
    anchor_stats['variance_times_cpu_ratio_anchored_to_direct']=float(
        (anchor_stats['anchored_cv']['sd_across_reps']**2*anchor_stats['mean_runtime_anchored_cv']) /
        (anchor_stats['direct_node_cv']['sd_across_reps']**2*anchor_stats['mean_runtime_direct_node_cv']))
    print(f'[V5.8] B2 — anchor vs direct CV sd: '
          f'{anchor_stats["anchored_cv"]["sd_across_reps"]:.5f} vs '
          f'{anchor_stats["direct_node_cv"]["sd_across_reps"]:.5f}; '
          f'var*CPU ratio {anchor_stats["variance_times_cpu_ratio_anchored_to_direct"]:.3f}',flush=True)

    # C — equal TOTAL outer path budget: pilot+training+evaluation are counted.
    dates=years_of_eligible_calls(p)
    strat_rows=[]
    total=budgets['date_all'];ntrain=total//5;npilot=total//5
    for k in range(budgets['reps']):
        st=simulate_outer(p,5878100+1901*k,total)
        y,l=discounted_payoffs(p,st,nodes,curve_fixed)
        row=date_control_equal_budget(y,l,ntrain,npilot)
        row.update(rep=k+1,dates=','.join(map(str,dates)))
        strat_rows.append(row)
    score_plain=moments([r['plain_all_paths'] for r in strat_rows])
    score_cv=moments([r['controlled'] for r in strat_rows])
    efficiency=(score_cv['sd_across_reps']/score_plain['sd_across_reps'])**2 if score_plain['sd_across_reps']>0 else None
    print(f'[V5.8] C — equal-cost date control vs full pooled: '
          f'{score_cv["sd_across_reps"]:.5f} vs {score_plain["sd_across_reps"]:.5f}',flush=True)

    # D — isolate the INTERPOLATION effect on premium using SAME inner
    # random bank for a dense curve; then subselect every second node.
    # This paired difference removes inner sampling noise, NOT the bias of
    # the finite-sample dense curve. Small held-out repricings use same bank.
    dense_nodes=np.linspace(nodes[0],nodes[-1],2*(len(nodes)-1)+1)
    bank=innovation_bank(int(round(p.renewal_phoenix.maturity_years*GRID)),
                         budgets['inner'],p.market.n_assets,5888111,'mc_bridge')
    _,dense,_,_,_,_=renewal_curve_gbm_cv(p,COUPON,GRID,dense_nodes,bank,ref.value)
    coarse_nodes=dense_nodes[::2];coarse_vals=dense[::2]
    ys,_=discounted_payoffs(p,fixed_state,dense_nodes,dense)
    yc,_=discounted_payoffs(p,fixed_state,coarse_nodes,coarse_vals)
    interp_rows=[]
    live=np.isfinite(fixed_state.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        live&=fixed_state.first_call_year<=p.acro.last_eligible_call_year+1e-9
    xs=fixed_state.factor_at_call[live]
    if not len(xs): raise RuntimeError('No eligible calls')
    for x in np.quantile(xs,np.linspace(.1,.9,budgets['heldout'])):
        # SAME bank as dense grid: measures interpolation relative to one
        # random finite-sample pricing function, not market/model truth.
        _,v,_,_,_,_=renewal_curve_gbm_cv(p,COUPON,GRID,np.array([float(x)]),bank,ref.value)
        interp_rows.append({'state_logvol':float(x), 'same_bank_direct':float(v[0]),
                            'linear_coarse':float(np.interp(x,coarse_nodes,coarse_vals)),
                            'linear_dense':float(np.interp(x,dense_nodes,dense)),
                            'difference_coarse_direct':float(np.interp(x,coarse_nodes,coarse_vals)-v[0]),
                            'difference_dense_direct':float(np.interp(x,dense_nodes,dense)-v[0])})
    interp={'premium_dense':float(ys.mean()),'premium_coarse':float(yc.mean()),
            'paired_premium_difference_coarse_minus_dense':float((yc-ys).mean()),
            'mean_abs_curve_error_coarse':float(np.mean(np.abs([r['difference_coarse_direct'] for r in interp_rows]))),
            'mean_abs_curve_error_dense':float(np.mean(np.abs([r['difference_dense_direct'] for r in interp_rows]))),
            'heldout_states':len(interp_rows)}
    print(f'[V5.8] D — interpolation premium coarse-dense={interp["paired_premium_difference_coarse_minus_dense"]:+.5f}',flush=True)

    # A shared GBM reference shifts every curve. Lipschitz contribution estimate.
    discounted_eligible=np.array([np.exp(-p.market.risk_free_rate*s.first_call_year[
                                      np.isfinite(s.first_call_year)]).sum()/len(s.first_call_year)
                                      for s in states])
    ref_se_bound=float(discounted_eligible.max()*p.acro.renewal_units*ref.std_error)
    result={'version':'5.8','model':'original one-factor stochastic-volatility model',
            'contract_coupon':COUPON,'time_steps_per_year':GRID,'budgets':budgets,
            'reference':{'gbm_price':ref.value,'gbm_reference_se':ref.std_error,
                         'max_premium_SE_contribution_by_lipschitz':ref_se_bound},
            'external_only':outer_stats,'internal_only':inner_stats,
            'independent_anchor_experiment':anchor_stats,
            'anchor_grid_check':anchor_grid_check,
            'date_control_equal_all_path_budget':{
                'plain_all_paths':score_plain,'controlled':score_cv,
                'variance_ratio_control_over_plain':efficiency,
                'description':'Independent train/pilot/eval; full pooled MC uses all paths.'},
            'interpolation_paired':interp,
            'elapsed_seconds':perf_counter()-start,
            'global_precision_0p02_certified':False,
            'limitations':[
                'Both A and B exclude common GBM reference uncertainty; same reference held fixed.',
                'Internal and external standard errors are diagnostic, not additive proof of total accuracy.',
                'No Jensen or temporal-discretization bias bound; 96 steps/year only.',
                'Interpolations share the same finite-simulation random pricing function.',
                'Equal-budget date stratification uses independent date controls; not direct conditional simulation.'
            ]}
    write_csv(out/'external_replicates.csv',outer_rows)
    write_csv(out/'internal_replicates.csv',inner_rows)
    write_csv(out/'anchor_comparison.csv',anchor_rows)
    write_csv(out/'date_control_equal_budget.csv',strat_rows)
    write_csv(out/'heldout_curve_interpolation.csv',interp_rows)
    (out/'AUDIT_VARIANCE_V5_8.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    md=[
        '# ACRO V5.8 — Variance de la prime complète', '',
        f'Coupon contractuel fixé **18 %**, grille **{GRID} pas/an**.',
        'Référence Black-Scholes indépendante; mêmes payoffs et même modèle que V5.7.', '',
        '| Source contrôlée | Prime moyenne (€/100) | Écart-type inter-réplications | IC95 de la moyenne |',
        '|---|---:|---:|---:|',
        f'| A : courbe fixée, externe indépendant | {outer_stats["mean"]:.4f} | {outer_stats["sd_across_reps"]:.4f} | [{outer_stats["ci95"][0]:.4f} ; {outer_stats["ci95"][1]:.4f}] |',
        f'| B : externe fixé, courbes indépendantes | {inner_stats["mean"]:.4f} | {inner_stats["sd_across_reps"]:.4f} | [{inner_stats["ci95"][0]:.4f} ; {inner_stats["ci95"][1]:.4f}] |',
        '', '## Ancrage indépendant + courbe relative', '',
        f'- Écart-type inter-réplications direct CV : {anchor_stats["direct_node_cv"]["sd_across_reps"]:.4f} € ; ancré : {anchor_stats["anchored_cv"]["sd_across_reps"]:.4f} €.',
        f'- Temps moyens par courbe : direct {anchor_stats["mean_runtime_direct_node_cv"]:.2f} s ; ancré {anchor_stats["mean_runtime_anchored_cv"]:.2f} s.',
        f'- Ratio de variance × temps (ancré/direct) : **{anchor_stats["variance_times_cpu_ratio_anchored_to_direct"]:.3f}**.',
        '- Les deux méthodes partagent ici la même référence GBM externe : son erreur commune est exclue.',
        f'- Nœud d’ancrage exact : {anchor_grid_check["anchor_node_exact"]} ; erreur maximum de restitution à x0 : {anchor_grid_check["max_abs_anchor_interp_error"]:.2e} €.',
        '', '## Contrôle par date de rappel, budget externe identique', '',
        f'- Monte Carlo **toutes trajectoires** : dispersion {score_plain["sd_across_reps"]:.4f} €.',
        f'- Contrôle dates (échantillons indépendants) : dispersion {score_cv["sd_across_reps"]:.4f} €.',
        f'- Rapport empirique des variances (contrôlé / classique) : **{efficiency:.3f}**.',
        '- Les trajectoires du pilote et de l’entraînement sont comptées ; aucun gain à coût égal n’est supposé.',
        '', '## Interpolation sur la prime', '',
        f'- Différence couplée grille espacée − grille dense : **{interp["paired_premium_difference_coarse_minus_dense"]:+.5f} €**.',
        f'- Erreur moyenne sur des états hors grille, mêmes tirages : espacée {interp["mean_abs_curve_error_coarse"]:.4f} €, dense {interp["mean_abs_curve_error_dense"]:.4f} €.',
        '- Les différences de grilles ne bornent pas l’erreur vis-à-vis de la valeur conditionnelle exacte.',
        '', '## Incertitude et décision', '',
        f'- Référence Black-Scholes : **{ref.value:.4f} €**, erreur-type **{ref.std_error:.4f} €**.',
        f'- Contribution maximale en erreur-type à la prime (Lipschitz, diagnostic) : **{ref_se_bound:.4f} €**.',
        '- Aucun de ces IC n’intègre simultanément les erreurs communes, le biais de Jensen, le biais temporel et le risque de modèle.',
        '**Tolérance totale ±0,02 €/100 non certifiée.**', '',
        'Pour conclure sur la prime du modèle, une série de grande précision et une validation temporelle indépendante restent nécessaires.', ''
    ]
    (out/'AUDIT_VARIANCE_V5_8_FR.md').write_text('\n'.join(md),encoding='utf-8')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    run(quick=args.quick,output=args.output)
