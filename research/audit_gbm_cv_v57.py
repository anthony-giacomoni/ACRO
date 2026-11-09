"""V5.7 paired 48/96/192 grid pricing audit (fixed 18% coupon).

Control: independent GBM reference, matched equity shocks for the SV/GBM
renewal payoff, common Brownian fine-grid paths for comparing time steps.
This benchmark DOES NOT certify a ±0.02 EUR total pricing error.

python audit_gbm_cv_v57.py --quick
python audit_gbm_cv_v57.py
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse,csv,json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from scipy.interpolate import PchipInterpolator
from acro import load_project,simulate_phoenix
from acro.gbm_control import gbm_reference,renewal_curve_gbm_cv,aggregated_bank
from research.audit_coupled import innovations_for_grid
from research.benchmark_rqmc_v56 import exercise_price

ROOT=Path(__file__).resolve().parents[1]
GRIDS=(48,96,192)
COUPON=.18


def ci(values):
    x=np.asarray(values,float)
    if len(x)<2:raise ValueError('Need multiple independent replications')
    se=float(np.std(x,ddof=1)/np.sqrt(len(x)))
    half=float(student_t.ppf(.975,len(x)-1)*se)
    return {'mean':float(x.mean()),'se_replication':se,'ci95':[float(x.mean()-half),float(x.mean()+half)],'n':len(x)}


def eligible_factors(outer,p):
    m=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        m &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    return outer.factor_at_call[m]


def run(quick=False, output=None):
    p=load_project(ROOT/'config.json')
    if p.market.dynamics!='stochastic_vol': raise ValueError('Only stochastic_vol benchmark')
    if p.initial_phoenix.capital_barrier_monitoring!='european' or p.renewal_phoenix.capital_barrier_monitoring!='european':
        raise ValueError('Only European capital barriers')
    p=replace(p,initial_coupon_policy='fixed',
              initial_phoenix=replace(p.initial_phoenix,coupon_rate=COUPON),
              acro=replace(p.acro,target_coupon_rate=COUPON))
    budget=(dict(reps=4,inner=768,outer=1600,nodes=19,ref=16384,scrambles=6)
            if quick else dict(reps=5,inner=1024,outer=2300,nodes=17,ref=32768,scrambles=8))
    output=Path(output or ROOT/('results_audit_v57_quick' if quick else 'results_audit_v57'))
    output.mkdir(parents=True,exist_ok=True)
    t0=perf_counter()
    # Same sparse-GBM price under all three grids, independently simulated.
    ref=gbm_reference(p,COUPON,192,budget['ref'],budget['scrambles'],base_seed=575701)
    print(f'[V5.7] Common independent GBM reference {ref.value:.4f} +/- '
          f'{1.96*ref.std_error:.4f} (normal approx)',flush=True)
    fine=192; T=p.initial_phoenix.maturity_years
    Ti=p.renewal_phoenix.maturity_years
    if abs(T-Ti)>1e-12: raise ValueError('Audit currently requires equal initial and renewal maturities')
    outer_data=[];inner_banks=[];observed_factors=[]
    outer_seconds=0.
    n_assets=p.market.n_assets
    # Independent repetitions; COMMON fine Brownian shocks within each repetition.
    for k in range(budget['reps']):
        bank_o=innovations_for_grid(int(round(T*fine)),budget['outer'],n_assets,573001+3001*k)
        bank_i=innovations_for_grid(int(round(Ti*fine)),budget['inner'],n_assets,574001+3001*k)
        inner_banks.append(bank_i)
        o={}
        for g in GRIDS:
            ob=aggregated_bank(bank_o,fine,g)
            t_outer=perf_counter()
            state=simulate_phoenix(p.market,p.initial_phoenix,budget['outer'],g,0,innovations=ob)
            outer_seconds+=perf_counter()-t_outer
            o[g]=state
            v=eligible_factors(state,p)
            if len(v):observed_factors.append(v)
        outer_data.append(o)
        del bank_o
    if not observed_factors:raise ValueError('No eligible autocall states')
    allstates=np.concatenate(observed_factors)
    lo=min(float(allstates.min())-.07,p.market.initial_logvol_factor-.05)
    hi=max(float(allstates.max())+.07,p.market.initial_logvol_factor+.05)
    nodes=np.unique(np.r_[np.linspace(lo,hi,budget['nodes']),p.market.initial_logvol_factor])
    rows=[];heldout=[];rep_diff=[];method_cost={'plain_s':0.,'cv_extra_s':ref.runtime_seconds}
    for k in range(budget['reps']):
        values={}; bank=inner_banks[k]
        for g in GRIDS:
            ib=aggregated_bank(bank,fine,g)
            st=perf_counter()
            raw,cv,vr,vd,svtime,gbmtime=renewal_curve_gbm_cv(p,COUPON,g,nodes,ib,ref.value)
            total=perf_counter()-st
            method_cost['plain_s']+=svtime
            method_cost['cv_extra_s']+=gbmtime
            o=outer_data[k][g]
            raw_price=exercise_price(p,o.first_call_year,o.factor_at_call,nodes,raw)
            cv_price=exercise_price(p,o.first_call_year,o.factor_at_call,nodes,cv)
            values[g]=cv_price
            mask=np.isfinite(o.first_call_year)
            if p.acro.last_eligible_call_year is not None:mask&=o.first_call_year<=p.acro.last_eligible_call_year+1e-10
            rows.append(dict(rep=k+1,steps_per_year=g,price_cv=cv_price,price_plain=raw_price,
                             call_probability=float(mask.mean()),conditional_nodes=len(nodes),
                             inner_paths=budget['inner'],outer_paths=budget['outer'],
                             inner_elapsed_seconds=total,
                             raw_anchor_var=float(vr[np.argmin(np.abs(nodes))]),
                             cv_anchor_variance_ratio=float(vd[np.argmin(np.abs(nodes))]/vr[np.argmin(np.abs(nodes))])))
            # Held-out direct values computed on EXACT same inner Brownian bank.
            if g==96:
                factors=eligible_factors(o,p)
                for x in np.quantile(factors,[.10,.35,.60,.85]):
                    single=simulate_phoenix(p.market,p.renewal_phoenix,budget['inner'],g,0,
                                            initial_factor=float(x),innovations=ib).pv(COUPON)
                    gbm=simulate_phoenix(replace(p.market,dynamics='gbm'),p.renewal_phoenix,
                                         budget['inner'],p.renewal_phoenix.observations_per_year,
                                         0,innovations=aggregated_bank(ib,g,p.renewal_phoenix.observations_per_year)).pv(COUPON)
                    direct=ref.value+float((single-gbm).mean())
                    approximate=float(np.interp(x,nodes,cv))
                    heldout.append(dict(rep=k+1,grid=g,x=float(x),direct_same_inner=direct,
                                        interpolated_same_inner=approximate,
                                        error_interpolation=approximate-direct,
                                        error_pchip=float(PchipInterpolator(nodes,cv)(x)-direct)))
        rep_diff.append(dict(rep=k+1,delta_96_48=values[96]-values[48],
                             delta_192_96=values[192]-values[96],
                             delta_192_48=values[192]-values[48]))
        print(f'[V5.7] {k+1}/{budget["reps"]} 48/96/192 = '
              + ' / '.join(f'{values[g]:.4f}' for g in GRIDS),flush=True)
    def to_csv(file,data):
        with (output/file).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    to_csv('grid_replicates.csv',rows);to_csv('paired_differences.csv',rep_diff)
    to_csv('interpolation_heldout.csv',heldout)
    summaries={str(g):ci([r['price_cv'] for r in rows if r['steps_per_year']==g]) for g in GRIDS}
    differences={key:ci([r[key] for r in rep_diff]) for key in ('delta_96_48','delta_192_96','delta_192_48')}
    interpolation=dict(mean_abs_error=float(np.mean(np.abs([z['error_interpolation'] for z in heldout]))),
                       mean_abs_error_pchip=float(np.mean(np.abs([z['error_pchip'] for z in heldout]))),
                       max_abs_error_observed=float(np.max(np.abs([z['error_interpolation'] for z in heldout]))),
                       number_states=len(heldout))
    elapsed=perf_counter()-t0
    # A common independent GBM reference is not included in replication CI.
    # Upper bound for its premium contribution is E[e^-r*tau I_call]*SE(ref).
    weights=[]
    for dat in outer_data:
        o=dat[192];m=np.isfinite(o.first_call_year)
        if p.acro.last_eligible_call_year is not None:m&=o.first_call_year<=p.acro.last_eligible_call_year+1e-10
        weights.append(float(np.exp(-p.market.risk_free_rate*o.first_call_year[m]).sum()/len(m)))
    ref_bound=max(weights)*p.acro.renewal_units*ref.std_error
    # Empirical means and noise at the same simulated inner/outer path counts.
    # The CV runtime includes the independent baseline, which plain MC does not need.
    plain_summaries={str(g):ci([r['price_plain'] for r in rows if r['steps_per_year']==g]) for g in GRIDS}
    cost_plain=method_cost['plain_s']+outer_seconds
    cost_cv=cost_plain+method_cost['cv_extra_s']
    # variance*CPU is a theoretical efficiency score; it is not a new
    # matched-runtime simulation, and CV reference error is shared across reps.
    efficiency={str(g):{'raw_se':plain_summaries[str(g)]['se_replication'],
                        'cv_se_excluding_common_ref':summaries[str(g)]['se_replication'],
                        'raw_variance_cpu_score':plain_summaries[str(g)]['se_replication']**2*cost_plain,
                        'cv_variance_cpu_score_excluding_reference_error':summaries[str(g)]['se_replication']**2*cost_cv}
                for g in GRIDS}
    target_halfwidth=.02
    # Strictly a failed/undetermined certification: no independently verified full-error CI.
    status='NON_VALIDÉ — global ±0.02 cannot be certified from these tests'
    out=dict(method='GBM conditional control; fixed 18% coupon',grids=GRIDS,budget=budget,
             results=summaries,raw_results=plain_summaries,efficiency_diagnostic=efficiency,
             paired_differences=differences,
             ref_gbm=dict(price=ref.value,se=ref.std_error,independent=True,
                          max_first_order_se_contribution_to_premium=ref_bound),
             heldout=interpolation,elapsed_seconds=elapsed,costs=dict(**method_cost,outer_s=outer_seconds,
                       raw_cost_s=cost_plain,cv_cost_s=cost_cv),
             note='CIs exclude common GBM ref, interpolation, inner Jensen bias, time error and model error.',
             tolerance_eur_per_100=target_halfwidth,convergence_status=status)
    (output/'AUDIT_GBM_CV_V5_7.json').write_text(json.dumps(out,indent=2,ensure_ascii=False),encoding='utf-8')
    md=['# ACRO — Audit V5.7', '', '**Coupon contractuel fixé à 18 %**. Mêmes Browniennes pour 48/96/192 pas/an.','',
        '| Grille | Prime moyenne €/100 | IC 95 % entre réplications |', '|---|---:|---:|']
    for g in GRIDS:
        v=summaries[str(g)];md.append(f'| {g} | {v["mean"]:.4f} | [{v["ci95"][0]:.4f}; {v["ci95"][1]:.4f}] |')
    md+=['','| Écart couplé | Moyenne €/100 | IC 95 % |','|---|---:|---:|']
    for key,v in differences.items():
        md.append(f'| {key} | {v["mean"]:+.4f} | [{v["ci95"][0]:+.4f}; {v["ci95"][1]:+.4f}] |')
    md += ['',f'- Référence GBM indépendante : **{ref.value:.4f} €**, erreur-type entre scrambles **{ref.std_error:.4f} €**.',
           f'- Borne Lipschitz sur la contribution de cette erreur-type commune à la prime : **{ref_bound:.4f} €**.',
           f'- Erreur moyenne absolue d’interpolation linéaire sur {len(heldout)} états hors grille (mêmes tirages internes) : **{interpolation["mean_abs_error"]:.4f} €** ; PCHIP : **{interpolation["mean_abs_error_pchip"]:.4f} €**.',
           f'- Coût observé sur la série : **{elapsed:.1f} s** (référence GBM incluse).',
           f'- Surcoût contrôlé estimé à partir des temps mesurés : brut **{cost_plain:.1f} s**, CV **{cost_cv:.1f} s** (temps commun de génération et de sauvegarde non inclus).',
           f'- À 96 pas/an : erreur-type inter-réplications brute **{efficiency["96"]["raw_se"]:.4f} €** contre CV **{efficiency["96"]["cv_se_excluding_common_ref"]:.4f} €**, sans l’erreur commune de référence.',
           '', '**Verdict : tolérance ±0,02 €/100 non certifiée.**',
           '', 'Les intervalles entre réplications contiennent les variations internes et externes, mais pas l’erreur commune de référence GBM ni les biais temporel, Jensen et interpolation. Les erreurs d’interpolation à mêmes tirages ne sont pas une borne absolue. Les résultats synthétiques ne sont pas des cotations de marché.']
    (output/'AUDIT_GBM_CV_V5_7_FR.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
    fig,ax=plt.subplots(figsize=(10,6))
    for k in range(1,budget['reps']+1):
        x=[r['steps_per_year'] for r in rows if r['rep']==k]
        y=[r['price_cv'] for r in rows if r['rep']==k]
        ax.plot(x,y,marker='.',alpha=.27,linewidth=1.1)
    mean=[summaries[str(g)]['mean'] for g in GRIDS]
    lo=[mean[i]-summaries[str(g)]['ci95'][0] for i,g in enumerate(GRIDS)]
    hi=[summaries[str(g)]['ci95'][1]-mean[i] for i,g in enumerate(GRIDS)]
    ax.errorbar(GRIDS,mean,yerr=[lo,hi],marker='o',linewidth=2,capsize=5,
                label='Moyenne et IC 95 % entre réplications')
    ax.set_xticks(GRIDS)
    ax.set_xlabel('Pas de simulation par an')
    ax.set_ylabel('Prime ACRO (€/100)')
    ax.set_title('Convergence temporelle à coupon fixé de 18 % — V5.7')
    ax.grid(alpha=.2)
    ax.legend(loc='best')
    fig.text(.1,.02,'Trajectoires browniennes couplées ; IC hors référence GBM commune et biais numériques.',fontsize=9)
    fig.tight_layout(rect=(0,.05,1,1))
    fig.savefig(output/'convergence_gbm_cv_v57.png',dpi=160)
    plt.close(fig)
    return out


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--quick',action='store_true');ap.add_argument('--output',type=Path)
    args=ap.parse_args();run(args.quick,args.output)
