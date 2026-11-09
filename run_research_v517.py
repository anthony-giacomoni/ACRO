"""V5.17 — Paired independent Q curves + comparable contingent-settlement stresses.

Run `python run_research_v517.py --quick` for a smoke test. By design,
this script makes NO claim of a fully calibrated Q/P or issuer credit model.
"""
from __future__ import annotations
import argparse, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from acro import load_project
from acro.controlled_pricing import controlled_quote, controlled_curve
from acro.investor_study import (
    InvestorScenario, discounted_acro_cost_and_call,
    calibrate_cash_fixed_on_independent_q, apply_investor_frictions,
    apply_common_contingent_settlement_stress, wealth_metrics,
    investor_target_metrics, interval_of_mean)
from research.run_investor_study import get_bank
from research.run_economic_study import basic_setup

ROOT=Path(__file__).resolve().parent


def summarize_q(rec, *, rate, F):
    c,d=discounted_acro_cost_and_call(rec,rate)
    H=rec['horizon']
    actual=(rec['wealth']['acro']-rec['wealth']['fixed'])*np.exp(-rate*H)
    return dict(pv_acro=float(c.mean()),pv_fixed=float((F*d).mean()),
                pv_theoretical_gap=float((c-F*d).mean()),
                pv_wealth_gap=float(actual.mean()),
                pv_cashflow_gap=float(np.mean(actual-(c-F*d))),
                eligible=int(np.sum(rec['eligible'])))


def calibrated_q_curve(p, seed, quick):
    paths=512 if quick else 2048
    scrambles=3 if quick else 6
    q,bank,_=controlled_quote(p,ref_paths=paths,scrambles=scrambles,seed_offset=seed)
    curve=controlled_curve(p,q,bank,np.array([-1.,1.]),ref_paths=paths,
                           scrambles=scrambles,seed_offset=seed)
    return q,curve


def grouped_ce_ci(groups, *, budget, rate, horizon, gamma=2, seed=991, boots=250):
    """Bank bootstrap; independent groups, *conditional* on pricing curves."""
    if len(groups)<2:raise ValueError('At least two independent banks required')
    x=[np.asarray(v,float) for v in groups]
    rng=np.random.default_rng(seed)
    def ce(selected):
        return wealth_metrics(np.concatenate([x[k] for k in selected]),budget=budget,
              rate=rate,horizon=horizon,gamma=gamma)['crra_ce']
    v=ce(range(len(x)))
    vals=[ce(rng.integers(0,len(x),len(x))) for _ in range(boots)]
    return dict(value=v,bank_bootstrap95=[float(v) for v in np.quantile(vals,[.025,.975])])


def run(p, out, *, banks=3, qpaths=1200, ppaths=800, quick=False,
        failure=.02, target=.02):
    out.mkdir(parents=True,exist_ok=True)
    rate=p.market.risk_free_rate
    H=p.initial_phoenix.maturity_years+p.renewal_phoenix.maturity_years
    # Two independent Q quotes/conditional curves; one no-autocall reference.
    qA,curveA,_,_,qnc,_=basic_setup(p,quick=quick)
    qB,curveB=calibrated_q_curve(p,65587,quick)
    # Contract, initial package budget and fixed indemnity are all calibrated
    # using curve A and independent HOLDOUT paths. Curve B is ONLY evaluation.
    calib=[]
    for b in range(banks):
        rec=get_bank(p,curveA,qA,qnc,0.,0.,seed=507001+1103*b,n=qpaths)
        calib.append(discounted_acro_cost_and_call(rec,rate))
    cd=np.concatenate([x[0] for x in calib]); d=np.concatenate([x[1] for x in calib])
    prices=calibrate_cash_fixed_on_independent_q(cd,d)
    premium,F=prices['premium'],prices['fixed_amount']
    qrows=[]
    for b in range(banks):
        seed=704001+1907*b
        a=get_bank(p,curveA,qA,qnc,premium,F,seed=seed,n=qpaths)
        z=get_bank(p,curveB,qA,qnc,premium,F,seed=seed,n=qpaths)
        for name,rec in [('A',a),('B',z)]:
            qrows.append(dict(bank=b,curve=name,**summarize_q(rec,rate=rate,F=F)))
    qdf=pd.DataFrame(qrows)
    qdf.to_csv(out/'controle_Q_deux_courbes.csv',index=False)
    p_scenarios=[('Défavorable',-.04,.10),('Neutre',0.,0.),('Favorable',.04,-.08)]
    outcomes=[];groups={}
    for si,(name,excess,facmean) in enumerate(p_scenarios):
        groups[name]={k:{'acro':[],'fixed':[]} for k in ('aucun','acro_seule','commun')}
        for b in range(banks):
            # Same Q price and same P market paths across all three settlement models.
            rec=get_bank(p,curveA,qA,qnc,premium,F,seed=882001+1313*b,n=ppaths,
                 excess=excess,factor_mean=facmean)
            s=InvestorScenario(name,excess,facmean,option_delivery_failure_probability=failure,
                renewal_fee_bps=10,acro_exercise_fee_bps=5)
            variants={
              'aucun':apply_investor_frictions(rec,rate=rate,premium=premium,
                  scenario=replace(s,option_delivery_failure_probability=0),
                  original_nominal=p.initial_phoenix.nominal,seed=777100+b),
              'acro_seule':apply_investor_frictions(rec,rate=rate,premium=premium,
                  scenario=s,original_nominal=p.initial_phoenix.nominal,seed=777100+b),
              'commun':apply_common_contingent_settlement_stress(rec,rate=rate,premium=premium,
                  scenario=s,original_nominal=p.initial_phoenix.nominal,seed=777100+b)}
            for mode,overlay in variants.items():
                a=np.asarray(overlay['wealth']['acro'])
                f=np.asarray(overlay['wealth']['fixed'])
                groups[name][mode]['acro'].append(a)
                groups[name][mode]['fixed'].append(f)
                diff=(a-f)*np.exp(-rate*H)
                eligible=np.asarray(rec['eligible'],bool)
                ashort=np.asarray(overlay['acro_shortfall'],float)
                fshort=np.asarray(overlay['fixed_shortfall'],float)
                outcomes.append(dict(regime=name,settlement_model=mode,bank=b,
                   n=ppaths,mean_pv_difference=float(diff.mean()),
                   p_acro_better=float(np.mean(diff>0)),
                   eligible_calls=int(np.sum(eligible)),
                   acro_unfunded_sum=float(np.sum(ashort[eligible])),
                   fixed_unfunded_sum=float(np.sum(fshort[eligible])),
                   acro_unfunded_frequency=int(np.sum(ashort[eligible]>1e-10)),
                   fixed_unfunded_frequency=int(np.sum(fshort[eligible]>1e-10)),
                   failures=int(np.sum(overlay['delivery_failed']))))
    odf=pd.DataFrame(outcomes)
    odf.to_csv(out/'scenarios_P_credit_apparies.csv',index=False)
    summary=[]
    budget=qA.price+premium
    for nm,ms in groups.items():
        for mode,ws in ms.items():
            acro=np.concatenate(ws['acro']); fixed=np.concatenate(ws['fixed'])
            ma=investor_target_metrics(acro,budget=budget,rate=rate,horizon=H,
                                      target_annual_growth=target)
            mf=investor_target_metrics(fixed,budget=budget,rate=rate,horizon=H,
                                      target_annual_growth=target)
            ca=grouped_ce_ci(ws['acro'],budget=budget,rate=rate,horizon=H,
                seed=4931+len(summary))
            cf=grouped_ce_ci(ws['fixed'],budget=budget,rate=rate,horizon=H,
                seed=5931+len(summary))
            # Paired bank bootstrap for CE *difference*, not difference of marginal CI.
            rng=np.random.default_rng(9931+len(summary));N=len(ws['acro'])
            ce_diff=[]
            for _ in range(300):
                idx=rng.integers(0,N,N)
                aa=np.concatenate([ws['acro'][i] for i in idx]);ff=np.concatenate([ws['fixed'][i] for i in idx])
                ce_diff.append(wealth_metrics(aa,budget=budget,rate=rate,horizon=H)['crra_ce']-
                               wealth_metrics(ff,budget=budget,rate=rate,horizon=H)['crra_ce'])
            e=odf[(odf.regime==nm)&(odf.settlement_model==mode)]
            total_calls=int(e.eligible_calls.sum())
            short_a=(float(e.acro_unfunded_sum.sum())/total_calls) if total_calls else None
            short_f=(float(e.fixed_unfunded_sum.sum())/total_calls) if total_calls else None
            summary.append(dict(regime=nm,settlement_model=mode,
                 eligible_calls=total_calls,
                 average_unfunded_cost_acro_when_eligible=short_a,
                 average_unfunded_cost_fixed_when_eligible=short_f,
                 p_unfunded_acro_when_eligible=(float(e.acro_unfunded_frequency.sum())/total_calls if total_calls else None),
                 p_unfunded_fixed_when_eligible=(float(e.fixed_unfunded_frequency.sum())/total_calls if total_calls else None),
                 pv_diff=ma['mean_pv']-mf['mean_pv'],
                 ce_diff=ca['value']-cf['value'],
                 ce_diff_ci_low=float(np.quantile(ce_diff,.025)),
                 ce_diff_ci_high=float(np.quantile(ce_diff,.975)),
                 es05_pv_difference=ma['es05_pv']-mf['es05_pv'],
                 p_missing_target_diff=ma['p_missing_target']-mf['p_missing_target'],
                 mean_target_shortfall_pv_diff=ma['mean_target_shortfall_pv']-mf['mean_target_shortfall_pv']))
    sdf=pd.DataFrame(summary)
    sdf.to_csv(out/'mesures_utilite_P.csv',index=False)
    # One diagram that actually compares paired P economics, not the full Phoenix variance.
    fig,ax=plt.subplots(figsize=(9.5,5.2))
    for i,(label,sub) in enumerate(sdf.groupby('settlement_model',sort=False)):
        sub=sub.set_index('regime').loc[['Défavorable','Neutre','Favorable']]
        x=np.arange(3)+(i-1)*.22
        ax.errorbar(x,sub.ce_diff.values,
                    yerr=[np.maximum(0.,sub.ce_diff.values-sub.ce_diff_ci_low.values),
                          np.maximum(0.,sub.ce_diff_ci_high.values-sub.ce_diff.values)],
                    fmt='o',capsize=4,label={'aucun':'Sans défaut contingent',
                       'acro_seule':'Défaillance ACRO seule','commun':'Défaillance commune des engagements'}[label])
    ax.axhline(0,color='gray',ls='--')
    ax.set_xticks(np.arange(3),['Défavorable','Neutre','Favorable'])
    ax.set_ylabel('ACRO − indemnité : équivalent-certain CRRA (€/100)')
    ax.set_title('Test de robustesse aux hypothèses de règlement contingent')
    ax.legend(fontsize=8);ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'01_utilite_stress_contingent.png',dpi=160);plt.close(fig)
    report=['# ACRO V5.17 — Audit Q/P indépendant (recherche)','',
      '**Aucune supériorité économique ni prime commerciale démontrée.**','',
      '## Deux courbes Q indépendantes','',
      f'- Coupon contractuel : {p.initial_phoenix.coupon_rate:.2%}. Budget initial commun : {budget:.4f} €/100.',
      f'- Quote Phoenix A : {qA.price:.4f}, quote indépendante B : {qB.price:.4f} €/100.',
      f'- Prime ACRO calibrée sous courbe A : {premium:.4f} €/100 ; indemnité F : {F:.4f} €/rappel admissible.',
      '', '| Courbe de vérification | PV ACRO | PV indemnité | Ecart payoff théorique | Ecart richesse effectivement simulée |',
      '|---|---:|---:|---:|---:|']
    for nm in ['A','B']:
        v=qdf[qdf.curve==nm]
        report.append(f'| {nm} | {v.pv_acro.mean():.4f} | {v.pv_fixed.mean():.4f} | {v.pv_theoretical_gap.mean():+.4f} | {v.pv_wealth_gap.mean():+.4f} |')
    report+=['', '**Important :** les trajectoires externes sont communes entre A et B, mais les quotes et courbes sont indépendantes. Le prix contractuel initial, la prime et F restent ceux du modèle A ; la comparaison B teste la sensibilité au pricing conditionnel. Aucune borne globale sur les biais, ni prix de marché.',
       '', '## Scénarios P hypothétiques — mêmes trajectoires, trois traitements du règlement','',
       'Le stress commun frappe les **deux engagements contingents à l’autocall**, et non l’intégralité des notes. **Ce n’est pas un modèle symétrique de défaut d’émetteur complet.**',
       '', '| Régime | Règlement | Écart de richesse PV | Écart CE CRRA | IC bootstrap des CE (banques) | Δ ES5% PV | Δ probabilité de manquer la cible |',
       '|---|---|---:|---:|---|---:|---:|']
    for row in summary:
        report.append(f'| {row["regime"]} | {row["settlement_model"]} | {row["pv_diff"]:+.3f} | {row["ce_diff"]:+.3f} | [{row["ce_diff_ci_low"]:+.3f}; {row["ce_diff_ci_high"]:+.3f}] | {row["es05_pv_difference"]:+.3f} | {row["p_missing_target_diff"]:+.1%} |')
    report+=['', f'**Objectif investisseur illustratif :** croissance annuelle nominale {target:.1%} sur {H:g} ans, pas garantie ; ES et probabilités calculés sous des P hypothétiques non calibrées.',
      '', '## Manque de financement au rappel (analyse ciblée)', '',
      '| Régime | Règlement | Nombre de rappels | Coût non financé ACRO | Coût non financé fixe |',
      '|---|---|---:|---:|---:|']
    for row in summary:
        def fnum(v): return 'n/a' if v is None else f'{v:.3f}'
        report.append(f'| {row["regime"]} | {row["settlement_model"]} | {row["eligible_calls"]} | {fnum(row["average_unfunded_cost_acro_when_eligible"])} | {fnum(row["average_unfunded_cost_fixed_when_eligible"])} |')
    report += ['', 'Cette mesure examine le **risque effectivement visé par ACRO**, contrairement à l’Expected Shortfall 5 % du portefeuille total, souvent identique dans les pires scénarios sans autocall.',
      '', '**Limites :** les intervalles bootstrap sont conditionnels à la courbe A et aux hypothèses de P ; 2 courbes seulement, non une distribution complète du risque de modèle ; pas de financement ni de défaut global calibré ; coûts d’émission approximatifs, besoin d’une vraie calibration des surfaces/skews et d’une estimation des préférences et primes de risque P.','',
      '## Résultats reproductibles','', '`controle_Q_deux_courbes.csv`, `scenarios_P_credit_apparies.csv`, `mesures_utilite_P.csv`, `01_utilite_stress_contingent.png`, `metrics_v517.json`.']
    (out/'AUDIT_QP_V5_17_FR.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    metrics=dict(version='V5.17', coupon=p.initial_phoenix.coupon_rate,
      q_quote_A=qA.price,q_quote_B=qB.price,premium_A=premium,fixed_amount=F,
      q_evaluation=qdf.groupby('curve').mean(numeric_only=True).to_dict(orient='index'),
      p_scenarios=summary,limitations=report[-3])
    (out/'metrics_v517.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf8')
    print('V5.17 Q: premium',round(premium,4),'F',round(F,4),'B holdout',round(float(qdf[qdf.curve=='B'].pv_acro.mean()),4))
    print('Output:',out)
    return metrics

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'config.json'))
    ap.add_argument('--output',default=None)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--banks',type=int,default=3)
    ap.add_argument('--qpaths',type=int,default=1200)
    ap.add_argument('--ppaths',type=int,default=800)
    ap.add_argument('--delivery-failure',type=float,default=.02)
    ap.add_argument('--target-growth',type=float,default=.02)
    args=ap.parse_args()
    p=load_project(args.config)
    p=replace(p,initial_coupon_policy='fixed',
        initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
        simulation=replace(p.simulation,steps_per_year=12,
            quote_paths=1200 if args.quick else 6500,
            outer_paths=1200 if args.quick else 7000,
            conditional_paths=1200 if args.quick else 6500,
            curve_nodes=11 if args.quick else 21,seed=91837))
    out=Path(args.output) if args.output else ROOT/('results_v517_quick' if args.quick else 'results_v517')
    run(p,out,banks=2 if args.quick else args.banks,
        qpaths=250 if args.quick else args.qpaths,
        ppaths=200 if args.quick else args.ppaths, quick=args.quick,
        failure=args.delivery_failure,target=args.target_growth)
