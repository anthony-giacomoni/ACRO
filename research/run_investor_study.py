"""ACRO V5.16: honest Q-coherence test followed by toy P investor scenarios.

Q pricing and fair coupons are never calibrated using P dynamics. The P cases
are explicit hypothetical drifts, NOT inferred forecasts or investment advice.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from acro import load_project
from acro.economic_study import simulate_full_market,compare_terminal_wealth
from acro.investor_study import (InvestorScenario,discounted_acro_cost_and_call,
    calibrate_cash_fixed_on_independent_q,paired_present_value_diff,
    apply_investor_frictions,compare_investor_outcomes,interval_of_mean)
from research.run_economic_study import basic_setup,KEYS

ROOT=Path(__file__).resolve().parents[1]

def get_bank(p,curve,q,qnc,premium,fixed, *, seed,n,excess=0.,factor_mean=None):
    H=p.initial_phoenix.maturity_years+p.renewal_phoenix.maturity_years
    log,fac=simulate_full_market(p,n,H,seed,equity_excess_drifts=[excess]*p.market.n_assets,
                                 factor_mean_override=factor_mean)
    rec=compare_terminal_wealth(p,log,fac,curve,coupon=p.initial_phoenix.coupon_rate,
        no_call_coupon=qnc.coupon_rate,premium=premium,fixed_amount=fixed,
        initial_note_price=q.price)
    return rec

def summary_ci(values):
    return dict(mean=interval_of_mean(values)[0],ci95=interval_of_mean(values)[2],
                n_banks=len(values))

def collect_q(p,q,curve,qnc,premium,fixed,*,banks,paths,seedbase):
    rows=[];pairs=[]
    for b in range(banks):
        rec=get_bank(p,curve,q,qnc,premium,fixed,seed=seedbase+b*1093,n=paths)
        v,d=discounted_acro_cost_and_call(rec,p.market.risk_free_rate)
        rows.append(dict(bank=b,n=paths,premium_pv=float(v.mean()),
            discounted_call=float(d.mean()),fixed_pv=float(fixed*d.mean()),
            diff_cost_fixed=float((v-fixed*d).mean()),
            paired_pv_wealth_diff=float(paired_present_value_diff(rec,rate=p.market.risk_free_rate).mean())))
        pairs.append(rec)
    return pd.DataFrame(rows),pairs

def make_plots(df,scenarios,qpilot,qcal,out,coupon,*,pilot_outer_se,holdout_bank_se):
    fig,ax=plt.subplots(figsize=(9,4.8))
    labels=['Pilote de pricing','Cotation contrôle Q','Évaluation Q finale']
    ys=[qpilot, qcal,df.premium_pv.mean()]
    yerrs=[1.96*pilot_outer_se,1.96*holdout_bank_se,
           1.96*df.premium_pv.std(ddof=1)/np.sqrt(len(df))]
    ax.errorbar(labels,ys,yerr=yerrs,fmt='o',capsize=5)
    ax.set_ylabel('Prime / 100 € (modèle synthétique sous Q)')
    ax.set_title('Cotation et contrôle hors échantillon')
    ax.text(0,-.20,'Barres : IC partiels (bruit externe seulement) ; erreur de courbe Q non incluse.',
            transform=ax.transAxes,fontsize=9,color='dimgray')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'01_controle_q.png',dpi=155);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.6,5))
    names=list(scenarios);v=[scenarios[n]['acro_minus_fixed_discounted_bankmeans'] for n in names]
    means=[np.mean(x) for x in v];ses=[student_se(x) for x in v]
    ax.errorbar(names,means,yerr=[1.96*x for x in ses],fmt='o',capsize=5)
    ax.axhline(0,color='gray',ls='--',lw=1)
    ax.set_ylabel('ACRO − indemnité fixe, richesse actualisée (€)')
    ax.set_title('Différence appariée par régime P — intervalles conditionnels')
    ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(out/'02_valeur_economique_scenarios_P.png',dpi=155);plt.close(fig)
    # Most informative graphic: incremental realised wealth, not total Phoenix wealth.
    data=scenarios['P neutre (illustratif)']['paired_calls']
    if data:
        a=pd.concat(data,ignore_index=True)
        z=a[a['eligible']].copy()
        if len(z):
            fig,ax=plt.subplots(figsize=(9.6,5.2))
            rr=np.random.default_rng(2026)
            shown=z.iloc[rr.choice(len(z),size=min(1500,len(z)),replace=False)]
            ax.scatter(shown['fair_coupon']*100,shown['acro_minus_fixed'],s=7,alpha=.10)
            bins=np.linspace(np.quantile(z.fair_coupon,.02),np.quantile(z.fair_coupon,.98),18)
            z['bin']=pd.cut(z.fair_coupon,bins)
            grouped=z.groupby('bin',observed=True)
            x=grouped.fair_coupon.median()*100
            ax.plot(x,grouped.acro_minus_fixed.median(),lw=2,label='Médiane conditionnelle')
            ax.fill_between(x,grouped.acro_minus_fixed.quantile(.10),
                            grouped.acro_minus_fixed.quantile(.90),alpha=.13,label='10e–90e percentiles')
            ax.axvline(coupon*100,ls='--',lw=1,color='gray',label='Coupon préservé')
            ax.axhline(0,color='gray',lw=1)
            ax.set(xlabel='Coupon de marché au rappel (%)',
                   ylabel='Richesse finale ACRO − indemnité fixe (€)',
                   title='Qui gagne quand le coupon de réinvestissement change ?')
            ax.legend();ax.grid(alpha=.15);fig.tight_layout()
            fig.savefig(out/'03_comparaison_conditionnelle.png',dpi=155);plt.close(fig)

def student_se(x):
    z=np.asarray(x,float)
    return float(np.std(z,ddof=1)/np.sqrt(len(z))) if len(z)>1 else float('nan')

def run(p,out, *, banks=4,paths=1200,quick=False,credit_stress=0.02):
    out.mkdir(parents=True,exist_ok=True)
    p.validate();r=p.market.risk_free_rate; H=p.initial_phoenix.maturity_years+p.renewal_phoenix.maturity_years
    q,curve,train,fixed_pilot,qnc,_=basic_setup(p,quick=quick)
    # Bank group A = pilot training (from old engine); group B = independent Q
    # cost/call calibration with fixed curve; group C = truly new Q evaluation.
    HOLD=990101;EVAL=880601
    hold,pairs_hold=collect_q(p,q,curve,qnc,train.price,fixed_pilot,
        banks=banks,paths=paths,seedbase=HOLD)
    # Crucial: fair fixed and premium used for Q/P comparison are calibrated on
    # holdout Q paths, not on the evaluation or subsequent P paths.
    calibrated=calibrate_cash_fixed_on_independent_q(
        np.concatenate([discounted_acro_cost_and_call(x,r)[0] for x in pairs_hold]),
        np.concatenate([discounted_acro_cost_and_call(x,r)[1] for x in pairs_hold]))
    prem=calibrated['premium'];fixed=calibrated['fixed_amount']
    qrows,qpairs=collect_q(p,q,curve,qnc,prem,fixed,banks=banks,paths=paths,seedbase=EVAL)
    qrows.to_csv(out/'controle_Q_banques.csv',index=False)
    qsummary=dict(pilot_premium=train.price,holdout_premium=prem,
        holdout_fixed_payment=fixed,holdout_discounted_call=calibrated['discounted_call_probability'],
        evaluation_premium=summary_ci(qrows.premium_pv.to_numpy()),
        evaluation_paired_diff=summary_ci(qrows.paired_pv_wealth_diff.to_numpy()),
        evaluation_cost_minus_fixed=summary_ci(qrows.diff_cost_fixed.to_numpy()),
        evaluation_discounted_fixed=summary_ci(qrows.fixed_pv.to_numpy()),
        warning='All intervals are between Q path banks CONDITIONAL on one shared renewal curve and GBM reference. Not market CIs.')
    scenarios=[
        InvestorScenario('P défavorable (illustratif)',-.04,+.10,
                         option_delivery_failure_probability=credit_stress,
                         renewal_fee_bps=10,acro_exercise_fee_bps=5),
        InvestorScenario('P neutre (illustratif)',0.,0.,
                         option_delivery_failure_probability=credit_stress,
                         renewal_fee_bps=10,acro_exercise_fee_bps=5),
        InvestorScenario('P favorable (illustratif)',+.04,-.08,
                         option_delivery_failure_probability=credit_stress,
                         renewal_fee_bps=10,acro_exercise_fee_bps=5)]
    reports={};allrows=[];riskrows=[]
    for si,scenario in enumerate(scenarios):
        inbank=[];calls=[]
        for b in range(banks):
            # Same Brownian across P scenarios, independent of Q pilot, holdout and evaluation.
            rec=get_bank(p,curve,q,qnc,prem,fixed,
                seed=660001+2011*b,n=paths,excess=scenario.excess_equity_drift,
                factor_mean=scenario.p_factor_mean)
            overlay=apply_investor_frictions(rec,rate=r,premium=prem,scenario=scenario,
                    original_nominal=p.initial_phoenix.nominal,seed=880010+1000*si+b)
            res=compare_investor_outcomes(rec,overlay,rate=r,gamma=2.)
            diff=(overlay['wealth']['acro']-overlay['wealth']['fixed'])*np.exp(-r*H)
            inbank.append(float(diff.mean()))
            riskrows.append(dict(scenario=scenario.name,bank=b,n=paths,
                  eligible_calls=res['eligible_calls'],acro_minus_fixed_pv=float(diff.mean()),
                  acro_ce=res['strategies']['acro']['crra_ce'],
                  fixed_ce=res['strategies']['fixed']['crra_ce'],
                  acro_p05=res['strategies']['acro']['p05'],
                  fixed_p05=res['strategies']['fixed']['p05'],
                  p_acro_below_fixed_at_call=res['p_acro_below_fixed_given_call'],
                  average_acro_shortfall=res['mean_shortfall_acro_given_call'],
                  average_fixed_shortfall=res['mean_shortfall_fixed_given_call']))
            elig=rec['eligible']
            calls.append(pd.DataFrame(dict(eligible=elig,
                fair_coupon=rec['fair_coupon'],
                acro_minus_fixed=np.asarray(overlay['wealth']['acro'])-np.asarray(overlay['wealth']['fixed']))))
        r_df=pd.DataFrame([rr for rr in riskrows if rr['scenario']==scenario.name]);
        reports[scenario.name]=dict(
            excess_equity_drift=scenario.excess_equity_drift,
            factor_mean_under_P=scenario.p_factor_mean,
            renewal_fee_bps=scenario.renewal_fee_bps,
            acro_exercise_fee_bps=scenario.acro_exercise_fee_bps,
            option_delivery_failure_probability=scenario.option_delivery_failure_probability,
            acro_minus_fixed_discounted_bankmeans=inbank,
            acro_minus_fixed_discounted=summary_ci(inbank),
            p_acro_below_fixed_at_call=float(r_df.p_acro_below_fixed_at_call.mean()),
            mean_acro_shortfall_at_call=float(r_df.average_acro_shortfall.mean()),
            mean_fixed_shortfall_at_call=float(r_df.average_fixed_shortfall.mean()),
            p05_acro_mean_bank=float(r_df.acro_p05.mean()),
            p05_fixed_mean_bank=float(r_df.fixed_p05.mean()),
            crra_ce_acro_mean_bank=float(r_df.acro_ce.mean()),
            crra_ce_fixed_mean_bank=float(r_df.fixed_ce.mean()),
            paired_calls=calls)
    pd.DataFrame(riskrows).to_csv(out/'scenarios_P_banques.csv',index=False)
    make_plots(qrows,reports,train.price,prem,out,p.initial_phoenix.coupon_rate,
        pilot_outer_se=train.outer_se,
        holdout_bank_se=float(hold.premium_pv.std(ddof=1)/np.sqrt(len(hold))))
    # do not serialize raw scenario DataFrames and all replications into JSON.
    for v in reports.values():
        v.pop('paired_calls');v.pop('acro_minus_fixed_discounted_bankmeans')
    result=dict(version='V5.16 Q-holdout + hypothetical P scenario sensitivity',
        assumption='Q pricing, hypothetical P drift/factor regimes. No P calibration.',
        banks=banks,paths_per_bank=paths,horizon=H,budget=q.price+prem,
        quoted_note=q.price,coupon=p.initial_phoenix.coupon_rate,
        q_diagnostics=qsummary,p_scenario_diagnostics=reports,
        limitations=['P risk premia and factor means are hypothetical, not estimated',
                     'One Q conditional renewal curve is shared between all banks and P regimes',
                     'Q holdout repricing is NOT a new independent conditional curve calibration',
                     'Q non-arbitrage expected PV equality only at fair, error-free prices',
                     'CRRA certainty equivalents compare terminal wealth with different time profiles',
                     'Execution-failure stress covers only option delivery, not full issuer default',
                     'Fees simplified; not based on executable market prices',
                     'No market-implied vol calibration, credit spreads, funding, skew or stochastic correlation'])
    (out/'metrics_investisseur.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,default=float),encoding='utf8')
    lines=['# ACRO V5.16 — Contrôle Q et scénarios P (recherche)', '',
       '**Aucune prime certifiée ni surperformance démontrée.** Hypothèses synthétiques.', '',
       '## Cohérence sous Q', '',
       f'- Prix pilote : **{train.price:.4f} €** ; repricing hors échantillon B : **{prem:.4f} €** ; évaluation C : **{qrows.premium_pv.mean():.4f} €**.',
       f'- Indemnité forfaitaire recalibrée sur B : **{fixed:.4f} €** par rappel admissible.',
       f'- Écart apparié ACRO − fixe, en PV sur C : **{qsummary["evaluation_paired_diff"]["mean"]:+.4f} €**, IC conditionnel 95 % **[{qsummary["evaluation_paired_diff"]["ci95"][0]:+.4f} ; {qsummary["evaluation_paired_diff"]["ci95"][1]:+.4f}] €**.', 
       '- Ce contrôle sépare les trois groupes de trajectoires, mais **pas** l’erreur de la courbe de renouvellement Q partagée. Il ne valide donc pas totalement le non-arbitrage.',
       '', '## Sensibilité des résultats à des scénarios P explicitement hypothétiques','',
       'On conserve les prix et coupons cotés sous Q, et on modifie seulement les dérives des actions et le niveau cible du facteur de volatilité sous P. **Ce ne sont pas des estimations de rendement réel.**',
       '', '| Scénario | ACRO − fixe (PV moyen, €) | Écart CE CRRA (€/H) | Manque de coupon : ACRO / fixe (€ au rappel) | P(ACRO < fixe) au rappel |',
       '|---|---:|---:|---:|---:|']
    for nm,v in reports.items():
        lines.append(f'| {nm} | {v["acro_minus_fixed_discounted"]["mean"]:+.3f} | {v["crra_ce_acro_mean_bank"]-v["crra_ce_fixed_mean_bank"]:+.3f} | {v["mean_acro_shortfall_at_call"]:.3f} / {v["mean_fixed_shortfall_at_call"]:.3f} | {v["p_acro_below_fixed_at_call"]:.1%} |')
    lines +=['','**Interprétation :** la protection ACRO porte sur le financement du maintien du coupon contractuel, pas sur un rendement certain. La comparaison à budget égal ne garantit pas une préférence sous P.','',
        '**Frais :** 10 pb à chaque renouvellement et 5 pb de frais supplémentaires si ACRO exercée. Un choc de non-livraison de '+str(credit_stress*100)+' % a été utilisé pour le seul droit ACRO ; il ne modélise PAS le défaut complet de l’émetteur.',
        '', '## Limites et protocole','',
        '- Contrôle Q : cotation pilote A, re-cotation B, évaluation C sur banques distinctes, mais avec la même courbe conditionnelle Q.',
        '- Sous P : trois régimes de dérive, CRRA γ=2, pertes extrêmes et richesse finale appariée. Aucune calibration P ni prime de risque empirique.',
        '- Les incertitudes des références GBM, de l’interpolation, du modèle, de la discrétisation et des banques P ne sont pas intégrées à un IC unique.',
        '- La stratégie sans autocall a une trajectoire de risque différente malgré le même horizon comptable.',
        '- Pas de mesure de crédit globale ; le stress « non-livraison » est volontairement partiel et asymétrique.',
        '- Pour finaliser : calibration de scénarios P plausibles, spreads de crédit/funding, coût d’émission et analyse du choix du client.','',
        '## Fichiers', '',
        '`controle_Q_banques.csv`, `scenarios_P_banques.csv`, `metrics_investisseur.json`, trois PNG.']
    (out/'RAPPORT_INVESTISSEUR_V5_16_FR.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('ACRO V5.16 Q pilot/holdout/evaluation:',round(train.price,4),round(prem,4),round(qrows.premium_pv.mean(),4))
    for nm,v in reports.items(): print(nm,': ACRO - fixed (PV)',round(v['acro_minus_fixed_discounted']['mean'],4))
    print('Results:',out)
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',default=str(ROOT/'config.json'))
    ap.add_argument('--output',default=None);ap.add_argument('--quick',action='store_true')
    ap.add_argument('--banks',type=int,default=4);ap.add_argument('--paths',type=int,default=2500)
    ap.add_argument('--coupon',type=float,default=.18)
    ap.add_argument('--delivery-failure',type=float,default=.02)
    args=ap.parse_args()
    p=load_project(args.config)
    p=replace(p,initial_coupon_policy='fixed',
        initial_phoenix=replace(p.initial_phoenix,coupon_rate=args.coupon),
        simulation=replace(p.simulation,steps_per_year=12,
            quote_paths=1800 if args.quick else 6500,
            outer_paths=1500 if args.quick else 7000,
            conditional_paths=1800 if args.quick else 6500,
            curve_nodes=11 if args.quick else 21,seed=91837))
    o=Path(args.output) if args.output else ROOT/('results_investor_quick' if args.quick else 'results_investor_v516')
    run(p,o,banks=2 if args.quick else args.banks,
        paths=300 if args.quick else args.paths,quick=args.quick,
        credit_stress=args.delivery_failure)
