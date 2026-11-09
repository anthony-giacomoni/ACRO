"""ACRO V5.15: independent Q-world same-horizon, same-budget economic study.

Run `python run_economic_study.py --quick` or F5 in Spyder.
This is NOT a real-world P-measure utility assessment or a market calibration.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
from time import perf_counter
import argparse,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from acro import load_project
from acro.engine import simulate_phoenix
from acro.config import grid_step
from acro.pricing import price_acro
from acro.controlled_pricing import controlled_quote,controlled_curve
from acro.economic_study import (simulate_full_market,compare_terminal_wealth,
                                  summarize_economics,fair_fixed_payment)

ROOT=Path(__file__).resolve().parents[1]
KEYS={'acro':'ACRO','fixed':'Indemnité fixe','reserve':'Sans ACRO (prime en cash)',
      'no_call':'Phoenix non autocallable'}


def basic_setup(p, quick=False):
    """Independent pilot for option premium, fixed indemnity and no-call coupon.

    Outputs remain noisy due to common conditional renewal curve; conditional
    error is not part of the scenario confidence intervals.
    """
    q,bank,diag=controlled_quote(p,ref_paths=512 if quick else 2048,
                                 scrambles=3 if quick else 6,
                                 seed_offset=14779)
    # One common target-value curve across all strategies, with independent
    # shocks from the evaluation scenarios. Include initial x0 exactly.
    curve=controlled_curve(p,q,bank,np.array([-1.0,1.0]),
                           ref_paths=512 if quick else 2048,
                           scrambles=3 if quick else 6,seed_offset=14779)
    train=price_acro(p,quote=q,curve_override=curve)
    # Pricer train first-call sample uses a seed 20; evaluation seeds do not.
    fixed=fair_fixed_payment(train.price,train.discounted_call_probability)
    nc=replace(p,initial_phoenix=replace(p.initial_phoenix,autocall_enabled=False),
               initial_coupon_policy='fair')
    qnc,_,_=controlled_quote(nc,ref_paths=512 if quick else 2048,
                              scrambles=3 if quick else 6,seed_offset=73921)
    return q,curve,train,fixed,qnc,diag


def plot_economics(df,stats,info,out):
    plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,
                         'savefig.bbox':'tight','figure.dpi':130})
    names=list(KEYS); colors=['#31688e','#35a88b','#ecab52','#7d67aa']
    call=df[df['eligible']]
    fig,ax=plt.subplots(figsize=(9,5.5))
    if len(call):
        cats=['ACRO','Indemnité fixe','Prime conservée en cash']
        keys=['shortfall_acro','shortfall_fixed','shortfall_reserve']
        y=[float(call[v].mean()) for v in keys]
        ax.barh(cats[::-1],y[::-1],color=['#e0ad5f','#7b9ec5','#36a48c'][::-1])
        for i,v in enumerate(y[::-1]):ax.text(v+.015,i,f'{v:.2f} €',va='center')
        ax.set_xlim(0,max(y+[.1])*1.28)
    ax.set_xlabel('Coût restant non financé à l’autocall (€/100), sous Q')
    ax.set_title('Protection du coupon cible — même budget initial',loc='left',fontweight='bold')
    ax.text(0,-.28,'Coût du renouvellement valorisé par le modèle ; hors risque de crédit et exécution.',
            transform=ax.transAxes,fontsize=9,color='#555')
    fig.savefig(out/'01_cout_renouvellement.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.5))
    x=np.arange(len(names)); w=.17
    for j,q in enumerate(['p05','p50','p95']):
        arr=[float(df[f'wealth_{z}'].quantile({'p05':.05,'p50':.5,'p95':.95}[q])) for z in names]
        ax.bar(x+(j-1)*w,arr,width=w,label={'p05':'5e percentile','p50':'Médiane','p95':'95e percentile'}[q],alpha=.83)
    ax.set_xticks(x, [KEYS[z] for z in names],rotation=12,ha='right')
    ax.set_ylabel('Richesse finale en € — coupons et capital reportés à H')
    ax.set_title('Distribution de richesse : quatre stratégies, horizon identique',loc='left',fontweight='bold')
    ax.legend(loc='upper left',ncol=3,fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out/'02_richesse_fin_horizon_commun.png');plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,5.5))
    for key,color in zip(names,colors):
        z=np.sort(df[f'wealth_{key}'].to_numpy()); f=(np.arange(len(z))+.5)/len(z)
        ax.plot(z,f,label=KEYS[key],color=color,lw=1.7)
    ax.axvline(info['budget']*np.exp(info['rate']*info['horizon']),c='#888',lw=1,ls='--',label='Budget capitalisé')
    ax.set_xlabel('Richesse finale (€)');ax.set_ylabel('Probabilité cumulée sous Q')
    ax.set_title('Risque de richesse finale — simulations appariées',loc='left',fontweight='bold')
    ax.legend(fontsize=9);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'03_distribution_wealth.png');plt.close(fig)

    fig,axs=plt.subplots(1,2,figsize=(11,4.7),gridspec_kw={'width_ratios':[1.15,1]})
    c=df.loc[df.eligible,'fair_coupon']*100
    axs[0].hist(c,bins=25,color='#528c9e',alpha=.85)
    axs[0].axvline(info['coupon']*100,color='#d47d36',ls='--',lw=1.6,label='Coupon contractuel')
    axs[0].set_xlabel('Coupon annuel au pair à la date d’autocall (%)')
    axs[0].set_ylabel('Nombre de scénarios');axs[0].legend(fontsize=8)
    t=df.loc[df.eligible,'call_year'].astype(int)
    if len(t):
        ys=t.value_counts().sort_index(); axs[1].bar(ys.index.astype(str),ys.values,color='#81a89d')
    axs[1].set_xlabel('Année du premier rappel');axs[1].set_ylabel('Nombre de scénarios')
    fig.suptitle('Conditions de réinvestissement et dates de rappel (sous Q)',fontweight='bold')
    fig.tight_layout();fig.savefig(out/'04_etats_au_rappel.png');plt.close(fig)


def run(p, out:Path, *, paths_per_bank=1800,banks=3,quick=False):
    t0=perf_counter();out.mkdir(exist_ok=True,parents=True)
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('Study implemented with stochastic_vol and controlled pricer; GBM control remains an independent zero check.')
    if p.initial_coupon_policy!='fixed':
        raise ValueError('The economic study requires a contractual coupon fixed ex ante (e.g. 18%).')
    p.validate()
    q,curve,train,fixed,qnc,_=basic_setup(p,quick)
    c=p.initial_phoenix.coupon_rate; H=p.initial_phoenix.maturity_years+p.renewal_phoenix.maturity_years
    r=p.market.risk_free_rate
    if not q.price>0: raise AssertionError('Invalid initial PV')
    data=[];rows=[]
    for bank in range(banks):
        # The full Q market path is independent of train quote and conditionals.
        log,factors=simulate_full_market(p,paths_per_bank,H,seed=260910+1009*bank)
        rec=compare_terminal_wealth(p,log,factors,curve,coupon=c,
                  no_call_coupon=qnc.coupon_rate,premium=train.price,
                  fixed_amount=fixed,initial_note_price=q.price)
        trial=pd.DataFrame(dict(bank=bank,eligible=rec['eligible'],called=rec['called'],
            call_year=np.where(rec['called'],rec['call_year'],np.nan),
            fair_coupon=rec['fair_coupon'],renewal_cost=rec['cost'],
            exercised=rec['exercise'],
            factor_at_call=np.where(rec['called'],rec['factor_at_call'],np.nan)))
        for k,x in rec['wealth'].items():trial[f'wealth_{k}']=x
        for k,x in rec['shortfall'].items():trial[f'shortfall_{k}']=x
        data.append(trial)
        for s in summarize_economics(rec,rate=r):
            s['bank']=bank; rows.append(s)
    df=pd.concat(data,ignore_index=True)
    cost=rec['budget'];call=df[df.eligible]
    stats={}
    for name in rec['wealth']:
        z=df[f'wealth_{name}'].to_numpy()
        zv=z*np.exp(-r*H)
        stats[name]=dict(wealth_mean=float(z.mean()),wealth_p05=float(np.quantile(z,.05)),
                         wealth_p50=float(np.quantile(z,.50)),wealth_p95=float(np.quantile(z,.95)),
                         discount_excess_vs_budget=float(zv.mean()-cost),
                         # Conditional on a shared trained renewal curve. Not total SE.
                         bank_conditional_se=float(pd.DataFrame(rows).query('strategy==@name')['discounted_excess_mean'].std(ddof=1)/np.sqrt(banks)) if banks>1 else None,
                         shortfall_at_call_mean=float(call[f'shortfall_{name}'].mean()) if name!='no_call' and len(call) else None)
    report=dict(version='V5.15 pilot',scenario_measure='synthetic risk-neutral Q, not physical P',
        premium_model=train.price,premium_train_outer_se_only=train.outer_se,
        fixed_payout_on_eligible_call=fixed,coupon_original=c,
        coupon_no_call=qnc.coupon_rate,initial_note_model_value=q.price,
        original_quote_se_partial=q.price_se,no_call_coupon_se_partial=qnc.coupon_se,
        budget=cost,horizon_years=H,rate=r,n_paths=len(df),banks=banks,
        paths_per_bank=paths_per_bank,eligible_autocall_probability=float(df.eligible.mean()),
        exercise_probability=float(df.exercised.mean()),
        mean_positive_renewal_cost_on_eligible=float(call.renewal_cost.mean()) if len(call) else None,
        renewal_cost_p95_on_eligible=float(call.renewal_cost.quantile(.95)) if len(call) else None,
        option_price_realized_eval_Q=float((df.renewal_cost*np.where(df.eligible,np.exp(-r*df.call_year.fillna(0)),0.)).mean()),
        fixed_pv_realized_eval_Q=float((fixed*np.where(df.eligible,np.exp(-r*df.call_year.fillna(0)),0.)).mean()),
        stats=stats,elapsed_seconds=perf_counter()-t0,
        caveats=['Model Q, synthetic market, no P measure or investor utility',
          'Fair coupon from interpolated conditional curve; calibration and interpolation risk unbounded',
          'One common conditional curve shared by all banks; bank SE excludes common pricing error',
          'Same initial budget; initial fixed-coupon note is valued at its theoretical price, not automatically par',
          'No credit/funding bid-ask, issuer delivery failure or taxation',
          'Coupon is conditional and nominal; preserving it does not guarantee yield',
          'No-call strategy reinvests available principal at year 5 in a fair Phoenix until common H',
          'Terminal Q wealth distributions do not establish risk preference or superior expected returns'])
    df.to_csv(out/'scenarios_economiques.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'repartition_par_banque.csv',index=False)
    (out/'metrics_economiques.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    plot_economics(df,stats,dict(budget=cost,horizon=H,rate=r,coupon=c),out)
    lines=['# ACRO V5.15 — Étude économique comparative (Q synthétique)','',
        '**Résumé :** comparaison à budget et horizon identiques ; aucune prétention de calibration ou de rendement garanti.','',
        f'- Coupon ACRO fixé : **{c:.2%}** ; coupon du Phoenix non autocallable au pair : **{qnc.coupon_rate:.2%}**.',
        f'- Prime ACRO pilote : **{train.price:.3f} €/100** (erreur MC externe seule ±{1.96*train.outer_se:.3f}).',
        f'- Indemnité fixe de même prix Q : **{fixed:.3f} €** par rappel admissible.',
        f'- Budget initial commun : **{cost:.3f} €** (Phoenix initial coté théoriquement {q.price:.3f} €).',
        f'- Horizon terminal commun : **{H:g} ans** ; {len(df)} scénarios Q sur {banks} banques, courbe commune.',
        f'- Rappels admissibles : **{df.eligible.mean():.1%}**, exercices ACRO positifs : **{df.exercised.mean():.1%}**.',
        f'- Option ACRO P.V. exercée sur banque d’évaluation : **{report["option_price_realized_eval_Q"]:.3f} €** (comparée au pilote : {train.price:.3f}).','',
        '## Manque de financement du coupon cible, à la date de rappel','',
        '| Stratégie | Manque moyen conditionnel au rappel (€/100) |','|---|---:|']
    for name in ('acro','fixed','reserve'):
        lines.append(f'| {KEYS[name]} | {stats[name]["shortfall_at_call_mean"]:.3f} |')
    lines+=['','La couverture totale de l’ACRO à la date de rappel est une **identité contractuelle** dans le modèle idéal. Ce n’est pas une supériorité économique gratuite : l’option est payée au départ.','',
         '## Richesse finale après réinvestissement à horizon égal','',
         '| Stratégie | Moyenne H (€) | p5 H (€) | p50 H (€) | Écart de PV moyen au budget (€) |',
         '|---|---:|---:|---:|---:|']
    for name in KEYS:
        z=stats[name]
        lines.append(f'| {KEYS[name]} | {z["wealth_mean"]:.2f} | {z["wealth_p05"]:.2f} | {z["wealth_p50"]:.2f} | {z["discount_excess_vs_budget"]:+.2f} |')
    lines+=['','Les écarts de PV sont des **contrôles Q** : pour des instruments correctement cotés, ils devraient tendre vers zéro au bruit et aux biais du modèle près. Un écart persistant indique un problème de précision, non un arbitrage exploitable.','',
        '## Limites décisives','',
        'Les scénarios sont simulés **sous Q**, pas sous P. On ne peut donc conclure ni au bénéfice réel pour l’investisseur, ni à une préférence de portefeuille sans mesure P et fonction d’utilité. Les courbes de renouvellement ne sont pas recalibrées sur données de marché et sont communes aux banques : l’incertitude statistique affichée est partielle.','',
        'Le Phoenix sans autocall est recalibré au pair, son principal à t=5 (potentiellement inférieur à 100) est réinvesti proportionnellement dans une nouvelle note au pair, puis tous les flux sont capitalisés jusqu’à H. Les stratégies ne procurent pas exactement la même exposition au risque au fil du temps.','',
        '## Fichiers','',
        '`scenarios_economiques.csv` : scénarios appariés ; `repartition_par_banque.csv` : diagnostics ; `metrics_economiques.json` : toutes les mesures ; figures 01–04 : graphiques autonomes.']
    (out/'RAPPORT_ECONOMIQUE_V5_15_FR.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(f'ACRO V5.15: premium={train.price:.3f} ; fixed={fixed:.3f} ; callQ={df.eligible.mean():.1%} ; H={H:g} years')
    print('Terminal wealth Q (mean): '+', '.join(f'{k}={v["wealth_mean"]:.2f}' for k,v in stats.items()))
    print(f'Outputs: {out.resolve()}')
    return report

if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('--config',default=str(ROOT/'config.json'))
    cli.add_argument('--output',default=None)
    cli.add_argument('--quick',action='store_true')
    cli.add_argument('--banks',type=int,default=4)
    cli.add_argument('--paths',type=int,default=2500)
    cli.add_argument('--steps',type=int,default=12)
    cli.add_argument('--coupon',type=float,default=.18,
                     help='Annual contractual coupon, decimal; default 0.18 for the reproducible research benchmark')
    arg,_=cli.parse_known_args()
    p=load_project(arg.config)
    if not 0 <= arg.coupon < 1:
        cli.error('--coupon must be in [0,1)')
    # The coupon is EX ANTE fixed for fair comparisons, unlike the default
    # stand-alone coupon-solver in config.json. Only pricing settings change.
    ns=replace(p.simulation,
        steps_per_year=arg.steps,
        quote_paths=1800 if arg.quick else 6500,
        outer_paths=1500 if arg.quick else 7000,
        conditional_paths=1800 if arg.quick else 6500,
        curve_nodes=11 if arg.quick else 21,
        seed=91837)
    p=replace(p,initial_coupon_policy='fixed',
        initial_phoenix=replace(p.initial_phoenix,coupon_rate=arg.coupon),simulation=ns)
    output=Path(arg.output) if arg.output else ROOT/('results_economic_quick' if arg.quick else 'results_economic_v515')
    run(p,output,paths_per_bank=(300 if arg.quick else arg.paths),
        banks=(2 if arg.quick else arg.banks),quick=arg.quick)
