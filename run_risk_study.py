"""ACRO: research sensitivities of a NEW issuance vs a LIVING note.

Spyder: open this file, set working directory to project folder, press F5.
CLI: python run_risk_study.py --quick

Not a market-calibrated desk hedge. Sensitivities are noisy finite differences
under an assumed risk-neutral model, and reference/inner errors are separate.
"""
from __future__ import annotations
import argparse, csv, json
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro import load_project
from acro.pricing import quote_initial, price_acro
from acro.controlled_pricing import price_acro_controlled
from acro.live_risk import LiveState,stock_delta_live,independent_hedge_stress

ROOT=Path(__file__).resolve().parent


def fixed_coupon_project(p, coupon):
    if not (0 <= coupon < 1): raise ValueError('Invalid contractual annual coupon')
    return replace(p,initial_coupon_policy='fixed',
                   initial_phoenix=replace(p.initial_phoenix,coupon_rate=coupon))


def bump_market(m, dimension, sign):
    if dimension=='vol_parallel_1pt':
        return replace(m,volatilities=tuple(v+sign*.01 for v in m.volatilities))
    if dimension=='dividend_parallel_50bp':
        return replace(m,dividends=tuple(v+sign*.005 for v in m.dividends))
    if dimension=='rate_50bp':
        return replace(m,risk_free_rate=m.risk_free_rate+sign*.005)
    if dimension=='equity_correlation_5pt':
        cor=np.asarray(m.correlations,dtype=float).copy();notdiag=~np.eye(m.n_assets,dtype=bool)
        cor[notdiag]+=sign*.05
        return replace(m,correlations=tuple(map(tuple,cor)))
    if dimension=='spot_vol_factor_corr_10pt':
        return replace(m,equity_factor_correlation=m.equity_factor_correlation+sign*.1)
    if dimension=='vol_of_vol_5pt':
        return replace(m,factor_volatility=m.factor_volatility+sign*.05)
    raise ValueError('Unknown risk dimension')


def model_price(p,pricing_method,ref_paths):
    if pricing_method=='gbm_cv' and p.market.dynamics=='stochastic_vol':
        r,diag=price_acro_controlled(p,ref_paths=ref_paths,scrambles=3)
    else:
        r=price_acro(p,quote_initial(p));diag={}
    return r,diag


def issue_risk(p,pricing_method='gbm_cv',ref_paths=512):
    """Contract coupon frozen across bumps. Reprice BOTH note and ACRO.

    Stock-level delta deliberately absent: a new note resets its reference levels,
    so `market.initial_spots` alone has no delta content in this simulator.
    """
    base,_=model_price(p,pricing_method,ref_paths)
    keys=['vol_parallel_1pt','dividend_parallel_50bp','rate_50bp',
          'equity_correlation_5pt','spot_vol_factor_corr_10pt','vol_of_vol_5pt']
    rows=[]
    for key in keys:
        down,_=model_price(replace(p,market=bump_market(p.market,key,-1)),pricing_method,ref_paths)
        up,_=model_price(replace(p,market=bump_market(p.market,key,+1)),pricing_method,ref_paths)
        if up.quote.coupon_rate!=p.initial_phoenix.coupon_rate or down.quote.coupon_rate!=p.initial_phoenix.coupon_rate:
            raise AssertionError('Issue risk cannot re-solve the contractual coupon')
        d_acro=(up.price-down.price)/2
        d_note=(up.quote.price-down.quote.price)/2
        # Paired CRN outer error conditional on independently estimated curves.
        pathdiff=(up.discounted_payoff-down.discounted_payoff)/2
        paired_se=float(pathdiff.std(ddof=1)/np.sqrt(len(pathdiff)))
        rows.append(dict(shock=key,bump='±1 vol pt' if 'vol_parallel' in key else
                '±50 bp' if ('rate' in key or 'dividend' in key) else
                '±5 correlation pts' if 'equity_correlation' in key else
                '±10 correlation pts' if 'factor_corr' in key else '±5 vol-of-vol pts',
                acro_value_minus=down.price,acro_value_plus=up.price,
                acro_change_for_up_bump=d_acro,
                note_change_for_up_bump=d_note,
                portfolio_change_for_up_bump=d_acro+d_note,
                acro_paired_outer_se_ONLY=paired_se))
    return base,rows


def fig_issue(rows,path):
    labels={'vol_parallel_1pt':'Vol. actions +1 pt','dividend_parallel_50bp':'Dividendes +50 pb',
            'rate_50bp':'Taux +50 pb','equity_correlation_5pt':'Corrélation actions +5 pts',
            'spot_vol_factor_corr_10pt':'Levier actions-vol +10 pts',
            'vol_of_vol_5pt':'Vol. du facteur +5 pts'}
    y=np.arange(len(rows));width=.37
    fig,ax=plt.subplots(figsize=(10,5.4))
    ax.barh(y-width/2,[r['note_change_for_up_bump'] for r in rows],height=width,label='Phoenix initial')
    ax.barh(y+width/2,[r['acro_change_for_up_bump'] for r in rows],height=width,label='Droit ACRO')
    ax.set_yticks(y, [labels[r['shock']] for r in rows]);ax.axvline(0,color='gray',lw=1)
    ax.set_xlabel('Différence centrale (V+ − V−)/2, en €/100')
    ax.set_title('Nouvelle émission | coupon contractuel fixe')
    ax.grid(axis='x',alpha=.16);ax.legend(loc='best');fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def fig_live(rows,path):
    xs=np.arange(len(rows));width=.25
    fig,ax=plt.subplots(figsize=(9,5.1))
    ax.bar(xs-width,[r['delta_note'] for r in rows],width,label='Phoenix vivant')
    ax.bar(xs,[r['delta_acro'] for r in rows],width,label='Option ACRO')
    ax.bar(xs+width,[r['delta_total'] for r in rows],width,label='Ensemble')
    ax.set_xticks(xs,[f'Action {r["asset"]}' for r in rows]);ax.axhline(0,color='gray',lw=1)
    ax.set_ylabel('Variation de valeur (€) pour +1 € de prix de l’action')
    ax.set_title('Position vivante | deltas à références contractuelles fixes')
    ax.grid(axis='y',alpha=.16);ax.legend();fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def main(config=ROOT/'config.json',output=ROOT/'results_risk',quick=False,coupon=None,pricing_method='gbm_cv',high_precision=False,risk_config=ROOT/'risk_scenarios.json'):
    settings=json.loads(Path(risk_config).read_text(encoding='utf8'))
    coupon=float(settings['coupon_fixed']) if coupon is None else coupon
    p=fixed_coupon_project(load_project(config),coupon)
    if quick:
        p=replace(p,simulation=replace(p.simulation,quote_paths=1152,outer_paths=1152,
                                       conditional_paths=768,curve_nodes=9,steps_per_year=12))
        n_live=700;refs=256
    elif high_precision:
        p=replace(p,simulation=replace(p.simulation,quote_paths=4500,outer_paths=4000,
                                       conditional_paths=3300,curve_nodes=15,steps_per_year=48))
        n_live=2500;refs=1024
    else:
        p=replace(p,simulation=replace(p.simulation,quote_paths=2400,outer_paths=2200,
                                       conditional_paths=1500,curve_nodes=11,steps_per_year=24))
        n_live=1300;refs=512
    p.validate()
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    base,issue=issue_risk(p,pricing_method,refs)
    state=LiveState(time_years=float(settings['live_time_years']),
                    spot_ratios=tuple(settings['live_spot_ratios']),
                    logvol_factor=float(settings['live_logvol_factor']),
                    missed_coupons=int(settings['missed_coupons']))
    state.validate(p)
    live,live_delta=stock_delta_live(p,state,relative_bump=float(settings['spot_bump_fraction']),
                                    n_paths=n_live,pricing_method=pricing_method,ref_paths=refs,
                                    scrambles=3)
    # Numerical out-of-sample stress: evaluation paths and conditional curve
    # independent from the training of the hedge delta.
    oos=independent_hedge_stress(p,state,live_delta,
                                bumps=(.005,.01,.02),n_paths=n_live,
                                seed=p.simulation.seed+33891,
                                pricing_method=pricing_method,ref_paths=refs,scrambles=3)
    with open(out/'hedge_oos.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(oos[0]));w.writeheader();w.writerows(oos)
    with open(out/'issue_risk.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(issue[0]));w.writeheader();w.writerows(issue)
    with open(out/'live_delta.csv','w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(live_delta[0]));w.writeheader();w.writerows(live_delta)
    fig_issue(issue,out/'01_sensibilites_emission.png')
    fig_live(live_delta,out/'02_deltas_note_vivante.png')
    fig,ax=plt.subplots(figsize=(9,4.8))
    indices=np.arange(len(live_delta));width=.35
    bump_pct=100*float(settings['spot_bump_fraction'])
    ax.bar(indices-width/2,[r['bump_up_unhedged'] for r in live_delta],width,
           label=f'P&L non couvert (action +{bump_pct:g} %)')
    ax.bar(indices+width/2,[r['bump_up_after_delta_hedge'] for r in live_delta],width,
           label=f'P&L après hedge local (action +{bump_pct:g} %)')
    ax.set_xticks(indices,[f'Action {r["asset"]}' for r in live_delta])
    ax.axhline(0,lw=1,color='gray');ax.grid(axis='y',alpha=.16)
    ax.set_title('Stress local in-sample | identité de différence centrale, pas un backtest')
    ax.set_ylabel('Variation du portefeuille long (€/100 de nominal)')
    ax.legend();fig.tight_layout();fig.savefig(out/'03_couverture_delta_locale.png',dpi=170);plt.close(fig)
    # Fair same-budget indemnity: this is Q-matched, not a P utility or hedge comparison.
    residual=np.zeros(len(base.intrinsic))
    mask=base.eligible
    residual[mask]=np.maximum(base.intrinsic[mask]-base.same_price_fixed_payment,0)
    payoutpositive=float(np.mean(base.eligible & (base.intrinsic>0)))
    metrics=dict(scope='synthetic Q model, NOT market-calibrated, not executable quote',
         coupon_contractuel=coupon,step_per_year=p.simulation.steps_per_year,
         pricing_method=pricing_method,issue_option_premium=base.price,
         issue_note_value=base.quote.price,long_investor_package_value=base.price+base.quote.price,
         issue_outer_se_ONLY=base.outer_se,
         eligible_call_probability=base.call_probability,
         option_exercise_unconditional_probability_Q=payoutpositive,
         equal_Q_cost_fixed_indemnity=base.same_price_fixed_payment,
         fixed_indemnity_uncovered_shortfall_mean_at_call_unconditional_Q=float(residual.mean()),
         fixed_indemnity_uncovered_shortfall_95p_at_call_unconditional_Q=float(np.quantile(residual,.95)),
         live_state=dict(time=state.time_years,ratios=state.spot_ratios,
                         logvol=state.logvol_factor,missed_coupons=state.missed_coupons),
         live_note=live.note_price,live_option=live.option_price,live_total=live.total_price,
         live_note_outer_se_ONLY=live.note_outer_se,live_option_outer_se_ONLY=live.option_outer_se,
         live_call_probability_remaining=live.eligible_call_probability,
         live_pricing_method=pricing_method,
         out_of_sample_delta_stress='hedge_oos.csv')
    (out/'metrics_risk.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf8')
    report=f'''# ACRO — Risques de structuration | V5.12

**Démonstration sous hypothèses synthétiques, pas une cotation de marché, pas des deltas de desk certifiés.**

## Différence fondamentale

- **Émission nouvelle** : on garde un coupon contractuel de **{coupon:.2%}** entre les chocs de volatilité, taux, dividendes et corrélation. On reprixe le Phoenix ET l'ACRO sans recalculer le coupon au pair.
- **Position vivante** : exemple après **{state.time_years:g} an**, ratios des actions par rapport aux références contractuelles **{state.spot_ratios}**, facteur de volatilité {state.logvol_factor:+.2f}, {state.missed_coupons} coupons mémorisés. Ses références originales restent fixes. Les coupons déjà versés sont exclus de la valorisation.

## Résultats illustratifs (€/100)

| Mesure | Estimation |
|---|---:|
| ACRO à l'émission | {base.price:.4f} |
| Valeur du Phoenix à coupon fixe | {base.quote.price:.4f} |
| Probabilité Q de rappel admissible | {base.call_probability:.2%} |
| Phoenix vivant (futurs flux uniquement) | {live.note_price:.4f} |
| ACRO attachée à la position vivante | {live.option_price:.4f} |
| Ensemble vivant | {live.total_price:.4f} |
| Indemnité forfaitaire, même juste valeur Q | {base.same_price_fixed_payment:.4f} par rappel admissible |

## Interprétation

- `01_sensibilites_emission.png` montre des sensibilités **de modèle à l'émission**, pas le delta d'un produit déjà émis.
- `02_deltas_note_vivante.png` donne des différences centrales en **€/€ d'action** sur une note survivante. Les chocs sont appariés (mêmes graines) et la courbe conditionnelle est commune entre bumps.
- Une indemnité forfaitaire de même valeur Q ne protège pas exactement les états de détérioration du coupon ; cela ne prouve **pas** une utilité supérieure sous la probabilité réelle P. Les payoffs ACRO restent conditionnels à l'autocall.

## Nouveau contrôle hors échantillon

- `hedge_oos.csv` évalue les deltas appris sur des trajectoires et une courbe de
  valorisation indépendantes à ±0,5 %, ±1 % et ±2 %. Ce sont des **stress numériques**,
  pas des P&L de couverture dynamique. Les erreurs-types rapportées excluent
  l'erreur de la courbe de renouvellement et celle de l'estimation initiale du delta.
- En mode `gbm_cv`, la valorisation de l'ACRO vivante utilise désormais une
  référence Black-Scholes indépendante (principal et annuité), avec corrections
  stochastiques sur les mêmes chocs pour toutes les valeurs du facteur.

## Incertitudes et limites

- Les erreurs-types figurant dans les CSV et JSON sont **externes conditionnelles** ; elles excluent le bruit commun de la courbe conditionnelle, l'erreur de référence GBM, l'interpolation, Jensen et la discrétisation.
- Les Greeks sont des **différences finies indicatives** ; ils peuvent être instables près des barrières et ne sont pas des instruments de couverture directement négociables.
- À l'émission, les chocs de spot sur `market.initial_spots` ne donnent pas de delta : la référence est remise à zéro. Une note vivante nécessite un état contractuel précis.
- Le modèle conserve un seul facteur de volatilité stochastique, des corrélations actions fixes, un taux déterministe, aucun risque de crédit, frais, financement ou friction.
- L'exemple de position vivante est **hypothétique** et non une position historique observée. Il ne démontre pas la cohérence avec un historique d'actions réel.
- Aucun backtest de couverture dynamique ni comparaison de richesse terminale à horizon égal n'est réalisé ici.

## Sorties

- `issue_risk.csv` : valeurs sous choc positif et négatif, dérivée centrale et erreur-type externe appariée.
- `live_delta.csv` : deltas, gammas indicatifs, actions de couverture pour le portefeuille long et stress local *in-sample* au bump configuré. Pour l'émetteur court, les signes des actions de hedge s'inversent.
- `hedge_oos.csv` : stress hors échantillon ±0,5 %, ±1 %, ±2 % à deltas estimés sur un autre jeu de chocs aléatoires. Ne pas qualifier ces résidus de backtest.
- `metrics_risk.json` : synthèse chiffrée et comparaison forfaitaire Q.
- `01_sensibilites_emission.png`, `02_deltas_note_vivante.png`, `03_couverture_delta_locale.png` : graphiques séparés. Le troisième est un simple stress instantané avec couverture locale, **pas** un backtest de couverture dynamique.
'''
    (out/'RAPPORT_RISQUES_V5_12_FR.md').write_text(report,encoding='utf8')
    print(f'ACRO risque: coupon contrat {coupon:.2%}; note nouvelle {base.quote.price:.4f}; option nouvelle {base.price:.4f}')
    print(f'Note vivante {live.note_price:.4f}; ACRO vivante {live.option_price:.4f}; total {live.total_price:.4f}')
    print('Rapport:',out/'RAPPORT_RISQUES_V5_12_FR.md')
    return metrics

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(ROOT/'config.json'))
    parser.add_argument('--output',default=str(ROOT/'results_risk'))
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--high-precision',action='store_true')
    parser.add_argument('--coupon',type=float,default=None)
    parser.add_argument('--risk-config',default=str(ROOT/'risk_scenarios.json'))
    parser.add_argument('--pricing-method',choices=('legacy','gbm_cv'),default='gbm_cv')
    args,_=parser.parse_known_args()
    main(args.config,args.output,args.quick,args.coupon,args.pricing_method,args.high_precision,args.risk_config)
