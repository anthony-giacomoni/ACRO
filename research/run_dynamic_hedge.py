"""ACRO: exploratory multi-date DELTA hedging under synthetic Q dynamics.

In Spyder: open this file and F5. Quick mode:
    python run_dynamic_hedge.py --quick

The live position starts AFTER the observation at year 1 (contractual references
unchanged), and is followed to year 2 or an earlier autocall. The scenario
paths are independent from the nested Monte Carlo hedging estimates. This is
not a historical backtest, market execution, or demonstration of P-utility.
"""
import argparse, csv, json, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro import load_project
from acro.live_risk import LiveState,_renewal_curve
from acro.dynamic_hedge import run_scenario_hedge,summarize_errors,paired_bootstrap_rmse
from run_risk_study import fixed_coupon_project

ROOT=Path(__file__).resolve().parents[1]


def save_records(path,records):
    if not records: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]))
        w.writeheader();w.writerows(records)


def plot_error(records,out):
    methods=['none','static','dynamic']
    vals=[[r['pnl_excess'] for r in records if r['component']=='package' and r['method']==m] for m in methods]
    fig,ax=plt.subplots(figsize=(9,5.2))
    ax.boxplot(vals,tick_labels=['Sans couverture','Delta initial fixe','Delta rééquilibré'],showmeans=True)
    ax.axhline(0,color='gray',linewidth=.8)
    ax.set_ylabel('Erreur de couverture à horizon commun (€/100)')
    ax.set_title('ACRO + Phoenix | Simulation Q hors échantillon')
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def plot_split(summary,out):
    comps=['phoenix','acro','package']
    methods=['none','static','dynamic']
    fig,ax=plt.subplots(figsize=(9.8,5.2))
    x=np.arange(3);w=.24
    for j,m in enumerate(methods):
        v=[next(r['rmse'] for r in summary if r['component']==c and r['method']==m) for c in comps]
        ax.bar(x+(j-1)*w,v,width=w,label={'none':'Non couvert','static':'Delta fixe','dynamic':'Delta rééquilibré'}[m])
    ax.set_xticks(x,['Phoenix seul','ACRO seule','Ensemble'])
    ax.set_ylabel('RMSE de couverture (€/100)')
    ax.set_title('Contribution des deux instruments | Couverture expérimentale')
    ax.grid(axis='y',alpha=.18);ax.legend()
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)


def plot_path(trades,marks,out):
    # Transparent example of the actual simulated underlying path and actions.
    f,axs=plt.subplots(2,1,figsize=(10,7.1),sharex=True,layout='constrained')
    for i in range(len(marks[0].ratios)):
        axs[0].plot([v.time for v in marks],[v.ratios[i] for v in marks],marker='.',ms=3,label=f'Action {i+1}')
    axs[0].axhline(1,ls='--',color='gray',label='Seuil d’autocall')
    axs[0].axhline(.7,ls=':',color='gray',label='Barrière coupon')
    axs[0].set_ylabel('Cours / référence contractuelle')
    axs[0].set_title('Un scénario simulé | Évolution des sous-jacents')
    axs[0].legend(loc='best',ncol=3,fontsize=8);axs[0].grid(alpha=.15)
    sel=[r for r in trades if r['component']=='package' and r['strategy']=='dynamic']
    for i in range(len(marks[0].ratios)):
        axs[1].step([x['t'] for x in sel],[x['hedge_shares'][i] for x in sel],where='post',label=f'Actions {i+1}')
    axs[1].set_ylabel('Nombre de titres détenus (couverture)')
    axs[1].set_xlabel('Année contractuelle')
    axs[1].set_title('Couverture delta estimée et rééquilibrée (non calibrée)')
    axs[1].legend(ncol=3,fontsize=8);axs[1].grid(alpha=.15)
    f.savefig(out,dpi=170);plt.close(f)


def main(config=ROOT/'config.json',risk_config=ROOT/'risk_scenarios.json',output=ROOT/'results_dynamic_v514',*,
         quick=False,scenario_count=None,valuation_paths=None,end_time=2.,fee_bps=5.,seed=20261107):
    start_time=time.perf_counter()
    settings=json.loads(Path(risk_config).read_text(encoding='utf8'))
    p=fixed_coupon_project(load_project(config),float(settings['coupon_fixed']))
    steps=12 if quick else 24
    p=replace(p,simulation=replace(p.simulation,steps_per_year=steps,quote_paths=900,
             outer_paths=500,conditional_paths=420 if quick else 900,
             curve_nodes=9 if quick else 15))
    p.validate()
    if scenario_count is None: scenario_count=6 if quick else 18
    if valuation_paths is None: valuation_paths=250 if quick else 750
    if scenario_count<2 or valuation_paths<50: raise ValueError('Too few paths')
    state=LiveState(float(settings['live_time_years']),tuple(settings['live_spot_ratios']),
                    float(settings['live_logvol_factor']),int(settings['missed_coupons']))
    state.validate(p)
    if end_time<=state.time_years or end_time>p.initial_phoenix.maturity_years:
        raise ValueError('End date must be later than live state and before maturity')
    hedge_every=steps//4
    if hedge_every<1: raise ValueError('Simulation grid must support quarterly hedge dates')
    # Ex-ante fixed factor grid: not fitted on future realized scenarios.
    anchor_factors=np.linspace(-1.15,1.15,25)
    curve=_renewal_curve(p,state,anchor_factors,pricing_method='gbm_cv',
                         ref_paths=256 if quick else 1024,scrambles=3,
                         seed_offset=100)
    # Independently sample the renewal value used for cash settlement. The
    # hedger cannot reuse this independent curve when pricing/hedging.
    settle_p=replace(p,simulation=replace(p.simulation,
               conditional_paths=p.simulation.conditional_paths*2,
               curve_nodes=max(13,p.simulation.curve_nodes)))
    settlement_curve=_renewal_curve(settle_p,state,anchor_factors,
                       pricing_method='gbm_cv',ref_paths=512 if quick else 2048,
                       scrambles=3,seed_offset=9100)
    records=[];trades=[];first_marks=None
    for idx in range(int(scenario_count)):
        scenario_seed=int(seed+10007*(idx+1))
        valuation_seed=int(seed+770003+18013*idx)
        rows,hedges,marks=run_scenario_hedge(p,state,end_time=end_time,
             scenario_seed=scenario_seed,valuation_seed=valuation_seed,curve=curve,
             settlement_curve=settlement_curve,
             rebalance_every_steps=hedge_every,n_paths=int(valuation_paths),
             bump=float(settings['spot_bump_fraction']),fee_bps=float(fee_bps))
        records.extend(rows)
        trades.extend(hedges)
        if first_marks is None: first_marks=marks
    summary=summarize_errors(records)
    bootstrap=paired_bootstrap_rmse(records) if int(scenario_count)>=3 else []
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    save_records(out/'erreurs_couverture_scenarios.csv',records)
    save_records(out/'erreurs_couverture_synthese.csv',summary)
    save_records(out/'positions_couverture.csv',trades)
    save_records(out/'bootstrap_comparaison_rmse.csv',bootstrap)
    save_records(out/'premier_scenario_marche.csv',[dict(t=v.time,
         spot_ratios=v.ratios,logvol_factor=v.factor,missed_coupons=v.missed,
         coupon=v.coupon,principal=v.principal,acro_cash=v.acro,
         alive=v.alive,autocall=v.autocall) for v in first_marks])
    plot_error(records,out/'01_distribution_erreurs_couverture.png')
    plot_split(summary,out/'02_risque_phoenix_et_acro.png')
    firstseed=records[0]['scenario_seed']
    plot_path([v for v in trades if v['scenario_seed']==firstseed],first_marks,
              out/'03_scenario_et_positions.png')
    package={r['method']:r for r in summary if r['component']=='package'}
    elapsed=time.perf_counter()-start_time
    b_dyn_static=bootstrap[0]
    report=f'''# ACRO V5.14 — Couverture dynamique (pilote de recherche)

**Ni backtest historique, ni stratégie de trading validée, ni mesure de performance sous P.**

Le droit ACRO reste attaché au Phoenix vivant. Les conditions de la note et le coupon contractuel **{p.initial_phoenix.coupon_rate:.2%}** sont fixes. Démarrage immédiatement après observation à l'année **{state.time_years:g}** ; fin au plus tard à **{end_time:g}**.

### Protocole

- {scenario_count} trajectoires de marché **Q synthétiques**, graines indépendantes des simulations de pricing.
- Deltas réestimés avec **{valuation_paths}** trajectoires Monte Carlo internes par état et par date, mêmes chocs pour les bumps ±{100*settings['spot_bump_fraction']:g} % ; une courbe conditionnelle GBM-CV commune préestimée pour la couverture, et une **seconde courbe indépendante**, plus précise, pour les paiements ACRO lors du rappel.
- Rebalancement trimestriel (ou dénouement lors d'un rappel), coût proportionnel {fee_bps:g} pb par valeur échangée, dividendes et cash rémunéré au taux du modèle.
- Prix et deltas estimés sur l'état *vivant* avec références contractuelles et coupons passés correctement traités. Paiements de coupons, principal et équivalent numéraire ACRO comptabilisés **une fois**.
- P&L excédentaire par rapport au placement sans risque de la valeur initiale théorique. Si rappel anticipé, P&L roulé au même horizon de comparaison.

### Diagnostic de couverture — **illustratif**

| Méthode sur portefeuille Phoenix + ACRO | RMSE €/100 | Erreur moyenne €/100 | Frais moyens €/100 |
|---|---:|---:|---:|
| Sans couverture | {package['none']['rmse']:.3f} | {package['none']['mean']:+.3f} | {package['none']['mean_fees']:.3f} |
| Hedge delta initial fixe | {package['static']['rmse']:.3f} | {package['static']['mean']:+.3f} | {package['static']['mean_fees']:.3f} |
| Hedge delta rééquilibré | {package['dynamic']['rmse']:.3f} | {package['dynamic']['mean']:+.3f} | {package['dynamic']['mean_fees']:.3f} |

**Test d'incertitude entre scénarios :** RMSE dynamique − RMSE statique = {b_dyn_static['rmse_a_minus_b']:+.3f} €/100, bootstrap apparié IC95 [{b_dyn_static['bootstrap_ci95_low']:+.3f} ; {b_dyn_static['bootstrap_ci95_high']:+.3f}]. Cet intervalle ne reflète **pas** l'erreur Monte Carlo de revalorisation ni le risque de modèle.

**Ne pas conclure que la couverture fonctionne ou non à partir de cet échantillon limité.** L'erreur contient le bruit des prix et deltas Monte Carlo, l'erreur de courbe de renouvellement partagée, l'approximation de discrétisation et le manque de modèle de marché calibré. La comptabilité du P&L ne prouve pas que les Greeks sont exécutables.

### Limites à lever avant tout usage professionnel

- Les scénarios de risque et les paramètres sont sous **Q**, pas des trajectoires de marché réelles sous **P**.
- Les dividendes sont approximés sur chaque intervalle ; le cash est rémunéré à un seul taux sans spread ni financement de titres.
- L'ACRO est réglée en numéraire à valeur modèle lors d'un autocall (équivalence théorique à la livraison physique, hors frictions/crédit).
- Les positions sont liquidées à la fin de la fenêtre. Absence de transaction réelle, bid-ask de notes, défaut émetteur, slippage, contraintes de hedge options/volatilité/corrélation.
- Deltas recalculés avec une précision limitée ; gammas non validés ; le sous-jacent volatilité stochastique n'est pas directement couvert.
- Les courbes et prix Q partagent une erreur d'estimation commune. Aucun IC global ni efficacité de couverture en marché réel.

**Temps calculé :** {elapsed:.1f} s. Graphiques et CSV joints pour reproductibilité. Il faudra comparer des banques de scénarios indépendantes et des budgets de Greeks plus élevés avant toute conclusion économique.
'''
    (out/'RAPPORT_COUVERTURE_V5_14_FR.md').write_text(report,encoding='utf8')
    print(f'ACRO V5.14 | {scenario_count} scénarios | {elapsed:.1f} s')
    for method in ('none','static','dynamic'):
        s=package[method];print(f'{method:>8}: RMSE {s["rmse"]:.3f} | mean {s["mean"]:+.3f} | fees {s["mean_fees"]:.3f} €/100')
    print(f'Sorties : {out}')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--config',type=Path,default=ROOT/'config.json')
    parser.add_argument('--risk-config',type=Path,default=ROOT/'risk_scenarios.json')
    parser.add_argument('--output',type=Path,default=ROOT/'results_dynamic_v514')
    parser.add_argument('--scenarios',type=int)
    parser.add_argument('--valuation-paths',type=int)
    parser.add_argument('--end-time',type=float,default=2.)
    parser.add_argument('--fee-bps',type=float,default=5.)
    parser.add_argument('--seed',type=int,default=20261107)
    args=parser.parse_args()
    main(config=args.config,risk_config=args.risk_config,output=args.output,
         quick=args.quick,scenario_count=args.scenarios,valuation_paths=args.valuation_paths,
         end_time=args.end_time,fee_bps=args.fee_bps,seed=args.seed)
