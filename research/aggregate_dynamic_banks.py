"""Aggregate independent Q scenario banks from run_dynamic_hedge.py.

This script does NOT rerun pricing. The renewal-valuation curve remains shared
across banks when their numerical configs/seeds are identical, so bootstrap
intervals describe the scenarios, not total calibration/pricer uncertainty.
"""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro.dynamic_hedge import summarize_errors,paired_bootstrap_rmse
ROOT=Path(__file__).resolve().parents[1]

def read_rows(path):
    with open(path,encoding='utf8',newline='') as f:
        r=list(csv.DictReader(f))
    for x in r:
        for field in ['pnl_excess','fees','initial_price','coupon_cash','principal_cash','acro_cash']:
            if field in x:x[field]=float(x[field])
        x['scenario_seed']=int(x['scenario_seed'])
    return r

def write(path,rows):
    with open(path,'w',encoding='utf8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(inputs,output):
    records=[]
    for folder in inputs:
        records+=read_rows(Path(folder)/'erreurs_couverture_scenarios.csv')
    seeds={x['scenario_seed'] for x in records}
    if len(seeds)*9!=len(records):
        raise ValueError('Duplicate/incomplete scenario banks; every bank must have unique seeds')
    summary=summarize_errors(records)
    boot=paired_bootstrap_rmse(records,reps=4000,seed=77)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write(out/'simulations_agregees.csv',records)
    write(out/'synthese_120_scenarios.csv',summary)
    write(out/'bootstrap_rmse_120.csv',boot)
    fig,ax=plt.subplots(figsize=(8.9,5.2))
    methods=('none','static','dynamic')
    x=[[r['pnl_excess'] for r in records if r['component']=='package' and r['method']==m] for m in methods]
    ax.boxplot(x,tick_labels=['Sans couverture','Delta fixe','Delta rééquilibré'],showmeans=True)
    ax.axhline(0,color='gray',linewidth=.7)
    ax.set_ylabel('Écart au cash capitalisé (€/100)')
    ax.set_title('Phoenix + ACRO | 3 banques × 40 scénarios Q')
    ax.grid(axis='y',alpha=.18)
    fig.tight_layout();fig.savefig(out/'erreurs_de_couverture_120.png',dpi=170);plt.close(fig)
    package={r['method']:r for r in summary if r['component']=='package'}
    option={r['method']:r for r in summary if r['component']=='acro'}
    pnl={r['method']:r for r in summary if r['component']=='phoenix'}
    dyn=boot[0]
    calls=[r for r in records if r['component']=='package' and r['method']=='dynamic']
    n_call=sum(str(r['autocall'])=='True' for r in calls)
    n_pay=sum(r['acro_cash']>0 for r in calls)
    report=f'''# ACRO V5.14 — Agrégation des trois banques indépendantes

**Étude d'illustration sous Q, non calibrée, sans validation d'exécution réelle.**

Total {len(seeds)} scénarios indépendants, 40 par banque ; deltas estimés par 700 trajectoires par date, rééquilibrage trimestriel de l'année 1 à 2, coupon fixe 18 %, 24 pas annuels, coût 5 pb par transaction (au nominal échangé). Les banques utilisent deux courbes communes mais **indépendantes entre elles** : une pour la couverture et une plus précise pour le paiement ACRO à l’autocall. Chacune reste partagée entre scénarios.

| Composante | Stratégie | RMSE €/100 | Erreur moyenne €/100 | Frais moyens €/100 |
|---|---|---:|---:|---:|
| Phoenix + ACRO | Sans hedge | {package['none']['rmse']:.3f} | {package['none']['mean']:+.3f} | {package['none']['mean_fees']:.3f} |
| Phoenix + ACRO | Delta fixe | {package['static']['rmse']:.3f} | {package['static']['mean']:+.3f} | {package['static']['mean_fees']:.3f} |
| Phoenix + ACRO | Delta dynamique | {package['dynamic']['rmse']:.3f} | {package['dynamic']['mean']:+.3f} | {package['dynamic']['mean_fees']:.3f} |
| ACRO seule | Sans hedge | {option['none']['rmse']:.3f} | {option['none']['mean']:+.3f} | {option['none']['mean_fees']:.3f} |
| ACRO seule | Delta fixe | {option['static']['rmse']:.3f} | {option['static']['mean']:+.3f} | {option['static']['mean_fees']:.3f} |
| ACRO seule | Delta dynamique | {option['dynamic']['rmse']:.3f} | {option['dynamic']['mean']:+.3f} | {option['dynamic']['mean_fees']:.3f} |
| Phoenix seul | Delta dynamique | {pnl['dynamic']['rmse']:.3f} | {pnl['dynamic']['mean']:+.3f} | {pnl['dynamic']['mean_fees']:.3f} |

Bootstrap apparié sur scénarios, **RMSE dynamique − RMSE fixe** : {dyn['rmse_a_minus_b']:+.3f} €/100 ; intervalle percentile 95 % [{dyn['bootstrap_ci95_low']:+.3f} ; {dyn['bootstrap_ci95_high']:+.3f}]. **Ce n'est pas un intervalle global de qualité de couverture**, car les erreurs de courbe, de delta et de modèle ne sont pas prises en compte.

Autocalls observés dans cette courte fenêtre : {n_call}/{len(calls)} ; paiements ACRO positifs : {n_pay}/{len(calls)}. Cette rareté interdit de valider la couverture de l'ACRO sur ce pilote. La trajectoire de risque n'est **pas** la durée totale de 5 ans.

Les trois banques ont des erreurs de couverture variables. Le P&L est celui d'un **portefeuille long** Phoenix+ACRO avec ventes d'actions pour neutraliser le delta ; le signe est inversé pour une banque courte de la note. L'expérience ne couvre pas la vega ni la corrélation, les frais options, l'exécution OTC, l'émetteur ou le crédit. Les prix et flux à l'autocall utilisent la **même courbe de valorisation synthétique**. Ne pas l'utiliser pour attribuer une performance économique réelle à ACRO.
'''
    (out/'RAPPORT_AGREGE_V5_14_FR.md').write_text(report,encoding='utf8')
    print(report)
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',nargs='+',type=Path,required=True)
    ap.add_argument('--output',type=Path,default=ROOT/'results_dynamic_120')
    args=ap.parse_args();main(args.input,args.output)
