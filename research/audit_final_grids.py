"""ACRO last numerical cross-grid check, coupon CONTRACTUALLY fixed at 18%.

Calls the DIRECT (not interpolated) exercise-policy lower/upper estimator separately
at 12/48/96 steps per year, using independent full nested banks at each grid.
It does NOT impose Brownian coupling across grids and does NOT bound the
continuous-time error.  --quick is a smoke test, not a precision audit.

CLI: python audit_final_grids.py --standard
Spyder: set RUN_STANDARD = False below, press F5 for smoke test.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro import load_project
from acro.total_premium_validation import run_bank

ROOT = Path(__file__).resolve().parents[1]
RUN_STANDARD = False

def quantile_summary(x):
    x=np.asarray(x,float)
    n=len(x)
    mid=float(x.mean()); se=float(x.std(ddof=1)/np.sqrt(n))
    return mid,se,[mid-t.ppf(.975,n-1)*se, mid+t.ppf(.975,n-1)*se]

def run(*, grids=(12,48,96), banks=12, outer=9000, selected=110,
        inner=2048,pilot=4096,ref=4096,scrambles=4,nodes=21,
        seed=5261107,output=ROOT/'results_final_grids'):
    if banks<3: raise ValueError('3 or more banks are required')
    project=load_project(ROOT/'config.json')
    project=replace(project,initial_coupon_policy='fixed',
      initial_phoenix=replace(project.initial_phoenix,coupon_rate=.18),
      renewal_phoenix=replace(project.renewal_phoenix,coupon_rate=.18))
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for step in grids:
        p=replace(project,simulation=replace(project.simulation,steps_per_year=step))
        p.validate()
        for b in range(banks):
            # Different full simulation banks across both grid and replicate.
            full_seed=int(seed+step*2_000_003+b*747_113)
            result,_=run_bank(p,seed=full_seed,outer_paths=outer,selected_states=selected,
                inner_paths=inner,pilot_inner=pilot,ref_paths=ref,
                scrambles=scrambles,pilot_nodes=nodes,margin=.6)
            result['steps_per_year']=step;result['replicate']=b
            rows.append(result)
            print(f'grid {step:3d} | bank {b+1:2d}/{banks} | '
                f'direct [{result["direct_lower"]:.4f},{result["direct_upper"]:.4f}] | '
                f'{result["time_seconds"]:.1f}s',flush=True)
        pd.DataFrame(rows).to_csv(output/'independent_banks.csv',index=False)
    data=pd.DataFrame(rows)
    summary=[]
    for step,df in data.groupby('steps_per_year'):
        lo,lo_se,lo_ci=quantile_summary(df['direct_lower'])
        hi,hi_se,hi_ci=quantile_summary(df['direct_upper'])
        # Conservative simultaneous enclosure for lower/upper expectations,
        # based on independent bank t approximation, NOT a rigorous bound.
        ci_factor=float(t.ppf(.9875,len(df)-1))
        bound=[lo-ci_factor*lo_se,hi+ci_factor*hi_se]
        summary.append(dict(steps=int(step),banks=len(df),lower=lo,upper=hi,
           midpoint=(lo+hi)/2,exercise_gap=hi-lo,
           lower_se=lo_se,upper_se=hi_se,
           conservative_lower=bound[0],conservative_upper=bound[1],
           average_seconds=float(df.time_seconds.mean())))
    s=pd.DataFrame(summary).sort_values('steps')
    s.to_csv(output/'summary_grids.csv',index=False)
    # Distinct grid experiments. Form conservative intervals for difference
    # in their expectation upper and lower endpoints; DO NOT miscall coupled.
    comparisons=[]
    from scipy.stats import t as student
    for a,b in zip(s.iloc[:-1].itertuples(),s.iloc[1:].itertuples()):
        # Independent sample normal/t approximation with conservative
        # Bonferroni bands around each mean, then interval arithmetic.
        delta=(b.midpoint-a.midpoint)
        band=(b.conservative_lower-a.conservative_upper,
              b.conservative_upper-a.conservative_lower)
        comparisons.append(dict(from_steps=a.steps,to_steps=b.steps,
             midpoint_difference=delta,conservative_difference_low=band[0],
             conservative_difference_high=band[1]))
    pd.DataFrame(comparisons).to_csv(output/'grid_comparisons.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5.1))
    ax.errorbar(s['steps'],s['midpoint'],
       yerr=[s['midpoint']-s['conservative_lower'],s['conservative_upper']-s['midpoint']],
       fmt='o-',lw=1.5,capsize=5)
    ax.set_xticks(s['steps'].to_list(),[str(i) for i in s['steps']])
    ax.set_xlabel('Pas de simulation par an')
    ax.set_ylabel('Prime ACRO (€/100)')
    ax.set_title('Prix du modèle discret : contrôle direct sur trois grilles')
    ax.grid(alpha=.2)
    fig.text(.1,.012,'Estimation sous Q synthétique, coupon fixe 18 %. Barres = encadrement statistique approximatif, PAS erreur continue ni prix marché.',fontsize=8.3)
    fig.tight_layout(rect=(0,.055,1,1))
    fig.savefig(output/'validation_grilles.png',dpi=170)
    plt.close(fig)
    objective=.02
    all_pairs_fit=all((r.conservative_difference_low>=-objective and
                       r.conservative_difference_high<=objective)
                      for r in pd.DataFrame(comparisons).itertuples())
    conf={"model":"stochastic_vol_synthetic_Q","fixed_coupon_annual":.18,
          "grids":list(grids),"banks_per_grid":banks,
          "outer_per_bank":outer,"direct_states_per_bank":selected,
          "inner_paths_per_half":inner,"pilot_paths_per_node":pilot,
          "reference_gbm_paths_per_scramble":ref,"scrambles":scrambles,
          "seed":seed,"brownian_coupled_between_grids":False,
          "target_eur_per_100":objective,
          "difference_intervals_all_within_target":bool(all_pairs_fit),
          "continuous_price_certified":False,"market_price_calibrated":False}
    (output/'protocol.json').write_text(json.dumps(conf,indent=2,ensure_ascii=False),encoding='utf8')
    fmt=lambda v: f'{v:.4f}'
    table='\n'.join(f'| {int(r.steps)} | {r.midpoint:.4f} | [{r.conservative_lower:.4f}; {r.conservative_upper:.4f}] | {r.exercise_gap:.4f} |' for r in s.itertuples())
    details='\n'.join(f'| {int(r.from_steps)}→{int(r.to_steps)} | {r.midpoint_difference:+.4f} | [{r.conservative_difference_low:+.4f}; {r.conservative_difference_high:+.4f}] |' for r in pd.DataFrame(comparisons).itertuples())
    report=f'''# Audit final du pas temporel — ACRO

**Portée :** modèle **Q synthétique**, coupon contractuel fixé à **18 %**, nominal 100 €.
Aucune calibration de marché. Les estimations directes sont évaluées aux états de rappel sans interpolation finale.

## Expérience indépendante

- {banks} banques totalement indépendantes par grille ({len(grids)} grilles). Chaque banque recalcule les trajectoires externes, le pilote, deux valeurs internes d'exercice et leurs références GBM.
- {outer:,} trajectoires externes, {selected} états directs par banque, {inner:,} chemins internes par moitié, {pilot:,} trajectoires pilotes/nœud.
- **Les grilles NE sont PAS couplées Browniennement.** Les différences entre grilles peuvent donc être bruitées. Les intervalles sont statistiques et approximatifs.

| Pas/an | Milieu descriptif (€) | Encadrement conservateur approx. (€) | Écart inférieur–supérieur (€) |
|---|---:|---:|---:|
{table}

## Comparaison entre grilles

| Grilles | Différence des milieux (€) | Intervalle conservateur par soustraction d'intervalles (€) |
|---|---:|---:|
{details}

Tolérance pré-définie pour les **différences entre grilles** : ±{objective:.2f} €/100.
**Critère strict démontré par ces comparaisons : {'OUI' if all_pairs_fit else 'NON / NON CONCLUANT'}.**

## Conclusion rigoureuse

Cet audit porte sur les prix du **modèle discrétisé**. Même en cas d'accord
entre grilles, il n'établit **aucune borne déterministe** sur la valeur du modèle continu.
Il ne couvre pas la calibration volatilité/skew/corrélation, dividendes observables,
crédit d'émetteur, liquidité, spread de financement ni frais commerciaux.
Le prix du pricer principal (qui peut recalibrer le coupon) est une autre expérience :
**ne pas mélanger ces chiffres**. Un éventuel échec à ±0,02 n'empêche pas de
terminer le projet de recherche sous réserve de présentation explicite.
'''
    (output/'RAPPORT_VALIDATION_TEMPORELLE.md').write_text(report,encoding='utf8')
    print('\n'+s[['steps','midpoint','conservative_lower','conservative_upper']].to_string(index=False))
    print('Comparaison stricte dans ±0,02 :',all_pairs_fit)
    return s

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--standard',action='store_true')
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default=str(ROOT/'results_final_grids'))
    a,_=parser.parse_known_args()
    standard = a.standard or RUN_STANDARD
    if standard:
        run(output=a.output)
    else:
        run(grids=(12,48),banks=3,outer=1200,selected=18,inner=256,
          pilot=512,ref=512,scrambles=2,nodes=9,
          output=a.output)
