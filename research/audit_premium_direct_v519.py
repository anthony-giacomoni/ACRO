"""Run from Spyder or CLI: direct, no-interpolation Q ACRO premium audit.

Default quick benchmark is illustrative, NOT market price certification.
Larger settings may require substantial CPU/RAM; --quick smoke is minimal.
"""
from __future__ import annotations
import argparse,json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro import load_project
from acro.total_premium_validation import run_bank,ci_banks

ROOT=Path(__file__).resolve().parents[1]

def execute(out,banks=6,outer=4500,selected=42,inner=768,pilot=2048,
            ref_paths=2048,scrambles=4,nodes=15,steps=12,seed=519013,margin=.6):
    p=load_project(ROOT/'config.json')
    # Contractual Q benchmark: avoid re-solving a noisy fair coupon on each grid.
    p=replace(p,initial_coupon_policy='fixed',
      initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
      renewal_phoenix=replace(p.renewal_phoenix,coupon_rate=.18),
      simulation=replace(p.simulation,steps_per_year=steps))
    p.validate()
    if banks<2:raise ValueError('At least 2 independent full banks required')
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    result=[];states=[]
    for k in range(banks):
        r,selected_rows=run_bank(p,seed=seed+k*743911,outer_paths=outer,
              selected_states=selected,inner_paths=inner,
              ref_paths=ref_paths,scrambles=scrambles,pilot_inner=pilot,
              pilot_nodes=nodes,margin=margin)
        r['bank']=k;result.append(r)
        for row in selected_rows:
            row['bank']=k;states.append(row)
        print(f"Bank {k+1}/{banks}: direct bounds [{r['direct_lower']:.4f}, {r['direct_upper']:.4f}], "
              f"pilot={r['pilot_premium']:.4f}, gap={r['exercise_policy_gap']:.4f}, "
              f"{r['time_seconds']:.1f}s",flush=True)
    df=pd.DataFrame(result);state=pd.DataFrame(states)
    df.to_csv(out/'banques_prime_directe.csv',index=False)
    state.to_csv(out/'etats_stratifies.csv',index=False)
    raw_lo,raw_lo_ci,_=ci_banks(df.raw_direct_lower)
    raw_hi,raw_hi_ci,_=ci_banks(df.raw_direct_upper)
    lo,lo_ci,lo_se=ci_banks(df.direct_lower)
    hi,hi_ci,hi_se=ci_banks(df.direct_upper)
    pilot_mean,pilot_ci,pilot_se=ci_banks(df.pilot_premium)
    d=df.direct_midpoint_diagnostic-df.pilot_premium
    t_joint=float(__import__('scipy.stats',fromlist=['t']).t.ppf(.9875,banks-1))
    # Bonferroni: each one-sided bound fails with <=2.5% approximately.
    # Model-discrete confidence enclosure, conditional on statistical t approximation.
    joint_bound=(lo-t_joint*lo_se,hi+t_joint*hi_se)
    variance_ratio=float(df.direct_lower.var(ddof=1)/df.raw_direct_lower.var(ddof=1))
    gap,gap_ci,_=ci_banks(df.exercise_policy_gap)
    delta,delta_ci,_=ci_banks(d)
    figures=[('Pilote seul (courbe interpolée)',pilot_mean,pilot_ci),
             ('Direct brut sans correction (bas)',raw_lo,raw_lo_ci),
             ('Direct inférieur (exercice hors échantillon)',lo,lo_ci),
             ('Direct supérieur (convexité)',hi,hi_ci)]
    fig,ax=plt.subplots(figsize=(9.5,5.2))
    for j,(name,m,ci) in enumerate(figures):
        ax.errorbar([m],[j],xerr=[[m-ci[0]],[ci[1]-m]],fmt='o',capsize=4)
    ax.set_yticks(np.arange(len(figures)),[z[0] for z in figures]);ax.invert_yaxis()
    ax.set_xlabel('Prime ACRO (€/100), IC95 entre banques entières')
    ax.set_title('Validation directe de la prime : sans courbe interpolée pour les bornes')
    ax.grid(axis='x',alpha=.25)
    fig.tight_layout();fig.savefig(out/'01_prime_directe_versus_pilote.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8.5,4))
    for name,grp in state.groupby('stratum'):
        ax.scatter(grp.factor_x,grp.pilot_moneyness,label=f'{name} ({len(grp)})',alpha=.3,s=17)
    ax.axhline(0,color='black',linewidth=.9);ax.axhspan(-margin,margin,color='grey',alpha=.1)
    ax.set_xlabel('Facteur log-volatilité à l’autocall')
    ax.set_ylabel('Moneyness du droit ACRO estimée au pilote (€)')
    ax.set_title('États de rappel sélectionnés — importance des zones proches de l’exercice')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'02_etats_selectionnes.png',dpi=160);plt.close(fig)
    obj={'title':'ACRO V5.19 direct nested check, NOT certified price',
         'coupon_fixed':.18,'steps_per_year':steps,'n_banks':banks,
         'outer_paths_per_bank':outer,'selected_states_per_bank':selected,
         'inner_paths_per_half':inner,'reference_paths_per_scramble':ref_paths,
         'scrambles_per_half':scrambles,'margin_pilot_eur':margin,
         'pilot_mean':pilot_mean,'pilot_ci95_between_banks':pilot_ci,
         'raw_direct_lower_mean':raw_lo,'raw_direct_lower_ci95':raw_lo_ci,
         'raw_direct_upper_mean':raw_hi,'raw_direct_upper_ci95':raw_hi_ci,
         'direct_lower_mean':lo,'direct_lower_ci95_between_banks':lo_ci,
         'direct_upper_mean':hi,'direct_upper_ci95_between_banks':hi_ci,
         'discrete_model_joint_band_95_approx':joint_bound,
         'variance_ratio_adjusted_to_raw':variance_ratio,
         'mean_exercise_gap':gap,'exercise_gap_ci95':gap_ci,
         'direct_mid_minus_pilot_mean_diagnostic':delta,
         'direct_mid_minus_pilot_ci95':delta_ci,
         'time_total_banks_seconds':float(df.time_seconds.sum()),
         'limitations':['Upper/lower bounds only in expectation for finite-grid model',
             'Randomized Sobol and MC references enter each full independent bank',
             'Student-t CI across independent whole-bank runs, with few banks',
             'Direct check estimates are Monte Carlo noisy and depend on stratified state selection',
             'No bound for time discretization, modelling, market calibration or tail rare-event error']}
    (out/'metrics_v519.json').write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf8')
    report=f'''# ACRO V5.19 — Prime directe sans interpolation

**Prix non certifié, modèle Q synthétique et coupon contractuel fixé à 18 %.**

L'audit V5.18 étudiait des erreurs sur la *valeur du Phoenix renouvelé*.
Ici le contrôle direct estime le **payoff ACRO complet** sur des états de rappel
avec pondération de sondage et deux valeurs conditionnelles indépendantes.
La courbe pilote (simulée séparément) sert à réduire la variance du sondage et à sur-échantillonner
les états où `V_new−100` est proche de zéro ; les poids rétablissent l'espérance
sur les **{outer:,} trajectoires initiales par banque**, y compris sans rappel.

## Résultats (€/100)

| Mesure | Moyenne | IC95 empirique entre {banks} banques |
|---|---:|---|
| Pilote interpolé | {pilot_mean:.4f} | [{pilot_ci[0]:.4f}; {pilot_ci[1]:.4f}] |
| Direct brut inférieur (non corrigé) | {raw_lo:.4f} | [{raw_lo_ci[0]:.4f}; {raw_lo_ci[1]:.4f}] |
| Direct inférieur **corrigé par différence** | {lo:.4f} | [{lo_ci[0]:.4f}; {lo_ci[1]:.4f}] |
| Direct supérieur **corrigé par différence** | {hi:.4f} | [{hi_ci[0]:.4f}; {hi_ci[1]:.4f}] |
| Écart direct supérieur − inférieur | {gap:.4f} | [{gap_ci[0]:.4f}; {gap_ci[1]:.4f}] |

Écart descriptif `(direct bas + haut)/2 − pilote` : {delta:.4f} €,
IC95 inter-banques [{delta_ci[0]:.4f}; {delta_ci[1]:.4f}].
Le **milieu de deux bornes n'est pas, en général, un estimateur sans biais**.

Intervalle conservateur approximatif simultané pour le **prix du modèle discret** :
**[{joint_bound[0]:.4f}; {joint_bound[1]:.4f}] €/100**. Il est construit avec
une borne inférieure t à 97,5 % d'un côté et une borne supérieure t à 97,5 %
de l'autre (Bonferroni). Les banques sont indépendantes entre elles, mais
ce résultat reste une approximation statistique, **pas une borne certaine**.

Rapport des variances inter-banques ajusté/direct brut : **{variance_ratio:.3f}**
(réduction observée de **{(1-variance_ratio)*100:.1f} %**). Le gain vaut dans ce
benchmark, pas nécessairement pour d'autres paramètres ou budgets.


Le prix corrigé est `prix pilote sur toutes les trajectoires + somme pondérée(prix direct - pilote)` sur les états échantillonnés.
L'interpolation du pilote se compense **en espérance** grâce aux poids ; elle n'est pas une hypothèse de pricing pour cette expérience.

Les deux valorisations du Phoenix renouvelé sont **directes à chaque état sélectionné**,
avec références Black-Scholes et tirages internes indépendants. Chaque paire
A/B partage en revanche ses innovations *entre états* pour diminuer le bruit,
et son erreur commune est comptée en répliquant toute la banque.

## Méthode de sélection

Le pilote classe les rappels admissibles en sous le strike, à proximité du strike,
au-dessus du strike. Une fraction disproportionnée des simulations directes
est affectée aux cas proches de l'exercice. Pour tout groupe h, les `m_h` états
sont tirés uniformément, sans remise, parmi les `N_h` états du groupe et leur
contribution est multipliée par `N_h/m_h`. Cela estime la prime **non conditionnelle**
aux rappels sans introduire de biais de sondage conditionnel au pilote.

## Limites et décision

- **Pas de tolérance globale ±0,02 démontrée** avec ce seul contrôle ; vérifier largeur des IC.
- Les IC inter-banques ne couvrent ni l'écart au modèle continu, ni la calibration marché.
- La variance sous-échantillonnage, les références GBM, l'interne et l'externe
  changent ensemble entre banques ; on évite de prétendre à des erreurs indépendantes.
- Les calculs sont faits à {steps} pas/an : non comparables mécaniquement aux grilles
  96/192/384 pas/an étudiées auparavant.
- Le pricer `gbm_cv` reste inchangé ; cette validation ne se substitue pas à lui.

Script à relancer : `python audit_premium_direct_v519.py --quick` pour la fumée,
`python audit_premium_direct_v519.py` pour l'expérience standard.
'''
    (out/'AUDIT_PRIME_DIRECTE_V5_19_FR.md').write_text(report,encoding='utf8')
    return obj

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--out',default='results_direct_v519')
    ap.add_argument('--banks',type=int,default=6)
    ap.add_argument('--outer',type=int,default=4500)
    ap.add_argument('--selected',type=int,default=42)
    ap.add_argument('--inner',type=int,default=768)
    ap.add_argument('--pilot',type=int,default=2048)
    ap.add_argument('--ref',type=int,default=2048)
    ap.add_argument('--scrambles',type=int,default=4)
    ap.add_argument('--nodes',type=int,default=15)
    ap.add_argument('--steps',type=int,default=12)
    ap.add_argument('--margin',type=float,default=.6)
    ap.add_argument('--seed',type=int,default=519013)
    a=ap.parse_args()
    if a.quick:
        a.banks=2;a.outer=600;a.selected=12;a.inner=128
        a.pilot=256;a.ref=256;a.scrambles=2;a.nodes=7
    execute(ROOT/a.out, a.banks,a.outer,a.selected,a.inner,
            a.pilot,a.ref,a.scrambles,a.nodes,a.steps,a.seed,a.margin)
