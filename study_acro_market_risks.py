"""Reproducible *static* sensitivity study: no stochastic rate/dividend/correlation.

Reprices the same Phoenix + ACRO in separate constant-market scenarios.
This is NOT a multi-factor ACRO, nor a decomposition of causal contributions.
The initial coupon is either the same fixed contractual coupon in every scenario,
or independently reset to each scenario's fair coupon. The fixed-coupon initial
Phoenix is generally not worth par after market parameters have been shocked.

Run: python study_acro_market_risks.py --output results_market_risks
For a faster exploratory run use --quick. Sensitivity outputs are research only.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import argparse, csv, json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro.config import load_project
from acro.controlled_pricing import price_acro_controlled
from acro.pricing import price_acro

BASE_COUPON = 0.182  # stable independent financial audit, fixed contract comparison


def scenarios(market):
    """Shocks are chosen ex ante, not reverse-engineered for biggest effect."""
    corr=np.asarray(market.correlations, dtype=float)
    def adjust_corr(v):
        new=corr.copy()
        mask=~np.eye(len(new),dtype=bool)
        new[mask] += v
        return tuple(map(tuple,new))
    return [
        ('Référence','base',market),
        ('Volatilité −20 %','vol', replace(market,volatilities=tuple(v*.8 for v in market.volatilities))),
        ('Volatilité +20 %','vol', replace(market,volatilities=tuple(v*1.2 for v in market.volatilities))),
        ('Corrélations −0,15','corr', replace(market,correlations=adjust_corr(-.15))),
        ('Corrélations +0,15','corr', replace(market,correlations=adjust_corr(+.15))),
        ('Dividendes −50 pb','div', replace(market,dividends=tuple(v-.005 for v in market.dividends))),
        ('Dividendes +50 pb','div', replace(market,dividends=tuple(v+.005 for v in market.dividends))),
        ('Taux −100 pb','rate', replace(market,risk_free_rate=market.risk_free_rate-.01)),
        ('Taux +100 pb','rate', replace(market,risk_free_rate=market.risk_free_rate+.01)),
    ]


def get_one(p, name, family, market, seed, *, ref_paths, scrambles):
    market_p=replace(p,market=market,
                     simulation=replace(p.simulation,seed=seed),
                     initial_coupon_policy='fair')
    result,diag=price_acro_controlled(market_p,ref_paths=ref_paths,scrambles=scrambles)
    q=result.quote
    # Reuse *identical* simulation paths and conditional curve to compare the
    # original fixed coupon. No new noise or refit of conditional market states.
    qp=replace(q,coupon_rate=BASE_COUPON,price=q.principal+BASE_COUPON*q.annuity)
    p_fixed=replace(market_p,initial_coupon_policy='fixed',
                    initial_phoenix=replace(p.initial_phoenix,coupon_rate=BASE_COUPON))
    fixed=price_acro(p_fixed,quote=qp,curve_override=result.curve)
    return {'name':name,'family':family,'seed':seed,
        'fair_coupon_pct':100*q.coupon_rate,
        'fair_option_eur':result.price,'fixed_option_eur':fixed.price,
        'fixed_initial_note_mtm_eur':qp.price,
        'fair_outer_se_only':result.outer_se,
        'fixed_outer_se_only':fixed.outer_se,
        'call_probability_pct':100*result.call_probability,
        'eligible_exercise_pct':100*np.mean((result.intrinsic>0)&result.eligible),
        'quote_reference_se_approx':q.coupon_se,
        'quote_paths':p.simulation.quote_paths,'outer_paths':p.simulation.outer_paths,
        'curve_nodes':len(result.curve.factors),'ref_scrambles':scrambles,
        'ref_paths_per_scramble':ref_paths}


def summarize(rows):
    names=list(dict.fromkeys(r['name'] for r in rows))
    fields=['fair_coupon_pct','fair_option_eur','fixed_option_eur','fixed_initial_note_mtm_eur','call_probability_pct']
    output=[]
    for name in names:
        subset=[r for r in rows if r['name']==name]
        d={'name':name,'family':subset[0]['family'],'replicates':len(subset)}
        for field in fields:
            v=np.array([r[field] for r in subset],dtype=float)
            d[field]=float(v.mean());d[field+'_between_seed_se']=float(v.std(ddof=1)/np.sqrt(len(v))) if len(v)>1 else None
        output.append(d)
    return output


def comparisons(rows,agg):
    base=next(r for r in agg if r['family']=='base')
    base_rows={r['seed']:r for r in rows if r['family']=='base'}
    result=[]
    for entry in agg:
        if entry['family']=='base': continue
        paired=[(r,base_rows[r['seed']]) for r in rows if r['name']==entry['name']]
        d={'name':entry['name'],'family':entry['family']}
        for field in ['fair_option_eur','fixed_option_eur','fair_coupon_pct','fixed_initial_note_mtm_eur']:
            delta=np.asarray([a[field]-b[field] for a,b in paired],dtype=float)
            d[field+'_vs_base']=float(delta.mean())
            d[field+'_diff_se']=float(delta.std(ddof=1)/np.sqrt(len(delta))) if len(delta)>1 else None
        result.append(d)
    return result


def plot(agg,path):
    # Matplotlib owns its colors. Error bars are seed-to-seed SE, not full pricing CI.
    names=[d['name'] for d in agg]
    vals=np.array([d['fixed_option_eur'] for d in agg])
    errs=np.array([d['fixed_option_eur_between_seed_se'] or 0 for d in agg])
    fig,ax=plt.subplots(figsize=(11,6))
    y=np.arange(len(agg))
    ax.barh(y,vals)
    ax.errorbar(vals,y,xerr=1.96*errs,fmt='none',ecolor='black',capsize=2)
    ax.set_yticks(y,labels=names)
    ax.invert_yaxis()
    ax.set_xlabel('Prime ACRO (€/100) à coupon contractuel fixé à 18,20 %')
    ax.set_title('ACRO : sensibilité à des changements de paramètres de marché constants')
    ax.grid(axis='x',alpha=.25)
    ax.text(.01,-.18,'Traits noirs = ±1,96 SE entre graines (n’inclut ni biais discret ni risque de modèle).\n'
               'Les chocs sont d’amplitudes différentes, donc pas de classement structurel universel.',
            transform=ax.transAxes,va='top',fontsize=9)
    fig.tight_layout()
    fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)


def write_report(rows,agg,deltas,out,mode):
    base=next(x for x in agg if x['family']=='base')
    fam=[]
    for f in ['vol','corr','div','rate']:
        group=[x for x in deltas if x['family']==f]
        eff=max(abs(x['fixed_option_eur_vs_base']) for x in group)
        fam.append((f,eff))
    ranking=sorted(fam,key=lambda a:a[1],reverse=True)
    lines=[
        '# ACRO — Stress-test comparatif de marché (hypothèses constantes)',
        '', '**Etude de sensibilité, pas calibration ni modèle multifactoriel.**',
        '',f'Mode : {mode} ; {len(set(r["seed"] for r in rows))} banques/graines indépendantes. '
             'Paramètres de marché *figés sur toute la vie* de chaque scénario. '
             'Le coupon initial contractuel est fixé à 18,20 % pour la colonne principale.',
        '', '## Résultats', '',
        '| Scénario | Coupon au pair (%) | Prime ACRO coupon fixé (€) | Prime ACRO coupon re-coté (€) | Phoenix initial coupon fixé (€) |',
        '|---|---:|---:|---:|---:|',
    ]
    for a in agg:
        lines.append(f'| {a["name"]} | {a["fair_coupon_pct"]:.2f} | '
                     f'{a["fixed_option_eur"]:.3f} | {a["fair_option_eur"]:.3f} | '
                     f'{a["fixed_initial_note_mtm_eur"]:.2f} |')
    lines+=['','## Amplitudes de sensibilité comparées','','Mesure : maximum de l’écart absolu de prime fixe à la référence pour les deux chocs prédéfinis. '
            '**Ce classement dépend des chocs choisis et ne mesure pas une importance universelle en pourcentage.**','']
    for f,eff in ranking:
        lines.append(f'- {f}: écart maximal de {eff:.3f} € pour 100 €')
    lines+=['','## Interprétation technique','',
        '1. **Contrat fixé** : le coupon préservé reste 18,20 % dans tous les scénarios. '
        'Le Phoenix initial ne vaut donc généralement plus 100 € lorsque les paramètres sont choqués. '
        'Cette expérience n’est pas un scénario dynamique de changement à une date future.',
        '2. **Coupon recalculé au pair** : on redéfinit le contrat initial pour qu’il vaille 100 € '
        'dans chaque scénario. Les primes de cette colonne ne comparent donc pas une même option.',
        '3. **Différences appariées** : chaque scénario partage la même graine que sa référence ; '
        'l’erreur entre graines est un diagnostic partiel, et n’inclut pas les erreurs systématiques '
        '(discrétisation, interpolation, bruit inner, Jensen, modèle).',
        '4. **Une sensibilité forte ne suffit pas à justifier un facteur stochastique supplémentaire** : '
        'une dynamique sous Q, des corrélations entre facteurs et une calibration indépendante sont requises. '
        'Ajouter arbitrairement une diffusion de corrélation ou de dividendes donnerait des primes trompeusement précises.',
        '5. **ExpOU** : la source d’aléa évoluant réellement dans cette version reste le facteur de volatilité, '
        'corrélé aux chocs actions. Taux, dividendes et corrélations actions sont des scénarios figés.',
        '', '## Reproduction','',
        '`python study_acro_market_risks.py --quick --seeds 20261105 20261371 20261537 20261791 20261987 --output results_market_risks`',
        '', 'Archive : le moteur initial ACRO n’est pas modifié. Fichiers détaillés : `raw_runs.csv`, '
        '`scenario_summary.csv`, `paired_differences.csv`.','']
    (out/'RAPPORT_SENSIBILITE_FINANCIERE.md').write_text('\n'.join(lines),encoding='utf-8')


def dump_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(quick=False,output='results_market_risks',seeds=None):
    root=Path(__file__).resolve().parent
    p=load_project(root/'config.json')
    if quick:
        s=replace(p.simulation,quote_paths=2500,outer_paths=3500,conditional_paths=1500,curve_nodes=7)
        ref_paths=1024; scrambles=4
        seed_vals=seeds if seeds is not None else [20261105,20261371]
    else:
        s=replace(p.simulation,quote_paths=5000,outer_paths=7000,conditional_paths=4000,curve_nodes=9)
        ref_paths=2048; scrambles=4
        seed_vals=seeds if seeds is not None else [20261105,20261371,20261537,20261791,20261987]
    p=replace(p,simulation=s)
    cases=scenarios(p.market)
    rows=[]
    for seed in seed_vals:
        for name,family,m in cases:
            # catches incompatible shocked correlation matrices before pricing
            row=get_one(p,name,family,m,seed,ref_paths=ref_paths,scrambles=scrambles)
            rows.append(row)
            print(f'{seed} {name}: fair-c={row["fair_coupon_pct"]:.2f}% '
                  f'fixed-option={row["fixed_option_eur"]:.3f} '
                  f'fair-option={row["fair_option_eur"]:.3f} EUR',flush=True)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    agg=summarize(rows);deltas=comparisons(rows,agg)
    dump_csv(out/'raw_runs.csv',rows);dump_csv(out/'scenario_summary.csv',agg)
    dump_csv(out/'paired_differences.csv',deltas)
    plot(agg,out/'sensibilite_prime_acro.png')
    write_report(rows,agg,deltas,out,'quick' if quick else 'standard')
    (out/'metadata.json').write_text(json.dumps({'coupon_fixed':BASE_COUPON,
        'seeds':seed_vals,'quote_paths':s.quote_paths,'outer_paths':s.outer_paths,
        'ref_paths':ref_paths,'scrambles':scrambles,
        'scope':'static Q parameters repriced from inception; NOT a new stochastic factor',
        },indent=2),encoding='utf-8')
    return rows,agg,deltas


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default='results_market_risks')
    parser.add_argument('--seeds',type=int,nargs='+',help='Independent random seeds used for paired scenarios')
    args,_=parser.parse_known_args()  # Spyder may pass additional command-line args
    run(quick=args.quick,output=args.output,seeds=args.seeds)
