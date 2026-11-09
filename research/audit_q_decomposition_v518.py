"""ACRO V5.18: isolate quote-level and conditional-curve-shape noise under Q.

This is a diagnostic, not a price certification: source model/payoffs unchanged.
Multiple QUOTE and SHAPE simulation banks are disjoint. A full crossing of
B0+A0*c anchor against B(x)-B(0)+c*(A(x)-A(0)) measures variation conditional
on a *single fixed external autocall sample*. An independent CRN comparison
checks selected out-of-grid states. No SE from this script covers discretisation,
model risk, market calibration, or interpolation against the unknown exact curve.
"""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import t as student_t
from acro import load_project
from acro.config import Project
from acro.engine import simulate_phoenix
from acro.pricing import same_contract
from acro.controlled_pricing import controlled_quote, controlled_curve
from acro.conditional_qmc import innovation_bank

ROOT = Path(__file__).resolve().parents[1]


def centered_premium(anchor_price, relative_change, discount, strike=100., units=1.):
    """Option payoff conditional on eligible autocall, no spurious reward outside."""
    rel=np.asarray(relative_change, dtype=float)
    d=np.asarray(discount, dtype=float)
    if rel.shape != d.shape:
        raise ValueError('relative_change and discount shapes differ')
    return float(np.mean(d * units * np.maximum(anchor_price+rel-strike, 0.)))


def reanchored_price_at(curve, qshape, coupon, x, new_q):
    """Preserve curve's relative shape, switch only independently quoted anchor."""
    a,b=curve.at(np.asarray(x, dtype=float))
    return (new_q.principal+coupon*new_q.annuity
            +(a-qshape.principal)+coupon*(b-qshape.annuity))


def split_curve(p, q, bank, factor_states, seed_offset, ref_paths, scrambles):
    c=controlled_curve(p,q,bank,factor_states,seed_offset=seed_offset,
                      ref_paths=ref_paths,scrambles=scrambles)
    x0=np.array([p.market.initial_logvol_factor])
    a,b=c.at(x0)
    if not np.allclose([a[0],b[0]],[q.principal,q.annuity],rtol=0,atol=1e-9):
        raise AssertionError('Curve anchor must equal its own independently computed quote')
    return c


def independent_relative_check(p:Project, factor_grid, coupon, *, banks, npaths, seed):
    """Independent pathwise CRN differences V(x)-V(x0), NOT interpolation."""
    x0=p.market.initial_logvol_factor
    terms=p.renewal_phoenix
    steps=p.simulation.steps_per_year
    data=[]
    for b in range(banks):
        bank=innovation_bank(int(round(terms.maturity_years*steps)),npaths,
                             p.market.n_assets,seed+1009*b,method='mc_bridge')
        base=simulate_phoenix(p.market,terms,npaths,steps,0,
                              initial_factor=x0,innovations=bank).pv(coupon)
        for x in factor_grid:
            res=simulate_phoenix(p.market,terms,npaths,steps,0,
                                 initial_factor=float(x),innovations=bank).pv(coupon)
            data.append(dict(bank=b,x=float(x),relative_value=float(np.mean(res-base))))
    return pd.DataFrame(data)


def run(p:Project,out:Path,*,n=5,outer_paths=8000,ref_paths=1024,scrambles=4,
        conditional_paths=4096,curve_nodes=21,check_banks=4,check_paths=4096,seed=518007):
    p.validate()
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('Requires stochastic_vol dynamics')
    if p.initial_coupon_policy!='fixed':
        raise ValueError('Freeze contractual coupon for a meaningful factorial comparison')
    if not same_contract(p.initial_phoenix,p.renewal_phoenix):
        raise ValueError('This version isolates matched-renewal contract noise only')
    if p.acro.target_coupon_rate not in (None,p.initial_phoenix.coupon_rate):
        raise ValueError('Target coupon must match fixed initial coupon in this audit')
    if n<2 or check_banks<2:
        raise ValueError('At least two quote and holdout banks required')
    out.mkdir(parents=True,exist_ok=True)
    st=perf_counter(); c=p.initial_phoenix.coupon_rate
    p=replace(p,simulation=replace(p.simulation,quote_paths=conditional_paths,
                                  conditional_paths=conditional_paths,
                                  curve_nodes=curve_nodes,outer_paths=outer_paths))
    steps=p.simulation.steps_per_year
    outer=simulate_phoenix(p.market,p.initial_phoenix,outer_paths,steps,seed+3)
    eligible=np.isfinite(outer.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= outer.first_call_year<=p.acro.last_eligible_call_year+1e-10
    factors=np.asarray(outer.factor_at_call[eligible],float)
    x0=p.market.initial_logvol_factor
    # Full, fixed external sample includes paths with no autocall (zero payoff).
    disc=np.exp(-p.market.risk_free_rate*outer.first_call_year[eligible])
    nall=len(eligible)
    anchor_quotes=[]
    for k in range(n):
        q,_,_=controlled_quote(p,ref_paths=ref_paths,scrambles=scrambles,
                               seed_offset=seed+100000+5107*k)
        anchor_quotes.append(q)
    curve_quotes=[];curves=[];shape_banks=[]
    for j in range(n):
        off=seed+900000+7057*j
        q,bank,_=controlled_quote(p,ref_paths=ref_paths,scrambles=scrambles,
                                 seed_offset=off)
        curve=split_curve(p,q,bank,factors,off,ref_paths,scrambles)
        curve_quotes.append(q); curves.append(curve);shape_banks.append(bank)
    # Cartesian crossings: anchors vary down rows, curve SHAPES across columns.
    mat=np.empty((n,n)); rows=[]; anchor_prices=[]
    for i, q in enumerate(anchor_quotes):
        anch=q.principal+c*q.annuity
        anchor_prices.append(anch)
        for j,(qs,curve) in enumerate(zip(curve_quotes,curves)):
            bv,av=curve.at(factors)
            relative=(bv-qs.principal)+c*(av-qs.annuity)
            payoff=np.zeros(nall)
            payoff[eligible]=p.acro.renewal_units*np.maximum(anch+relative-p.acro.strike,0)*disc
            val=float(payoff.mean())
            mat[i,j]=val
            rows.append(dict(anchor_bank=i,shape_bank=j,anchor_price=anch,
                             price_per_100=val,outer_paths=nall,
                             eligible_calls=int(eligible.sum())))
    pd.DataFrame(rows).to_csv(out/'prix_q_croisements.csv',index=False)
    mean=float(mat.mean());anchor_means=mat.mean(axis=1);shape_means=mat.mean(axis=0)
    # Pool the *conditional valuations* before the convex max. Averaging the
    # premiums is different and can retain positive-part simulation bias.
    rel_pool=np.zeros(len(factors))
    for qs,cv in zip(curve_quotes,curves):
        b,a=cv.at(factors)
        rel_pool+=(b-qs.principal)+c*(a-qs.annuity)
    rel_pool/=n
    pooled_payoffs=np.zeros(nall)
    pooled_payoffs[eligible]=(p.acro.renewal_units*np.maximum(
        float(np.mean(anchor_prices))+rel_pool-p.acro.strike,0.)*disc)
    pooled_price=float(np.mean(pooled_payoffs))
    if pooled_price > mean+1e-10:
        raise AssertionError('Convex positive-part/Jensen inequality violated')
    # Descriptive effects: NOT a population ANOVA because repeated cells share quotes/curves.
    effects=dict(mean=mean,pooled_conditional_price=pooled_price,
       convexity_gap_mean_individual_minus_pooled=mean-pooled_price,anchor_sd=float(anchor_means.std(ddof=1)),
       shape_sd=float(shape_means.std(ddof=1)),
       anchor_span=float(np.ptp(anchor_means)),shape_span=float(np.ptp(shape_means)),
       quote_price_sd=float(np.std(anchor_prices,ddof=1)),
       quote_price_range=[float(np.min(anchor_prices)),float(np.max(anchor_prices))],
       fixed_external_mc_se_not_estimated=True,
       remark='Crossing isolates quote anchor and relative curve shape conditional on one external sample; shared bank effects are not a global variance decomposition.')
    # Compare relative-shape curves to out-of-grid CRN independent valuation.
    # Factor quantiles are preselected from the common outer sample, not chosen to fit curves.
    if len(factors)<8:
        raise ValueError('Too few eligible calls to validate the conditional curve')
    xs=np.quantile(factors,[.15,.5,.85])
    xs=np.asarray([x for x in xs if abs(x-x0)>1e-7],float)
    holdout=independent_relative_check(p,xs,c,banks=check_banks,
                                        npaths=check_paths,seed=seed+4500000)
    holdout.to_csv(out/'validation_relative_directe_banques.csv',index=False)
    valrows=[]
    for x in xs:
        v=holdout[holdout.x==float(x)].relative_value.to_numpy()
        v_mean=float(np.mean(v));se=float(np.std(v,ddof=1)/np.sqrt(len(v)))
        for j,(qs,cv,bank) in enumerate(zip(curve_quotes,curves,shape_banks)):
            B,A=cv.at(np.array([x]))
            shape=float(B[0]-qs.principal+c*(A[0]-qs.annuity))
            # Same-bank exact conditional repricing: isolate interpolation from
            # path-sampling error using the *same* quote/curve innovations.
            sam=simulate_phoenix(p.market,p.renewal_phoenix,
                 bank[0].shape[1],steps,0,initial_factor=float(x),
                 innovations=bank).pv(c)
            direct_d=sam-qs.paths.pv(c)
            samebank=float(np.mean(direct_d))
            mcse=float(np.std(direct_d,ddof=1)/np.sqrt(len(direct_d)))
            valrows.append(dict(x=float(x),shape_bank=j,
                interpolated_relative=shape,
                direct_samebank=samebank,
                interpolation_difference_samebank=shape-samebank,
                shape_mc_se_at_x=mcse,
                independent_mean=v_mean,
                independent_se=se,difference_vs_independent=shape-v_mean))
    vdf=pd.DataFrame(valrows)
    vdf.to_csv(out/'validation_forme_hors_grille.csv',index=False)
    max_interp=float(vdf.interpolation_difference_samebank.abs().max())
    max_vs_holdout=float(vdf.difference_vs_independent.abs().max())

    # Visual 1: obvious crossing patterns, not a payoff surface or vague heatmap.
    fig,ax=plt.subplots(figsize=(8,5))
    for j in range(n):
        xx=np.arange(1,n+1)+(j-(n-1)/2)*.085
        ax.scatter(xx,mat[:,j],alpha=.7,s=38,label=f'Forme {j+1}')
    ax.scatter(np.arange(1,n+1),anchor_means,color='black',marker='_',s=400,
               label='Moyenne par ancrage',zorder=5)
    ax.set_xticks(range(1,n+1),[f'Ancrage {i+1}' for i in range(n)])
    ax.set_ylabel('Prime ACRO sous Q (€/100)')
    ax.set_title('Bruit de cotation et de forme — mêmes trajectoires externes')
    ax.legend(ncol=min(n,3),fontsize=8)
    ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'01_croisement_ancrage_forme.png',dpi=150);plt.close(fig)
    # Visual 2: check shape only, different axes/meaning.
    fig,ax=plt.subplots(figsize=(8,5))
    for j,(qs,cv) in enumerate(zip(curve_quotes,curves)):
        xx=np.linspace(float(xs.min()),float(xs.max()),75)
        b,a=cv.at(xx)
        ax.plot(xx,(b-qs.principal)+c*(a-qs.annuity),lw=1.3,alpha=.65,
                label=f'Courbe {j+1}')
    for x in xs:
        sub=holdout[holdout.x==float(x)].relative_value.to_numpy()
        ym=float(sub.mean());err=float(student_t.ppf(.975,len(sub)-1)*sub.std(ddof=1)/np.sqrt(len(sub)))
        ax.errorbar([x],[ym],yerr=[err],color='black',fmt='s',capsize=4,ms=5)
    ax.axhline(0,color='gray',linestyle='--',lw=.6)
    ax.axvline(x0,color='gray',linestyle=':',lw=.6)
    ax.set_xlabel('Facteur de log-volatilité au rappel')
    ax.set_ylabel('Variation de valeur du Phoenix renouvelé vs x₀ (€/100)')
    ax.set_title('Forme conditionnelle : courbes vs repricings indépendants')
    ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(out/'02_validation_forme_independante.png',dpi=150);plt.close(fig)
    metrics={'version':'V5.18','status':'diagnostic_only',
         'coupon_fixed':float(c),'steps_per_year':steps,
         'banks_quote':n,'banks_shape':n,'outer_paths':outer_paths,
         'eligible_calls':int(eligible.sum()),'ref_paths':ref_paths,
         'scrambles':scrambles,'inner_paths':conditional_paths,
         'curve_nodes':curve_nodes,'validation_banks':check_banks,
         'validation_paths_per_bank':check_paths,'factorial':effects,
         'max_interp_difference_same_bank':max_interp,
         'max_difference_vs_independent_holdout':max_vs_holdout,
         'time_seconds':round(perf_counter()-st,2),
         'limitations':['Single external bank fixed across crossings',
           'Quote and shape relative curves computed under same synthetic Q only',
           'Holdout checks interpolation/relative shape at selected states but not total error',
           'No certified error on quote, anchor, model dynamics, time step or entire ACRO premium']}
    (out/'metrics_v518.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf8')
    report=[
      '# ACRO V5.18 — Diagnostic factoriel de la prime Q', '',
      '**Aucun prix de marché, aucun intervalle global de précision certifié.**',
      '', f'Coupon contractuel fixé : {c:.1%} ; {steps} pas/an ; {outer_paths} scénarios externes communs.',
      f'{n} cotations initiales et {n} formes de courbes provenant de banques de tirage **disjointes**.',
      '', '## Croisement des deux erreurs', '',
      f'- Moyenne descriptive des {n*n} combinaisons : **{mean:.4f} €/100**.',
      f'- **Moyenne des valorisations conditionnelles avant application du max** : **{pooled_price:.4f} €/100**.',
      f'- Écart de convexité entre moyenne des primes et prime de la valorisation moyenne : **{mean-pooled_price:.4f} €**.',
      f'- Dispersion des moyennes lorsque seul **l’ancrage** varie : **{effects["anchor_sd"]:.4f} €**.',
      f'- Dispersion des moyennes lorsque seule **la forme relative** varie : **{effects["shape_sd"]:.4f} €**.',
      f'- Étendue des cotations indépendantes du Phoenix initial : **{effects["quote_price_range"][0]:.3f} à {effects["quote_price_range"][1]:.3f} €**.',
      '', 'Ces dispersions sont descriptives, calculées conditionnellement au **même tirage externe** ;',
      'ce ne sont pas des erreurs-types globales ni une décomposition ANOVA indépendante en population.',
      '', '## Valorisation directe de la forme hors grille', '',
      'Les carrés noirs du second graphique mesurent V(x)−V(x₀) par simulations directes',
      'à chocs communs, *distinctes* des banques utilisées pour construire les courbes.',
      'Les intervalles sur les carrés représentent uniquement la dispersion entre banques directes.',
      'Un accord visuel ne valide pas le niveau de référence ni le biais temporel.',
      '', '| Facteur x | Relative directe | IC 95 % direct | Écart absolu max des courbes |',
      '|---:|---:|---:|---:|']
    for x in xs:
        sub=holdout[holdout.x==float(x)].relative_value.to_numpy()
        m=float(sub.mean());half=float(student_t.ppf(.975,len(sub)-1)*sub.std(ddof=1)/np.sqrt(len(sub)))
        dif=[abs(r['difference_vs_independent']) for r in valrows if abs(r['x']-float(x))<1e-12]
        report.append(f'| {x:+.3f} | {m:+.3f} | [{m-half:+.3f}; {m+half:+.3f}] | {max(dif):.3f} |')
    report.extend(['',
       f'- Écart maximal d’interpolation **à trajectoires identiques** : **{max_interp:.4f} €**.',
       f'- Écart maximal face au repricing indépendant : **{max_vs_holdout:.4f} €**.',
       'Ce dernier écart inclut le bruit MC des deux méthodes ; il n’est pas un biais d’interpolation.',
       'Les erreurs-types conditionnelles des courbes figurent dans le CSV hors grille.',
       '', '## Décision', '',
       'Ne pas sélectionner automatiquement la plus petite prime ni supprimer les courbes divergentes.',
       'Augmenter le budget de la composante qui domine ; valider ensuite les autres sources d’erreur',
       'avec des banques externes et internes réestimées. La valeur du projet reste la conception et l’analyse',
       'économique d’un droit de réémission, pas un chiffre de prime unique.', '',
       'Fichiers : `prix_q_croisements.csv`, `validation_relative_directe_banques.csv`,',
       '`validation_forme_hors_grille.csv`, deux PNG et `metrics_v518.json`.'])
    (out/'AUDIT_Q_FACTORIEL_V5_18_FR.md').write_text('\n'.join(report)+'\n',encoding='utf8')
    return metrics


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',default=str(ROOT/'config.json'))
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--output',default=None)
    ap.add_argument('--seeds',type=int,default=None)
    args=ap.parse_args()
    p=load_project(args.config)
    p=replace(p,initial_coupon_policy='fixed',
        initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
        simulation=replace(p.simulation,steps_per_year=12,seed=518025))
    output=Path(args.output or ROOT/('results_q_v518_quick' if args.quick else 'results_q_v518'))
    k=args.seeds or (3 if args.quick else 6)
    kw=dict(n=k,outer_paths=1500 if args.quick else 12000,
           conditional_paths=1024 if args.quick else 6000,
           ref_paths=256 if args.quick else 2048,
           scrambles=3 if args.quick else 6,
           curve_nodes=11 if args.quick else 21,
           check_banks=3 if args.quick else 6,
           check_paths=1024 if args.quick else 6000)
    result=run(p,output,**kw)
    print('V5.18 Q factorization:',json.dumps(result['factorial'],ensure_ascii=False))
    print('Files:',output)

if __name__=='__main__':
    main()
