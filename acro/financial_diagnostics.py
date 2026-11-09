"""Transparent Q-risk diagnostics for the INITIAL Phoenix.

These are direct sample cash-flow estimates, NOT new tradeable option quotes.
They use the unmodified SV quote paths even when the note uses GBM control
variates for its official quote. Hence this raw decomposition may not sum
exactly to the controlled issue price or match its variance-reduced B and A.
"""
from __future__ import annotations
import numpy as np
from .engine import Paths
from .config import Market, Phoenix


def phoenix_risk_diagnostics(paths: Paths, market: Market, terms: Phoenix, coupon_rate: float) -> dict:
    """Reconstruct risk-free-at-exit, terminal loss and coupon PV on ONE sample.

    The zero-loss leg pays 100% of face at the actual first autocall or maturity;
    it is *not* a five-year zero-coupon. Loss is the difference between this
    hypothetical leg and actual principal PV, so it also works for continuously
    monitored capital barriers and optional autocall schedules.
    """
    years=np.asarray(paths.first_call_year,dtype=float)
    principal=np.asarray(paths.principal_pv,dtype=float)
    annuity=np.asarray(paths.coupon_annuity_pv,dtype=float)
    if years.ndim!=1 or len(years)==0 or principal.shape!=years.shape or annuity.shape!=years.shape:
        raise ValueError('Cash-flow arrays must be nonempty and aligned')
    if not all(np.isfinite(a).all() for a in (principal,annuity)):
        raise ValueError('Cash-flow arrays must be finite')
    called=np.isfinite(years)
    exit_year=np.where(called,years,terms.maturity_years)
    baseline=terms.nominal*np.exp(-market.risk_free_rate*exit_year)
    pv_loss=baseline-principal
    if np.any(pv_loss < -1e-7 * terms.nominal):
        raise ValueError('Principal payoff exceeds full nominal at actual exit')
    pv_loss=np.maximum(pv_loss,0.)
    lost=pv_loss>1e-9*terms.nominal
    if np.any(lost&called):
        raise ValueError('Capital loss on early autocall is inconsistent')
    raw_baseline=float(baseline.mean())
    raw_loss=float(pv_loss.mean())
    raw_principal=float(principal.mean())
    raw_coupons=float(coupon_rate*annuity.mean())
    no_call=int((~called).sum())
    loss_count=int(lost.sum())
    # Contract face minus actual principal, not a loss after coupon payments.
    loss_in_euros=(pv_loss[lost]*np.exp(market.risk_free_rate*terms.maturity_years))
    return {
        'source':'direct_uncontrolled_quote_paths_Q',
        'basket_rule':terms.basket_rule,
        'sample_paths':int(len(years)),
        'call_probability_Q':float(called.mean()),
        'no_autocall_probability_Q':float(1.-called.mean()),
        'capital_loss_probability_Q':float(lost.mean()),
        'capital_loss_probability_given_no_autocall_Q':float(loss_count/no_call) if no_call else None,
        'mean_nominal_loss_if_loss':float(loss_in_euros.mean()) if loss_count else None,
        'pv_redemption_before_loss':raw_baseline,
        'pv_capital_loss':raw_loss,
        'pv_actual_principal_uncontrolled':raw_principal,
        'pv_conditional_coupons_uncontrolled':raw_coupons,
        'pv_note_uncontrolled':raw_principal+raw_coupons,
        'decomposition_residual':raw_baseline-raw_loss-raw_principal,
        'zero_coupon_risk_free_at_maturity':float(terms.nominal*np.exp(-market.risk_free_rate*terms.maturity_years)),
        'note':'Q probability of principal loss, not historical P or net loss after coupons. '
               'Raw sample estimates are not the variance-reduced model quote.'
    }


def draw_risk_decomposition(diagnostics: dict, outfile) -> str:
    """Publish a source-labelled financial waterfall; no new pricing calculations."""
    from pathlib import Path
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    full=diagnostics['pv_redemption_before_loss']
    loss=diagnostics['pv_capital_loss']
    coupon=diagnostics['pv_conditional_coupons_uncontrolled']
    total=diagnostics['pv_note_uncontrolled']
    steps=[(0.,full,'#245e96'),(full-loss,full,'#c35850'),
           (full-loss,full-loss+coupon,'#34806c'),(0.,total,'#293b55')]
    labels=['Nominal sans pertes\nà la date de sortie',
            f'Pertes contingentes\n{diagnostics.get("basket_rule","worst_of")}',
            'Coupons\nconditionnels',
            'Prix Phoenix\n(simulation brute)']
    fig,ax=plt.subplots(figsize=(11.5,6.4))
    fig.subplots_adjust(left=.09,right=.95,bottom=.24,top=.73)
    fig.text(.09,.96,'11 / D’où vient le coupon du Phoenix ?',weight='bold',fontsize=19,va='top')
    fig.text(.09,.885,f'Décomposition des flux actualisés sur {diagnostics["sample_paths"]:,} scénarios Q non contrôlés',
             fontsize=11.8,color='#475569',va='top')
    pct=diagnostics['capital_loss_probability_Q']
    missing=diagnostics['no_autocall_probability_Q']
    cond=diagnostics['capital_loss_probability_given_no_autocall_Q']
    note=(f'P(perte sur le NOMINAL) sous Q ≈ {pct:.1%}     '
          f'P(aucun autocall) ≈ {missing:.1%}     '
          f'P(perte | pas de rappel) ≈ {cond:.1%}' if cond is not None else
          f'P(perte sur le NOMINAL) sous Q ≈ {pct:.1%}     P(aucun autocall) ≈ {missing:.1%}')
    fig.text(.09,.805,note,fontsize=11.4,weight='bold',color='#a53734',va='top')
    for i,(low,high,col) in enumerate(steps):
        ax.add_patch(Rectangle((i-.32,low),.64,high-low,facecolor=col,alpha=.88))
        value=full if i==0 else (-loss if i==1 else coupon if i==2 else total)
        ax.text(i,high+2.5,f'{value:+.1f} €' if i in (1,2) else f'{value:.1f} €',
                ha='center',fontsize=14,weight='bold',color='#263850')
    # Waterfall connectors show intermediate running value.
    for i,y in enumerate([full,full-loss,full-loss+coupon]):
        ax.plot([i+.32,i+1-.32],[y,y],color='#8d9bad',lw=1.5,ls='--')
    ax.set_xlim(-.7,3.7);ax.set_ylim(0,max(full,total)+14)
    ax.set_xticks(range(4),labels)
    ax.set_ylabel('Valeur actuelle (€ par contrat initial)',fontsize=11.5)
    ax.spines[['top','right','bottom']].set_visible(False)
    ax.tick_params(axis='x',length=0,labelsize=10.4,pad=10)
    ax.tick_params(axis='y',labelsize=10)
    ax.grid(axis='y',alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.09,.085,'IMPORTANT : jambe de perte = exposition conditionnelle au panier avec barrière et extinction à l’autocall, pas put vanilla coté.\n'
             'Échantillon brut distinct du prix contrôlé B + cA ; Q ≠ probabilité historique P ; pertes avant coupons.',
             color='#526071',fontsize=10,va='bottom',linespacing=1.6)
    target=Path(outfile);target.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(target,dpi=160,facecolor='white')
    plt.close(fig)
    return target.name
