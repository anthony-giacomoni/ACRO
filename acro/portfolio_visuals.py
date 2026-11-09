"""Explanatory ACRO graphics, reusing actual model and pricing state.

These sample paths are *additional* independent Q paths, not new pricing
observations and never added to the ACRO price estimate. The distribution
chart uses exactly the already-valued external paths of the pricer.
"""
from __future__ import annotations

import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

from .engine import simulate_phoenix, basket_level

FIGURE_NAMES = [
    '08_trajectoires_monte_carlo.png',
    '09_neuf_scenarios_phoenix.png',
    '10_distribution_payoffs_acro.png',
]


def payoff_statistics(result):
    """Unconditional Q probabilities. Monetary amounts are per entire initial contract."""
    discounted=np.asarray(result.discounted_payoff,dtype=float)
    exercised=discounted>1e-12
    eligible=np.asarray(result.eligible,dtype=bool)
    n=len(discounted)
    return {
        'positive_payoff_probability_Q':float(exercised.mean()),
        'exercise_probability_given_eligible_call_Q':float(exercised.sum()/eligible.sum()) if eligible.any() else 0.,
        'zero_payoff_probability_Q':float((~exercised).mean()),
        'conditional_discounted_payoff_if_positive':float(discounted[exercised].mean()) if exercised.any() else 0.,
        'payoff_discounted_p95_Q':float(np.percentile(discounted,95)) if n else 0.,
        'payoff_discounted_p99_Q':float(np.percentile(discounted,99)) if n else 0.,
        'payoff_discounted_mean_check':float(discounted.mean()) if n else 0.,
        'path_count_for_distribution':n,
    }


def _plot_paths(p, sim, out):
    t=sim.observation_times
    ratios=sim.observed_equity_ratios
    with np.errstate(invalid='ignore'):
        w=basket_level(np.log(ratios), p.initial_phoenix.basket_rule)
    fig, ax=plt.subplots(figsize=(12.8,6.8))
    n_paths = len(ratios)
    opacity = max(.055, min(.32, 20 / max(1, n_paths)))
    line_width = .7 if n_paths > 200 else 1.1
    call_x, call_y = [], []
    for i in range(n_paths):
        called=np.isfinite(sim.first_call_year[i])
        ax.plot(t,w[i]*100,color='#2762a8' if called else '#7c8a9c',
                alpha=opacity,lw=line_width)
        if called:
            j=int(round(sim.first_call_year[i]*p.initial_phoenix.observations_per_year))
            call_x.append(float(sim.first_call_year[i]))
            call_y.append(float(w[i,j]*100))
    if call_x:
        ax.scatter(call_x, call_y, s=12 if n_paths > 200 else 16,
                   color='#dc7838', alpha=.67, zorder=4)
    ax.axhline(p.initial_phoenix.autocall_barrier*100,color='#bc5d28',ls='--',lw=1.6,label='Seuil de rappel')
    ax.axhline(p.initial_phoenix.coupon_barrier*100,color='#398877',ls=':',lw=1.6,label='Barrière de coupon')
    ax.axhline(p.initial_phoenix.capital_barrier*100,color='#77619b',ls='-.',lw=1.3,label='Barrière capital à maturité (européenne dans le cas de base)')
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0],[0],color='#2762a8',label='Chemins rappelés'),
                       Line2D([0],[0],color='#7c8a9c',label='Chemins jusqu’à maturité'),
                       Line2D([0],[0],color='#bc5d28',ls='--',label='Seuil de rappel'),
                       Line2D([0],[0],color='#398877',ls=':',label='Barrière coupon'),
                       Line2D([0],[0],color='#77619b',ls='-.',label='Barrière capital'),
                       Line2D([0],[0],marker='o',lw=0,color='#dc7838',label='Point de rappel')],
              loc='upper right',frameon=False,fontsize=8)
    ax.set(xlabel='Années depuis émission',ylabel=f'{p.initial_phoenix.basket_rule} (en % du niveau initial)',
           title=f'08 | {len(ratios)} trajectoires Monte Carlo du Phoenix initial')
    ax.grid(alpha=.16)
    fig.text(.5,.015,'Simulation Q illustrative, indépendante du pricing externe. Chaque trajectoire s’arrête au rappel ou à maturité.',
             ha='center',size=9,color='#45556b')
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(out/FIGURE_NAMES[0],dpi=160)
    plt.close(fig)


def _illustrative_payoffs(p, r, sim):
    # The same conditional state/curve and contract as in the pricer. Clipping
    # outside precomputed nodes is explicitly disclosed as an approximation.
    x=sim.factor_at_call
    eligible=np.isfinite(sim.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= sim.first_call_year<=p.acro.last_eligible_call_year+1e-10
    payoff=np.zeros(len(x),dtype=float)
    clipped=0
    values=np.full(len(x),np.nan,dtype=float)
    clipped_flags=np.zeros(len(x),dtype=bool)
    if eligible.any():
        factors=x[eligible]
        clipped_flags[eligible]=(factors<r.curve.factors[0]) | (factors>r.curve.factors[-1])
        clipped=int(clipped_flags.sum())
        B=np.interp(factors,r.curve.factors,r.curve.principal)
        A=np.interp(factors,r.curve.factors,r.curve.annuity)
        values[eligible]=B+r.target_coupon*A
        payoff[eligible]=p.acro.renewal_units*np.maximum(values[eligible]-p.acro.strike,0)
    return payoff, clipped, values, clipped_flags


def _pick_representative(sim, max_examples=9):
    years=sim.first_call_year
    final=sim.final_worst
    selected=[]
    # First demonstrate years where there is an early redemption.
    for year in sorted(set(years[np.isfinite(years)].tolist())):
        cands=np.flatnonzero(np.isclose(years,year))
        if len(cands):
            selected.append(int(cands[len(cands)//2]))
        if len(selected)>=5:break
    alive=np.flatnonzero(~np.isfinite(years))
    if len(alive):
        order=alive[np.argsort(final[alive])]
        for idx in (order[0],order[len(order)//2],order[-1]):
            if int(idx) not in selected: selected.append(int(idx))
    for idx in range(len(years)):
        if len(selected)>=max_examples:break
        if idx not in selected: selected.append(idx)
    return selected[:max_examples]


def _plot_scenarios(p, sim, individual_payoffs, out, clipped=0):
    selected=_pick_representative(sim)
    ncols=3
    nrows=int(np.ceil(len(selected)/ncols))
    fig, axes=plt.subplots(nrows,ncols,figsize=(15,4*nrows),sharex=True,sharey=True)
    axes=np.asarray(axes).reshape(-1)
    colors=['#2367b1','#dc853b','#31937b','#a05eac','#d35461']
    times=sim.observation_times
    for k,idx in enumerate(selected):
        ax=axes[k]
        values=sim.observed_equity_ratios[idx]*100
        for j in range(p.market.n_assets):
            ax.plot(times,values[:,j],label=f'Action {j+1}' if k==0 else None,
                    color=colors[j%len(colors)],lw=1.7,alpha=.9)
        ax.axhline(p.initial_phoenix.autocall_barrier*100,color='#bc5d28',ls='--',alpha=.6,lw=1)
        ax.axhline(p.initial_phoenix.coupon_barrier*100,color='#398877',ls=':',alpha=.6,lw=1)
        coupon_positions=np.flatnonzero(sim.coupon_units_at_observation[idx]>0)
        if len(coupon_positions):
            obs_i=coupon_positions+1
            ax.scatter(times[obs_i],100*basket_level(np.log(sim.observed_equity_ratios[idx,obs_i]), p.initial_phoenix.basket_rule),marker='o',
                       s=23,facecolors='none',edgecolors='#22866b',lw=1.2,zorder=4)
        if np.isfinite(sim.first_call_year[idx]):
            outcome=f'Rappel an {sim.first_call_year[idx]:.0f}'
            ax.axvline(sim.first_call_year[idx],ls='--',lw=1,color='#bc5d28',alpha=.6)
        else: outcome='Maturité sans rappel'
        ax.set_title(f'Traj. {idx} : {outcome}\nACRO à l’exercice ≈ {individual_payoffs[idx]:.2f} €',fontsize=10)
        ax.grid(alpha=.12)
        ax.set_xlim(0,p.initial_phoenix.maturity_years)
    if selected:
        upper = max(float(np.nanmax(sim.observed_equity_ratios[idx]*100)) for idx in selected)
        # A shared upper bound must be chosen AFTER all nine paths are plotted.
        axes[0].set_ylim(0, max(upper*1.07, p.initial_phoenix.autocall_barrier*110))
    for ax in axes[len(selected):]:ax.axis('off')
    if selected:
        handles,labels=axes[0].get_legend_handles_labels()
        fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.955),ncol=p.market.n_assets,frameon=False)
    fig.suptitle(f'09 | Neuf scénarios : {p.market.n_assets} action(s), règle {p.initial_phoenix.basket_rule}',fontsize=15,y=.995)
    note=('Exemples pédagogiques sélectionnés (non représentatifs). Ronds verts = coupon conditionnel payé. '
          'Payoffs via interpolation, non utilisés pour estimer la prime.'
          + (f' Attention : {clipped} état(s) hors-grille, interpolation bornée.' if clipped else ''))
    fig.text(.5,.035,note,
             ha='center',fontsize=9,color='#45556b')
    fig.supxlabel('Années depuis émission',y=.055)
    fig.supylabel('Actions / niveau initial (%)',x=.006)
    fig.tight_layout(rect=(.035,.075,.98,.955),h_pad=2.0)
    fig.savefig(out/FIGURE_NAMES[1],dpi=150)
    plt.close(fig)


def _plot_distribution(result, out, initial_nominal):
    vals=np.asarray(result.discounted_payoff,dtype=float)
    pos=vals[vals>1e-12]
    zero=len(vals)-len(pos)
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13.2,5.7),gridspec_kw={'width_ratios':[1,2]})
    ax1.bar(['Aucun versement','Payoff positif'],[zero,len(pos)],color=['#8493aa','#2378ac'],width=.65)
    ax1.set(ylabel='Trajectoires externes sous Q',title='Payoffs nuls vs positifs')
    for i,n in enumerate((zero,len(pos))):
        ax1.text(i,n+max(1,len(vals)*.01),f'{n/len(vals):.1%}',ha='center',fontsize=12,fontweight='bold')
    ax1.set_ylim(0,len(vals)*1.16)
    if len(pos):
        ax2.hist(pos,bins=min(28,max(7,int(np.sqrt(len(pos))))),color='#2779ad',edgecolor='white',alpha=.9)
        ax2.axvline(pos.mean(),color='#b8563e',lw=2,ls='--',label=f'Moyenne conditionnelle : {pos.mean():.2f} €')
        ax2.legend(frameon=False,fontsize=9)
    else:
        ax2.text(.5,.5,'Aucun payoff positif dans cet échantillon',transform=ax2.transAxes,ha='center',va='center')
    ax2.set(xlabel=f'Payoff ACRO actualisé à t=0 (€ / contrat initial, nominal {initial_nominal:g} €)',ylabel='Nombre de chemins',
            title='Distribution des payoffs positifs')
    ax1.grid(axis='y',alpha=.15); ax2.grid(axis='y',alpha=.15)
    fig.suptitle(f'10 | Payoff simulé et prime ACRO : E^Q[payoff actualisé] = {np.mean(vals):.3f} € / nominal {initial_nominal:g} €',fontsize=14)
    fig.text(.5,.015,'Prix = moyenne de TOUS les chemins, y compris ceux à zéro. Interpolation/erreurs MC internes non comprises dans l’ET externe.',
             ha='center',fontsize=9,color='#45556b')
    fig.tight_layout(rect=(0,.04,1,.93))
    fig.savefig(out/FIGURE_NAMES[2],dpi=160)
    plt.close(fig)


def make_portfolio_figures(p, result, folder, count=48):
    """Adds three educational visuals; pricing estimator is not modified."""
    folder=Path(folder)
    sampled=simulate_phoenix(p.market,p.initial_phoenix,count,p.simulation.steps_per_year,
                             int(p.simulation.seed)+820001,record_observations=True)
    individual, clipped, renewal_values, clipped_flags=_illustrative_payoffs(p,result,sampled)
    _plot_paths(p,sampled,folder)
    _plot_scenarios(p,sampled,individual,folder,clipped=clipped)
    _plot_distribution(result,folder,p.initial_phoenix.nominal)
    with (folder/'illustrative_equity_trajectories.csv').open('w',newline='',encoding='utf8') as file:
        wr=csv.writer(file)
        wr.writerow(['illustrative_path_id','observation_year',*[f'action_{j+1}_ratio' for j in range(p.market.n_assets)],
                     'basket_reference_ratio','coupon_units_at_observation','first_autocall_year',
                     'eligible_call','logvol_factor_at_call','renewal_value_approx','curve_factor_clipped',
                     'approx_acro_intrinsic_at_call','not_part_of_pricing_sample'])
        for i in range(count):
            for j,time in enumerate(sampled.observation_times):
                row=sampled.observed_equity_ratios[i,j]
                if not np.isfinite(row).all():continue
                worst=float(basket_level(np.log(row),p.initial_phoenix.basket_rule))
                wr.writerow([i,float(time),*map(float,row),worst,
                             float(sampled.coupon_units_at_observation[i,j-1]) if j else 0.,
                             float(sampled.first_call_year[i]),
                             bool(np.isfinite(renewal_values[i])),float(sampled.factor_at_call[i]),
                             float(renewal_values[i]),bool(clipped_flags[i]),float(individual[i]),True])
    return {
        'illustrative_paths':count,
        'illustrative_curve_factors_clipped':clipped,
        'illustrative_intrinsic_not_used_for_price':True,
        'portfolio_figures':FIGURE_NAMES,
    }
