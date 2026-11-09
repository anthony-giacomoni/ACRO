"""ACRO: interpretable research figures from the most recent run.

Run after run_demo.py or standalone with `python plot_results.py`.
Uses CSV/JSON produced by the existing pricer; does NOT reprice the notes.
Every ACRO payoff picture assumes the original Phoenix survived to the
stated observation date, and the holder has an eligible first autocall.
"""
from pathlib import Path
import argparse, json, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

ROOT=Path(__file__).resolve().parent


def make_figures(results=ROOT/'results'):
    out=Path(results)
    s=pd.read_csv(out/'scenario_paths.csv')
    g=pd.read_csv(out/'conditional_curve.csv').sort_values('logvol_factor')
    m=json.loads((out/'metrics.json').read_text(encoding='utf8'))
    source_config = (out/'effective_config.json' if (out/'effective_config.json').exists()
                     else out/'run_config.json' if (out/'run_config.json').exists()
                     else ROOT/'config.json')
    config=json.loads(source_config.read_text(encoding='utf8'))
    nominal=config['initial_phoenix']['nominal']
    strike=config['acro']['strike']
    units=config['acro']['renewal_units']
    c0=float(m['annual_coupon']) if config['acro']['target_coupon_rate'] is None else float(config['acro']['target_coupon_rate'])
    fixed=float(m['fixed_payment_same_initial_option_price'])
    eligible=s[s['eligible']==1].copy()
    if len(eligible)==0:
        raise RuntimeError('Pas de rappel admissible dans les simulations : graphiques conditionnels indisponibles.')
    colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
    foot='Marché synthétique, non calibré. Les payoffs sont conditionnels au premier rappel admissible ; ils ne sont pas des prix de marché.'
    # Large single-topic charts, legible alone rather than as a tiny contact sheet.
    def new(title,subtitle,figsize=(11.0,6.3)):
        fig,ax=plt.subplots(figsize=figsize)
        fig.subplots_adjust(left=.105,right=.96,bottom=.27,top=.79)
        fig.text(.105,.945,title,fontsize=19,fontweight='bold',va='top')
        fig.text(.105,.885,subtitle,fontsize=11.3,color='#525252',va='top')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.17)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=11)
        return fig,ax
    def save(fig,name,note=foot):
        fig.text(.105,.045,'\n'.join(textwrap.wrap(note,width=112)),fontsize=9.2,color='#575757',va='bottom',linespacing=1.35)
        fig.savefig(out/name,dpi=180,facecolor='white')
        plt.close(fig)
        made.append(name)
    made=[]
    xnodes=g['logvol_factor'].to_numpy(float)
    bn=g['principal_pv'].to_numpy(float)
    an=g['coupon_annuity_pv'].to_numpy(float)
    # Renewed-note price at a prescribed stochastic market state.
    def top_up(logvol):
        B=np.interp(logvol,xnodes,bn)
        A=np.interp(logvol,xnodes,an)
        return units*np.maximum(B+c0*A-strike,0.0)
    def fair_coupon(logvol):
        B=np.interp(logvol,xnodes,bn)
        A=np.interp(logvol,xnodes,an)
        return (config['renewal_phoenix']['nominal']-B)/A

    # FIG 1 -- the distinctive 2D payoff, at ONE eligible observation date.
    # Current configured basket reference controls autocall; factor controls repricing.
    fig,ax=new('01 / Quand l’ACRO se déclenche-t-elle ?',
               'Payoff à une date d’observation, si le Phoenix était encore vivant avant cette date',
               (11.1,7.0))
    rule=config['initial_phoenix'].get('basket_rule','worst_of')
    threshold=config['initial_phoenix']['autocall_barrier']
    x=np.linspace(max(.50,threshold-.35),threshold+.35,260)
    y=np.linspace(max(0.3,np.exp(xnodes[0])),min(2.3,np.exp(xnodes[-1])),190)
    X,Y=np.meshgrid(x,y)
    z=np.where(X>=threshold,top_up(np.log(Y)),0.)
    im=ax.pcolormesh(X*100,Y,z,shading='auto',cmap='viridis',vmin=0)
    ax.axvline(threshold*100,linestyle='--',lw=2.3,color='white')
    ax.text(threshold*100-1,y[-1]-.04,'PAS DE RAPPEL',ha='right',va='top',fontsize=11,color='white',weight='bold',bbox={'facecolor':'black','alpha':.3,'pad':5,'edgecolor':'none'})
    ax.text(threshold*100+1,y[-1]-.04,'RAPPEL POSSIBLE',ha='left',va='top',fontsize=11,color='white',weight='bold',bbox={'facecolor':'black','alpha':.3,'pad':5,'edgecolor':'none'})
    ax.set_xlabel(f'Référence {rule} à la date d’observation (% du niveau initial)',fontsize=12,labelpad=9)
    ax.set_ylabel('Multiplicateur de volatilité instantanée',fontsize=12,labelpad=9)
    cb=fig.colorbar(im,ax=ax,pad=.02,shrink=.93)
    cb.set_label(f'Payoff ACRO à cette date (€ par contrat de nominal initial {nominal:g} €)',fontsize=11)
    cb.ax.tick_params(labelsize=10)
    fig.subplots_adjust(left=.105,right=.915,bottom=.20,top=.78)
    save(fig,'01_carte_payoff_worstof_vol.png',
         'Vue instantanée conditionnelle : l’exercice dépend du facteur de volatilité, pas du niveau de référence au-delà du seuil. Aucun exercice sans rappel.')

    # FIG 2 -- cash value at call vs fair renewal coupon, from actual model states.
    fig,ax=new('02 / Combien vaut le droit de renouvellement ?',
               f'Coupon contractuel conservé : {100*c0:.2f} % • Paiement uniquement au premier autocall admissible')
    qlow,qhigh=np.quantile(eligible['fair_coupon'].to_numpy(float),[.01,.99])
    xs=np.linspace(max(0,qlow-.01),qhigh+.01,400)
    logspace=np.linspace(xnodes.min(),xnodes.max(),2000)
    fair=100*fair_coupon(logspace)
    vals=top_up(logspace)
    sort=np.argsort(fair)
    # This is multi-valued in principle; scatter from actual states is more honest.
    take=eligible.sample(n=min(len(eligible),1200),random_state=17)
    ax.scatter(take['fair_coupon']*100,take['acro_intrinsic'],s=15,alpha=.26,
               color=colors[0],label='États simulés de rappel')
    ax.axvline(100*c0,ls='--',lw=2,color=colors[1],label=f'Coupon préservé {100*c0:.2f} %')
    ax.axhline(fixed,ls=':',lw=2,color=colors[2],label=f'Indemnité forfaitaire à prime égale : {fixed:.2f} €')
    ax.set_xlim(xs.min()*100,xs.max()*100)
    ax.set_ylim(bottom=-.08)
    ax.set_xlabel('Coupon équitable du nouveau Phoenix à la date du rappel (%)',fontsize=12,labelpad=9)
    ax.set_ylabel(f'Valeur d’exercice (€ par contrat de nominal initial {nominal:g} €)',fontsize=12,labelpad=9)
    ax.legend(frameon=False,fontsize=10,loc='upper right')
    save(fig,'02_payoff_au_rappel.png',
         'Chaque point utilise un état de marché réellement simulé. Le seuil de coupon au pair et le seuil d’exercice ne coïncident que si strike = nominal renouvelé ; la prime payée en année 0 est exclue.')

    # FIG 3 -- call timing and contribution to initial price, with factual labels.
    years=np.sort(eligible['first_autocall_year'].unique())
    n=len(s)
    probcall=[];probexercise=[];prices=[]
    for yr in years:
        z=eligible[np.isclose(eligible['first_autocall_year'],yr)]
        probcall.append(100*len(z)/n)
        probexercise.append(100*np.sum(z['acro_intrinsic']>1e-12)/n)
        prices.append(z['discounted_acro'].sum()/n)
    fig,ax=new('03 / Quelles années comptent vraiment ?',
               'Probabilité de rappel et d’exercice, avec contribution de chaque date à la prime initiale')
    pos=np.arange(len(years),dtype=float)
    wid=.34
    ax.bar(pos-wid/2,probcall,width=wid,label='Premier rappel admissible',color=colors[0])
    ax.bar(pos+wid/2,probexercise,width=wid,label='Rappel + exercice de l’ACRO',color=colors[1])
    for i,p in enumerate(prices):
        ax.text(i,max(probcall[i],probexercise[i])+.65,f'Prime : {p:.3f} €',fontsize=11,ha='center',weight='bold')
    ax.set_xticks(pos,[f'Année {yr:g}' for yr in years])
    ax.set_ylabel('Probabilité sous Q (% des trajectoires initiales)',fontsize=12,labelpad=9)
    ax.set_ylim(0,max(probcall)*1.27 if probcall else 1)
    ax.legend(frameon=False,fontsize=11,loc='upper right')
    save(fig,'03_rappel_exercice_et_prix.png',
         f'Probabilité totale de rappel admissible : {100*m["eligible_call_probability_Q"]:.1f} %. Somme des contributions : {sum(prices):.3f} € par nominal initial {nominal:g} €. Biais de valorisation interne non inclus.')

    # FIG 4 -- fair renewal coupon distribution by first eligible autocall year.
    fig,ax=new('04 / Quelles conditions après le rappel ?',
               'Distribution du coupon équitable du nouveau Phoenix, par première date de rappel')
    data=[100*eligible.loc[np.isclose(eligible['first_autocall_year'],yr),'fair_coupon'].to_numpy() for yr in years]
    b=ax.boxplot(data,positions=np.arange(len(years)),widths=.5,patch_artist=True,
              showfliers=False,medianprops=dict(lw=2.3,color=colors[1]))
    for box in b['boxes']:
        box.set_facecolor(colors[0]);box.set_alpha(.18);box.set_edgecolor(colors[0])
    for i,ys in enumerate(data):
        # unobtrusive points for sample composition; few drawn for readability
        sub=np.random.default_rng(120+i).choice(ys,size=min(len(ys),90),replace=False)
        ax.scatter(np.random.default_rng(900+i).normal(i,.052,len(sub)),sub,
                   s=11,alpha=.13,color=colors[0])
    ax.axhline(100*c0,ls='--',lw=2.2,color=colors[2],label=f'Coupon contractuel : {100*c0:.2f} %')
    ax.set_xticks(range(len(years)),[f'Année {yr:g}\n(n={len(z):,})' for yr,z in zip(years,data)])
    ax.set_ylabel('Coupon équitable au pair (%)',fontsize=12,labelpad=9)
    ax.legend(frameon=False,fontsize=11)
    save(fig,'04_distribution_coupon_renouvellement.png',
         'Les groupes sont CONDITIONNELS au premier rappel dans l’année considérée. Boîtes = quartiles ; moustaches = 1,5 × IQR (sans valeurs aberrantes).')

    # FIG 5 -- same-premium fixed indemnity: loss exceedance conditional on call.
    owed=eligible['acro_intrinsic'].to_numpy(float)
    residual_fixed=np.maximum(owed-fixed,0)
    xtop=max(np.quantile(owed,.995),np.quantile(residual_fixed,.995),.05)
    cut=np.linspace(0,xtop,300)
    p0=np.array([np.mean(owed>v) for v in cut])
    pf=np.array([np.mean(residual_fixed>v) for v in cut])
    fig,ax=new('05 / ACRO ou indemnité forfaitaire ?',
               'Probabilité qu’un manque à gagner reste non compensé après le premier rappel')
    ax.plot(cut,100*p0,lw=2.8,color=colors[0],label='Sans protection')
    ax.plot(cut,100*pf,lw=2.8,color=colors[1],label=f'Indemnité fixe de {fixed:.2f} €')
    ax.axhline(0,ls=':',lw=2,color=colors[2],label='ACRO : déficit contractuel nul*')
    ax.set_ylim(0,102)
    ax.set_xlim(0,xtop)
    ax.set_ylabel('Probabilité de dépassement | rappel (%)',fontsize=12,labelpad=9)
    ax.set_xlabel(f'Déficit de financement non couvert (€ par nominal initial {nominal:g} €)',fontsize=12,labelpad=9)
    ax.legend(frameon=False,fontsize=11,loc='upper right')
    save(fig,'05_couverture_versus_forfait.png',
         '*Nul par construction du payoff, hors prime, défaut et exécution. L’indemnité forfaitaire est calibrée à la même prime théorique initiale, pas au même paiement dans chaque état.')

    # FIG 6 -- show dependence on equity/log-vol shock correlation (outer only).
    sens=pd.read_csv(out/'sensitivity.csv')
    if len(sens):
        sens=sens.sort_values('rho')
        fig,ax=new('06 / Quel risque fait varier la prime ?',
                   'Sensibilité au lien entre chocs actions et volatilité — coupon de départ conservé')
        xs=sens['rho'].to_numpy(float)
        ys=sens['premium'].to_numpy(float)
        errs=1.96*sens['outer_se'].to_numpy(float)
        ax.plot(xs,ys,marker='o',lw=2.5,markersize=8,color=colors[0],label='Prime ACRO théorique')
        ax.fill_between(xs,ys-errs,ys+errs,alpha=.16,color=colors[0],label='± 1,96 SE externe seule (courbe interne recalculée par ρ ; incertitude interne exclue)')
        ax.axvline(config['market']['equity_factor_correlation'],ls='--',lw=1.8,
                   color=colors[1],label='Corrélation du scénario de référence')
        ax.set_xlabel('Corrélation des chocs panier actions / facteur de volatilité',fontsize=12,labelpad=9)
        ax.set_ylabel(f'Prime ACRO à l’origine (€ / nominal initial {nominal:g} €)',fontsize=12,labelpad=9)
        ax.set_xlim(min(-.98,xs.min()-.05),max(.4,xs.max()+.05))
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False,fontsize=10,loc='upper right')
        save(fig,'06_sensibilite_dependance.png',
             'Même estimateur que la prime de référence, coupon initial fixé au niveau de base. Les bandes NE couvrent PAS le bruit MC interne, les autres biais, ni l’incertitude de modèle. ρ n’est pas la corrélation entre actions.')

    # FIG 7 -- THE cost of flexibility at issue. Initial Phoenix has not
    # changed when varying the renewed note's target coupon. All values in
    # this figure are discounted expected premiums, NOT exercise payoffs.
    fig,ax=new('07 / Combien coûte la promesse de renouvellement ?',
               'Prix aujourd’hui du droit ACRO, selon le coupon contractuel promis au renouvellement')
    base=float(m['annual_coupon'])
    bundle=m.get('bundled_fair_coupon')
    ref_coupons=[base,c0]+([float(bundle)] if bundle is not None else [])
    grid_coupon=np.linspace(max(0,min(ref_coupons)-.06),max(ref_coupons)+.06,101)
    factors=eligible['logvol_factor_at_call'].to_numpy(float)
    B=np.interp(factors,xnodes,bn)
    A=np.interp(factors,xnodes,an)
    weights=np.exp(-float(config['market']['risk_free_rate'])*eligible['first_autocall_year'].to_numpy(float))
    premiums=[float(np.sum(weights*units*np.maximum(B+c*A-strike,0))/len(s)) for c in grid_coupon]
    ax.plot(grid_coupon*100,premiums,lw=2.7,color=colors[0])
    ax.axvline(base*100,lw=1.8,ls='--',color=colors[1],label=f'Coupon du Phoenix seul : {base:.2%}')
    ax.scatter([c0*100],[float(m['acro_price'])],s=70,color=colors[1],zorder=4,
               label=f'ACRO du contrat : {float(m["acro_price"]):.2f} €')
    if bundle is not None:
        xbundle=float(bundle)*100
        ax.axvline(xbundle,lw=1.4,ls=':',color=colors[2],
                   label=f'Coupon de la note groupée au pair : {float(bundle):.2%}')
    ax.set_xlabel('Coupon annuel contractuel du nouveau Phoenix (%)',fontsize=12,labelpad=9)
    ax.set_ylabel(f'Prime ACRO à l’émission (€ par nominal initial {nominal:g} €)',fontsize=12,labelpad=9)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False,fontsize=10)
    save(fig,'07_prix_promesse_coupon.png',
         'Courbe calculée à dates de rappel et états de marché identiques, seul le coupon du NOUVEAU Phoenix varie. Prix en t=0, différent du payoff au rappel. Biais d’interpolation / Monte Carlo encore présent.')

    return made

if __name__=='__main__':
    a=argparse.ArgumentParser()
    a.add_argument('--results',default=str(ROOT/'results'))
    x=a.parse_args()
    for f in make_figures(x.results): print('Créé :',Path(x.results)/f)
