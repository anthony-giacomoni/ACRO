"""ACRO V5.13: finite-difference live Greeks near the contractual barriers.

Run from Spyder (F5) or CLI:
  python audit_greeks_v513.py --quick
  python audit_greeks_v513.py
  python audit_greeks_v513.py --banks 6 --outer 10000 --inner 1500

The result is a *numerical validation study*, not an option hedge backtest.
"""
from __future__ import annotations
import argparse,csv,json,time
from pathlib import Path
from dataclasses import replace
from collections import defaultdict
import numpy as np
from scipy.stats import t as student_t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from acro import load_project
from acro.live_risk import LiveState
from acro.greek_validation import live_greek_bank

ROOT=Path(__file__).resolve().parents[1]

def fix_coupon(p,c=.18,steps=24,inner=800,nodes=11):
    return replace(p,initial_coupon_policy='fixed',
        initial_phoenix=replace(p.initial_phoenix,coupon_rate=c),
        simulation=replace(p.simulation,steps_per_year=int(steps),
                 conditional_paths=int(inner),curve_nodes=int(nodes)))

def scenarios(p):
    # Build near-barrier examples from the CONTRACT, not from a hardcoded 70/100.
    steps=p.simulation.steps_per_year
    coupon_barrier=p.initial_phoenix.coupon_barrier
    call_barrier=p.initial_phoenix.autocall_barrier
    obs_gap=steps//p.initial_phoenix.observations_per_year
    maturity=int(round(p.initial_phoenix.maturity_years*steps))
    future_coupon_steps=[k for k in range(1,maturity) if k%obs_gap==0 and k>steps]
    if not future_coupon_steps:
        raise ValueError('No future coupon date for a near-barrier example')
    coupon_k=min(future_coupon_steps,key=lambda k:abs(k-1.5*steps))
    calls=sorted(k for k in p.initial_phoenix.call_steps(steps) if k>0)
    if not calls:
        raise ValueError('Autocall schedule required for this stress study')
    call_k=min(calls,key=lambda k:abs(k-2.*steps))
    def triple(level,second,third):
        values=[float(level),float(second),float(third)]
        return tuple(values[:p.market.n_assets]) + tuple([max(level+.05,.85)]*max(0,p.market.n_assets-3))
    return {
      'Position survivante, hors seuils':LiveState(1.,triple(.88,.92,.84),.05),
      'Avant coupon (worst-of ≈ barrière)':LiveState((coupon_k-1)/steps,
           triple(coupon_barrier-.001,max(coupon_barrier+.08,.92),max(coupon_barrier+.06,.89)),.05),
      'Même date (worst-of +8 points)':LiveState((coupon_k-1)/steps,
           triple(coupon_barrier+.08,max(coupon_barrier+.08,.92),max(coupon_barrier+.06,.89)),.05),
      'Avant rappel (worst-of ≈ seuil)':LiveState((call_k-1)/steps,
           triple(call_barrier-.003,call_barrier+.03,call_barrier+.02),.05)
    }

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        key=(r['scenario'],r['asset'],r['component'],r['bump_percent'])
        groups[key].append(r)
    out=[]
    for (name,asset,component,bump),bank in groups.items():
        n=len(bank)
        rec={'scenario':name,'asset':asset,'component':component,
             'bump_percent':bump,'banks':n,'n_paths_each':bank[0]['n_paths'],
             'description':'Independent inner and outer banks; same paths between finite-difference bumps'}
        for metric in ('delta','gamma'):
            v=np.array([float(r[metric]) for r in bank])
            mean=float(v.mean())
            se=float(v.std(ddof=1)/np.sqrt(n)) if n>1 else float('nan')
            radius=float(student_t.ppf(.975,n-1)*se) if n>1 else float('nan')
            rec[metric+'_mean']=mean
            rec[metric+'_se_banks']=se
            rec[metric+'_ci95_low']=mean-radius
            rec[metric+'_ci95_high']=mean+radius
            rec[metric+'_mean_conditional_outer_se']=float(np.mean([x[metric+'_outer_se'] for x in bank]))
        out.append(rec)
    return sorted(out,key=lambda v:(v['scenario'],v['asset'],v['bump_percent'],v['component']))

def plot_metric(summary,metric,out):
    states=list(dict.fromkeys(r['scenario'] for r in summary))
    bumps=sorted({r['bump_percent'] for r in summary})
    n_assets=max(r['asset'] for r in summary)
    fig,axes=plt.subplots(len(states),1,figsize=(10.5,3.9*len(states)),squeeze=False)
    for z,state in enumerate(states):
        ax=axes[z,0]
        for asset in range(1,n_assets+1):
            rr=[r for r in summary if r['scenario']==state and r['asset']==asset and r['component']=='package']
            rr.sort(key=lambda x:x['bump_percent'])
            xs=[x['bump_percent'] for x in rr];ys=[x[metric+'_mean'] for x in rr]
            cis=np.array([[(x[metric+'_mean']-x[metric+'_ci95_low']),
                          (x[metric+'_ci95_high']-x[metric+'_mean'])] for x in rr]).T
            ax.errorbar(xs,ys,yerr=cis,marker='o',capsize=3,label=f'Action {asset}')
        ax.axhline(0,color='gray',lw=.7)
        ax.set_title(state)
        ax.set_ylabel('€/€' if metric=='delta' else '€/€²')
        ax.set_xlabel('Choc relatif symétrique (± % du spot)')
        ax.grid(alpha=.18)
        ax.legend(ncol=3,fontsize=8)
    fig.suptitle(('Deltas' if metric=='delta' else 'Gammas') +
       ' du portefeuille Phoenix + ACRO | IC 95 % inter-banques',y=.997)
    fig.tight_layout(rect=(0,0,1,.985))
    fig.savefig(out,dpi=155)
    plt.close(fig)

def plot_acro_delta(summary,path):
    """Option-only delta (not dominated by the delta of the underlying Phoenix)."""
    states=list(dict.fromkeys(r['scenario'] for r in summary))
    assets=sorted(set(r['asset'] for r in summary))
    fig,ax=plt.subplots(figsize=(11.8,5.3))
    x=np.arange(len(states),dtype=float)
    width=.23
    for idx,asset in enumerate(assets):
        r=[v for v in summary if v['scenario'] in states and v['asset']==asset and
                  v['component']=='acro' and abs(v['bump_percent']-1)<1e-8]
        r=sorted(r,key=lambda v:states.index(v['scenario']))
        means=np.array([v['delta_mean'] for v in r])
        err=np.array([[v['delta_mean']-v['delta_ci95_low'] for v in r],
                      [v['delta_ci95_high']-v['delta_mean'] for v in r]])
        ax.bar(x+(idx-(len(assets)-1)/2)*width,means,width,yerr=err,capsize=3,
               label=f'Action {asset}')
    ax.set_xticks(x,[v.replace('Position survivante, ','').replace('Avant ','Avant ') for v in states],rotation=12,ha='right')
    ax.set_ylabel('Delta ACRO : € par € de prix de l’action')
    ax.set_title('Risque directionnel propre à l’option ACRO | bump ±1 %')
    ax.axhline(0,color='gray',lw=.8)
    ax.legend(ncol=3);ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(path,dpi=170);plt.close(fig)


def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf8') as f:
        if not rows:return
        w=csv.DictWriter(f,list(rows[0].keys()));w.writeheader();w.writerows(rows)

def main(output=ROOT/'results_greeks_v513',quick=False,banks=None,outer=None,inner=None,steps=24,ref_paths=None):
    if quick:
        banks=banks or 3;outer=outer or 1200;inner=inner or 384;ref_paths=ref_paths or 256
    else:
        banks=banks or 5;outer=outer or 2800;inner=inner or 640;ref_paths=ref_paths or 512
    p=fix_coupon(load_project(ROOT/'config.json'),steps=steps,inner=inner,nodes=11)
    p.validate(); out=Path(output);out.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter(); raw=[]
    for name,state in scenarios(p).items():
        for bank in range(banks):
            rows=live_greek_bank(p,state,bumps=(.005,.01,.02),n_paths=outer,
                 outer_seed=507000+101*bank+3*list(scenarios(p)).index(name),
                 curve_seed=509000+739*bank+11*list(scenarios(p)).index(name),
                 pricing_method='gbm_cv',ref_paths=ref_paths,scrambles=3)
            for r in rows:
                raw.append(dict(scenario=name,bank=bank,**r))
    summary=summarize(raw)
    write_csv(out/'greeks_banks.csv',raw)
    write_csv(out/'greeks_summary.csv',summary)
    plot_metric(summary,'delta',out/'01_deltas_barrieres.png')
    plot_metric(summary,'gamma',out/'02_gammas_barrieres.png')
    plot_acro_delta(summary,out/'03_deltas_acro_seuls.png')
    # The criterion is BOTH noise and bump robustness. Do not declare pass based on just 1 source.
    bullet=[]; n=0
    for state in scenarios(p):
        subset=[r for r in summary if r['scenario']==state and r['component']=='package']
        for asset in range(1,p.market.n_assets+1):
            rr=sorted((r for r in subset if r['asset']==asset),key=lambda r:r['bump_percent'])
            delta_spread=max(r['delta_mean'] for r in rr)-min(r['delta_mean'] for r in rr)
            gamma_spread=max(r['gamma_mean'] for r in rr)-min(r['gamma_mean'] for r in rr)
            # Check if CI spans zero for the estimated gamma at standard bump.
            rr1=min(rr,key=lambda r:abs(r['bump_percent']-1))
            bullet.append((state,asset,delta_spread,gamma_spread,
                           rr1['delta_mean'],rr1['delta_ci95_low'],rr1['delta_ci95_high'],
                           rr1['gamma_mean'],rr1['gamma_ci95_low'],rr1['gamma_ci95_high']))
            n+=1
    report=['# Audit ACRO V5.13 — Greeks de position vivante', '',
      f'Modèle synthétique, coupon contractuel figé à 18 %, grille {steps} pas/an.',
      f'{banks} banques indépendantes par état, {outer} trajectoires externes et {inner} internes par banque.',
      'GBM-CV utilisé pour la courbe de valorisation du Phoenix renouvelé. Références contractuelles fixes.',
      '', '## Résultats locaux à ±1 % du spot', '',
      '|État|Action|Delta total (IC95 inter-banques)|Gamma total (IC95 inter-banques)|Amplitude delta entre bumps|',
      '|---|---:|---:|---:|---:|']
    for row in bullet:
        state,asset,sd,sg,d,dl,du,g,gl,gu=row
        report.append(f'|{state}|{asset}|{d:+.3f} [{dl:+.3f}; {du:+.3f}]|{g:+.4f} [{gl:+.4f}; {gu:+.4f}]|{sd:.3f}|')
    report.extend(['', '## Méthodologie et limites', '',
       '- Delta = (V(S+h)−V(S−h))/(2h), gamma = (V(S+h)−2V(S)+V(S−h))/h².',
       '- Chocs symétriques de ±0,5 %, ±1 % et ±2 % ; mêmes chocs Brownien pour base, up et down dans une banque.',
       '- Les banques changent à la fois les chocs externes et les cotations internes (référence GBM incluse).',
       '- Les intervalles Student-t décrivent la dispersion entre banques dans le **modèle discret**, pas l’incertitude totale.',
       '- Les états proches de 70 % et 100 % sont observés un pas **avant** une date contractuelle ; le comparateur à +8 points du coupon garde **la même date et les mêmes autres facteurs**.',
       '- Les notes proches des seuils sont hypothétiquement survivantes ; aucun historique réel complet n’est reconstitué.',
       '- Les gammas peuvent être mal définis à une discontinuité contractuelle. La dépendance au bump n’est pas seulement du bruit.',
       '- L’ACRO seule peut avoir des deltas trop faibles pour être distingués du bruit sur ces budgets.',
       '- Absence de calibration marché, de risque de crédit, de financement, de coûts de couverture, de P&L dynamique.',
       '- Aucun résultat ne constitue une recommandation de couverture exécutable.', '',
       '## Décision', '',
       'La validation porte sur la reproductibilité des différences finies et la visibilité de leur incertitude. '
       'Aucun Greek n’est déclaré « validé » sans stabilité aux graines et aux tailles de bump. '
       'Les prochains travaux doivent cibler le P&L de couverture, les risques vega et corrélation, puis la comparaison économique.',
       '',f'Temps de calcul de l’audit : {time.perf_counter()-t0:.1f} s.'])
    (out/'RAPPORT_GREEKS_V5_13_FR.md').write_text('\n'.join(report),encoding='utf8')
    metrics={'banks':banks,'outer_paths':outer,'inner_paths':inner,'steps_per_year':steps,'ref_paths':ref_paths,
             'raw_rows':len(raw),'elapsed_seconds':time.perf_counter()-t0,
             'reference':'synthetic model; statistical uncertainty only; no hedge claims'}
    (out/'audit_meta.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False),encoding='utf8')
    print('ACRO V5.13:',len(raw),'observations; elapsed',round(metrics['elapsed_seconds'],2),'sec')
    print('Results:',out)
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(ROOT/'results_greeks_v513'))
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--banks',type=int)
    parser.add_argument('--outer',type=int)
    parser.add_argument('--inner',type=int)
    parser.add_argument('--steps',type=int,default=24)
    parser.add_argument('--ref-paths',type=int)
    a=parser.parse_args()
    main(a.output,a.quick,a.banks,a.outer,a.inner,a.steps,a.ref_paths)
