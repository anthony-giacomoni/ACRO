"""ACRO V5.5 -- unbiased independent-pilot call-date control variate.

The financial payoff and Q dynamics in acro/ are not modified. The coupon
is CONTRACTUALLY FIXED at 18%, not re-estimated between grids. The purpose of
this audit is to isolate/possibly reduce OUTER Monte Carlo sampling noise.

A pilot sample provides Q call-date probabilities. A disjoint training sample
provides conditional call-date average payoffs (beta); a disjoint evaluation
sample provides the residual. Conditional on the two MC renewal curves, this
control-variate estimator is unbiased for the discretized payoff valued with
those curves. Its reported SE does NOT include inner, interpolation, time or
model/calibration uncertainty. In the report all comparisons use the SAME
market paths, coupons and curve for plain and controlled estimators.

    python audit_outer_v55.py --quick
    python audit_outer_v55.py
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import t as student_t

from acro import load_project, simulate_phoenix
from acro.variance_reduction import controlled_mean
from research.audit_coupled import innovations_for_grid, aggregate_innovations

ROOT = Path(__file__).resolve().parents[1]
GRIDS = (48, 96, 192)
COUPON = 0.18


def config_budget(quick: bool):
    return (dict(reps=3, pilot=6000, train=800, eval=1600, inner=1600,
                 anchor=25000, anchor_train=2500, nodes=11, batch=600) if quick else
            dict(reps=4, pilot=9000, train=1000, eval=2200, inner=1700,
                 anchor=30000, anchor_train=3000, nodes=11, batch=800))


def years_of_eligible_calls(p):
    annual=p.initial_phoenix
    steps=annual.call_steps(192)
    last=p.acro.last_eligible_call_year
    years=tuple(sorted(s/192 for s in steps if last is None or s/192<=last+1e-10))
    if not years: raise ValueError('No eligible autocall dates in this configuration')
    return years


def labels_for_years(years, available):
    """Event label 0 = no covered autocall, k=1,...,K = first autocall at date k."""
    y=np.asarray(years,float)
    out=np.zeros(y.size,dtype=np.int16)
    for j,t in enumerate(available,1):
        out[np.isfinite(y)&np.isclose(y,t,atol=1e-9,rtol=0)]=j
    return out


def two_sample_cv(y_pilot, l_pilot, y_train, l_train, y_eval, l_eval, K):
    """Independent-pilot control variate; variance conditional on trained beta.

    Estimator = mean(Y_eval - beta[label_eval]) + mean(beta[label_pilot]).
    beta[label=0] is zero. Both pilot and eval must be independent of train.
    This holds for *any* beta, so noisy/small training cannot cause bias.
    """
    ys=(np.asarray(y_pilot,float),np.asarray(y_train,float),np.asarray(y_eval,float))
    ls=(np.asarray(l_pilot,int),np.asarray(l_train,int),np.asarray(l_eval,int))
    for z,l in zip(ys,ls):
        if z.ndim!=1 or z.size!=l.size or not np.isfinite(z).all():
            raise ValueError('Invalid sample')
        if z.size<2 or (l<0).any() or (l>K).any():
            raise ValueError('Invalid label/sample length')
    beta=np.zeros(K+1)
    for k in range(1,K+1):
        sel=ls[1]==k
        beta[k]=float(ys[1][sel].mean()) if sel.any() else 0.0
    pilot_control=beta[ls[0]]
    residual=ys[2]-beta[ls[2]]
    estimate=float(pilot_control.mean()+residual.mean())
    # Independent-sample variance; trained betas treated as fixed in each replication.
    var=float(pilot_control.var(ddof=1)/len(pilot_control)
              + residual.var(ddof=1)/len(residual))
    naive=float(ys[2].mean())
    naive_se=float(ys[2].std(ddof=1)/np.sqrt(len(ys[2])))
    pooled=np.concatenate((ys[0],ys[1],ys[2]))
    return dict(controlled=estimate, cv_se=np.sqrt(var), naive=naive,
                naive_se=naive_se, all_paths=float(pooled.mean()),
                all_paths_se=float(pooled.std(ddof=1)/np.sqrt(len(pooled))),
                beta=beta.tolist(), correction=float(estimate-naive),
                variance_ratio=float(var/(naive_se**2)) if naive_se>0 else None)


def paired_outer_sample(p, amount, grids, batch, seed):
    """Use same fine Brownian paths across time grids, processed in bounded chunks."""
    nasset=p.market.n_assets
    steps=int(round(192*p.initial_phoenix.maturity_years))
    events={g:[] for g in grids}; factors={g:[] for g in grids}
    processed=0; rng=np.random.default_rng(seed)
    while processed<amount:
        take=min(batch,amount-processed)
        # Generate independent chunk seeds; within a chunk the Brownian drivers
        # are shared across grids. No cross-batch state or histories retained.
        bank=innovations_for_grid(steps,take,nasset,int(rng.integers(1,2**31-1)))
        for g in grids:
            innovations=aggregate_innovations(bank,192,g,p.initial_phoenix.maturity_years)
            s=simulate_phoenix(p.market,p.initial_phoenix,take,g,0,innovations=innovations)
            events[g].append(s.first_call_year)
            factors[g].append(s.factor_at_call)
        processed+=take
        del bank
    return {g:(np.concatenate(events[g]),np.concatenate(factors[g])) for g in grids}


def paired_anchor(p, grid, n_train, n_eval, batch=1000):
    """Independent high-budget renewal-note quote at the initial factor.

    An out-of-sample regression on stopped stock Q-martingales reduces noise.
    At any finite grid the stopped normalized stocks have EXACT known mean 1.
    """
    m=int(round(192*p.renewal_phoenix.maturity_years))
    def generate(n,seed):
        yy=[];controls=[];rng=np.random.default_rng(seed)
        count=0
        while count<n:
            take=min(batch,n-count)
            bank=innovations_for_grid(m,take,p.market.n_assets,int(rng.integers(1,2**31-1)))
            shocks=aggregate_innovations(bank,192,grid,p.renewal_phoenix.maturity_years)
            r=simulate_phoenix(p.market,p.renewal_phoenix,take,grid,0,
                               initial_factor=p.market.initial_logvol_factor,innovations=shocks)
            yy.append(r.pv(COUPON));controls.append(r.stopped_stock_martingales)
            count+=take
        return np.concatenate(yy),np.concatenate(controls)
    ytr,mtr=generate(n_train,169001)
    ye,me=generate(n_eval,277001)
    return controlled_mean(ytr,mtr,ye,me)


def conditional_curve_pair(p, grid, nodes, n_inner, seeds,anchor):
    """Independent CRN factor DIFFERENCE curves; share precise anchor value.

    Instead of quoting the entire noisy Phoenix separately at every factor, use
    V(x) = V(x0) + E[PV(x)-PV(x0)] with exactly common innovations.
    Each difference is control-variate adjusted using the difference between
    stopped stock martingales, which has known expectation zero.
    """
    result=[]
    nasset=p.market.n_assets
    steps=int(round(192*p.renewal_phoenix.maturity_years))
    for seed in seeds:
        bank=innovations_for_grid(steps,n_inner,nasset,seed)
        innovations=aggregate_innovations(bank,192,grid,p.renewal_phoenix.maturity_years)
        base=simulate_phoenix(p.market,p.renewal_phoenix,n_inner,grid,0,
                             initial_factor=p.market.initial_logvol_factor,innovations=innovations)
        base_pv=base.pv(COUPON)
        h=n_inner//2
        values=[]
        for x in nodes:
            if abs(x-p.market.initial_logvol_factor)<1e-14:
                values.append(float(anchor));continue
            sim=simulate_phoenix(p.market,p.renewal_phoenix,n_inner,grid,0,
                                 initial_factor=float(x),innovations=innovations)
            dif=sim.pv(COUPON)-base_pv
            # Controls for the DIFFERENCE both have exactly zero expectation.
            cm=sim.stopped_stock_martingales-base.stopped_stock_martingales+1.
            v1=controlled_mean(dif[:h],cm[:h],dif[h:],cm[h:]).estimate
            v2=controlled_mean(dif[h:],cm[h:],dif[:h],cm[:h]).estimate
            values.append(float(anchor)+.5*(v1+v2))
        result.append(np.asarray(values))
        del bank
    return result[0],result[1]


def call_payoff(p, years, factors, nodes, values, dates):
    years=np.asarray(years,float)
    factor=np.asarray(factors,float)
    label=labels_for_years(years,dates)
    Y=np.zeros(years.size)
    idx=label!=0
    if idx.any():
        vol=factor[idx]
        if np.min(vol)<nodes[0]-1e-10 or np.max(vol)>nodes[-1]+1e-10:
            raise ValueError('Out-of-bounds factor: use a wider conditional grid')
        v=np.interp(vol,nodes,values)
        Y[idx]=p.acro.renewal_units*np.maximum(v-p.acro.strike,0)*np.exp(
            -p.market.risk_free_rate*years[idx])
    return Y,label


def report_ci(x):
    x=np.asarray(x,float)
    if x.size<2: raise ValueError('Need at least 2 replications')
    se=float(x.std(ddof=1)/np.sqrt(x.size))
    h=float(student_t.ppf(0.975,x.size-1)*se)
    mu=float(x.mean())
    return dict(mean=mu,ci95=[mu-h,mu+h],halfwidth=h,replications=int(x.size))


def run(quick=False,output=None):
    p=load_project(ROOT/'config.json')
    if p.market.dynamics!='stochastic_vol' or p.acro.target_coupon_rate is not None:
        raise ValueError('V5.5 benchmark is for the reference 1-factor and matching target coupon')
    if p.initial_phoenix != p.renewal_phoenix:
        raise ValueError('V5.5 benchmark assumes the same renewal terms (reset)')
    dates=years_of_eligible_calls(p)
    b=config_budget(quick)
    out=Path(output or (ROOT/('results_audit_v55_quick' if quick else 'results_audit_v55')))
    out.mkdir(exist_ok=True,parents=True)
    start=perf_counter()
    # Hold the conditional curves/anchor FIXED while changing outer paths.
    # That makes replicate dispersion a legitimate diagnostic of OUTER error.
    all_samples=[]
    for rep in range(b['reps']):
        sub=[]
        for j,amount in enumerate([b['pilot'],b['train'],b['eval']]):
            sub.append(paired_outer_sample(p,amount,GRIDS,b['batch'],101100+rep*100000+j*3191))
        all_samples.append(sub)
    factor_nodes={};curves={};anchors={}
    for g in GRIDS:
        xs=np.concatenate([sub[j][g][1][np.isfinite(sub[j][g][0])]
                            for sub in all_samples for j in range(3)])
        lo=min(float(xs.min())-.12,p.market.initial_logvol_factor-.05)
        hi=max(float(xs.max())+.12,p.market.initial_logvol_factor+.05)
        nodes=np.unique(np.r_[np.linspace(lo,hi,b['nodes']),p.market.initial_logvol_factor])
        anchor=paired_anchor(p,g,b['anchor_train'],b['anchor'],b['batch'])
        curve_a,curve_b=conditional_curve_pair(p,g,nodes,b['inner'],
                                               [77001,88001],anchor.estimate)
        factor_nodes[g]=nodes;curves[g]=(curve_a,curve_b);anchors[g]=anchor
        print(f'[V5.5] reference note at grid {g}: CV={anchor.estimate:.4f} '
              f'+/- {1.96*anchor.std_error:.4f} (naive SE={anchor.naive_std_error:.4f})',flush=True)
    samples=[]
    for rep,sub in enumerate(all_samples):
        data={}
        for g in GRIDS:
            nodes=factor_nodes[g]
            curve_a,curve_b=curves[g]
            avg=(curve_a+curve_b)/2
            yt=[];lt=[]
            for j in range(3):
                y,l=call_payoff(p,*sub[j][g],nodes,avg,dates)
                yt.append(y);lt.append(l)
            cv=two_sample_cv(yt[0],lt[0],yt[1],lt[1],yt[2],lt[2],len(dates))
            # Cross-fitted lower and upper estimators based on two independent
            # conditional valuations. Uses the SAME outer evaluation paths.
            ye,le=yt[2],lt[2]
            ca,_=call_payoff(p,*sub[2][g],nodes,curve_a,dates)
            cb,_=call_payoff(p,*sub[2][g],nodes,curve_b,dates)
            # Direct cross-fitting of the option's conditional value. For each
            # independent A/B curve, exercise on A and evaluate payoff on B.
            eligible=(le>0)
            va=np.interp(sub[2][g][1][eligible],nodes,curve_a)
            vb=np.interp(sub[2][g][1][eligible],nodes,curve_b)
            lower_pay=np.zeros_like(ye)
            disc=p.acro.renewal_units*np.exp(-p.market.risk_free_rate*sub[2][g][0][eligible])
            lower_pay[eligible]=.5*disc*((va>p.acro.strike)*(vb-p.acro.strike)+
                                         (vb>p.acro.strike)*(va-p.acro.strike))
            upper_pay=.5*(ca+cb)
            cv.update(dict(grid=g,rep=rep+1,
                           crossfit_lower=float(lower_pay.mean()),
                           convex_upper=float(upper_pay.mean()),
                           inner_gap=float((upper_pay-lower_pay).mean()),
                           call_probability_eval=float(np.mean(eligible)),
                           max_factor=float(nodes.max()),min_factor=float(nodes.min())))
            samples.append(cv)
            data[g]=cv
            print(f'[V5.5] rep {rep+1}/{b["reps"]}, grid {g}: '
                  f'CV={cv["controlled"]:.4f}±{1.96*cv["cv_se"]:.4f} '
                  f'plain(eval)={cv["naive"]:.4f} '
                  f'inner gap={cv["inner_gap"]:.4f}',flush=True)
        # store coupled fine-coarse difference for both estimators
        for g in (96,192):
            data[g]['paired_cv_difference']=data[g]['controlled']-data[g//2]['controlled']
            data[g]['paired_plain_difference']=data[g]['naive']-data[g//2]['naive']
    from csv import DictWriter
    columns=list(dict.fromkeys(k for row in samples for k in row));
    with open(out/'per_replication.csv','w',newline='',encoding='utf-8') as f:
        w=DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(samples)
    by_grid={}
    for g in GRIDS:
        rr=[s for s in samples if s['grid']==g]
        by_grid[str(g)]={k:report_ci([r[k] for r in rr]) for k in ('controlled','naive','all_paths','crossfit_lower','convex_upper','inner_gap')}
    by_diff={str(g):{
        'controlled':report_ci([s['paired_cv_difference'] for s in samples if s['grid']==g]),
        'naive':report_ci([s['paired_plain_difference'] for s in samples if s['grid']==g])}
        for g in (96,192)}
    # Outer-error estimate conditional on the computed curve. Displayed with
    # warnings: it does not include interpolation/inner MC/grid uncertainty.
    summary=dict(version='5.5',coupon_fixed=COUPON,budgets=b,
                 notes='Only the OUTER variance is targeted; a noisy common renewal curve remains. All 3 grids are Brownian-coupled on outer paths. Quoted SEs exclude common inner and model errors.',
                 prices=by_grid,paired_differences=by_diff,
                 mean_variance_ratio_vs_eval_plain={str(g):float(np.mean([s['variance_ratio'] for s in samples if s['grid']==g])) for g in GRIDS},
                 anchors={str(g):dict(value=anchors[g].estimate, se=anchors[g].std_error,
                                      raw_value=anchors[g].naive, raw_se=anchors[g].naive_std_error,
                                      variance_ratio=anchors[g].variance_ratio) for g in GRIDS},
                 runtime_seconds=perf_counter()-start,
                 tolerance_002_certified=False)
    (out/'audit.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=[
      '# ACRO V5.5 — contrôle de variance externe à coupon contractuel fixé',
      '',f'- Coupon fixé **ex ante à {100*COUPON:.2f} %**, sans cotation au pair entre grilles.',
      f'- Trajectoires externes couplées : {", ".join(map(str,GRIDS))} pas/an.',
      f'- Pilot / training / evaluation : {b["pilot"]} / {b["train"]} / {b["eval"]} trajectoires par réplication ; {b["reps"]} réplications.',
      '- Courbe conditionnelle : deux différences CRN indépendantes et ancrage de grande taille ; **imprécision résiduelle des courbes/ancrages non incluse dans les IC de réplication externe**.',
      '', '| Grille | CV par date de rappel | MC simple (évaluation) | MC simple (toutes trajectoires) |',
      '|---:|---:|---:|---:|']
    for g in GRIDS:
        z=by_grid[str(g)];lines.append(f'| {g} | {z["controlled"]["mean"]:.4f} | {z["naive"]["mean"]:.4f} | {z["all_paths"]["mean"]:.4f} |')
    lines+=['','## Ancrage du Phoenix renouvelé à facteur initial','',
           '| Grille | Valeur de la note (coupon 18 %) | Erreur-type CV de l’ancrage | Erreur-type brute |',
           '|---:|---:|---:|---:|']
    for g in GRIDS:
        a=anchors[g];lines.append(f'| {g} | {a.estimate:.4f} | {a.std_error:.4f} | {a.naive_std_error:.4f} |')
    lines+=['','## Écarts de grille avec couplage Brownien','']
    for g in (96,192):
        z=by_diff[str(g)]['controlled'];lines.append(f'- {g} − {g//2} : **{z["mean"]:+.4f} €**, IC empirique 95 % [{z["ci95"][0]:+.4f} ; {z["ci95"][1]:+.4f}].')
    lines+=['','## Efficacité de la réduction de variance','']
    for g in GRIDS:
        lines.append(f'- {g} pas/an : rapport moyen Var(CV) / Var(MC simple évaluation) = **{summary["mean_variance_ratio_vs_eval_plain"][str(g)]:.3f}**.')
    lines+=['',
      '**Ce rapport de variance concerne les échantillons de validation de même taille.** Le CV requiert un pilote et un entraînement additionnels ; `all_paths` reflète ce coût et constitue une comparaison plus honnête à budget total similaire.',
      '', '## Contrôle de convexité (deux courbes indépendantes)',
      '', 'Les bornes cross-fit (exercice déterminé sur A, règlement évalué sur B) sont valables *en espérance* pour le prix de la valeur conditionnelle **interpolée**, sous indépendance des deux courbes. Elles ne bornent ni la discrétisation du temps, ni l’erreur d’interpolation des vraies valeurs du Phoenix.',
      '', f'- Gap intérieur moyen 192 pas/an : {by_grid["192"]["inner_gap"]["mean"]:.4f} € (hors IC global).',
      '', '## Verdict',
      '', '**La tolérance ±0,02 € n’est pas certifiée.** Ce protocole sépare mieux le bruit des dates de rappel, mais la précision totale nécessite un ancrage indépendant bien plus précis des valorisations conditionnelles et un budget de convergence temporelle accru.',
      '', 'Données individuelles : `per_replication.csv`; statistiques : `audit.json`.']
    (out/'AUDIT_ESTIMATEUR_V5_5_FR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('[V5.5] Output:',out,'elapsed:',round(summary['runtime_seconds'],1),'seconds',flush=True)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--output',default=None)
    args=ap.parse_args()
    run(args.quick,args.output)
