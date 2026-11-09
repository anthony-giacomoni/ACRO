"""Path-coupled numerical diagnostics for ACRO V5.

Run: python audit_coupled.py --quick   (or without --quick for larger budgets)

Controls fixed Brownian innovations across time discretizations. Separately
probes inner-MC variability and held-out factor interpolation with a fixed
initial quote and fixed outer paths. It does NOT remove any numerical bias.
"""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import argparse, json, csv, time
import numpy as np
from acro import load_project, simulate_phoenix

ROOT = Path(__file__).resolve().parents[1]


def innovations_for_grid(max_steps, n_paths, n_assets, seed):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal((max_steps, n_paths, n_assets)),
            rng.standard_normal((max_steps, n_paths)))


def aggregate_innovations(bank, fine_per_year, steps_per_year, tenor):
    eq, fac = bank
    assert fine_per_year % steps_per_year == 0
    k = fine_per_year // steps_per_year
    coarse_steps = int(round(tenor*steps_per_year))
    if eq.shape[0] != coarse_steps*k:
        raise ValueError('Bank tenor does not match terms')
    a = eq.reshape(coarse_steps,k,*eq.shape[1:]).sum(axis=1) / np.sqrt(k)
    b = fac.reshape(coarse_steps,k,fac.shape[1]).sum(axis=1) / np.sqrt(k)
    return a,b


def quote_from_bank(p, bank, steps, fine_steps):
    inn=aggregate_innovations(bank,fine_steps,steps,p.initial_phoenix.maturity_years)
    path=simulate_phoenix(p.market,p.initial_phoenix,inn[0].shape[1],steps,0,innovations=inn)
    B=float(path.principal_pv.mean()); A=float(path.coupon_annuity_pv.mean())
    return B,A,(p.initial_phoenix.nominal-B)/A


def outer_from_bank(p, bank, steps, fine_steps):
    inn=aggregate_innovations(bank,fine_steps,steps,p.initial_phoenix.maturity_years)
    path=simulate_phoenix(p.market,p.initial_phoenix,inn[0].shape[1],steps,0,innovations=inn)
    eligible=np.isfinite(path.first_call_year)
    if p.acro.last_eligible_call_year is not None:
        eligible &= path.first_call_year <= p.acro.last_eligible_call_year + 1e-12
    return path,eligible


def renewal_differences(p, bank, steps, fine_steps, nodes):
    """Common-inner-draw differences relative to initial log-vol state.

    This avoids using the quote's random draws as the anchor and keeps the
    inner noise isolated, while preserving the exact t=0 par identity.
    """
    inn=aggregate_innovations(bank,fine_steps,steps,p.renewal_phoenix.maturity_years)
    n=inn[0].shape[1]; x0=p.market.initial_logvol_factor
    baseline=simulate_phoenix(p.market,p.renewal_phoenix,n,steps,0,
                              initial_factor=x0,innovations=inn)
    db=[];da=[]
    for x in nodes:
        if abs(x-x0)<1e-14:
            db.append(0.);da.append(0.)
        else:
            v=simulate_phoenix(p.market,p.renewal_phoenix,n,steps,0,
                               initial_factor=float(x),innovations=inn)
            db.append(float(np.mean(v.principal_pv-baseline.principal_pv)))
            da.append(float(np.mean(v.coupon_annuity_pv-baseline.coupon_annuity_pv)))
    return np.asarray(db),np.asarray(da)


def price_from_components(p, q, outer, eligible, nodes, db, da):
    B,A,c=q
    target = c if p.acro.target_coupon_rate is None else p.acro.target_coupon_rate
    x = outer.factor_at_call[eligible]
    val = B + np.interp(x,nodes,db) + target*(A+np.interp(x,nodes,da))
    z = np.zeros(len(outer.first_call_year), dtype=float)
    z[eligible] = np.exp(-p.market.risk_free_rate*outer.first_call_year[eligible])*p.acro.renewal_units*np.maximum(val-p.acro.strike,0.)
    return float(z.mean())


def csv_write(path,rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run(config, output, quick):
    p=load_project(config)
    if p.market.dynamics!='stochastic_vol':
        raise ValueError('This audit addresses the reference stochastic-vol configuration')
    if p.initial_phoenix != p.renewal_phoenix or p.acro.target_coupon_rate is not None:
        raise ValueError('This audit currently requires identical Phoenix terms and target coupon = initial coupon')
    if p.initial_phoenix.maturity_years != int(p.initial_phoenix.maturity_years):
        raise ValueError('Use integer-year Phoenix maturities for coupled grid experiment')
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    t0=time.monotonic(); fine=48
    grids=[12,24,48]
    if quick:
        n_q,n_outer,n_inner,nodes_n,reps=1800,1500,1500,9,2
    else:
        n_q,n_outer,n_inner,nodes_n,reps=4500,3500,4500,13,3
    tenor=p.initial_phoenix.maturity_years
    length=int(round(tenor*fine)); na=p.market.n_assets
    rows=[]; all_states=[]
    # Paired path banks per repeat; reuse exact fine-grid Gaussian drivers.
    for rep in range(reps):
        bank_q=innovations_for_grid(length,n_q,na,1001+rep*10000)
        bank_o=innovations_for_grid(length,n_outer,na,2001+rep*10000)
        bank_i=innovations_for_grid(length,n_inner,na,3001+rep*10000)
        states={}; calls={}; quotes={}
        for step in grids:
            quotes[step]=quote_from_bank(p,bank_q,step,fine)
            calls[step]=outer_from_bank(p,bank_o,step,fine)
            x=calls[step][0].factor_at_call[calls[step][1]]
            all_states.extend(x.tolist())
            states[step]=x
        valid=[x for x in states.values() if len(x)]
        if not valid: raise RuntimeError('No eligible autocall scenarios in this paired experiment')
        min_x=min(float(x.min()) for x in valid)-.08
        max_x=max(float(x.max()) for x in valid)+.08
        nodes=np.unique(np.r_[np.linspace(min_x,max_x,nodes_n),p.market.initial_logvol_factor])
        for step in grids:
            db,da=renewal_differences(p,bank_i,step,fine,nodes)
            pr=price_from_components(p,quotes[step],*calls[step],nodes,db,da)
            rows.append(dict(repetition=rep+1,steps_per_year=step,price=pr,
                             initial_coupon=quotes[step][2],
                             autocall_probability=float(calls[step][1].mean()),
                             paths_quote=n_q,paths_outer=n_outer,paths_inner=n_inner,nodes=len(nodes)))
        print('Paired repetition',rep+1,'/',reps,'complete',flush=True)
    csv_write(out/'paired_time_grids.csv',rows)
    deltas=[]
    for rep in range(reps):
        d={r['steps_per_year']:r['price'] for r in rows if r['repetition']==rep+1}
        deltas.append(dict(repetition=rep+1,difference_24_minus_12=d[24]-d[12],
                           difference_48_minus_24=d[48]-d[24],
                           difference_48_minus_12=d[48]-d[12]))
    csv_write(out/'paired_differences.csv',deltas)
    # Isolate the conditional value approximation error: all quote and outer
    # paths are fixed; only the inner MC innovation bank varies.
    step=24
    q_bank=innovations_for_grid(int(tenor*step),n_q,na,91001)
    o_bank=innovations_for_grid(int(tenor*step),n_outer,na,92001)
    q=quote_from_bank(p,q_bank,step,step)
    o,eligible=outer_from_bank(p,o_bank,step,step)
    xs=o.factor_at_call[eligible]
    if not len(xs): raise RuntimeError('No autocall in controlled inner experiment')
    # Freeze all candidate outer scenarios before choosing common grid support;
    # avoid silently extrapolating prices when varying outer samples.
    outer_candidates=[(o,eligible)]
    for j in range(1,4):
        candidate_bank=innovations_for_grid(int(tenor*step),n_outer,na,92001+j*12101)
        outer_candidates.append(outer_from_bank(p,candidate_bank,step,step))
    candidate_factors=[v.factor_at_call[mask] for v,mask in outer_candidates if mask.any()]
    xs_all=np.concatenate(candidate_factors)
    nodes=np.unique(np.r_[np.linspace(float(xs_all.min())-.10,float(xs_all.max())+.10,nodes_n),p.market.initial_logvol_factor])
    inner_rows=[]; heldout=[]
    fixed_db=fixed_da=None
    # Pointwise direct conditional checks using identical inner innovations:
    # interpolation residual compares the same MC sample at interior states.
    check_states=np.quantile(xs,[.1,.3,.5,.7,.9])
    for rep in range(3 if quick else 4):
        inn=innovations_for_grid(int(tenor*step),n_inner,na,99001+rep*10000)
        db,da=renewal_differences(p,inn,step,step,nodes)
        if rep==0: fixed_db,fixed_da=db,da
        price=price_from_components(p,q,o,eligible,nodes,db,da)
        inner_rows.append(dict(repetition=rep+1,price=price,
                               same_quote=True,same_outer=True,
                               inner_paths=n_inner,nodes=len(nodes)))
        # Evaluate 5 off-grid states directly with SAME simulation bank:
        # no independent-MC noise is introduced into the comparison.
        exact_db,exact_da=renewal_differences(p,inn,step,step,check_states)
        target=q[2]; B,A,_=q
        for x,dbx,dax in zip(check_states,exact_db,exact_da):
            approximate=B+np.interp(x,nodes,db)+target*(A+np.interp(x,nodes,da))
            direct=B+dbx+target*(A+dax)
            heldout.append(dict(repetition=rep+1,logvol_factor=float(x),
                                curve_value=float(approximate),heldout_value=float(direct),
                                interpolation_difference=float(approximate-direct)))
    csv_write(out/'fixed_outer_inner_mc.csv',inner_rows)
    csv_write(out/'heldout_interpolation.csv',heldout)
    # One-factor-at-a-time empirical variability, not confidence bounds.
    outer_rows=[]
    for j,(candidate,mask) in enumerate(outer_candidates):
        value=price_from_components(p,q,candidate,mask,nodes,fixed_db,fixed_da)
        outer_rows.append(dict(repetition=j+1,price=value,what_varied='outer_paths',
                               eligible_count=int(mask.sum())))
    quote_rows=[]
    for j in range(4):
        if j==0: candidate_quote=q
        else:
            qj=innovations_for_grid(int(tenor*step),n_q,na,91001+j*12101)
            candidate_quote=quote_from_bank(p,qj,step,step)
        value=price_from_components(p,candidate_quote,o,eligible,nodes,fixed_db,fixed_da)
        quote_rows.append(dict(repetition=j+1,price=value,what_varied='initial_quote',
                               annual_coupon=candidate_quote[2]))
    csv_write(out/'fixed_quote_outer_variation.csv',outer_rows)
    csv_write(out/'fixed_outer_quote_variation.csv',quote_rows)
    values=np.asarray([r['price'] for r in inner_rows]); err=np.asarray([r['interpolation_difference'] for r in heldout])
    def summarise(seq):
        a=np.asarray(seq,dtype=float)
        return dict(n=len(a),mean=float(a.mean()),sample_sd=float(a.std(ddof=1)) if len(a)>1 else None,
                    min=float(a.min()),max=float(a.max()))
    report={
        'scope':'Numerical diagnosis, synthetic Q model; not market calibration or total bias bound',
        'reference_step_24_fixed_outer_inner_variability':summarise(values),
        'fixed_quote_and_inner_outer_variability':summarise([v['price'] for v in outer_rows]),
        'fixed_outer_and_inner_quote_variability':summarise([v['price'] for v in quote_rows]),
        'heldout_interpolation_abs_diff_in_renewal_note_eur':summarise(abs(err)),
        'paired_step_differences':{
            k:summarise([v[k] for v in deltas]) for k in ('difference_24_minus_12','difference_48_minus_24','difference_48_minus_12')},
        'runtime_s':time.monotonic()-t0,
        'caveats':[
            'Paired time-grid differences include Monte Carlo residuals from discontinuous autocall events and finite inner-node simulation; Brownian coupling reduces sampling variance but does not remove it.',
            'For different time steps the simulated option value uses a recalculated par coupon and matching market paths. This measures TOTAL model discretization sensitivity, not a pure Euler weak-error coefficient.',
            'One-factor-at-a-time variation compares finite batches and the empirical standard deviations are not confidence bounds for true price.',
            'Inner-MC spread is measured with a frozen quote and outer paths; it is not a confidence interval for the true option price.',
            'Held-out interpolation residual uses same inner trajectories and measures numerical curve interpolation only for sampled states at a finite inner MC budget.',
            'Q pricing not P investor return. No market calibration, hedging, transaction cost or credit risk.']}
    (out/'audit_coupled.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    lines=['# ACRO V5.1 — Audit numérique couplé',
           '', '**Statut : diagnostic exploratoire, pas validation du prix ni calibration aux marchés.**',
           '', '## 1. Grilles temporelles couplées',
           '', 'Les trois discrétisations 12/24/48 utilisent **les mêmes incréments browniens agrégés**. Cela évite de comparer des simulations totalement indépendantes.',
           '', '| Répétition | Prix 12 | Prix 24 | Prix 48 | Écart 48−12 |',
           '|---:|---:|---:|---:|---:|']
    for rep in range(reps):
        d={r['steps_per_year']:r['price'] for r in rows if r['repetition']==rep+1}
        lines.append(f'| {rep+1} | {d[12]:.4f} | {d[24]:.4f} | {d[48]:.4f} | {d[48]-d[12]:+.4f} |')
    lines+=['', 'Ces différences ne prouvent pas la convergence : les événements d’autocall rendent la fonction de payoff discontinue et le bruit résiduel demeure.',
            '', '## 2. Bruit MC interne à quote et scénarios externes fixes',
            '', f'- {len(values)} courbes internes indépendantes : moyenne {np.mean(values):.4f} €, écart-type empirique {np.std(values,ddof=1):.4f} €, plage [{np.min(values):.4f}, {np.max(values):.4f}] € par 100.',
            '- Il s’agit d’une mesure de variabilité des valeurs conditionnelles, **pas d’une borne du biais de Jensen**.',
            '', '### Décomposition empirique des sources (autres sources gelées)',
            f"- Quote initiale seule : écart-type des prix sur {len(quote_rows)} réplications = {np.std([r['price'] for r in quote_rows],ddof=1):.4f} €.",
            f"- Scénarios externes seuls : écart-type des prix sur {len(outer_rows)} réplications = {np.std([r['price'] for r in outer_rows],ddof=1):.4f} €.",
            '- Ces écarts-types sont indicatifs et ne s’additionnent pas simplement pour produire une erreur totale.',
            '', '## 3. Interpolation hors nœuds, mêmes innovations internes',
            '', f'- Écart absolu moyen sur {len(err)} tests : {np.mean(abs(err)):.4f} € de valeur de la note renouvelée ; maximum {np.max(abs(err)):.4f} €.',
            '- Cette comparaison tient fixes les chocs des simulations internes et distingue mieux interpolation et bruit de repricing indépendant.',
            '', '## 4. Décision',
            '', '**Ne pas figer une prime ACRO, un intervalle de confiance global ni une réduction du coupon unique.** Vérifier d’abord la stabilité numérique avec des budgets significativement supérieurs et une grille assez fine, puis seulement calibrer le modèle de marché.',
            '', 'Fichiers détaillés : `paired_time_grids.csv`, `paired_differences.csv`, `fixed_outer_inner_mc.csv`, `fixed_quote_outer_variation.csv`, `fixed_outer_quote_variation.csv`, `heldout_interpolation.csv`, `audit_coupled.json`.']
    (out/'AUDIT_COUPLED_FR.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Finished in',round(time.monotonic()-t0,1),'s',flush=True)
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--config',default=str(ROOT/'config.json'))
    ap.add_argument('--output',default=str(ROOT/'results_audit_coupled'))
    a=ap.parse_args()
    run(a.config,a.output,a.quick)
