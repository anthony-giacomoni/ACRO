"""State-aware risk of a LIVING Phoenix and its contingent renewal option.

Research model only.  Contractual issue reference levels remain fixed.  A live note
cannot be repriced by bumping `market.initial_spots` in the reset-note engine.

Dates are on the simulation grid; the position is valued immediately AFTER any
coupon/call observation on valuation date.  Prior coupons are excluded from PV.
The state 'alive' and missed-coupon history are supplied, never inferred.
"""
from dataclasses import dataclass, replace
import numpy as np
from .config import Project, grid_step
from .engine import Paths, basket_level
from .pricing import quote_initial, conditional_curve, ConditionalCurve


@dataclass(frozen=True)
class LiveState:
    time_years: float
    spot_ratios: tuple  # current S_i / ORIGINAL FIXED contractual S_i(0)
    logvol_factor: float
    missed_coupons: int = 0

    def validate(self, p: Project):
        s = p.simulation.steps_per_year
        # Reject off-grid dates rather than silently rounding the live state.
        ticks = float(self.time_years) * s
        if not np.isfinite(ticks) or not np.isclose(ticks, round(ticks), rtol=0., atol=1e-10):
            raise ValueError('Live valuation date must align exactly with the simulation grid')
        start = grid_step(self.time_years, s)
        maturity = grid_step(p.initial_phoenix.maturity_years, s)
        if start < 0 or start >= maturity:
            raise ValueError('Live note must be valued strictly before maturity')
        ratios = np.asarray(self.spot_ratios, float)
        if ratios.shape != (p.market.n_assets,) or not np.isfinite(ratios).all() or min(ratios) <= 0:
            raise ValueError('A positive current/reference spot ratio is needed for every asset')
        if not np.isfinite(self.logvol_factor) or self.missed_coupons < 0:
            raise ValueError('Invalid factor or missed coupon count')
        terms = p.initial_phoenix
        if not terms.memory and self.missed_coupons:
            raise ValueError('A note without coupon memory cannot have missed coupons carried forward')
        if terms.capital_barrier_monitoring != 'european':
            raise ValueError('Continuous capital barrier needs full historical running-min state: not supported')
        if start in terms.call_steps(s) and basket_level(np.log(ratios),terms.basket_rule) >= terms.autocall_barrier:
            raise ValueError('Note cannot still be alive after an autocall trigger at valuation date')
        return start


def simulate_live_phoenix(p: Project, state: LiveState, n_paths: int, seed: int) -> Paths:
    """Condition on a genuine *outstanding* note; future cash flows only, PV at t.

    Uses the same discrete risk-neutral dynamics as engine.simulate_phoenix.
    Does NOT reset original issue reference levels. Supports European capital barrier.
    """
    p.validate()
    k0 = state.validate(p)
    mkt = p.market; terms = p.initial_phoenix; steps = p.simulation.steps_per_year
    n = int(n_paths); n_assets = mkt.n_assets
    if n < 2: raise ValueError('Need >=2 paths')
    maturity = grid_step(terms.maturity_years, steps)
    C = np.asarray(mkt.correlations)
    L = np.linalg.cholesky(C)
    bnorm = np.sqrt(np.ones(n_assets) @ C @ np.ones(n_assets))
    rng = np.random.default_rng(seed)
    issue_logratios=np.log(np.asarray(state.spot_ratios, float))
    logs = np.repeat(issue_logratios[None,:],n,axis=0)
    x = np.full(n, state.logvol_factor if mkt.dynamics=='stochastic_vol' else 0.,dtype=float)
    live=np.ones(n,dtype=bool); missed=np.full(n,state.missed_coupons,dtype=int)
    B=np.zeros(n); A=np.zeros(n)
    calls=np.full(n,np.nan); factors=np.full(n,np.nan); wcall=np.full(n,np.nan)
    final=np.full(n,np.nan)
    marts=np.full((n,n_assets),np.nan)
    rate=mkt.risk_free_rate; q=np.asarray(mkt.dividends)
    sig0=np.asarray(mkt.volatilities); rho=mkt.equity_factor_correlation
    dt=1/steps; sqdt=np.sqrt(dt)
    coupon_spacing=steps//terms.observations_per_year
    call_steps=terms.call_steps(steps)
    for k in range(k0+1,maturity+1):
        if not live.any(): break
        idx=np.flatnonzero(live)
        # Draw for all n to pair bumped states with the same innovation streams.
        dw=(rng.standard_normal((n,n_assets)) @ L.T)[idx]
        sig=sig0[None,:] if mkt.dynamics=='gbm' else sig0[None,:]*np.exp(x[idx,None])
        logs[idx]+=(rate-q[None,:]-.5*sig*sig)*dt+sig*sqdt*dw
        w=basket_level(logs[idx],terms.basket_rule)
        if mkt.dynamics=='stochastic_vol':
            independent=rng.standard_normal(n)[idx]
            z=rho*dw.sum(axis=1)/bnorm + np.sqrt(1-rho*rho)*independent
            x[idx]+=mkt.factor_reversion*(mkt.factor_mean-x[idx])*dt+mkt.factor_volatility*sqdt*z
        discount=np.exp(-rate*(k-k0)*dt)
        if k%coupon_spacing==0:
            good=w >= terms.coupon_barrier
            paying=idx[good]
            if terms.memory:
                units=missed[paying]+1
                missed[paying]=0
                missed[idx[~good]]+=1
            else: units=np.ones(len(paying))
            A[paying]+=terms.nominal*units/terms.observations_per_year*discount
        if k in call_steps:
            trigger=w >= terms.autocall_barrier
            triggered=idx[trigger]
            B[triggered]+=terms.nominal*discount
            calls[triggered]=k*dt
            factors[triggered]=x[triggered]
            wcall[triggered]=w[trigger]
            marts[triggered]=np.exp(logs[triggered]-issue_logratios[None,:]-(rate-q[None,:])*((k-k0)*dt))
            live[triggered]=False
        if k==maturity:
            remaining=live[idx]; r_idx=idx[remaining]; w_final=w[remaining]
            B[r_idx]+=terms.nominal*np.where(w_final<terms.capital_barrier,np.minimum(1.,w_final),1.)*discount
            final[r_idx]=w_final
            marts[r_idx]=np.exp(logs[r_idx]-issue_logratios[None,:]-(rate-q[None,:])*((k-k0)*dt))
            live[r_idx]=False
    if not np.isfinite(marts).all(): raise AssertionError('Unsettled scenario')
    return Paths(B,A,calls,factors,final,wcall,marts)


@dataclass
class LiveValuation:
    note_price: float
    option_price: float
    total_price: float
    note_outer_se: float
    option_outer_se: float
    eligible_call_probability: float
    discounted_option_payoffs: np.ndarray
    discounted_note_payoffs: np.ndarray


def _renewal_curve(p:Project,state:LiveState,eligible_factors,seed_offset=0,*,
                   pricing_method='legacy',ref_paths=512,scrambles=3):
    """Reprice a NEW, reset renewal note (not the outstanding Phoenix).

    gbm_cv independently prices the renewal note's GBM principal and annuity,
    then estimates SV-minus-GBM components using *paired* Brownian innovations.
    The reference and conditional bank are independent. A single curve is shared
    by all spot bumps, because reset-note value depends on the volatility factor,
    not the old issue's contractual spot references.
    """
    adjusted=replace(p,market=replace(p.market,initial_logvol_factor=state.logvol_factor),
                     initial_coupon_policy='fixed')
    factors=np.asarray(eligible_factors,float)
    if pricing_method not in ('legacy','gbm_cv'):
        raise ValueError('pricing_method must be legacy or gbm_cv')
    if pricing_method=='legacy' or p.market.dynamics=='gbm':
        q=quote_initial(adjusted)
        return conditional_curve(adjusted,factors,q,seed_offset)
    from .controlled_pricing import _reference, _make_bank
    from .gbm_control import aggregated_bank
    from .engine import simulate_phoenix
    terms=adjusted.renewal_phoenix
    sim=adjusted.simulation
    steps=sim.steps_per_year
    if terms.capital_barrier_monitoring!='european':
        raise ValueError('GBM control currently supports a European renewal barrier')
    target=(adjusted.initial_phoenix.coupon_rate if adjusted.acro.target_coupon_rate is None
            else adjusted.acro.target_coupon_rate)
    # Exact same renewal terms for every node. No initial Phoenix quote is reused.
    ref=_reference(adjusted,terms,target,scrambles,int(ref_paths),
                   seed=2941007+int(seed_offset))
    bank=_make_bank(adjusted,terms,sim.conditional_paths,steps,
                    seed=2943007+int(seed_offset))
    gbm=simulate_phoenix(replace(adjusted.market,dynamics='gbm'),terms,
                         sim.conditional_paths,terms.observations_per_year,0,
                         innovations=aggregated_bank(bank,steps,terms.observations_per_year))
    mid=state.logvol_factor
    low=min(float(factors.min())-.08,mid-.1) if factors.size else mid-.5
    high=max(float(factors.max())+.08,mid+.1) if factors.size else mid+.5
    # Preserve the exact valuation-date factor as a grid node.
    nodes=np.unique(np.r_[np.linspace(low,high,sim.curve_nodes),mid])
    principal=[]; annuity=[]
    for x in nodes:
        sv=simulate_phoenix(adjusted.market,terms,sim.conditional_paths,steps,0,
                            initial_factor=float(x),innovations=bank)
        principal.append(ref.B+float(np.mean(sv.principal_pv-gbm.principal_pv)))
        annuity.append(ref.A+float(np.mean(sv.coupon_annuity_pv-gbm.coupon_annuity_pv)))
    return ConditionalCurve(nodes,np.asarray(principal),np.asarray(annuity),
                            sim.conditional_paths)


def value_live(p:Project, state:LiveState, *, n_paths:int|None=None,
               seed:int|None=None, curve:ConditionalCurve|None=None,
               pricing_method='legacy',ref_paths=512,scrambles=3,curve_seed_offset=0):
    """First eligible future autocall triggers ACRO. No historical coupons included.

    Curve may be supplied to pair *stock bumps*. Its stochastic error is COMMON
    and NOT counted in returned Monte Carlo SE. Not a full risk confidence interval.
    """
    p.validate();state.validate(p)
    n=n_paths or p.simulation.outer_paths
    paths=simulate_live_phoenix(p,state,n,seed if seed is not None else p.simulation.seed+8201)
    coupon=p.initial_phoenix.coupon_rate  # CONTRACT FIXED, never re-solved when bumping
    note=paths.pv(coupon)
    calls=paths.first_call_year
    elig=np.isfinite(calls)
    if p.acro.last_eligible_call_year is not None:
        elig &= calls<=p.acro.last_eligible_call_year + 1e-10
    option=np.zeros(n)
    if elig.any():
        if curve is None:
            curve=_renewal_curve(p,state,paths.factor_at_call[elig],curve_seed_offset,
                                 pricing_method=pricing_method,ref_paths=ref_paths,
                                 scrambles=scrambles)
        B,A=curve.at(paths.factor_at_call[elig])
        target=coupon if p.acro.target_coupon_rate is None else p.acro.target_coupon_rate
        exercise=np.maximum(B+target*A-p.acro.strike,0)*p.acro.renewal_units
        option[elig]=np.exp(-p.market.risk_free_rate*(calls[elig]-state.time_years))*exercise
    return LiveValuation(float(note.mean()),float(option.mean()),float((note+option).mean()),
                         float(note.std(ddof=1)/np.sqrt(n)),float(option.std(ddof=1)/np.sqrt(n)),
                         float(elig.mean()),option,note)


def stock_delta_live(p:Project,state:LiveState,*,relative_bump=.01,n_paths=1500,
                     pricing_method='legacy',ref_paths=512,scrambles=3,seed=None,
                     curve_seed_offset=0,curve:ConditionalCurve|None=None):
    """Paired finite-difference deltas of *outstanding* note and ACRO.

    /share delta uses ORIGINAL issue references; current S = issue S0 * ratio.
    Uses one common renewal value curve across bumps, and therefore omits
    shared inner-curve uncertainty from the paired SEs.
    """
    if not 0 < relative_bump < .1: raise ValueError('Invalid relative bump')
    state.validate(p)
    states=[state]
    for i in range(p.market.n_assets):
        for sign in (-1,1):
            ratios=list(state.spot_ratios);ratios[i]*=1+sign*relative_bump
            states.append(replace(state,spot_ratios=tuple(ratios)))
    # Factor coverage from every scenario in order to build just one shared curve.
    seed=p.simulation.seed+8201 if seed is None else int(seed)
    if curve is None:
        factors=[]
        for st in states:
            paths=simulate_live_phoenix(p,st,n_paths,seed)
            elig=np.isfinite(paths.first_call_year)
            if p.acro.last_eligible_call_year is not None:
                elig &= paths.first_call_year<=p.acro.last_eligible_call_year+1e-10
            factors.extend(paths.factor_at_call[elig].tolist())
        curve=(_renewal_curve(p,state,np.asarray(factors),curve_seed_offset,
                              pricing_method=pricing_method,ref_paths=ref_paths,
                              scrambles=scrambles) if factors else None)
    vals=[value_live(p,st,n_paths=n_paths,seed=seed,curve=curve,
                     pricing_method=pricing_method) for st in states]
    rows=[]
    for i in range(p.market.n_assets):
        down,up=vals[1+2*i],vals[2+2*i]
        spot=float(p.market.initial_spots[i]*state.spot_ratios[i])
        denom=2*relative_bump*spot
        # Paired pathwise differences; does not include common curve error.
        diff_o=(up.discounted_option_payoffs-down.discounted_option_payoffs)/denom
        diff_n=(up.discounted_note_payoffs-down.discounted_note_payoffs)/denom
        delta=float((diff_o+diff_n).mean())
        shift=relative_bump*spot
        # Long portfolio hedge: short delta shares, self-financed at t.
        # These are instantaneous single-name shocks, NOT multi-day hedging backtests.
        unhedged_up=up.total_price-vals[0].total_price
        unhedged_down=down.total_price-vals[0].total_price
        rows.append(dict(asset=i+1,current_spot=spot,delta_acro=float(diff_o.mean()),
                         delta_note=float(diff_n.mean()),delta_total=delta,
                         delta_total_paired_outer_se=float((diff_o+diff_n).std(ddof=1)/np.sqrt(n_paths)),
                         shares_to_short_for_long_portfolio=-delta,
                         hedge_notional_for_long_portfolio=-delta*spot,
                         stress_bump_fraction=relative_bump,
                         bump_up_unhedged=unhedged_up,
                         bump_up_after_delta_hedge=unhedged_up-delta*shift,
                         bump_down_after_delta_hedge=unhedged_down+delta*shift,
                         # Deprecated aliases retained for existing external analyses.
                         one_pct_up_unhedged=unhedged_up,
                         one_pct_up_after_delta_hedge=unhedged_up-delta*shift,
                         one_pct_down_after_delta_hedge=unhedged_down+delta*shift,
                         finite_difference_gamma=(up.total_price+down.total_price-2*vals[0].total_price)/(shift**2)))
    return vals[0],rows


def independent_hedge_stress(p: Project, state: LiveState, hedge_rows, *,
                            bumps=(.005,.01,.02), n_paths=1600, seed=3351031,
                            pricing_method='gbm_cv',ref_paths=512,scrambles=3):
    """Out-of-sample *instantaneous stress*, not dynamic trading P&L.

    The hedge ratios are fitted externally. A new independent renewal curve and
    independent outer Brownian paths are used to evaluate V(S+h)-V(S)-delta*h.
    Same evaluation seed within the baseline and its shocks reduces MC noise.
    """
    state.validate(p)
    if len(hedge_rows) != p.market.n_assets: raise ValueError('One delta per asset')
    if not bumps or any(h<=0 or h>=.10 for h in bumps):
        raise ValueError('Stress bumps must lie between 0 and 10%')
    cases=[state]; tags=[(0,0.,0)]
    for i in range(p.market.n_assets):
        for h in bumps:
            for sign in (-1,1):
                r=list(state.spot_ratios);r[i]*=(1+sign*h)
                cases.append(replace(state,spot_ratios=tuple(r)))
                tags.append((i,sign*h,1))
    factors=[]
    for case in cases:
        sim=simulate_live_phoenix(p,case,n_paths,seed)
        elig=np.isfinite(sim.first_call_year)
        if p.acro.last_eligible_call_year is not None:
            elig &= sim.first_call_year<=p.acro.last_eligible_call_year+1e-10
        factors.extend(sim.factor_at_call[elig].tolist())
    curve=_renewal_curve(p,state,np.asarray(factors),seed_offset=811,
                         pricing_method=pricing_method,ref_paths=ref_paths,
                         scrambles=scrambles) if factors else None
    estimates=[value_live(p,case,n_paths=n_paths,seed=seed,curve=curve) for case in cases]
    base=estimates[0]
    result=[]
    for (i,h,_),v in zip(tags[1:],estimates[1:]):
        spot=float(p.market.initial_spots[i]*state.spot_ratios[i])
        predicted=float(hedge_rows[i]['delta_total'])*spot*h
        unhedged=v.total_price-base.total_price
        # Pathwise paired SE of evaluation *residual* (hedge is fixed).
        delta_payoffs=(v.discounted_note_payoffs+v.discounted_option_payoffs
                       -base.discounted_note_payoffs-base.discounted_option_payoffs)
        se=float(delta_payoffs.std(ddof=1)/np.sqrt(n_paths))
        result.append(dict(asset=i+1,shock_percent=100*h,delta_trained=hedge_rows[i]['delta_total'],
                           unhedged_pnl=unhedged,hedged_pnl=unhedged-predicted,
                           outer_se_only=se,
                           evaluation_seed=seed,training_seed_different=True))
    return result
