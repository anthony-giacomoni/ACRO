"""ACRO V5.15: comparable-budget investor outcomes under the synthetic Q model.

An *economic research experiment*, not a claim of positive expected investment
return or a calibrated P-measure utility analysis. Initial note, market curve,
ACRO premium and fixed indemnity are quoted on a SEPARATE training sample.
The same independent underlying market scenarios are then used for every
strategy, and all cash flows are carried to the same terminal horizon.
"""
from __future__ import annotations
from .engine import basket_level
from dataclasses import dataclass, replace
import numpy as np
from .config import Project, Phoenix, grid_step


@dataclass
class ContractWealth:
    """Future wealth at horizon H: principal component B and coupons c*A."""
    B: np.ndarray
    A: np.ndarray
    call_step: np.ndarray
    redemption: np.ndarray


def simulate_full_market(p: Project, n: int, horizon: float, seed: int, *,
                         equity_excess_drifts=None, factor_mean_override=None):
    """Vectorized market dynamics exactly matching the Euler scheme in engine.py.

    Includes a full post-autocall continuation of the original equities and
    common market volatility factor for the physical renewal note.
    """
    m=p.market; steps=p.simulation.steps_per_year
    # The baseline is unchanged Q. P-scenarios modify the equity drift only,
    # without changing pricing curves (always calculated under Q).
    excess=(np.zeros(m.n_assets) if equity_excess_drifts is None
            else np.asarray(equity_excess_drifts,dtype=float))
    if excess.shape!=(m.n_assets,) or not np.all(np.isfinite(excess)):
        raise ValueError('Equity excess drifts must be finite, one per underlying')
    factor_target=(m.factor_mean if factor_mean_override is None else float(factor_mean_override))
    if not np.isfinite(factor_target): raise ValueError('Invalid P factor mean')
    kmax=grid_step(horizon,steps)
    L=np.linalg.cholesky(np.asarray(m.correlations,float))
    norm=np.sqrt(np.ones(m.n_assets)@np.asarray(m.correlations,float)@np.ones(m.n_assets))
    dt=1./steps; sr=np.sqrt(dt); r=m.risk_free_rate
    q=np.asarray(m.dividends,float); sig=np.asarray(m.volatilities,float)
    log=np.zeros((kmax+1,n,m.n_assets),float)
    factors=np.zeros((kmax+1,n),float)
    factors[0]=m.initial_logvol_factor if m.dynamics=='stochastic_vol' else 0.
    rng=np.random.default_rng(seed)
    for k in range(1,kmax+1):
        shocks=rng.standard_normal((n,m.n_assets))@L.T
        v=sig[None,:] * (np.exp(factors[k-1])[:,None] if m.dynamics=='stochastic_vol' else 1.)
        log[k]=log[k-1]+(r-q[None,:]+excess[None,:]-.5*v*v)*dt+sr*v*shocks
        if m.dynamics=='stochastic_vol':
            eps=rng.standard_normal(n)
            z=m.equity_factor_correlation*shocks.sum(axis=1)/norm + np.sqrt(1-m.equity_factor_correlation**2)*eps
            factors[k]=factors[k-1]+m.factor_reversion*(factor_target-factors[k-1])*dt + m.factor_volatility*sr*z
    return log,factors


def contract_future_wealth(log: np.ndarray, terms: Phoenix, steps: int,
                           rate: float, start_steps: np.ndarray):
    """Payoffs on the common full-path time grid, with path-specific issue dates.

    start_steps=-1 denotes no note. Coupon A is per ONE unit annual coupon.
    Redemption is undiscounted principal paid at the note's stop date.
    """
    n=log.shape[1]; horizon=(log.shape[0]-1)/steps
    start=np.asarray(start_steps,dtype=int)
    if start.shape!=(n,): raise ValueError('One start step per path required')
    if np.any((start>=0)&(start+grid_step(terms.maturity_years,steps)>=len(log))):
        raise ValueError('Horizon must cover all issued notes until maturity')
    terms.validate(steps)
    coupon_step=steps//terms.observations_per_year
    call_step_set=terms.call_steps(steps)
    live=start>=0
    missed=np.zeros(n,dtype=int)
    running_min=np.ones(n)
    calls=np.full(n,-1,dtype=int)
    redemption=np.zeros(n)
    B=np.zeros(n); A=np.zeros(n)
    idx=np.arange(n)
    for k in range(1,len(log)):
        offset=k-start
        active=live & (offset>0) & (offset<=grid_step(terms.maturity_years,steps))
        if not active.any():continue
        ix=np.flatnonzero(active)
        rel=np.exp(log[k,ix,:]-log[start[ix],ix,:])
        worst=basket_level(np.log(rel),terms.basket_rule)
        if terms.capital_barrier_monitoring=='continuous':
            running_min[ix]=np.minimum(running_min[ix],worst)
        growth=np.exp(rate*(horizon-k/steps))
        coupon=(offset[ix]%coupon_step)==0
        good=coupon & (worst>=terms.coupon_barrier)
        bad=coupon & ~good
        if np.any(good):
            good_idx=ix[good]
            units=(missed[good_idx]+1) if terms.memory else np.ones(len(good_idx))
            A[good_idx]+=terms.nominal*units/terms.observations_per_year*growth
            missed[good_idx]=0
        if terms.memory and np.any(bad):missed[ix[bad]]+=1
        call=np.isin(offset[ix],tuple(call_step_set)) & (worst>=terms.autocall_barrier)
        if np.any(call):
            cix=ix[call]
            B[cix]+=terms.nominal*growth
            redemption[cix]=terms.nominal
            calls[cix]=k
            live[cix]=False
        ending=(offset[ix]==grid_step(terms.maturity_years,steps)) & ~call
        if np.any(ending):
            eix=ix[ending]; wf=worst[ending]
            barrier=wf<terms.capital_barrier if terms.capital_barrier_monitoring=='european' else running_min[eix]<=terms.capital_barrier
            pay=terms.nominal*np.where(barrier,np.minimum(1.,wf),1.)
            B[eix]+=pay*growth
            redemption[eix]=pay
            live[eix]=False
    if live.any():raise AssertionError('Outstanding notes at the common horizon')
    return ContractWealth(B,A,calls,redemption)


def exercise_cost(eligible, B, A, target, strike, units=1.):
    out=np.zeros(len(eligible),float)
    z=np.asarray(eligible,bool)
    out[z]=units*np.maximum(B[z]+target*A[z]-strike,0.)
    return out


def equal_budget_contract_shortfalls(cost, eligible, call_year, premium, fixed_amount, rate):
    """Net shortage to buy the TARGET renewal note at its fair model value.

    Unprotected investor kept the ACRO premium in a money-market account.
    Fixed insurer pays F on first eligible autocall, independently of C.
    ACRO directly provides the right (theoretical shortfall = 0 on exercise).
    """
    if premium<0 or fixed_amount<0:raise ValueError('Negative protection cost')
    c=np.asarray(cost,float); z=np.asarray(eligible,bool)
    dt=np.maximum(np.asarray(call_year,float),0.)
    growth=np.exp(rate*dt)
    return {'acro':np.zeros(len(c)),
            'fixed':np.where(z,np.maximum(c-fixed_amount,0.),0.),
            'reserve':np.where(z,np.maximum(c-premium*growth,0.),0.)}


def fair_fixed_payment(premium, discounted_eligible_probability):
    if premium<0:raise ValueError('Negative option premium')
    if discounted_eligible_probability<=0:
        if premium==0:return 0.
        raise ValueError('No insured eligible events')
    return float(premium/discounted_eligible_probability)


def compare_terminal_wealth(p: Project, log, factors, curve, *, coupon:float,
                            no_call_coupon:float, premium:float, fixed_amount:float,
                            initial_note_price:float):
    """Four strategies, one original basket path and common horizon H.

    - ACRO: at first eligible autocall buy reset R at K=face; pick c0 or c_fair.
    - FIXED: buy fair-market R and invest flat insurer payment F risk-free.
    - RESERVE: buy fair-market R and invest saved original premium risk-free.
    - NO_CALL: buy fairly priced no-call A, then reinvest its maturity principal
      proportionately in a fair-market R for five years. Any unused budget is cash.

    Coupon and redemption cash flows of all notes roll to H at the same risk-free
    rate. All four receive the same initial budget initial_price + premium.
    """
    steps=p.simulation.steps_per_year; H=(len(log)-1)/steps; r=p.market.risk_free_rate
    a=p.initial_phoenix; renew=p.renewal_phoenix; n=log.shape[1]
    if not(np.isclose(renew.nominal,a.nominal) and np.isclose(p.acro.strike,a.nominal)
           and np.isclose(p.acro.renewal_units,1.)):
        raise ValueError('Economic pilot requires one renewal unit and strike equal to both note nominals. The base pricing remains general.')
    if p.acro.target_coupon_rate is not None and not np.isclose(p.acro.target_coupon_rate,coupon):
        raise ValueError('Pilot requires preserving original contractual coupon')
    if not(np.isclose(p.market.initial_logvol_factor,0.) or p.market.dynamics=='gbm'):
        pass
    first=contract_future_wealth(log,a,steps,r,np.zeros(n,dtype=int))
    all_call=first.call_step>=0
    when=np.where(all_call,first.call_step,0)
    eligible=all_call.copy()
    if p.acro.last_eligible_call_year is not None:
        eligible &= when<=grid_step(p.acro.last_eligible_call_year,steps)
    years=when/steps
    call_factor=factors[when,np.arange(n)]
    BR=np.zeros(n);AR=np.ones(n)
    if np.any(all_call):
        b,aa=curve.at(call_factor[all_call]);BR[all_call]=b;AR[all_call]=aa
    fair=np.full(n,np.nan)
    fair[all_call]=(renew.nominal-BR[all_call])/AR[all_call]
    if np.any(fair[all_call]<0):raise ValueError('Negative fair coupon: study requires nonnegative par coupon')
    cost=exercise_cost(eligible,BR,AR,coupon,p.acro.strike)
    # Reissue on ALL first autocalls (even those not eligible for ACRO).
    renewed=contract_future_wealth(log,renew,steps,r,np.where(all_call,when,-1))
    # Principal returned by A upon call is immediately used for R subscription.
    # Contract is restricted to K=nominal; one original note funds one replacement unit.
    remove_principal=np.where(all_call,a.nominal*np.exp(r*(H-years)),0.)
    base=first.B+coupon*first.A-remove_principal
    fair_coupon=np.where(all_call,fair,0.)
    fair_renewed=renewed.B+fair_coupon*renewed.A
    chosen_coupon=np.where(eligible,np.maximum(coupon,fair_coupon),fair_coupon)
    acro_renewed=renewed.B+chosen_coupon*renewed.A
    extra_fixed=np.where(eligible,fixed_amount*np.exp(r*(H-years)),0.)
    budget=initial_note_price+premium
    W_acro=base+acro_renewed
    W_fixed=base+fair_renewed+extra_fixed
    W_reserve=base+fair_renewed+premium*np.exp(r*H)
    # Non-autocall initial Phoenix is independently priced at its own nominal.
    noncall=replace(a,autocall_enabled=False,coupon_rate=no_call_coupon)
    first_nc=contract_future_wealth(log,noncall,steps,r,np.zeros(n,dtype=int))
    # Reinvestment after maturity t=TA, proportionate to actual renewal nominal.
    tmat=grid_step(noncall.maturity_years,steps)
    renew_after=contract_future_wealth(log,renew,steps,r,np.full(n,tmat,dtype=int))
    last_factor=factors[tmat]
    b5,a5=curve.at(last_factor)
    fair5=(renew.nominal-b5)/a5
    if np.any(fair5<0):raise ValueError('Negative no-call reinvestment coupon')
    fractions=first_nc.redemption/renew.nominal
    # The maturity principal is reinvested (not counted twice).
    principal_nc_roll=first_nc.redemption*np.exp(r*(H-noncall.maturity_years))
    W_noncall=(first_nc.B+no_call_coupon*first_nc.A-principal_nc_roll +
              fractions*(renew_after.B+fair5*renew_after.A) + (budget-a.nominal)*np.exp(r*H))
    shortfall=equal_budget_contract_shortfalls(cost,eligible,years,premium,fixed_amount,r)
    return dict(wealth={'acro':W_acro,'fixed':W_fixed,'reserve':W_reserve,'no_call':W_noncall},
                call_year=years,called=all_call,eligible=eligible,
                cost=cost,shortfall=shortfall, fair_coupon=fair,
                exercise=eligible&(cost>1e-10),factor_at_call=call_factor,
                budget=budget,horizon=H,fixed_amount=float(fixed_amount),
                no_call_renewal_fraction=fractions,
                no_call_renewal_time=noncall.maturity_years)


def summarize_economics(records, *, rate):
    """Returns transparent per-strategy Q-distribution diagnostics."""
    wealth=records['wealth']; H=records['horizon']; budget=records['budget']
    discounted_budget=float(budget)
    out=[]
    for name,w in wealth.items():
        dv=np.asarray(w)*np.exp(-rate*H)
        err=dv-discounted_budget
        tail=np.quantile(dv,.05)
        es=float(np.mean(dv[dv<=tail]))
        out.append(dict(strategy=name,n=len(dv),wealth_mean=float(np.mean(w)),
                        discounted_excess_mean=float(err.mean()),
                        discounted_excess_se=float(err.std(ddof=1)/np.sqrt(len(err))),
                        wealth_p05=float(np.quantile(w,.05)),
                        wealth_p50=float(np.quantile(w,.5)),
                        wealth_p95=float(np.quantile(w,.95)),
                        discounted_es05=es,
                        terminal_below_grown_initial_budget_probability=float(np.mean(w<budget*np.exp(rate*H)))))
    return out
