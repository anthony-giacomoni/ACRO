"""Exploratory risk-neutral multi-date delta-hedging experiment for a LIVE ACRO package.

A single realized scenario is generated under the project's synthetic Q dynamics;
conditional valuations and bump deltas use separate independent random streams.
Trading is self-financing: stock purchases/sales, cash financing, deterministic
dividends, coupons, principal, renewal cash equivalent and transaction fees.

Not a historical backtest, professional desk hedge, or evidence of P-measure utility.
European capital protection only; no borrowing spread, issuer credit or bid/ask.
"""
from __future__ import annotations
from .engine import basket_level
from dataclasses import dataclass, replace
from typing import Sequence
import numpy as np
from .config import Project, grid_step
from .live_risk import LiveState, stock_delta_live, value_live
from .pricing import ConditionalCurve


@dataclass(frozen=True)
class MarketMark:
    tick: int
    time: float
    ratios: tuple
    factor: float
    missed: int
    coupon: float
    principal: float
    acro: float
    alive: bool
    autocall: bool

    def state(self):
        return LiveState(self.time, self.ratios, self.factor, self.missed)


def generate_scenario(p: Project, initial: LiveState, *, end_time: float,
                      seed: int, renewal_curve: ConditionalCurve) -> list[MarketMark]:
    """One path of the current outstanding note, conditioned on supplied alive state.

    Cash paid on coupon/call observations is recorded *after* that observation.
    ACRO is cash-settled at fair model value solely for the research P&L.
    """
    k0=initial.validate(p)
    steps=p.simulation.steps_per_year
    kend=grid_step(end_time,steps)
    if kend<=k0 or kend>grid_step(p.initial_phoenix.maturity_years,steps):
        raise ValueError('end_time must be after live_time and no later than maturity')
    terms=p.initial_phoenix; m=p.market
    if terms.capital_barrier_monitoring!='european':
        raise ValueError('Historical running-min is needed for continuously monitored capital protection')
    ratios=np.asarray(initial.spot_ratios,float).copy()
    logsp=np.log(ratios)
    x=float(initial.logvol_factor if m.dynamics=='stochastic_vol' else 0.)
    missed=int(initial.missed_coupons)
    out=[MarketMark(k0,k0/steps,tuple(ratios),x,missed,0.,0.,0.,True,False)]
    rng=np.random.default_rng(seed)
    L=np.linalg.cholesky(np.asarray(m.correlations,float))
    n=m.n_assets
    bnorm=np.sqrt(np.ones(n)@np.asarray(m.correlations)@np.ones(n))
    r=float(m.risk_free_rate)
    dividends=np.asarray(m.dividends,float)
    vol0=np.asarray(m.volatilities,float)
    dt=1./steps
    cstep=steps//terms.observations_per_year
    calls=terms.call_steps(steps)
    for k in range(k0+1,kend+1):
        dw=rng.standard_normal(n)@L.T
        vol=vol0 if m.dynamics=='gbm' else vol0*np.exp(x)
        logsp+=(r-dividends-.5*vol**2)*dt+vol*np.sqrt(dt)*dw
        ratios=np.exp(logsp)
        if m.dynamics=='stochastic_vol':
            z=m.equity_factor_correlation*dw.sum()/bnorm + np.sqrt(1-m.equity_factor_correlation**2)*rng.standard_normal()
            x+=m.factor_reversion*(m.factor_mean-x)*dt+m.factor_volatility*np.sqrt(dt)*z
        w=float(basket_level(logsp,terms.basket_rule))
        coupon=0.
        if k%cstep==0:
            if w>=terms.coupon_barrier:
                units=missed+1 if terms.memory else 1
                coupon=float(terms.nominal*terms.coupon_rate*units/terms.observations_per_year)
                missed=0
            elif terms.memory:
                missed+=1
        call=(k in calls and w>=terms.autocall_barrier)
        principal=0.; acro=0.
        if call:
            principal=float(terms.nominal)
            if p.acro.last_eligible_call_year is None or k/steps<=p.acro.last_eligible_call_year+1e-10:
                b,a=renewal_curve.at(np.asarray([x]))
                target=(terms.coupon_rate if p.acro.target_coupon_rate is None else p.acro.target_coupon_rate)
                acro=float(p.acro.renewal_units*max(float(b[0]+target*a[0]-p.acro.strike),0.))
        elif k==grid_step(terms.maturity_years,steps):
            principal=float(terms.nominal*(min(1.,w) if w<terms.capital_barrier else 1.))
        alive=(not call and k!=grid_step(terms.maturity_years,steps))
        out.append(MarketMark(k,k/steps,tuple(ratios),float(x),missed,coupon,principal,acro,alive,call))
        if not alive:
            break
    return out


@dataclass
class CashBook:
    """Hedged LONG asset. Stock holdings are *negative* of estimated delta."""
    component: str
    method: str
    shares: np.ndarray
    cash: float
    initial_price: float
    fees: float=0.
    transactions: int=0

    @classmethod
    def open(cls, component: str, method: str, initial_price: float,
             spot: np.ndarray, delta: np.ndarray, fee_bps: float):
        shares=(-delta.copy() if method!='none' else np.zeros_like(spot))
        # Sale proceeds are placed into cash; initial long mark remains V0.
        c=-float(np.dot(shares,spot))
        obj=cls(component,method,shares,c,float(initial_price))
        cost=obj._fee(shares,spot,fee_bps)
        obj.cash-=cost
        obj.fees+=cost
        obj.transactions=int(bool(np.any(shares)))
        return obj

    @staticmethod
    def _fee(changed: np.ndarray, spot: np.ndarray, fee_bps: float):
        return float(np.sum(np.abs(changed)*spot)*fee_bps/10000.)

    def accrue(self, old_spot, new_spot, dt, rate, dividends, flow):
        # Flat continuously compounded cash account; deterministic continuous
        # dividend yields approximated by the start-of-step spot over dt.
        div=float(np.dot(self.shares,old_spot*np.expm1(dividends*dt)))
        self.cash=self.cash*np.exp(rate*dt)+div+float(flow)

    def trade(self, target, spot, fee_bps):
        change=np.asarray(target,float)-self.shares
        cost=self._fee(change,spot,fee_bps)
        self.cash-=float(np.dot(change,spot))+cost
        self.shares=np.asarray(target,float).copy()
        self.fees+=cost
        self.transactions+=int(bool(np.any(change)))

    def wealth(self, mark, spot):
        return float(mark+self.cash+np.dot(self.shares,spot))


def _prices_deltas(p, state, curve, *, n_paths, seed, bump):
    v, rr=stock_delta_live(p,state,relative_bump=bump,n_paths=n_paths,
                          seed=seed,curve=curve)
    prices={'phoenix':v.note_price,'acro':v.option_price,'package':v.total_price}
    deltas={tag:np.array([row[col] for row in rr],float) for tag,col in
            [('phoenix','delta_note'),('acro','delta_acro'),('package','delta_total')]}
    return prices,deltas


def run_scenario_hedge(p: Project, initial: LiveState, *, end_time: float,
                       scenario_seed: int, valuation_seed: int,
                       curve:ConditionalCurve,rebalance_every_steps:int,
                       settlement_curve:ConditionalCurve|None=None,
                       n_paths:int, bump:float, fee_bps:float):
    """Run NO HEDGE, INITIAL STATIC and DYNAMIC hedges on the same Q path.

    Estimated deltas at each hedge time use no future scenario innovations.
    Cash flows enter the account AFTER every contractual observation.
    Existing shares are unwound at the terminal evaluation time.
    """
    if not 0<bump<.1 or not 0<=fee_bps<1000:
        raise ValueError('Invalid bump or transaction cost')
    steps=p.simulation.steps_per_year
    if not isinstance(rebalance_every_steps,int) or rebalance_every_steps<1:
        raise ValueError('Invalid hedge frequency')
    hist=generate_scenario(p,initial,end_time=end_time,seed=scenario_seed,
                           renewal_curve=curve if settlement_curve is None else settlement_curve)
    m=p.market
    spots=np.asarray(m.initial_spots,float)
    dividends=np.asarray(m.dividends,float)
    first=hist[0]
    prices0,delta0=_prices_deltas(p,initial,curve,n_paths=n_paths,seed=valuation_seed,bump=bump)
    methods=('none','static','dynamic')
    components=('phoenix','acro','package')
    books={(tag,method):CashBook.open(tag,method,prices0[tag],spots*np.asarray(first.ratios),delta0[tag],fee_bps)
           for tag in components for method in methods}
    trades=[];mark0=spots*np.asarray(first.ratios)
    for tag in components:
        trades.append(dict(t=first.time,scenario_seed=scenario_seed,component=tag,
                           strategy='dynamic',event='initial',market_value=prices0[tag],
                           hedge_shares=list(books[tag,'dynamic'].shares),cash=books[tag,'dynamic'].cash))
    for j in range(1,len(hist)):
        prior=hist[j-1]; cur=hist[j]
        dt=cur.time-prior.time
        spot_prev=spots*np.asarray(prior.ratios)
        spot_cur=spots*np.asarray(cur.ratios)
        for tag in components:
            flow=cur.coupon+cur.principal if tag=='phoenix' else cur.acro if tag=='acro' else cur.coupon+cur.principal+cur.acro
            for method in methods:
                books[tag,method].accrue(spot_prev,spot_cur,dt,m.risk_free_rate,dividends,flow)
        if cur.alive and j<len(hist)-1 and (cur.tick-first.tick)%rebalance_every_steps==0:
            newprice,newdelta=_prices_deltas(p,cur.state(),curve,n_paths=n_paths,
                                             seed=valuation_seed+1103*cur.tick+7,bump=bump)
            for tag in components:
                b=books[tag,'dynamic']
                b.trade(-newdelta[tag],spot_cur,fee_bps)
                trades.append(dict(t=cur.time,scenario_seed=scenario_seed,component=tag,
                    strategy='dynamic',event='rebalance',market_value=newprice[tag],
                    hedge_shares=list(b.shares),cash=b.cash))
    last=hist[-1]
    final_spot=spots*np.asarray(last.ratios)
    if last.alive:
        # Terminal mark is a valuation, not a hedge rebalance: computing bumped
        # deltas here can retrospectively trigger an already-observed autocall.
        final_value=value_live(p,last.state(),n_paths=n_paths,
                               seed=valuation_seed+99131,
                               curve=settlement_curve if settlement_curve is not None else curve)
        final_prices={'phoenix':final_value.note_price,
                      'acro':final_value.option_price,
                      'package':final_value.total_price}
    else:
        final_prices={tag:0. for tag in components}
    results=[]
    elapsed=last.time-first.time
    for tag in components:
        for method in methods:
            b=books[tag,method]
            # Close all stock hedges at end of experimental risk window.
            if method!='none': b.trade(np.zeros_like(b.shares),final_spot,fee_bps)
            val=b.wealth(final_prices[tag],final_spot)
            # Subtract deterministic financing of theoretical initial mark.
            error=val-b.initial_price*np.exp(m.risk_free_rate*elapsed)
            # After autocall, roll residual cash to common comparison horizon.
            error*=np.exp(m.risk_free_rate*(end_time-last.time))
            results.append(dict(scenario_seed=scenario_seed,component=tag,method=method,
                from_time=first.time,to_time=end_time,termination_time=last.time,terminated=not last.alive,
                autocall=last.autocall,coupon_cash=sum(h.coupon for h in hist),
                principal_cash=sum(h.principal for h in hist),acro_cash=sum(h.acro for h in hist),
                pnl_excess=error,terminal_wealth=val,fees=b.fees,
                transactions=b.transactions,initial_price=b.initial_price))
    return results,trades,hist


def summarize_errors(records: Sequence[dict]):
    result=[]
    groups=sorted(set((x['component'],x['method']) for x in records))
    for comp,method in groups:
        rows=[x for x in records if (x['component'],x['method'])==(comp,method)]
        z=np.array([r['pnl_excess'] for r in rows],float)
        fees=np.array([r['fees'] for r in rows],float)
        result.append(dict(component=comp,method=method,n=len(z),mean=float(z.mean()),
             std=float(z.std(ddof=1)) if len(z)>1 else float('nan'),
             rmse=float(np.sqrt(np.mean(z*z))),
             p05=float(np.quantile(z,.05)),p50=float(np.quantile(z,.50)),
             p95=float(np.quantile(z,.95)),mean_fees=float(fees.mean())))
    return result


def paired_bootstrap_rmse(records,*,component='package',reps=2000,seed=87123):
    """Paired scenario bootstrap. Conditions on ONE simulated renewal curve;
    not a total pricing/hedging error interval.
    """
    rows={(r['scenario_seed'],r['method']):r['pnl_excess'] for r in records
          if r['component']==component}
    seeds=sorted({k[0] for k in rows})
    if len(seeds)<3: raise ValueError('Need at least three independent scenarios')
    arr=np.array([[rows[s,m] for m in ('none','static','dynamic')] for s in seeds],float)
    rng=np.random.default_rng(seed)
    out=[]
    for a,b in [(2,1),(2,0),(1,0)]:
        difference=float(np.sqrt(np.mean(arr[:,a]**2))-np.sqrt(np.mean(arr[:,b]**2)))
        vals=[]
        for _ in range(reps):
            idx=rng.integers(0,len(seeds),len(seeds))
            vals.append(float(np.sqrt(np.mean(arr[idx,a]**2))-np.sqrt(np.mean(arr[idx,b]**2))))
        out.append(dict(component=component,method_a=('none','static','dynamic')[a],
            method_b=('none','static','dynamic')[b],rmse_a_minus_b=difference,
            bootstrap_ci95_low=float(np.quantile(vals,.025)),
            bootstrap_ci95_high=float(np.quantile(vals,.975)),
            n=len(seeds),ci_scope='paired scenarios only; fixed conditional curve and model'))
    return out
