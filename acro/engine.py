"""Multi-equity Phoenix under Q, with cash-flow affine decomposition B + c*A."""
from dataclasses import dataclass
import numpy as np
from .config import Market, Phoenix, grid_step

def basket_level(log_returns, basket_rule):
    """Phoenix reference ratio from log(S_i/S_i,reset), last axis = asset.

    worst-of and best-of use min/max of individual ratios; equal_weight
    uses the *arithmetic mean of normalized stock ratios*, not log returns.
    """
    z=np.asarray(log_returns,dtype=float)
    if basket_rule=='worst_of': return np.exp(np.min(z,axis=-1))
    if basket_rule=='best_of': return np.exp(np.max(z,axis=-1))
    if basket_rule=='equal_weight': return np.mean(np.exp(z),axis=-1)
    raise ValueError(f'Unknown basket rule: {basket_rule}')


@dataclass
class Paths:
    principal_pv: np.ndarray
    coupon_annuity_pv: np.ndarray
    first_call_year: np.ndarray
    factor_at_call: np.ndarray
    final_worst: np.ndarray
    first_call_worst: np.ndarray
    stopped_stock_martingales: np.ndarray | None = None
    observed_equity_ratios: np.ndarray | None = None
    observation_times: np.ndarray | None = None
    coupon_units_at_observation: np.ndarray | None = None
    def pv(self,coupon): return self.principal_pv + coupon*self.coupon_annuity_pv


def simulate_phoenix(market:Market, terms:Phoenix, n_paths:int,
                     steps_per_year:int, seed:int, initial_factor=None, innovations=None,
                     record_observations:bool=False) -> Paths:
    """At each issue, reset each underlying to 100% of its own starting spot.

    Risk neutral S dynamics: dS/S=(r-q)dt+sigma_i exp(x)dW_i;
    dx=kappa(theta-x)dt+eta dZ, Z correlated with equal-weight W basket.
    Flat rates, dividends, correlations. Euler discrete grid.
    """
    terms.validate(steps_per_year)
    n=int(n_paths); rng=np.random.default_rng(seed)
    a=market.n_assets; C=np.asarray(market.correlations,dtype=float)
    L=np.linalg.cholesky(C)
    basket_norm=np.sqrt(np.ones(a)@C@np.ones(a))
    x=np.full(n,market.initial_logvol_factor if initial_factor is None else 0.,dtype=float)
    if initial_factor is not None:
        x[:]=np.broadcast_to(np.asarray(initial_factor,dtype=float),(n,))
    if market.dynamics=='gbm': x[:]=0.0
    logs=np.zeros((n,a)); running_min=np.ones(n); live=np.ones(n,dtype=bool)
    missed=np.zeros(n,dtype=int); B=np.zeros(n); A=np.zeros(n)
    call_year=np.full(n,np.nan); call_factor=np.full(n,np.nan); final_worst=np.full(n,np.nan); call_worst=np.full(n,np.nan)
    martingales=np.full((n,a),np.nan)
    m=grid_step(terms.maturity_years,steps_per_year)
    every_coupon=steps_per_year//terms.observations_per_year
    call_steps=terms.call_steps(steps_per_year)
    # Optional, small visualization-only observations, never allocated for the
    # high-budget pricing runs. Ratios are to original contractual spots.
    n_obs = int(round(terms.maturity_years*terms.observations_per_year))
    observed = None
    observation_times = None
    coupon_records = None
    if record_observations:
        observed=np.full((n,n_obs+1,a),np.nan)
        observed[:,0,:]=1.
        coupon_records=np.zeros((n,n_obs))
        observation_times=np.arange(n_obs+1,dtype=float)/terms.observations_per_year
    dt=terms.maturity_years/m; sqdt=np.sqrt(dt); r=market.risk_free_rate
    if innovations is not None:
        eq_eps, factor_eps = innovations
        if eq_eps.shape != (m, n, a) or factor_eps.shape != (m, n):
            raise ValueError('Innovation bank has incompatible shape for time grid, paths or assets')
    q=np.asarray(market.dividends); sigma0=np.asarray(market.volatilities)
    rho=market.equity_factor_correlation
    for k in range(1,m+1):
        if not live.any(): break
        # Full-size random draws maintain common random numbers across factor-grid nodes
        idx=np.flatnonzero(live)
        shocks = rng.standard_normal((n,a)) if innovations is None else eq_eps[k-1]
        dw=(shocks@L.T)[idx]
        sigma=sigma0[None,:] if market.dynamics=='gbm' else sigma0[None,:]*np.exp(x[idx,None])
        logs[idx]+=(r-q[None,:]-0.5*sigma*sigma)*dt + sigma*sqdt*dw
        w=basket_level(logs[idx],terms.basket_rule)
        if terms.capital_barrier_monitoring=='continuous':
            running_min[idx]=np.minimum(running_min[idx],w)
        if market.dynamics=='stochastic_vol':
            independent=(rng.standard_normal(n) if innovations is None else factor_eps[k-1])[idx]
            z=rho*(dw.sum(axis=1)/basket_norm)+np.sqrt(1-rho*rho)*independent
            x[idx]+=market.factor_reversion*(market.factor_mean-x[idx])*dt + market.factor_volatility*sqdt*z
        discount=np.exp(-r*k*dt)
        if k%every_coupon==0:
            obs_i=k//every_coupon
            if record_observations:
                observed[idx,obs_i,:]=np.exp(logs[idx])
            good=w>=terms.coupon_barrier
            good_idx=idx[good]
            if terms.memory:
                coupon_units=missed[good_idx]+1
                missed[good_idx]=0
                missed[idx[~good]]+=1
            else: coupon_units=np.ones(len(good_idx))
            A[good_idx]+=terms.nominal*coupon_units/terms.observations_per_year*discount
            if record_observations:
                coupon_records[good_idx,obs_i-1]=coupon_units
        if k in call_steps:
            triggered=w>=terms.autocall_barrier
            cidx=idx[triggered]
            B[cidx]+=terms.nominal*discount
            call_year[cidx]=k*dt; call_factor[cidx]=x[cidx]; call_worst[cidx]=w[triggered]
            # Each discounted / dividend-adjusted normalized stock is a Q-martingale.
            # The first-call-or-maturity time is bounded and stopping-adapted; hence
            # E_Q[exp(log(S_i(tau)/S_i(0))-(r-q_i)*tau)] = 1.
            # This gives exact-mean control variates for the note PV, with no change
            # to prices, cash flows, barriers or the stochastic diffusion.
            martingales[cidx]=np.exp(logs[cidx]-(r-q[None,:])*(k*dt))
            live[cidx]=False
        if k==m:
            remaining=live[idx]; midx=idx[remaining]; wf=w[remaining]
            if terms.capital_barrier_monitoring=='european':
                breached=wf<terms.capital_barrier
            else:
                breached=running_min[midx]<=terms.capital_barrier
            redemption=np.where(breached,np.minimum(1.,wf),1.)
            B[midx]+=terms.nominal*redemption*discount
            final_worst[midx]=wf
            martingales[midx]=np.exp(logs[midx]-(r-q[None,:])*(k*dt))
            live[midx]=False
    if not np.isfinite(martingales).all():
        raise AssertionError('All paths should stop by the maturity time')
    return Paths(B,A,call_year,call_factor,final_worst,call_worst,martingales,
                 observed,observation_times,coupon_records)
