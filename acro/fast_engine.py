"""Optional Numba path simulator, exactly the same Euler/GBM model and payoffs.

No new model assumptions: fed the SAME innovation bank as the NumPy engine,
returns the same discounted principal, coupon annuity, autocall timing, etc.
This module is used only in the high-precision research validation pipeline.
The original numpy simulator remains the default for normal Spyder runs.
"""
from __future__ import annotations
import math
import numpy as np
from .engine import Paths
from .config import Market,Phoenix,grid_step
try:
    from numba import njit
except ImportError:
    njit=None

if njit is not None:
    @njit(cache=True)
    def _simulate(eq,fac,L,q,sigma0,r,rho,basket_norm,
                  xstart,theta,kappa,eta,gbm,dt,
                  coupon_every,nominal,coupon_barrier,memory,call_mask,
                  autocall_barrier,capital_barrier,continuous,basket_mode):
        nsteps,n,a=eq.shape
        B=np.zeros(n);A=np.zeros(n);calls=np.empty(n);callf=np.empty(n)
        callw=np.empty(n);lastw=np.empty(n);mart=np.empty((n,a))
        for i in range(n):
            calls[i]=np.nan;callf[i]=np.nan;callw[i]=np.nan;lastw[i]=np.nan
            logs=np.zeros(a);x=xstart;missed=0;minimum=1.0
            for k in range(nsteps):
                step=k+1
                vol=1.0 if gbm else math.exp(x)
                total_dW=0.0
                for j in range(a):
                    dw=0.0
                    for h in range(a):
                        dw+=L[j,h]*eq[k,i,h]
                    total_dW+=dw
                    sigma=sigma0[j]*vol
                    logs[j]+=(r-q[j]-.5*sigma*sigma)*dt+sigma*math.sqrt(dt)*dw
                w=math.inf if basket_mode==0 else (-math.inf if basket_mode==1 else 0.0)
                for j in range(a):
                    zz=math.exp(logs[j])
                    if basket_mode==0 and zz<w: w=zz
                    elif basket_mode==1 and zz>w: w=zz
                    elif basket_mode==2: w+=zz/a
                if continuous and w<minimum:minimum=w
                if not gbm:
                    # Recompute sum dw: identical sum of correlated equity shocks.
                    z=rho*total_dW/basket_norm+math.sqrt(1-rho*rho)*fac[k,i]
                    x+=kappa*(theta-x)*dt+eta*math.sqrt(dt)*z
                disc=math.exp(-r*step*dt)
                if step%coupon_every==0:
                    if w>=coupon_barrier:
                        units=missed+1 if memory else 1
                        A[i]+=nominal*units*disc/(1./(coupon_every*dt))
                        missed=0
                    elif memory:
                        missed+=1
                stopped=False
                if call_mask[step] and w>=autocall_barrier:
                    B[i]+=nominal*disc;calls[i]=step*dt
                    callf[i]=x;callw[i]=w;stopped=True
                elif step==nsteps:
                    level=minimum if continuous else w
                    red=min(1.0,w) if level<capital_barrier else 1.0
                    B[i]+=nominal*red*disc;lastw[i]=w;stopped=True
                if stopped:
                    for j in range(a):
                        mart[i,j]=math.exp(logs[j]-(r-q[j])*(step*dt))
                    break
        return B,A,calls,callf,lastw,callw,mart
else:
    _simulate=None


def simulate_phoenix_fast(market:Market,terms:Phoenix,n_paths:int,steps_per_year:int,
                          seed:int=0,initial_factor=None,innovations=None):
    """Execute the original contract with a compiled path loop.

    This is an optional accelerator. Provide innovation bank for exact
    paired benchmarking; if omitted, innovations are generated independently.
    """
    if _simulate is None:
        raise ImportError('Numba not installed: install numba or use the NumPy pricer')
    terms.validate(steps_per_year)
    n=int(n_paths);m=grid_step(terms.maturity_years,steps_per_year)
    if innovations is None:
        rng=np.random.default_rng(seed)
        eq=rng.standard_normal((m,n,market.n_assets));fac=rng.standard_normal((m,n))
    else:
        eq,fac=innovations
    if eq.shape!=(m,n,market.n_assets) or fac.shape!=(m,n):
        raise ValueError('Innovation bank has incompatible shape')
    eq=np.ascontiguousarray(eq,float);fac=np.ascontiguousarray(fac,float)
    C=np.asarray(market.correlations,float)
    L=np.linalg.cholesky(C)
    a=market.n_assets
    basket=float(np.sqrt(np.ones(a)@C@np.ones(a)))
    callmask=np.zeros(m+1,dtype=np.bool_)
    for k in terms.call_steps(steps_per_year):callmask[k]=True
    dt=terms.maturity_years/m
    x=0.0 if market.dynamics=='gbm' else (market.initial_logvol_factor if initial_factor is None else float(initial_factor))
    B,A,t,f,w,wc,mar=_simulate(eq,fac,L,np.asarray(market.dividends),
        np.asarray(market.volatilities),market.risk_free_rate,
        market.equity_factor_correlation,basket,float(x),market.factor_mean,
        market.factor_reversion,market.factor_volatility,market.dynamics=='gbm',
        dt,steps_per_year//terms.observations_per_year,terms.nominal,
        terms.coupon_barrier,terms.memory,callmask,terms.autocall_barrier,
        terms.capital_barrier,terms.capital_barrier_monitoring=='continuous',
        {'worst_of':0,'best_of':1,'equal_weight':2}[terms.basket_rule])
    return Paths(B,A,t,f,w,wc,mar)
