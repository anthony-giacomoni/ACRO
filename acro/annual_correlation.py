"""EXPERIMENTAL annual stochastic *equity cross-correlation* extension of ACRO.

The validated flat-correlation engine is deliberately left unchanged.

Under Q, within a calendar year, equity diffusion correlation is C(Y_t),
where Y_t is an annual OU state, independently shocked at whole-year dates.
At an autocall at a whole-year date the *post-update* Y_t is stored, and
becomes the initial market state of the new Phoenix. Thus the renewed note
is conditional on (X_call, Y_call), not just stochastic logvol X_call.

C(Y) = (1-|u|) C0 + max(u,0) 11' + max(-u,0) I,
with u = amplitude*tanh(Y), 0 <= amplitude < 1. Convexity ensures PD.

Neither this model nor its parameters are calibrated to option/correlation
markets. The conditional 2D surface and positive part carry MC/interpolation
bias. Prices must be labelled *research* only.
"""
from __future__ import annotations
from dataclasses import dataclass
from dataclasses import replace
import math
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from .config import Market, Phoenix, Project, grid_step
from .engine import basket_level


@dataclass(frozen=True)
class AnnualCorrelation:
    reversion: float = 0.8
    volatility: float = 0.50
    amplitude: float = 0.80
    mean: float = 0.0
    initial_state: float = 0.0

    def validate(self):
        vals = (self.reversion, self.volatility, self.amplitude,
                self.mean, self.initial_state)
        if not np.all(np.isfinite(vals)):
            raise ValueError('Correlation factor inputs must be finite')
        if self.reversion < 0 or self.volatility < 0 or not 0 <= self.amplitude < 1:
            raise ValueError('Correlation reversion/volatility must be nonnegative; amplitude < 1')


def correlation_matrices(base, states, amplitude):
    """Batch strictly PD correlation matrices for any finite latent state."""
    C0 = np.asarray(base, dtype=float)
    y = np.atleast_1d(np.asarray(states, dtype=float))
    if y.ndim != 1 or not np.all(np.isfinite(y)):
        raise ValueError('1D finite correlation states required')
    if not 0 <= amplitude < 1:
        raise ValueError('Invalid correlation mixing amplitude')
    u = amplitude * np.tanh(y)
    w0 = (1 - np.abs(u))[:, None, None]
    w1 = np.maximum(u, 0)[:, None, None]
    w2 = np.maximum(-u, 0)[:, None, None]
    return w0 * C0[None, :, :] + w1 * np.ones_like(C0)[None, :, :] + w2 * np.eye(len(C0))[None, :, :]


@dataclass
class AnnualPaths:
    principal_pv: np.ndarray
    coupon_annuity_pv: np.ndarray
    first_call_year: np.ndarray
    factor_at_call: np.ndarray
    correlation_at_call: np.ndarray
    final_worst: np.ndarray
    first_call_worst: np.ndarray
    stopped_stock_martingales: np.ndarray
    def pv(self, c):
        return self.principal_pv + c * self.coupon_annuity_pv


def simulate_phoenix_annual(market: Market, terms: Phoenix, regime: AnnualCorrelation,
                            n_paths: int, steps_per_year: int, seed: int,
                            *, initial_factor=None, initial_corr_state=None,
                            innovations=None) -> AnnualPaths:
    """Simulate note and annually refreshed correlations under Q.

    Innovations if supplied: (equity[step,path,asset], vol[step,path],
    annual_corr[year,path]); each bank is independent standard normal.
    Annual-corr independent of equity & spot-vol shocks is an explicit,
    uncalibrated modelling choice; it is NOT a fitted joint equity model.
    """
    regime.validate(); terms.validate(steps_per_year)
    n = int(n_paths); a = market.n_assets
    if n < 2: raise ValueError('Need at least two simulation paths')
    m = grid_step(terms.maturity_years, steps_per_year)
    n_annual = m // steps_per_year
    if innovations is not None:
        eq_eps, vol_eps, corr_eps = innovations
        if (np.shape(eq_eps)!=(m,n,a) or np.shape(vol_eps)!=(m,n)
                or np.shape(corr_eps)!=(n_annual,n)):
            raise ValueError('Wrong innovation bank shapes')
    eq_rng = np.random.default_rng(seed)
    # An independent generator keeps the original equity/vol innovation stream
    # unchanged when annual-correlation randomness is switched on/off.
    corr_rng = np.random.default_rng(seed + 15485863)
    C0 = np.asarray(market.correlations, dtype=float)
    L0 = np.linalg.cholesky(C0)
    x0 = market.initial_logvol_factor if initial_factor is None else initial_factor
    y0 = regime.initial_state if initial_corr_state is None else initial_corr_state
    x = np.broadcast_to(np.asarray(x0, dtype=float), (n,)).copy()
    y = np.broadcast_to(np.asarray(y0, dtype=float), (n,)).copy()
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Initial latent states must be finite')
    if market.dynamics=='gbm': x[:]=0.
    L = np.linalg.cholesky(correlation_matrices(C0,y,regime.amplitude))
    basket_norm = np.sqrt(correlation_matrices(C0,y,regime.amplitude).sum(axis=(1,2)))
    logs = np.zeros((n,a)); running_min=np.ones(n)
    live = np.ones(n,dtype=bool)
    missed = np.zeros(n, dtype=int)
    B=np.zeros(n); A=np.zeros(n)
    calls=np.full(n,np.nan); callx=np.full(n,np.nan); cally=np.full(n,np.nan)
    final_worst=np.full(n,np.nan); call_worst=np.full(n,np.nan)
    mart=np.full((n,a),np.nan)
    q=np.asarray(market.dividends,dtype=float)
    sigma0=np.asarray(market.volatilities,dtype=float)
    dt=terms.maturity_years/m; sqrt_dt=math.sqrt(dt)
    r=market.risk_free_rate; rho=market.equity_factor_correlation
    coupon_step=steps_per_year//terms.observations_per_year
    call_steps=terms.call_steps(steps_per_year)
    kappa=regime.reversion
    decay=math.exp(-kappa)
    annual_sd=regime.volatility * (math.sqrt(-math.expm1(-2*kappa)/(2*kappa))
                                    if kappa>1e-12 else 1.)
    annual_idx=0
    for k in range(1,m+1):
        if not np.any(live): break
        idx=np.flatnonzero(live)
        shocks=(eq_rng.standard_normal((n,a)) if innovations is None else eq_eps[k-1])
        dw=np.einsum('nij,nj->ni',L[idx],shocks[idx])
        sigma=(sigma0[None,:] if market.dynamics=='gbm' else
               sigma0[None,:]*np.exp(x[idx,None]))
        logs[idx]+=(r-q[None,:]-0.5*sigma*sigma)*dt+sigma*sqrt_dt*dw
        w=basket_level(logs[idx],terms.basket_rule)
        if terms.capital_barrier_monitoring=='continuous':
            running_min[idx]=np.minimum(running_min[idx],w)
        if market.dynamics=='stochastic_vol':
            independent=(eq_rng.standard_normal(n) if innovations is None
                         else vol_eps[k-1])[idx]
            basket=dw.sum(axis=1)/basket_norm[idx]
            z=rho*basket+np.sqrt(1-rho*rho)*independent
            x[idx]+=market.factor_reversion*(market.factor_mean-x[idx])*dt + \
                    market.factor_volatility*sqrt_dt*z
        # At t=1,2,... sample the new correlation state. It affects the next
        # year's diffusion AND the value of the renewed Phoenix if called now.
        # The equity move and coupon at this same date used the *old* state.
        if k%steps_per_year==0:
            new=(corr_rng.standard_normal(n) if innovations is None
                 else corr_eps[annual_idx])
            y[idx]=regime.mean+decay*(y[idx]-regime.mean)+annual_sd*new[idx]
            cm=correlation_matrices(C0,y[idx],regime.amplitude)
            L[idx]=np.linalg.cholesky(cm)
            basket_norm[idx]=np.sqrt(cm.sum(axis=(1,2)))
            annual_idx+=1
        discount=math.exp(-r*k*dt)
        if k%coupon_step==0:
            good=w>=terms.coupon_barrier
            goodidx=idx[good]
            if terms.memory:
                units=missed[goodidx]+1
                missed[goodidx]=0
                missed[idx[~good]]+=1
            else:
                units=np.ones(len(goodidx))
            A[goodidx]+=terms.nominal*units/terms.observations_per_year*discount
        if k in call_steps:
            trigger=w>=terms.autocall_barrier
            cidx=idx[trigger]
            B[cidx]+=terms.nominal*discount
            calls[cidx]=k*dt
            callx[cidx]=x[cidx]
            cally[cidx]=y[cidx]
            call_worst[cidx]=w[trigger]
            mart[cidx]=np.exp(logs[cidx]-(r-q[None,:])*(k*dt))
            live[cidx]=False
        if k==m:
            remaining=live[idx]
            midx=idx[remaining]; wf=w[remaining]
            breached=(wf<terms.capital_barrier if
                      terms.capital_barrier_monitoring=='european' else
                      running_min[midx]<=terms.capital_barrier)
            redemption=np.where(breached,np.minimum(1.,wf),1.)
            B[midx]+=terms.nominal*redemption*discount
            final_worst[midx]=wf
            mart[midx]=np.exp(logs[midx]-(r-q[None,:])*(k*dt))
            live[midx]=False
    if not np.isfinite(mart).all():
        raise AssertionError('Every path must have terminated')
    return AnnualPaths(B,A,calls,callx,cally,final_worst,call_worst,mart)


@dataclass
class ConditionalSurface2D:
    logvol_nodes: np.ndarray
    correlation_nodes: np.ndarray
    principal: np.ndarray
    annuity: np.ndarray
    paths_per_node: int

    def at(self, x, y):
        xx=np.atleast_1d(np.asarray(x,dtype=float))
        yy=np.atleast_1d(np.asarray(y,dtype=float))
        if xx.shape!=yy.shape: raise ValueError('Same-shaped state vectors required')
        pts=np.column_stack([xx,yy])
        valB=RegularGridInterpolator((self.logvol_nodes,self.correlation_nodes),
                                     self.principal,bounds_error=True)
        valA=RegularGridInterpolator((self.logvol_nodes,self.correlation_nodes),
                                     self.annuity,bounds_error=True)
        return valB(pts),valA(pts)


@dataclass
class AnnualACROResult:
    coupon_rate: float
    fair_coupon_rate: float
    initial_note_price: float
    initial_coupon_se: float
    acro_premium: float
    outer_se_only: float
    call_probability: float
    exercise_probability: float
    surface: ConditionalSurface2D
    outer_paths: AnnualPaths
    discounted_payoffs: np.ndarray


def _anchored_nodes(values, anchor, count):
    vals=np.asarray(values,dtype=float)
    lower=min(float(np.min(vals))-.035,anchor-.06) if len(vals) else anchor-.6
    upper=max(float(np.max(vals))+.035,anchor+.06) if len(vals) else anchor+.6
    nodes=np.linspace(lower,upper,count)
    k=int(np.argmin(abs(nodes-anchor)))
    if k==0 or k==count-1:
        raise AssertionError('Anchor should be an interior node')
    nodes[k]=anchor
    if np.any(np.diff(nodes)<=0):
        raise ValueError('Ill-posed conditional grid')
    return nodes


def price_acro_annual_correlations(project: Project,
                                    regime: AnnualCorrelation=AnnualCorrelation(),
                                    *, logvol_nodes: int=9,
                                    corr_nodes: int=7,
                                    conditional_paths: int|None=None,
                                    seed_offset: int=0,
                                    fixed_coupon: float|None=None) -> AnnualACROResult:
    """2D-conditional valuation, anchored to the same-model issue quote.

    Optional fixed_coupon is a *contractual* coupon shared across scenarios;
    the corresponding initial note need not be issued at par.
    """
    project.validate(); regime.validate()
    if logvol_nodes<3 or corr_nodes<3:
        raise ValueError('Need >=3 nodes per market state')
    s=project.simulation;m=project.market
    ncond = s.conditional_paths if conditional_paths is None else int(conditional_paths)
    if ncond<2: raise ValueError('Need >=2 conditional paths')
    seed=s.seed+seed_offset
    qpaths=simulate_phoenix_annual(m,project.initial_phoenix,regime,
                                   s.quote_paths,s.steps_per_year,seed+10)
    B0=float(np.mean(qpaths.principal_pv));A0=float(np.mean(qpaths.coupon_annuity_pv))
    if A0<=0:raise ValueError('No coupon annuity')
    fair=(project.initial_phoenix.nominal-B0)/A0
    if fixed_coupon is not None:
        c=float(fixed_coupon)
    elif project.initial_coupon_policy=='fair':
        c=fair
    else:
        c=project.initial_phoenix.coupon_rate
    if c<0: raise ValueError('Negative fixed/fair coupon')
    note=B0+c*A0
    q_se=float(qpaths.pv(c).std(ddof=1)/math.sqrt(s.quote_paths)/A0)
    outer=simulate_phoenix_annual(m,project.initial_phoenix,regime,
                                  s.outer_paths,s.steps_per_year,seed+20)
    call=outer.first_call_year
    eligible=np.isfinite(call)
    if project.acro.last_eligible_call_year is not None:
        eligible&=call<=project.acro.last_eligible_call_year+1e-10
    xvals=outer.factor_at_call[eligible]; yvals=outer.correlation_at_call[eligible]
    # Renewal state y is ALWAYS the correlation value known at the call;
    # its subsequent evolution in the new Phoenix starts from this same state.
    x0=m.initial_logvol_factor if m.dynamics=='stochastic_vol' else 0.
    y0=regime.initial_state
    xs=_anchored_nodes(xvals,x0,logvol_nodes)
    ys=_anchored_nodes(yvals,y0,corr_nodes)
    from .pricing import same_contract
    same=same_contract(project.initial_phoenix,project.renewal_phoenix)
    # High-precision quote bank and smaller relative-node bank can differ:
    # the *same* relative bank is used for (x,y) and (x0,y0), cancelling
    # most node noise. Exact quote B0/A0 anchors (x0,y0) without a spurious
    # time-zero renewal spread. This allows a large quote budget cheaply.
    n = ncond
    offset = 100
    base=simulate_phoenix_annual(m,project.renewal_phoenix,regime,n,
                                 s.steps_per_year,seed+offset,
                                 initial_factor=x0,initial_corr_state=y0)
    Bs=np.empty((len(xs),len(ys)));As=np.empty_like(Bs)
    for i,x in enumerate(xs):
        for j,y in enumerate(ys):
            if same and x==x0 and y==y0:
                b,a=B0,A0
            else:
                sim=simulate_phoenix_annual(m,project.renewal_phoenix,regime,n,
                            s.steps_per_year,seed+offset,
                            initial_factor=float(x),initial_corr_state=float(y))
                if same:
                    b=B0+float(np.mean(sim.principal_pv-base.principal_pv))
                    a=A0+float(np.mean(sim.coupon_annuity_pv-base.coupon_annuity_pv))
                else:
                    b=float(np.mean(sim.principal_pv))
                    a=float(np.mean(sim.coupon_annuity_pv))
            Bs[i,j]=b;As[i,j]=a
    surf=ConditionalSurface2D(xs,ys,Bs,As,n)
    payoffs=np.zeros(s.outer_paths)
    exercised=np.zeros(s.outer_paths,dtype=bool)
    if eligible.any():
        b,a=surf.at(xvals,yvals)
        renewal_coupon=(c if project.acro.target_coupon_rate is None
                         else project.acro.target_coupon_rate)
        intrinsic=project.acro.renewal_units*np.maximum(b+renewal_coupon*a-project.acro.strike,0.)
        payoffs[eligible]=np.exp(-m.risk_free_rate*call[eligible])*intrinsic
        exercised[eligible]=intrinsic>0
    return AnnualACROResult(c,fair,note,q_se,float(payoffs.mean()),
           float(payoffs.std(ddof=1)/math.sqrt(s.outer_paths)),
           float(eligible.mean()),float(exercised.mean()),surf,outer,payoffs)
