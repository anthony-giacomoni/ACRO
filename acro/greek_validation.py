"""Out-of-sample numerical analysis of Greeks for a LIVE ACRO package.

Common outer shocks across base / up / down, one common independently generated
renewal curve per bank. Values and errors are for a DISCRETE research model, not
market quotes. No dynamic hedging is claimed.
"""
from dataclasses import replace
import numpy as np
from .live_risk import LiveState, simulate_live_phoenix, _renewal_curve


def path_payoffs(project, state, paths, curve):
    """Discounted note and ACRO payoffs conditional on the supplied live state."""
    note = paths.pv(project.initial_phoenix.coupon_rate)
    eligible = np.isfinite(paths.first_call_year)
    last = project.acro.last_eligible_call_year
    if last is not None:
        eligible &= paths.first_call_year <= last + 1e-10
    option = np.zeros_like(note)
    if eligible.any():
        if curve is None:
            raise ValueError('An independent renewal curve is required for eligible calls')
        b,a = curve.at(paths.factor_at_call[eligible])
        c = (project.initial_phoenix.coupon_rate if project.acro.target_coupon_rate is None
             else project.acro.target_coupon_rate)
        intrinsic = np.maximum(b + c*a-project.acro.strike, 0.)
        option[eligible] = (project.acro.renewal_units * intrinsic
            * np.exp(-project.market.risk_free_rate
                     * (paths.first_call_year[eligible]-state.time_years)))
    return np.stack([note, option, note+option], axis=0)


def paired_estimate(a):
    a=np.asarray(a,dtype=float)
    if a.ndim != 1 or len(a)<2 or not np.isfinite(a).all():
        raise ValueError('Invalid paired payoff vector')
    return float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a)))


def paired_greek_triplet(down,center,up,spot,relative_bump):
    """Return paired delta and gamma estimates (mean,se), for each component.

    down,center,up: shape [3, paths] for note, ACRO, total.
    delta = dV/dS in EUR/EUR ; gamma = d²V/dS² in EUR/EUR².
    """
    h=float(spot)*float(relative_bump)
    if not h>0 or not np.isfinite(h):
        raise ValueError('Positive finite absolute spot bump required')
    values=[np.asarray(a,float) for a in (down,center,up)]
    if not all(v.ndim==2 and v.shape==values[0].shape for v in values):
        raise ValueError('Incompatible payoff arrays')
    delta=(values[2]-values[0])/(2*h)
    gamma=(values[2]-2*values[1]+values[0])/(h*h)
    return [(paired_estimate(delta[i]),paired_estimate(gamma[i])) for i in range(values[0].shape[0])]


def live_greek_bank(p,state,*,bumps=(.005,.01,.02),n_paths=1500,
                    outer_seed=800001,curve_seed=600000,
                    pricing_method='gbm_cv',ref_paths=512,scrambles=3):
    """Single joint Greek bank with a common outer bank AND a common inner curve.

    Repeated independent banks should vary both outer_seed and curve_seed.
    Repeated outer-only banks can hold curve_seed fixed to diagnose errors.
    """
    state.validate(p)
    bumps=tuple(float(v) for v in bumps)
    if not bumps or any(v <= 0 or v >=.1 for v in bumps):
        raise ValueError('bumps must lie between 0 and 10%')
    states={(0,0,0):state}
    for asset in range(p.market.n_assets):
        for j,bump in enumerate(bumps):
            for sign in (-1,1):
                ratios=list(state.spot_ratios)
                ratios[asset]*=(1+sign*bump)
                states[(asset,j,sign)]=replace(state,spot_ratios=tuple(ratios))
    path_map={}; factors=[]
    for key,s in states.items():
        paths=simulate_live_phoenix(p,s,int(n_paths),int(outer_seed))
        path_map[key]=paths
        mask=np.isfinite(paths.first_call_year)
        last=p.acro.last_eligible_call_year
        if last is not None:
            mask &= paths.first_call_year<=last+1e-10
        factors.extend(paths.factor_at_call[mask])
    curve = _renewal_curve(p,state,np.asarray(factors),int(curve_seed),
                pricing_method=pricing_method,ref_paths=ref_paths,
                scrambles=scrambles) if factors else None
    base=path_payoffs(p,state,path_map[(0,0,0)],curve)
    result=[]
    for i in range(p.market.n_assets):
        spot=float(p.market.initial_spots[i]*state.spot_ratios[i])
        for j,bump in enumerate(bumps):
            down=path_payoffs(p,states[(i,j,-1)],path_map[(i,j,-1)],curve)
            up=path_payoffs(p,states[(i,j,1)],path_map[(i,j,1)],curve)
            res=paired_greek_triplet(down,base,up,spot,bump)
            for comp,((delta,se_delta),(gamma,se_gamma)) in zip(('phoenix','acro','package'),res):
                result.append(dict(asset=i+1,component=comp,bump_percent=100*bump,
                    delta=delta,delta_outer_se=se_delta,
                    gamma=gamma,gamma_outer_se=se_gamma,
                    current_spot=spot,n_paths=int(n_paths),
                    outer_seed=int(outer_seed),curve_seed=int(curve_seed)))
    return result
