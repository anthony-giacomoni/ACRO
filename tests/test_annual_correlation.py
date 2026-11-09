"""Independent checks of the OPTIONAL annual cross-correlation model."""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from acro.config import load_project
from acro.engine import simulate_phoenix
from acro.annual_correlation import (AnnualCorrelation, correlation_matrices,
   simulate_phoenix_annual,price_acro_annual_correlations)

P = load_project(Path(__file__).parents[1]/'config.json')


def test_correlation_matrix_is_pd_for_all_extreme_regimes():
    y=np.array([-1e4,-10.,-1.,0.,1.,10.,1e4])
    cs=correlation_matrices(P.market.correlations,y,0.8)
    assert cs.shape==(len(y),3,3)
    np.testing.assert_allclose(np.diagonal(cs,axis1=1,axis2=2),1.)
    for c in cs:
        np.linalg.cholesky(c)
    np.testing.assert_allclose(cs[3],P.market.correlations)
    assert cs[0,0,1]<cs[3,0,1]<cs[-1,0,1]


def test_reject_invalid_amplitude_and_negative_reversion():
    with pytest.raises(ValueError):AnnualCorrelation(amplitude=1.).validate()
    with pytest.raises(ValueError):AnnualCorrelation(reversion=-.1).validate()
    with pytest.raises(ValueError):correlation_matrices(P.market.correlations,[0],1.)


def test_frozen_annual_model_matches_validated_base_engine_path_by_path():
    market=P.market;terms=P.initial_phoenix
    base=simulate_phoenix(market,terms,500,12,654)
    annual=simulate_phoenix_annual(market,terms,AnnualCorrelation(volatility=0.),500,12,654)
    for a,b in [(base.principal_pv,annual.principal_pv),
                (base.coupon_annuity_pv,annual.coupon_annuity_pv),
                (base.first_call_year,annual.first_call_year),
                (base.factor_at_call,annual.factor_at_call),
                (base.stopped_stock_martingales,annual.stopped_stock_martingales)]:
        np.testing.assert_allclose(a,b,rtol=1e-12,atol=1e-12,equal_nan=True)
    assert np.nanmax(np.abs(annual.correlation_at_call))<1e-14


def test_annual_regime_changes_only_at_year_boundary_and_captured_on_call():
    n=100
    rg=AnnualCorrelation(reversion=0.0,volatility=0.5)
    seed=321
    a=P.market.n_assets
    m=60
    # Force every path to autocall at year 1: deterministic upward equity shocks.
    eq=np.ones((m,n,a))*2.
    fac=np.zeros((m,n)); corr=np.zeros((5,n))
    corr[0,:]=1.0
    term=replace(P.initial_phoenix,autocall_barrier=.75)
    result=simulate_phoenix_annual(P.market,term,rg,n,12,seed,
                                    innovations=(eq,fac,corr))
    assert np.all(result.first_call_year==1.)
    np.testing.assert_allclose(result.correlation_at_call,0.5)


def test_risk_neutral_discounted_stock_martingale_approximation():
    sim=simulate_phoenix_annual(P.market,P.initial_phoenix,
                                AnnualCorrelation(),20000,12,20262027)
    mart=sim.stopped_stock_martingales
    se=mart.std(axis=0,ddof=1)/np.sqrt(len(mart))
    assert np.max(np.abs(mart.mean(axis=0)-1)/se)<4.5


def test_annual_price_surface_anchor_and_positive_payoffs():
    q=replace(P.simulation,quote_paths=500,outer_paths=300,conditional_paths=200,
              curve_nodes=5,seed=124)
    p=replace(P,simulation=q)
    res=price_acro_annual_correlations(p,logvol_nodes=5,corr_nodes=5)
    assert np.isfinite(res.acro_premium) and res.acro_premium>=0
    assert np.isfinite(res.coupon_rate)
    assert abs(res.initial_note_price-100.)<1e-10
    assert np.all(res.discounted_payoffs>=0)
    ix=np.where(res.surface.logvol_nodes==p.market.initial_logvol_factor)[0][0]
    iy=np.where(res.surface.correlation_nodes==0)[0][0]
    b,a=res.surface.at([0.],[0.])
    assert abs(float(b[0]+res.coupon_rate*a[0])-100.)<1e-10
    assert ix>0 and iy>0


def test_renewal_values_depend_on_both_market_states():
    q=replace(P.simulation,quote_paths=500,outer_paths=300,seed=125)
    res=price_acro_annual_correlations(replace(P,simulation=q),
                                        logvol_nodes=5,corr_nodes=5)
    v=res.surface.principal+res.coupon_rate*res.surface.annuity
    assert np.ptp(v,axis=0).max()>0.05  # vary log volatility
    assert np.ptp(v,axis=1).max()>0.01  # vary correlation factor


def test_fixed_coupon_does_not_claim_initial_note_still_at_par():
    q=replace(P.simulation,quote_paths=300,outer_paths=200,seed=126)
    res=price_acro_annual_correlations(replace(P,simulation=q),
              logvol_nodes=3,corr_nodes=3,fixed_coupon=.182)
    assert abs(res.coupon_rate-.182)<1e-12
    assert np.isfinite(res.initial_note_price)


def test_gbm_flat_correlation_zero_identity_but_stochastic_correlation_value():
    q=replace(P.simulation,quote_paths=650,outer_paths=650,conditional_paths=450,seed=99)
    p=replace(P,market=replace(P.market,dynamics='gbm'),simulation=q)
    frozen=price_acro_annual_correlations(p,AnnualCorrelation(volatility=0.),
                                           logvol_nodes=3,corr_nodes=5)
    variable=price_acro_annual_correlations(p,AnnualCorrelation(volatility=.5),
                                             logvol_nodes=3,corr_nodes=5)
    assert frozen.acro_premium == 0.0
    assert variable.acro_premium > 0.0
