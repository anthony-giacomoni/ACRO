"""Multi-equity basket contract extension: independent checks and compatibility."""
from dataclasses import replace
import numpy as np
import pytest
from acro.config import Market, Phoenix, load_project
from acro.engine import simulate_phoenix, basket_level
from acro.annual_correlation import simulate_phoenix_annual, AnnualCorrelation
from acro_gui import resize_market_config

BASE=load_project(__import__('pathlib').Path(__file__).resolve().parents[1]/'config.json')

def project_assets(n):
    from dataclasses import asdict
    d=asdict(BASE)
    resized=resize_market_config(d,n)
    m=resized['market']
    return replace(BASE, market=replace(BASE.market,
      initial_spots=tuple(m['initial_spots']),volatilities=tuple(m['volatilities']),
      dividends=tuple(m['dividends']),correlations=tuple(tuple(x) for x in m['correlations'])))

@pytest.mark.parametrize('n', [1,2,3,4,5,6])
def test_resize_market_and_pd(n):
    p=project_assets(n)
    p.validate()
    assert p.market.n_assets==n
    np.linalg.cholesky(p.market.correlations)
    assert np.asarray(p.market.correlations).shape==(n,n)

@pytest.mark.parametrize('rule,expected', [('worst_of',0.55),('equal_weight',1.),('best_of',1.45)])
def test_arithmetic_reference_selection(rule,expected):
    assert basket_level(np.log([.55,1.,1.45]),rule)==pytest.approx(expected)

@pytest.mark.parametrize('n',[1,2,4,6])
@pytest.mark.parametrize('rule',['worst_of','equal_weight','best_of'])
def test_simulator_all_n_and_styles(n,rule):
    p=project_assets(n)
    term=replace(p.initial_phoenix, basket_rule=rule, autocall_enabled=False,
                 maturity_years=1, capital_barrier_monitoring='european')
    out=simulate_phoenix(p.market,term,350,12,39000)
    assert np.all(np.isfinite(out.principal_pv))
    assert np.all(np.isfinite(out.coupon_annuity_pv))
    assert np.all(np.isfinite(out.final_worst))
    assert np.all(out.principal_pv<=100*np.exp(-p.market.risk_free_rate)+1e-12)

@pytest.mark.parametrize('n',[1,2,4,6])
def test_pathwise_principal_and_coupons_ordering(n):
    p=project_assets(n)
    m=replace(p.market,dynamics='gbm')
    rng=np.random.default_rng(2026)
    bank=(rng.standard_normal((12,350,n)),np.zeros((12,350)))
    arrays={}
    for rule in ('worst_of','equal_weight','best_of'):
        t=replace(p.initial_phoenix,maturity_years=1,autocall_enabled=False,
                  basket_rule=rule)
        arrays[rule]=simulate_phoenix(m,t,350,12,0,innovations=bank)
    for a,b in [('worst_of','equal_weight'),('equal_weight','best_of')]:
        assert np.all(arrays[a].principal_pv<=arrays[b].principal_pv+1e-11)
        assert np.all(arrays[a].coupon_annuity_pv<=arrays[b].coupon_annuity_pv+1e-11)
    if n==1:
        for attr in ('principal_pv','coupon_annuity_pv','final_worst'):
            np.testing.assert_array_equal(getattr(arrays['worst_of'],attr),getattr(arrays['best_of'],attr))

@pytest.mark.parametrize('rule',['worst_of','equal_weight','best_of'])
def test_original_and_annual_no_stochastic_corr_exact(rule):
    p=project_assets(4)
    t=replace(p.initial_phoenix,basket_rule=rule,maturity_years=1,first_autocall_year=1)
    eq=np.random.default_rng(51).standard_normal((12,400,4))
    vol=np.random.default_rng(52).standard_normal((12,400))
    y=np.zeros((1,400))
    base=simulate_phoenix(p.market,t,400,12,0,innovations=(eq,vol))
    annual=simulate_phoenix_annual(p.market,t,AnnualCorrelation(volatility=0),400,12,0,
                                    innovations=(eq,vol,y))
    for key in ('principal_pv','coupon_annuity_pv','first_call_year','final_worst'):
        np.testing.assert_allclose(getattr(base,key),getattr(annual,key),rtol=1e-12,atol=1e-12,equal_nan=True)

def test_unknown_aggregator_rejected():
    t=replace(BASE.initial_phoenix,basket_rule='median')
    with pytest.raises(ValueError,match='basket_rule'):
        t.validate(12)

def test_original_default_unchanged():
    assert BASE.market.n_assets==3
    assert BASE.initial_phoenix.basket_rule==BASE.renewal_phoenix.basket_rule=='worst_of'

def test_mixed_reference_initial_and_renewal_are_priced_as_distinct_notes():
    from acro.pricing import quote_initial, price_acro
    p=project_assets(2)
    p=replace(p,initial_phoenix=replace(p.initial_phoenix,basket_rule='worst_of'),
              renewal_phoenix=replace(p.renewal_phoenix,basket_rule='best_of'),
              simulation=replace(p.simulation,quote_paths=550,outer_paths=450,
                                 conditional_paths=550,curve_nodes=5))
    q=quote_initial(p)
    r=price_acro(p,q)
    assert np.isfinite(r.price) and r.price>=0
    assert np.isfinite(q.coupon_rate)
    assert r.zero_gbm_identity_applies is False
