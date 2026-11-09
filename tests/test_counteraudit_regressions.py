"""Independent regression tests supplied by the external ACRO annual-correlation audit.

Run from the ACRO_GitHub_Final folder:
    python -m pytest -q tests/test_counteraudit_regressions.py

Checks unusual Phoenix terms, simulated equity correlation, exact annual OU
transition and non-anticipation at first autocall. Retained unmodified in spirit
from the independent auditor, with only project-relative imports adapted.
"""
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
from acro.config import load_project
from acro.engine import simulate_phoenix
from acro.annual_correlation import (
    simulate_phoenix_annual, AnnualCorrelation, correlation_matrices
)

P = load_project(Path(__file__).resolve().parents[1] / 'config.json')


@pytest.mark.parametrize('memory,barrier', [(False, 'european'),
                                             (True, 'european'),
                                             (True, 'continuous')])
def test_pathwise_recovery_nondefault_terms(memory, barrier):
    terms = replace(P.initial_phoenix, memory=memory,
                    capital_barrier_monitoring=barrier,
                    observations_per_year=4)
    a = simulate_phoenix(P.market, terms, 1400, 12, 884)
    b = simulate_phoenix_annual(P.market, terms, AnnualCorrelation(volatility=0.),
                                1400, 12, 884)
    for x, y in [(a.principal_pv, b.principal_pv),
                 (a.coupon_annuity_pv, b.coupon_annuity_pv),
                 (a.first_call_year, b.first_call_year),
                 (a.factor_at_call, b.factor_at_call),
                 (a.first_call_worst, b.first_call_worst),
                 (a.stopped_stock_martingales, b.stopped_stock_martingales)]:
        np.testing.assert_allclose(x, y, rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize('y', [-2., 0., 2.])
def test_empirical_equity_correlation(y):
    market = replace(P.market, dynamics='gbm')
    terms = replace(P.initial_phoenix, maturity_years=1,
                    autocall_enabled=False)
    regime = AnnualCorrelation(reversion=0., volatility=0., initial_state=y)
    paths = simulate_phoenix_annual(market, terms, regime, 12000, 12, 2029)
    vols = np.asarray(market.volatilities)
    shock = np.log(paths.stopped_stock_martingales) / vols + .5 * vols
    target = correlation_matrices(market.correlations, [y], .8)[0]
    assert np.max(np.abs(np.corrcoef(shock.T) - target)) < .02


def test_stochastic_y_pathwise_transition():
    terms = replace(P.initial_phoenix, autocall_barrier=.1)
    regime = AnnualCorrelation(reversion=.8, volatility=.5, initial_state=.15)
    n = 90
    eps = np.full((60, n, 3), 2.)
    vol = np.zeros((60, n))
    corr = np.zeros((5, n))
    corr[0] = np.linspace(-2, 2, n)
    paths = simulate_phoenix_annual(P.market, terms, regime, n, 12, 123,
                                    innovations=(eps, vol, corr))
    expected = (np.exp(-.8) * .15
                + .5 * np.sqrt((-np.expm1(-1.6)) / 1.6) * corr[0])
    assert np.all(paths.first_call_year == 1.)
    np.testing.assert_allclose(paths.correlation_at_call, expected,
                               rtol=0, atol=1e-15)


def test_no_future_correlation_shock_in_renewal_at_year_one():
    n = 110
    terms = replace(P.initial_phoenix, autocall_barrier=.1)
    eq = np.full((60, n, 3), 2.)
    vol = np.zeros((60, n))
    corr = np.zeros((5, n))
    corr[0] = np.linspace(-2, 2, n)
    future = corr.copy()
    future[1:] = 100.0
    regime = AnnualCorrelation(volatility=.5)
    a = simulate_phoenix_annual(P.market, terms, regime, n, 12, 123,
                               innovations=(eq, vol, corr))
    b = simulate_phoenix_annual(P.market, terms, regime, n, 12, 123,
                               innovations=(eq, vol, future))
    assert np.all(a.first_call_year == 1.)
    np.testing.assert_array_equal(a.correlation_at_call, b.correlation_at_call)
    np.testing.assert_array_equal(a.principal_pv, b.principal_pv)
    np.testing.assert_array_equal(a.coupon_annuity_pv, b.coupon_annuity_pv)
