"""Regression tests for the educational visualization layer.

They do not validate an actual market price; they enforce that explanatory
outputs are consistent with the existing simulated cash flows.
"""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import numpy as np

from acro.config import load_project
from acro.engine import simulate_phoenix
from acro.pricing import price_acro
from acro.portfolio_visuals import make_portfolio_figures, payoff_statistics, FIGURE_NAMES

ROOT=Path(__file__).resolve().parents[1]

class PortfolioVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=load_project(ROOT/'config.json')
        cls.p=replace(p, simulation=replace(p.simulation,steps_per_year=4,
                                             quote_paths=128,outer_paths=256,
                                             conditional_paths=64,curve_nodes=5))

    def test_recording_is_optional_and_preserves_pricing(self):
        a=simulate_phoenix(self.p.market,self.p.initial_phoenix,32,4,17)
        b=simulate_phoenix(self.p.market,self.p.initial_phoenix,32,4,17,
                           record_observations=True)
        self.assertIsNone(a.observed_equity_ratios)
        for name in ('principal_pv','coupon_annuity_pv','first_call_year',
                     'factor_at_call','stopped_stock_martingales'):
            np.testing.assert_allclose(getattr(a,name),getattr(b,name),equal_nan=True)

    def test_observed_paths_start_at_one_and_stop_on_call(self):
        z=simulate_phoenix(self.p.market,self.p.initial_phoenix,64,4,102,
                           record_observations=True)
        np.testing.assert_allclose(z.observed_equity_ratios[:,0,:],1.)
        self.assertEqual(z.observed_equity_ratios.shape,
                         (64,1+int(self.p.initial_phoenix.maturity_years*2),3))
        self.assertTrue((z.observed_equity_ratios[np.isfinite(z.observed_equity_ratios)]>0).all())
        for i,call in enumerate(z.first_call_year):
            if np.isfinite(call):
                k=int(round(call*self.p.initial_phoenix.observations_per_year))
                self.assertTrue(np.isnan(z.observed_equity_ratios[i,k+1:]).all())

    def test_coupon_records_reconstruct_annuity(self):
        z=simulate_phoenix(self.p.market,self.p.initial_phoenix,64,4,103,
                           record_observations=True)
        terms=self.p.initial_phoenix
        discounted=terms.nominal/terms.observations_per_year*(
            z.coupon_units_at_observation *
            np.exp(-self.p.market.risk_free_rate*z.observation_times[1:]))
        np.testing.assert_allclose(discounted.sum(axis=1),z.coupon_annuity_pv,atol=1e-10)

    def test_discounted_payoff_mean_matches_price(self):
        r=price_acro(self.p)
        stats=payoff_statistics(r)
        self.assertAlmostEqual(stats['payoff_discounted_mean_check'],r.price,places=12)
        self.assertAlmostEqual(stats['positive_payoff_probability_Q']+
                               stats['zero_payoff_probability_Q'],1.0,places=12)
        self.assertAlmostEqual(stats['path_count_for_distribution'],256)

    def test_educational_charts_export_independently(self):
        # Small budget, no pricing input modified by rendering.
        r=price_acro(self.p)
        price_before=r.price
        with tempfile.TemporaryDirectory() as tmp:
            info=make_portfolio_figures(self.p,r,tmp,count=17)
            self.assertEqual(info['illustrative_paths'],17)
            self.assertEqual(info['portfolio_figures'],FIGURE_NAMES)
            for name in FIGURE_NAMES:
                self.assertGreater((Path(tmp)/name).stat().st_size,2500)
            self.assertGreater((Path(tmp)/'illustrative_equity_trajectories.csv').stat().st_size,250)
        self.assertEqual(price_before,r.price)

    def test_payoff_stats_works_with_only_zero_values(self):
        class ZeroResult:
            eligible=np.array([False,True,False,True])
            discounted_payoff=np.zeros(4)
        info=payoff_statistics(ZeroResult())
        self.assertEqual(info['positive_payoff_probability_Q'],0.)
        self.assertEqual(info['conditional_discounted_payoff_if_positive'],0.)
        self.assertEqual(info['zero_payoff_probability_Q'],1.)

if __name__=='__main__':unittest.main()
