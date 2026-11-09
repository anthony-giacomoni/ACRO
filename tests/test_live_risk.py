import unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
from acro import load_project
from acro.engine import simulate_phoenix
from acro.live_risk import LiveState, simulate_live_phoenix, value_live, stock_delta_live, independent_hedge_stress, _renewal_curve
from run_risk_study import fixed_coupon_project,bump_market

ROOT=Path(__file__).resolve().parents[1]

class TestLiveRisk(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=load_project(ROOT/'config.json')
        cls.p=fixed_coupon_project(replace(base,simulation=replace(base.simulation,
            quote_paths=256,outer_paths=240,conditional_paths=192,curve_nodes=5,steps_per_year=12)),.18)

    def test_t0_live_engine_matches_reset_engine(self):
        p=self.p;state=LiveState(0.,(1.,)*p.market.n_assets,0.)
        fresh=simulate_phoenix(p.market,p.initial_phoenix,260,12,912)
        live=simulate_live_phoenix(p,state,260,912)
        for attr in ['principal_pv','coupon_annuity_pv','first_call_year','factor_at_call','final_worst','stopped_stock_martingales']:
            np.testing.assert_allclose(getattr(fresh,attr),getattr(live,attr),atol=1e-12,equal_nan=True)

    def test_contract_reference_is_not_reset_for_live_note(self):
        p=self.p
        v1=value_live(p,LiveState(1.,(.78,.92,.84),0.),n_paths=250,seed=67)
        v2=value_live(p,LiveState(1.,(.88,.92,.84),0.),n_paths=250,seed=67)
        self.assertGreater(abs(v2.note_price-v1.note_price),.01)
        # Changing initial_spots in reset-note engine alone has no effect on new-issue prices.
        m=replace(p.market,initial_spots=tuple(x*1.08 for x in p.market.initial_spots))
        a=simulate_phoenix(p.market,p.initial_phoenix,100,12,456).pv(.18)
        b=simulate_phoenix(m,p.initial_phoenix,100,12,456).pv(.18)
        np.testing.assert_array_equal(a,b)

    def test_rejected_inconsistent_surviving_state(self):
        p=self.p
        with self.assertRaises(ValueError):
            LiveState(1,(1.10,1.10,1.10),0).validate(p)
        with self.assertRaises(ValueError):
            LiveState(5,(.8,.8,.8),0).validate(p)
        with self.assertRaises(ValueError):
            LiveState(.5,(.8,.9,.9),0,missed_coupons=1).validate(p)

    def test_no_future_autocall_means_option_zero(self):
        p=replace(self.p,initial_phoenix=replace(self.p.initial_phoenix,autocall_enabled=False))
        z=value_live(p,LiveState(1,(.8,.9,.88),.12),n_paths=200,seed=88)
        self.assertEqual(z.option_price,0.)
        self.assertEqual(z.eligible_call_probability,0.)

    def test_shocks_preserve_coupon(self):
        p=self.p
        for label in ['vol_parallel_1pt','dividend_parallel_50bp','rate_50bp','equity_correlation_5pt',
                      'spot_vol_factor_corr_10pt','vol_of_vol_5pt']:
            for sign in (-1,1):
                m=bump_market(p.market,label,sign)
                self.assertTrue(np.isfinite(m.risk_free_rate))
                self.assertEqual(p.initial_phoenix.coupon_rate,.18)

    def test_live_stock_deltas_reproducible(self):
        p=self.p;st=LiveState(1.,(.87,.90,.81),.04)
        a,rows=stock_delta_live(p,st,n_paths=180)
        b,rows2=stock_delta_live(p,st,n_paths=180)
        self.assertAlmostEqual(a.note_price,b.note_price,places=12)
        self.assertEqual(len(rows),p.market.n_assets)
        for x,y in zip(rows,rows2):
            self.assertAlmostEqual(x['delta_total'],y['delta_total'],places=12)
            self.assertTrue(np.isfinite(x['delta_total']))
            # Central-difference hedge removes the odd first-order response for symmetric shocks.
            self.assertAlmostEqual(x['one_pct_up_after_delta_hedge'],
                                   x['one_pct_down_after_delta_hedge'],places=10)
            self.assertAlmostEqual(x['shares_to_short_for_long_portfolio'],
                                   -x['delta_total'],places=12)

    def test_live_martingale_normalized_at_valuation_date(self):
        p=self.p
        st=LiveState(1.,(.8,.94,.82),.05)
        paths=simulate_live_phoenix(p,st,6000,1234)
        np.testing.assert_allclose(paths.stopped_stock_martingales.mean(axis=0),
                                   np.ones(p.market.n_assets),rtol=.055,atol=.055)

    def test_strict_live_date_grid(self):
        p=self.p
        with self.assertRaisesRegex(ValueError,'grid'):
            LiveState(1.000001,(.80,.88,.91),.1).validate(p)

    def test_controlled_live_curve_anchor_and_determinism(self):
        p=self.p
        st=LiveState(1.,(.88,.92,.84),.05)
        factors=np.array([-.1,0.,.05,.13])
        a=_renewal_curve(p,st,factors,pricing_method='gbm_cv',ref_paths=128,scrambles=2)
        b=_renewal_curve(p,st,factors,pricing_method='gbm_cv',ref_paths=128,scrambles=2)
        self.assertTrue(np.any(np.isclose(a.factors,st.logvol_factor)))
        np.testing.assert_array_equal(a.principal,b.principal)
        np.testing.assert_array_equal(a.annuity,b.annuity)
        self.assertTrue(np.isfinite(a.at(factors)[0]).all())

    def test_controlled_live_value_and_no_call(self):
        p=self.p;st=LiveState(1.,(.88,.92,.84),.05)
        x=value_live(p,st,n_paths=220,seed=207,pricing_method='gbm_cv',ref_paths=128,scrambles=2)
        y=value_live(p,st,n_paths=220,seed=207,pricing_method='gbm_cv',ref_paths=128,scrambles=2)
        self.assertEqual(x.total_price,y.total_price)
        self.assertGreaterEqual(x.option_price,0.)
        disabled=replace(p,initial_phoenix=replace(p.initial_phoenix,autocall_enabled=False))
        v=value_live(disabled,st,n_paths=220,seed=207,pricing_method='gbm_cv',ref_paths=128,scrambles=2)
        self.assertEqual(v.option_price,0.)

    def test_controlled_live_delta_out_of_sample_stress(self):
        p=self.p; st=LiveState(1.,(.88,.92,.84),.05)
        v,rows=stock_delta_live(p,st,n_paths=160,pricing_method='gbm_cv',
                                ref_paths=128,scrambles=2)
        self.assertTrue(np.isfinite(v.total_price))
        self.assertEqual(len(rows),p.market.n_assets)
        o=independent_hedge_stress(p,st,rows,bumps=(.005,.01,.02),n_paths=160,
                                   seed=99117,pricing_method='gbm_cv',ref_paths=128,scrambles=2)
        self.assertEqual(len(o),6*p.market.n_assets)
        self.assertTrue(all(np.isfinite(r['hedged_pnl']) for r in o))
        self.assertTrue(all(r['outer_se_only']>=0 for r in o))
        self.assertTrue(all(r['evaluation_seed']==99117 for r in o))

    def test_controlled_route_gbm_is_compatible(self):
        p=replace(self.p,market=replace(self.p.market,dynamics='gbm'))
        st=LiveState(1.,(.88,.92,.84),0.)
        a=value_live(p,st,n_paths=220,seed=209,pricing_method='legacy')
        b=value_live(p,st,n_paths=220,seed=209,pricing_method='gbm_cv')
        self.assertAlmostEqual(a.total_price,b.total_price,places=12)

if __name__=='__main__': unittest.main()
