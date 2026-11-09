import unittest
from dataclasses import replace
import numpy as np
from pathlib import Path
from acro import load_project
from acro.pricing import ConditionalCurve
from acro.live_risk import LiveState
from acro.dynamic_hedge import CashBook,generate_scenario,run_scenario_hedge,summarize_errors
from run_risk_study import fixed_coupon_project

ROOT=Path(__file__).resolve().parents[1]

class TestDynamicHedge(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=load_project(ROOT/'config.json')
        cls.p=fixed_coupon_project(replace(p,simulation=replace(p.simulation,
              steps_per_year=12,outer_paths=128,conditional_paths=128,
              curve_nodes=5,quote_paths=128)),.18)
        cls.state=LiveState(1.,(.88,.92,.84),.05)
        cls.curve=ConditionalCurve(np.array([-2.,2.]),
                    np.array([101.,101.]),np.array([10.,10.]),100)

    def test_cash_account_exact_self_financing_without_fees(self):
        spot=np.array([100.,100.]);h=np.array([1.,-.5])
        b=CashBook.open('package','dynamic',20.,spot,h,0.)
        self.assertAlmostEqual(b.wealth(20.,spot),20.)
        new=np.array([110.,95.]);b.accrue(spot,new,1.,0.,np.array([0.,0.]),flow=3.)
        b.trade(-h,new,0.)
        self.assertAlmostEqual(b.wealth(25.,new),25.+3.+(-h)@new-(-h)@spot)

    def test_cash_fees_and_trading_ledger(self):
        b=CashBook.open('x','static',5.,np.array([100.]),np.array([1.]),10.)
        self.assertAlmostEqual(b.fees,.1)
        self.assertAlmostEqual(b.wealth(5.,np.array([100.])),4.9)
        b.trade(np.array([0.]),np.array([110.]),10.)
        self.assertAlmostEqual(b.fees,.21)
        self.assertEqual(b.transactions,2)

    def test_scenario_reproducible_and_fixed_contract_references(self):
        a=generate_scenario(self.p,self.state,end_time=2.,seed=72,renewal_curve=self.curve)
        b=generate_scenario(self.p,self.state,end_time=2.,seed=72,renewal_curve=self.curve)
        self.assertEqual(a,b)
        self.assertTrue(a[0].alive)
        self.assertEqual(a[0].time,1.)
        self.assertEqual(a[-1].time,2.)
        self.assertEqual(a[-1].principal,0. if a[-1].alive else 100.)
        for x in a:
            self.assertGreater(min(x.ratios),0.)
            self.assertGreaterEqual(x.acro,0.)

    def test_known_autocall_coupon_and_acro_cash_only_once(self):
        m=replace(self.p.market,dynamics='gbm',risk_free_rate=.5,
                  volatilities=tuple([1e-6]*self.p.market.n_assets))
        p=replace(self.p,market=m)
        st=LiveState(1.,(.95,.95,.95),0.)
        crv=ConditionalCurve(np.array([-2.,2.]),np.array([110.,110.]),
                             np.array([0.,0.]),100)
        history=generate_scenario(p,st,end_time=2.,seed=66,renewal_curve=crv)
        self.assertTrue(history[-1].autocall)
        self.assertEqual(history[-1].principal,100.)
        self.assertEqual(history[-1].acro,10.)
        self.assertAlmostEqual(sum(x.coupon for x in history),18.)
        self.assertFalse(history[-1].alive)

    def test_package_unhedged_equals_note_plus_option_and_reproducible(self):
        kwargs=dict(p=self.p,initial=self.state,end_time=2.,scenario_seed=812,
                    valuation_seed=9012,curve=self.curve,rebalance_every_steps=3,
                    n_paths=100,bump=.01,fee_bps=5.)
        a,_,_=run_scenario_hedge(**kwargs)
        b,_,_=run_scenario_hedge(**kwargs)
        self.assertEqual(a,b)
        self.assertEqual(len(a),9)
        def err(comp,method):
            return next(x['pnl_excess'] for x in a if x['component']==comp and x['method']==method)
        self.assertAlmostEqual(err('package','none'),err('phoenix','none')+err('acro','none'),places=10)
        self.assertTrue(all(x['fees']>=0 for x in a))
        self.assertTrue(all(np.isfinite(x['pnl_excess']) for x in a))

    def test_dynamic_cash_flows_and_invalid_inputs(self):
        with self.assertRaises(ValueError):
            generate_scenario(self.p,self.state,end_time=1.,seed=1,renewal_curve=self.curve)
        with self.assertRaises(ValueError):
            run_scenario_hedge(self.p,self.state,end_time=2.,scenario_seed=1,
                valuation_seed=2,curve=self.curve,rebalance_every_steps=0,
                n_paths=100,bump=.01,fee_bps=5.)

    def test_summary_reports_rmse_and_fees(self):
        rows=[dict(component='package',method='dynamic',pnl_excess=x,fees=.1)
              for x in [-1.,2.,1.]]
        v=summarize_errors(rows)[0]
        self.assertEqual(v['n'],3)
        self.assertAlmostEqual(v['mean'],2/3)
        self.assertAlmostEqual(v['rmse'],np.sqrt(2))
        self.assertAlmostEqual(v['mean_fees'],.1)

if __name__=='__main__':unittest.main()
