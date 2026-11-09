"""V5.16: test Q/P separation, out-of-sample fair indemnity and cost accounting."""
from dataclasses import replace
from pathlib import Path
import unittest
import numpy as np
from acro import load_project
from acro.economic_study import simulate_full_market,compare_terminal_wealth
from acro.pricing import ConditionalCurve
from acro.investor_study import (InvestorScenario,interval_of_mean,
    discounted_acro_cost_and_call,calibrate_cash_fixed_on_independent_q,
    apply_investor_frictions,wealth_metrics,compare_investor_outcomes,
    paired_present_value_diff)

ROOT=Path(__file__).resolve().parents[1]

class TestInvestorV516(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=load_project(ROOT/'config.json')
        p=replace(p,initial_coupon_policy='fixed',
             initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
             simulation=replace(p.simulation,steps_per_year=12,quote_paths=100,
                conditional_paths=100,outer_paths=100))
        cls.p=p
        cls.curve=ConditionalCurve(np.array([-3.,3.]),np.array([91.,91.]),
                                   np.array([90.,90.]),100)

    def small_rec(self):
        p=self.p;log,fac=simulate_full_market(p,90,10.,1101)
        return compare_terminal_wealth(p,log,fac,self.curve,coupon=.18,
            no_call_coupon=.15,premium=.7,fixed_amount=1.8,
            initial_note_price=99.7)

    def test_q_baseline_equal_explicit_zero_risk_premia(self):
        a,x=simulate_full_market(self.p,60,10.,199)
        b,y=simulate_full_market(self.p,60,10.,199,
                 equity_excess_drifts=[0.]*3,factor_mean_override=0.)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(x,y)

    def test_p_drift_shifts_log_spot_exactly_for_coupled_shocks(self):
        a,x=simulate_full_market(self.p,40,2.,882)
        b,y=simulate_full_market(self.p,40,2.,882,equity_excess_drifts=[.03]*3)
        np.testing.assert_allclose(b[-1]-a[-1],.06,atol=1e-12)
        np.testing.assert_array_equal(x,y)

    def test_bad_drifts_and_bad_credit_stress_rejected(self):
        with self.assertRaises(ValueError):
            simulate_full_market(self.p,10,2.,13,equity_excess_drifts=[.03])
        with self.assertRaises(ValueError):
            InvestorScenario('bad',0.,0.,1.1).validate()

    def test_calibration_identity_and_paired_pv(self):
        c=np.array([0.,.1,.5,.9]);d=np.array([0.,.5,1.,1.])
        z=calibrate_cash_fixed_on_independent_q(c,d)
        self.assertAlmostEqual(float((c-z['fixed_amount']*d).mean()),0.,places=14)
        rec=self.small_rec();self.assertTrue(np.isfinite(paired_present_value_diff(rec,rate=.025)).all())
        c,d=discounted_acro_cost_and_call(rec,.025)
        self.assertEqual(c.shape,d.shape)
        self.assertTrue(np.all(c[~rec['eligible']]==0))

    def test_no_frictions_leaves_all_wealth_unchanged(self):
        rec=self.small_rec()
        result=apply_investor_frictions(rec,rate=.025,premium=.7,
            scenario=InvestorScenario('none',0,0),original_nominal=100,seed=10)
        for key,val in rec['wealth'].items():
            np.testing.assert_array_equal(result['wealth'][key],val)
        self.assertFalse(result['delivery_failed'].any())

    def test_full_delivery_failure_falls_back_to_fair_renewal(self):
        rec=self.small_rec()
        r=apply_investor_frictions(rec,rate=.025,premium=.7,
            scenario=InvestorScenario('fail',0,0,1.),original_nominal=100,seed=10)
        fair=np.asarray(rec['wealth']['reserve'])-.7*np.exp(.025*rec['horizon'])
        np.testing.assert_allclose(r['wealth']['acro'],fair)
        self.assertTrue(np.all(r['acro_shortfall']>=0))

    def test_fee_difference_and_no_call_fraction(self):
        rec=self.small_rec()
        a=apply_investor_frictions(rec,rate=.025,premium=.7,
            scenario=InvestorScenario('fee',0,0,0.,renewal_fee_bps=10,
                                      acro_exercise_fee_bps=5),original_nominal=100,seed=3)
        b=apply_investor_frictions(rec,rate=.025,premium=.7,
            scenario=InvestorScenario('zero',0,0),original_nominal=100,seed=3)
        self.assertTrue(np.all(a['wealth']['fixed']<=b['wealth']['fixed']+1e-12))
        self.assertTrue(np.all(a['wealth']['no_call']<=b['wealth']['no_call']+1e-12))
        self.assertTrue(np.all(a['wealth']['acro']<=b['wealth']['acro']+1e-12))
        np.testing.assert_allclose((b['wealth']['no_call']-a['wealth']['no_call'])/
             np.exp(.025*(rec['horizon']-rec['no_call_renewal_time'])),
             .1*rec['no_call_renewal_fraction'])

    def test_wealth_metrics_and_conditional_preferences(self):
        z=wealth_metrics([100.,100.,100.],budget=100.,rate=0.,horizon=10.)
        self.assertAlmostEqual(z['crra_ce'],100.)
        a=compare_investor_outcomes(self.small_rec(),
              apply_investor_frictions(self.small_rec(),rate=.025,premium=.7,
                 scenario=InvestorScenario('none',0,0),original_nominal=100,seed=1),rate=.025)
        self.assertEqual(a['n'],90)

    def test_intervals_require_multiple_independent_banks(self):
        with self.assertRaises(ValueError): interval_of_mean([1.])
        mu,se,ci=interval_of_mean([.2,.3,.4,.1])
        self.assertTrue(ci[0] < mu < ci[1]);self.assertGreater(se,0.)

if __name__=='__main__':unittest.main()
