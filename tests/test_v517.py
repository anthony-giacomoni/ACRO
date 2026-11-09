"""Audit-specific tests: common contingent default and investor targets."""
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from acro import load_project
from acro.economic_study import simulate_full_market,compare_terminal_wealth
from acro.pricing import ConditionalCurve
from acro.investor_study import (
    InvestorScenario,apply_investor_frictions,
    apply_common_contingent_settlement_stress,investor_target_metrics)
from run_research_v517 import summarize_q,grouped_ce_ci

ROOT=Path(__file__).resolve().parents[1]
class TestV517(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=load_project(ROOT/'config.json')
        cls.p=replace(p,initial_coupon_policy='fixed',
            initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
            simulation=replace(p.simulation,steps_per_year=12,
                quote_paths=100,conditional_paths=100,outer_paths=100))
        cls.curve=ConditionalCurve(np.array([-3.,3.]),np.array([91.,91.]),
                                    np.array([90.,90.]),100)
    def record(self):
        log,fac=simulate_full_market(self.p,120,10,4001)
        return compare_terminal_wealth(self.p,log,fac,self.curve,coupon=.18,
            no_call_coupon=.15,premium=.7,fixed_amount=1.8,initial_note_price=99.7)

    def test_zero_failure_same_as_original_no_failure(self):
        rec=self.record();s=InvestorScenario('none',0,0,0.,10,5)
        a=apply_investor_frictions(rec,rate=.025,premium=.7,
                scenario=s,original_nominal=100,seed=11)
        b=apply_common_contingent_settlement_stress(rec,rate=.025,premium=.7,
                scenario=s,original_nominal=100,seed=11)
        for k in a['wealth']:
            np.testing.assert_allclose(a['wealth'][k],b['wealth'][k],atol=1e-10)

    def test_full_failure_removes_fixed_payment_and_acro_right(self):
        rec=self.record();s=InvestorScenario('full',0,0,1.,10,5)
        a=apply_common_contingent_settlement_stress(rec,rate=.025,premium=.7,
            scenario=s,original_nominal=100,seed=11)
        base=apply_investor_frictions(rec,rate=.025,premium=.7,
            scenario=replace(s,option_delivery_failure_probability=0),
            original_nominal=100,seed=11)
        yc=rec['call_year']; H=rec['horizon']; el=rec['eligible']
        np.testing.assert_allclose(base['wealth']['fixed']-a['wealth']['fixed'],
           np.where(el,1.8*np.exp(.025*(H-yc)),0))
        self.assertEqual(int(np.sum(a['fixed_payment_failed'])),int(np.sum(el)))
        np.testing.assert_array_equal(a['delivery_failed'],a['fixed_payment_failed'])
        np.testing.assert_array_equal(a['wealth']['reserve'],base['wealth']['reserve'])

    def test_symmetric_failure_does_not_alter_no_call(self):
        rec=self.record();s=InvestorScenario('x',0,0,0.5,10,5)
        a=apply_common_contingent_settlement_stress(rec,rate=.025,premium=.7,
            scenario=s,original_nominal=100,seed=11)
        b=apply_investor_frictions(rec,rate=.025,premium=.7,
            scenario=replace(s,option_delivery_failure_probability=0),
            original_nominal=100,seed=11)
        np.testing.assert_array_equal(a['wealth']['no_call'],b['wealth']['no_call'])

    def test_target_metrics_and_q_summary(self):
        z=investor_target_metrics([100,99,101],budget=100,rate=0,horizon=1,
          target_annual_growth=0)
        self.assertAlmostEqual(z['mean_pv'],100)
        self.assertAlmostEqual(z['p_missing_target'],1/3)
        rec=self.record();q=summarize_q(rec,rate=.025,F=1.8)
        self.assertTrue(np.isfinite(q['pv_wealth_gap']))
        self.assertGreaterEqual(q['eligible'],0)

    def test_bank_bootstrap_finite(self):
        z=grouped_ce_ci([np.array([95.,100.,101.]),
                         np.array([96.,99.,100.])],budget=100,rate=0,
                        horizon=5,boots=30)
        self.assertTrue(np.isfinite(z['value']))
        self.assertEqual(len(z['bank_bootstrap95']),2)

if __name__=='__main__':unittest.main()
