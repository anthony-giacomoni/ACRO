import unittest
import numpy as np
from pathlib import Path
from acro import load_project
from research.audit_outer_v55 import (
    years_of_eligible_calls, labels_for_years, two_sample_cv,
    call_payoff, report_ci, config_budget,
)
ROOT=Path(__file__).resolve().parents[1]

class V55OuterTests(unittest.TestCase):
    def test_date_labels_and_expiry(self):
        y=np.array([np.nan,1.,2.,4.,5.])
        np.testing.assert_array_equal(labels_for_years(y,(1.,2.,4.)),[0,1,2,3,0])
    def test_reference_eligible_years(self):
        p=load_project(ROOT/'config.json')
        self.assertEqual(years_of_eligible_calls(p),(1.,2.,3.,4.))
    def test_cv_estimator_zero_control_exactly_naive(self):
        p=[0,1,3,5]; l=[0,0,0,0]
        r=two_sample_cv(p,l,[2,3],[0,0],[1,3,5,7],[0,0,0,0],2)
        self.assertAlmostEqual(r['controlled'],r['naive'])
        self.assertAlmostEqual(r['cv_se'],r['naive_se'])
    def test_control_variate_unbiased_when_stratum_means_known(self):
        # pilot frequencies 50/50, all eligible payoffs exactly 2 on type 1
        # and 4 on type 2. Training estimates exactly beta=[0,2,4].
        r=two_sample_cv([2,4]*10,[1,2]*10,[2,4]*5,[1,2]*5,
                        [2,2,4,4,4,4],[1,1,2,2,2,2],2)
        self.assertAlmostEqual(r['controlled'],3.0)
        self.assertAlmostEqual(r['cv_se'],r['beta'][2]/np.sqrt(20)*0+1/np.sqrt(19),delta=.5)
        self.assertAlmostEqual(r['naive'],20/6)
    def test_cv_input_validations(self):
        with self.assertRaises(ValueError):
            two_sample_cv([1,2],[1],[1,2],[1,1],[1,2],[1,1],2)
        with self.assertRaises(ValueError):
            two_sample_cv([1,2],[4,0],[1,2],[1,1],[1,2],[1,1],2)
    def test_payoff_one_in_money(self):
        p=load_project(ROOT/'config.json')
        years=np.array([np.nan,1.,3.])
        factors=np.array([np.nan,0.,.5])
        y,l=call_payoff(p,years,factors,np.array([-.5,0.,.5]),
                        np.array([90.,112.,95.]),(1.,2.,3.,4.))
        np.testing.assert_allclose(y,[0,12*np.exp(-.025),0])
        np.testing.assert_array_equal(l,[0,1,3])
    def test_ci_and_mode(self):
        d=report_ci([.5,.75,.9])
        self.assertTrue(d['ci95'][0]<d['mean']<d['ci95'][1])
        self.assertGreater(config_budget(False)['pilot'],config_budget(True)['pilot'])

class V55MartingaleControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=load_project(ROOT/'config.json')
    def test_stopped_stocks_have_unit_mean_under_q(self):
        from acro import simulate_phoenix
        s=simulate_phoenix(self.p.market,self.p.renewal_phoenix,12000,48,2027)
        control=s.stopped_stock_martingales
        self.assertEqual(control.shape,(12000,3))
        for k in range(3):
            se=control[:,k].std(ddof=1)/np.sqrt(12000)
            self.assertLess(abs(control[:,k].mean()-1),4*se)
    def test_out_of_sample_control_on_numerical_example(self):
        from acro.variance_reduction import controlled_mean
        rng=np.random.default_rng(18)
        train=rng.normal(size=(2000,2))+1.
        ev=rng.normal(size=(5000,2))+1.
        # Financial variable with known expectation 5.
        ytr=5+3*(train[:,0]-1)-2*(train[:,1]-1)
        ye=5+3*(ev[:,0]-1)-2*(ev[:,1]-1)
        res=controlled_mean(ytr,train,ye,ev)
        self.assertAlmostEqual(res.estimate,5,places=7)
        self.assertLess(res.variance_ratio,.01)
    def test_factor_difference_at_initial_state_zero(self):
        from research.audit_outer_v55 import conditional_curve_pair
        x0=self.p.market.initial_logvol_factor
        nodes=np.array([x0-.1,x0,x0+.1])
        a,b=conditional_curve_pair(self.p,48,nodes,120,[31,32],98.5)
        self.assertEqual(a[1],98.5)
        self.assertEqual(b[1],98.5)
    def test_notes_have_unchanged_cash_flows(self):
        from acro import simulate_phoenix
        s=simulate_phoenix(self.p.market,self.p.initial_phoenix,200,48,55)
        self.assertTrue(np.all(np.isfinite(s.principal_pv)))
        self.assertTrue(np.all(np.isfinite(s.coupon_annuity_pv)))
        self.assertTrue(np.all(np.isfinite(s.stopped_stock_martingales)))
