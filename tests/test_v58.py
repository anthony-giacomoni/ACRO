"""V5.8 checks for independent anchor and equal-budget controls."""
import unittest
from dataclasses import replace
import numpy as np
from pathlib import Path
from acro import load_project,simulate_phoenix
from acro.renewal_anchor import renewal_curve_anchor
from research.audit_variance_v58 import discounted_payoffs, date_control_equal_budget

ROOT=Path(__file__).resolve().parents[1]


class TestV58(unittest.TestCase):
    def test_date_control_constant_cashflow(self):
        # In every class the payout is deterministic; adequate training has
        # seen each class and the estimator recovers pilot call frequency.
        labels=np.tile(np.array([0,1]),100)
        y=3.0*(labels==1)
        out=date_control_equal_budget(y,labels,60,60)
        self.assertAlmostEqual(out['controlled'],1.5,places=12)
        self.assertAlmostEqual(out['plain_all_paths'],1.5,places=12)
        self.assertEqual(out['total_paths'],200)

    def test_date_control_counts_all_paths(self):
        rng=np.random.default_rng(151)
        l=rng.integers(0,5,500)
        y=np.where(l==0,0.,l*rng.random(500))
        out=date_control_equal_budget(y,l,100,100)
        self.assertEqual(out['train']+out['pilot']+out['eval'],500)
        self.assertAlmostEqual(out['plain_all_paths'],float(y.mean()),places=14)
        self.assertTrue(np.isfinite(out['controlled']))

    def test_anchored_curve_reproducible(self):
        p=load_project(ROOT/'config.json')
        p=replace(p,initial_coupon_policy='fixed',
                  acro=replace(p.acro,target_coupon_rate=.18))
        nodes=np.array([-.1,0.,.1])
        args=dict(p=p,coupon=.18,steps=12,nodes=nodes,reference_value=100.,
                  reference_se=.05,anchor_paths=80,relative_paths=80,seed=1234)
        a=renewal_curve_anchor(**args)
        b=renewal_curve_anchor(**args)
        self.assertTrue(np.array_equal(a.prices,b.prices))
        self.assertAlmostEqual(a.prices[1],a.anchor,places=12)
        self.assertAlmostEqual(a.relative_node_se[1],0.,places=12)

    def test_anchored_zero_factor_matches_gbm(self):
        p=load_project(ROOT/'config.json')
        market=replace(p.market,initial_logvol_factor=0.,factor_mean=0.,
                       factor_volatility=0.)
        p=replace(p,market=market)
        x=np.array([0.,.15])
        res=renewal_curve_anchor(p,.18,12,x,99.25,0.01,80,80,12)
        self.assertAlmostEqual(res.anchor,99.25,places=9)
        self.assertTrue(np.isfinite(res.prices).all())

    def test_payoff_no_autocall_is_zero(self):
        p=load_project(ROOT/'config.json')
        state=simulate_phoenix(p.market,p.initial_phoenix,300,12,123)
        state.first_call_year[:]=np.nan
        y,labels=discounted_payoffs(p,state,np.array([-1.,1.]),np.array([80.,120.]))
        self.assertTrue(np.all(y==0))
        self.assertTrue(np.all(labels==0))

if __name__=='__main__':unittest.main()
