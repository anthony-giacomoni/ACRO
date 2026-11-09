import unittest
from dataclasses import replace
import numpy as np
from pathlib import Path
from acro import load_project
from acro.total_premium_validation import (
    stratified_indices,selected_bounds,direct_values,ci_banks,run_bank,
    difference_corrected_bounds)
ROOT=Path(__file__).resolve().parents[1]

class TestDirectPremiumV519(unittest.TestCase):
    def test_full_census_weights_are_1(self):
        score=np.array([-2.,-1.,-.2,0.,.1,2.,3.])
        ids,w,classes,cnt=stratified_indices(score,7,.5,np.random.default_rng(2))
        self.assertEqual(len(ids),7)
        self.assertTrue(np.all(w==1))
        self.assertEqual(len(set(ids)),7)
        self.assertEqual(cnt.sum(),7)

    def test_stratified_weights_reconstruct_population_means(self):
        rng=np.random.default_rng(5)
        s=np.r_[np.full(10,-2.),np.zeros(6),np.full(14,2.)]
        for _ in range(5):
            ids,w,grp,cnt=stratified_indices(s,15,.5,rng)
            for g in range(3):
                self.assertAlmostEqual(w[grp==g].sum(),cnt[g])
            self.assertEqual(len(set(ids)),15)

    def test_bounds_deterministic_or_no_call(self):
        d=np.array([1.,.9])
        lo,up=selected_bounds([112,104],[112,104],d,[1,1],total_paths=4)
        self.assertAlmostEqual(lo,up)
        self.assertAlmostEqual(lo,(12+4*.9)/4)
        lo,up=selected_bounds([],[],[],[],total_paths=4)
        self.assertEqual(lo,0.);self.assertEqual(up,0.)

    def test_census_cancels_arbitrarily_wrong_pilot(self):
        # Even a severely biased interpolated pilot cancels if all states sampled.
        correct=(0.55,0.60)
        lo,up=difference_corrected_bounds(3.25,3.25,*correct)
        self.assertAlmostEqual(lo,correct[0]);self.assertAlmostEqual(up,correct[1])

    def test_independent_exercise_generates_gap(self):
        lo,up=selected_bounds([108],[92],[.8],[1],total_paths=1)
        self.assertLess(lo,up)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            stratified_indices([-1.,0.,1.],2,.4,np.random.default_rng(0))
        with self.assertRaises(ValueError):
            selected_bounds([1],[2],[1,2],[1],total_paths=3)
        with self.assertRaises(ValueError):
            ci_banks([1])

    def test_direct_values_reproducible_small(self):
        p=load_project(ROOT/'config.json')
        p=replace(p,initial_coupon_policy='fixed',
            initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
            renewal_phoenix=replace(p.renewal_phoenix,coupon_rate=.18))
        a=direct_values(p,np.array([-.2,0.,.3]),.18,inner_paths=32,
                        ref_paths=64,scrambles=2,seed=719,fast=True)
        b=direct_values(p,np.array([-.2,0.,.3]),.18,inner_paths=32,
                        ref_paths=64,scrambles=2,seed=719,fast=True)
        self.assertTrue(np.array_equal(a[0],b[0]))
        self.assertTrue(np.all(a[1]>=0))
        self.assertGreaterEqual(a[2],0)

if __name__=='__main__':unittest.main()
