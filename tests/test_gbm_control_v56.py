import unittest
from dataclasses import replace
import numpy as np

from acro import load_project,simulate_phoenix
from acro.conditional_qmc import innovation_bank, bridge_innovations
from acro.gbm_control import (GBMReference,gbm_observation_grid,
                              aggregated_bank, gbm_reference,renewal_curve_gbm_cv)
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class TestGBMSurrogateV56(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=load_project(ROOT/'config.json')

    def test_gbm_fine_vs_coupon_only_identical_payoff(self):
        p=self.p
        fine=48
        n=1024
        inn=innovation_bank(int(p.renewal_phoenix.maturity_years*fine),n,p.market.n_assets,9287,'mc_bridge')
        m=replace(p.market,dynamics='gbm')
        v_fine=simulate_phoenix(m,p.renewal_phoenix,n,fine,0,innovations=inn).pv(.18)
        sparse=gbm_observation_grid(p,fine)
        coarsened=aggregated_bank(inn,fine,sparse)
        v_coupon=simulate_phoenix(m,p.renewal_phoenix,n,sparse,0,innovations=coarsened).pv(.18)
        np.testing.assert_allclose(v_fine,v_coupon,rtol=0,atol=1e-10)

    def test_control_vanishes_when_stochastic_factor_has_zero_vol(self):
        p=self.p
        m=replace(p.market,factor_volatility=0.,factor_mean=0.,initial_logvol_factor=0.)
        p=replace(p,market=m)
        n=600
        inn=innovation_bank(int(p.renewal_phoenix.maturity_years*48),n,p.market.n_assets,5160,'mc_bridge')
        raw,control,vr,vd,_,_=renewal_curve_gbm_cv(p,.18,48,np.array([0.]),inn,100.)
        self.assertAlmostEqual(control[0],100.,places=10)
        self.assertLess(vd[0],1e-17)
        self.assertGreater(vr[0],0.)

    def test_gbm_reference_seed_reproducible_and_finite(self):
        a=gbm_reference(self.p,.18,48,paths_per_scramble=256,scrambles=4,base_seed=7281)
        b=gbm_reference(self.p,.18,48,paths_per_scramble=256,scrambles=4,base_seed=7281)
        self.assertEqual(a.value,b.value)
        self.assertEqual(a.std_error,b.std_error)
        self.assertGreaterEqual(a.std_error,0.)
        self.assertLess(a.ci95[0],a.ci95[1])

    def test_bridge_reproducible_and_same_moments(self):
        n=2048
        eq,fac=innovation_bank(10,n,3,73002,'sobol_bridge')
        eq2,fac2=innovation_bank(10,n,3,73002,'sobol_bridge')
        np.testing.assert_array_equal(eq,eq2)
        np.testing.assert_array_equal(fac,fac2)
        self.assertEqual(eq.shape,(10,n,3))
        self.assertLess(abs(eq.mean()),0.03)
        self.assertLess(abs(eq.std()-1.),0.03)

    def test_no_nonpower_of_two_sobol(self):
        with self.assertRaises(ValueError):
            innovation_bank(8,1000,3,29,'sobol_bridge')

    def test_sparse_reference_excludes_continuous_barrier(self):
        p=replace(self.p,renewal_phoenix=replace(self.p.renewal_phoenix,
                    capital_barrier_monitoring='continuous'))
        with self.assertRaises(ValueError):
            gbm_observation_grid(p,48)

    def test_common_gbm_baseline_across_factor_values(self):
        p=self.p
        inn=innovation_bank(240,300,p.market.n_assets,884,'mc_bridge')
        a,b,_,_,_,_=renewal_curve_gbm_cv(p,.18,48,np.array([-.2,.0,.2]),inn,100.)
        self.assertEqual(len(b),3)
        self.assertTrue(np.isfinite(b).all())
        # For identical innovation bank and a fixed GBM estimate, the only
        # difference is the SV(x)-GBM pathwise mean.
        self.assertTrue(np.allclose(b-a,np.full(len(b),b[1]-a[1]),atol=1e-10))

if __name__=='__main__':unittest.main()
