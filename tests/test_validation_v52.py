"""Fast focused tests of the NEW audit mechanics, with no market calibration."""
import unittest
import numpy as np
from dataclasses import replace
from acro import simulate_phoenix
from research.audit_pricing_v52 import conditional_direct_values, describe_replicates
from research.audit_coupled import innovations_for_grid, aggregate_innovations, price_from_components
from tests.test_acro import make


class TestV52Audit(unittest.TestCase):
    def test_common_brownian_bank_aggregation(self):
        fine = innovations_for_grid(24, 20, 2, 93000)
        c12 = aggregate_innovations(fine, 24, 12, 1.)
        c24 = aggregate_innovations(fine, 24, 24, 1.)
        np.testing.assert_allclose(c12[0]*np.sqrt(2),c24[0][0::2]+c24[0][1::2])
        np.testing.assert_allclose(c12[1]*np.sqrt(2),c24[1][0::2]+c24[1][1::2])

    def test_fixed_coupon_parameter_used_unaltered(self):
        p=make(dynamics='gbm')
        n=40
        outer=simulate_phoenix(p.market,p.initial_phoenix,n,12,981)
        mask=np.isfinite(outer.first_call_year)
        nodes=np.array([-1.,0.,1.]); zero=np.zeros(3)
        base=(99.,5.,.20)
        a=price_from_components(p,base,outer,mask,nodes,zero,zero)
        b=price_from_components(p,(99.,5.,.40),outer,mask,nodes,zero,zero)
        self.assertGreaterEqual(b,a)

    def test_direct_raw_and_anchored_at_same_factor(self):
        p=make(dynamics='gbm')
        x=p.market.initial_logvol_factor
        raw,anchored,h1,h2=conditional_direct_values(p,x,.2,200,12,811,anchor=100.)
        self.assertAlmostEqual(anchored,100.,places=12)
        self.assertAlmostEqual(raw,.5*(h1+h2),places=12)

    def test_jensen_halves_nonnegative(self):
        a,b=95.,105.; strike=100.
        self.assertGreaterEqual(.5*(max(a-strike,0)+max(b-strike,0))-
                                max(.5*(a+b)-strike,0),0)

    def test_independent_replication_statistics(self):
        d=describe_replicates([.7,.75,.8,.85])
        self.assertEqual(d['n'],4)
        self.assertLess(d['ci95_statistical'][0],d['mean'])
        self.assertGreater(d['ci95_statistical'][1],d['mean'])
        with self.assertRaises(ValueError): describe_replicates([.7])

if __name__=='__main__':unittest.main()
