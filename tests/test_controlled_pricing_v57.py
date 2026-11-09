import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from acro import load_project
from acro.controlled_pricing import (controlled_quote,controlled_curve,
                                     price_acro_controlled)
from acro.pricing import price_acro

ROOT=Path(__file__).resolve().parents[1]

class TestControlledRoute(unittest.TestCase):
    def setUp(self):
        self.p=load_project(ROOT/'config.json')
        self.p=replace(self.p, simulation=replace(self.p.simulation,
           quote_paths=256,outer_paths=400,conditional_paths=192,curve_nodes=5,steps_per_year=12))

    def test_quote_fair_at_par(self):
        q,inn,info=controlled_quote(self.p,ref_paths=256,scrambles=3)
        self.assertAlmostEqual(q.price,100.,places=8)
        self.assertGreater(q.annuity,0)
        self.assertEqual(inn[0].shape[1],256)
        self.assertGreater(info['gbm_reference_se_at_selected_coupon'],0)

    def test_reproducibility_full_pipeline(self):
        a,da=price_acro_controlled(self.p,ref_paths=256,scrambles=3)
        b,db=price_acro_controlled(self.p,ref_paths=256,scrambles=3)
        self.assertAlmostEqual(a.price,b.price,places=12)
        np.testing.assert_allclose(a.curve.principal,b.curve.principal,atol=0,rtol=0)
        self.assertTrue(np.isfinite(a.price))
        self.assertGreaterEqual(a.price,0.)

    def test_constant_factor_gbm_zero_acro(self):
        # When factor volatility = 0, x stays at 0 and renewal terms = initial.
        m=replace(self.p.market,factor_volatility=0.,initial_logvol_factor=0.,factor_mean=0.)
        p=replace(self.p,market=m)
        r,_=price_acro_controlled(p,ref_paths=256,scrambles=3)
        self.assertAlmostEqual(r.price,0.,places=8)

    def test_paired_repricing_same_quote_and_curve(self):
        r,_=price_acro_controlled(self.p,ref_paths=256,scrambles=3)
        rerun=price_acro(self.p,quote=r.quote,curve_override=r.curve)
        self.assertAlmostEqual(r.price,rerun.price,places=12)
        self.assertAlmostEqual(np.interp(self.p.market.initial_logvol_factor,
                                        r.curve.factors,r.curve.principal)+
             r.quote.coupon_rate*np.interp(self.p.market.initial_logvol_factor,
                                           r.curve.factors,r.curve.annuity),100.,places=8)

    def test_other_terms_and_fixed_coupon(self):
        renew=replace(self.p.renewal_phoenix,capital_barrier=.5)
        initial=replace(self.p.initial_phoenix,coupon_rate=.18)
        p=replace(self.p,renewal_phoenix=renew,initial_phoenix=initial,
                  initial_coupon_policy='fixed')
        r,_=price_acro_controlled(p,ref_paths=256,scrambles=3)
        self.assertAlmostEqual(r.quote.coupon_rate,.18,places=12)
        self.assertTrue(np.isfinite(r.curve.principal).all())
        self.assertGreaterEqual(r.price,0.)

    def test_continuous_barrier_is_not_silently_accepted(self):
        p=replace(self.p,initial_phoenix=replace(self.p.initial_phoenix,
                  capital_barrier_monitoring='continuous'))
        with self.assertRaises(ValueError):
            price_acro_controlled(p,ref_paths=256,scrambles=3)

if __name__=='__main__':unittest.main()
