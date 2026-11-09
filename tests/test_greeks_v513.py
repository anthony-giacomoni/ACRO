import unittest
from dataclasses import replace
import numpy as np
from pathlib import Path
from acro import load_project
from acro.live_risk import LiveState
from acro.greek_validation import paired_greek_triplet,live_greek_bank
ROOT=Path(__file__).resolve().parents[1]

class GreekValidation(unittest.TestCase):
    def test_exact_quadratic(self):
        # Finite difference exact for quadratic and linear functions.
        h=.02;s=100.;n=40
        down=np.tile(np.array([4.,10.,14.])[:,None],(1,n))
        cen=np.tile(np.array([5.,12.,17.])[:,None],(1,n))
        up=np.tile(np.array([6.4,14.3,20.7])[:,None],(1,n))
        vals=paired_greek_triplet(down,cen,up,s,h)
        self.assertAlmostEqual(vals[0][0][0],(6.4-4.)/4.,places=12)
        self.assertAlmostEqual(vals[0][1][0],(.4)/4.,places=12)
        self.assertAlmostEqual(vals[2][0][0],vals[0][0][0]+vals[1][0][0],places=12)
        self.assertAlmostEqual(vals[2][1][0],vals[0][1][0]+vals[1][1][0],places=12)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            paired_greek_triplet(np.zeros((3,3)),np.zeros((3,3)),np.zeros((3,3)),100,0)
    def test_live_greeks_are_reproducible_and_additive(self):
        p=load_project(ROOT/'config.json')
        p=replace(p,initial_coupon_policy='fixed',
                  initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
                  simulation=replace(p.simulation,steps_per_year=12,conditional_paths=96,curve_nodes=5))
        state=LiveState(1.,(.88,.92,.84),.05)
        kwargs=dict(bumps=(.01,),n_paths=100,outer_seed=90001,curve_seed=50001,
                    ref_paths=64,scrambles=2)
        a=live_greek_bank(p,state,**kwargs)
        b=live_greek_bank(p,state,**kwargs)
        self.assertEqual(a,b)
        self.assertEqual(len(a),3*p.market.n_assets)
        for i in range(p.market.n_assets):
            group=a[3*i:3*i+3]
            self.assertAlmostEqual(group[2]['delta'],group[0]['delta']+group[1]['delta'],places=10)
            self.assertAlmostEqual(group[2]['gamma'],group[0]['gamma']+group[1]['gamma'],places=10)
    def test_barrier_state_bumps_do_not_mutate_issue_references(self):
        p=load_project(ROOT/'config.json')
        p=replace(p,initial_coupon_policy='fixed',
                  initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
                  simulation=replace(p.simulation,steps_per_year=24,conditional_paths=96,curve_nodes=5))
        state=LiveState(2-1/24,(.995,1.02,1.03),.05)
        state.validate(p)
        initial=p.market.initial_spots
        got=live_greek_bank(p,state,bumps=(.01,),n_paths=90,ref_paths=64,scrambles=2)
        self.assertEqual(initial,p.market.initial_spots)
        self.assertTrue(all(np.isfinite(r['gamma']) for r in got))


    def test_same_date_comparator_keeps_all_other_factors(self):
        from research.audit_greeks_v513 import scenarios,fix_coupon
        p=load_project(ROOT/'config.json')
        p=fix_coupon(p,steps=24,inner=96,nodes=5)
        d=scenarios(p)
        low=d['Avant coupon (worst-of ≈ barrière)']
        high=d['Même date (worst-of +8 points)']
        self.assertAlmostEqual(low.time_years,high.time_years)
        self.assertEqual(low.logvol_factor,high.logvol_factor)
        self.assertEqual(low.spot_ratios[1:],high.spot_ratios[1:])
        self.assertAlmostEqual(high.spot_ratios[0]-low.spot_ratios[0],.081)

if __name__=='__main__':unittest.main()
