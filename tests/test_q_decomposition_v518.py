import unittest
import numpy as np
from acro.pricing import ConditionalCurve, Quote
from research.audit_q_decomposition_v518 import centered_premium, reanchored_price_at

class TestQFactorialV518(unittest.TestCase):
    def setUp(self):
        self.nodes=np.array([-1.,0.,1.])
        self.curve=ConditionalCurve(self.nodes, np.array([91.,92.,88.]),
            np.array([45.,46.,39.]), 1024)
        self.anchor=Quote(.18, 100.,0.,0.,92.,46.,None)
        self.changed=Quote(.18, 99.,0.,0.,93.,46.,None)

    def test_identity_reanchored_curve(self):
        xs=np.array([-.5,0.,.5]); b,a=self.curve.at(xs)
        out=reanchored_price_at(self.curve,self.anchor,.18,xs,self.anchor)
        np.testing.assert_allclose(out,b+.18*a)

    def test_new_quote_changes_level_not_relative_shape(self):
        xs=np.array([-.5,0.,.5])
        base=reanchored_price_at(self.curve,self.anchor,.18,xs,self.anchor)
        shift=reanchored_price_at(self.curve,self.anchor,.18,xs,self.changed)
        np.testing.assert_allclose(shift-base,np.full(3,1.))

    def test_exact_initial_anchor_and_no_kink(self):
        at0=reanchored_price_at(self.curve,self.anchor,.18,np.array([0]),self.changed)[0]
        self.assertAlmostEqual(at0,93.+.18*46.)
        r=reanchored_price_at(self.curve,self.anchor,.18,np.array([1e-7,-1e-7]),self.changed)
        self.assertLess(abs(r[0]-r[1]),1e-5)

    def test_zero_payoff_outside_call(self):
        z=centered_premium(105.,[0.,0.,0.],[0.,0.5,0.5],strike=100.)
        self.assertAlmostEqual(z,5/3.)
        self.assertEqual(centered_premium(99.,[0.,-1.],[1.,1.]),0.)

    def test_invalid_shapes(self):
        with self.assertRaises(ValueError):centered_premium(100.,[0],[0,1])

    def test_small_factorial_run_and_jensen(self):
        import tempfile
        from dataclasses import replace
        from pathlib import Path
        from acro import load_project
        from research.audit_q_decomposition_v518 import run
        p=load_project(Path(__file__).resolve().parents[1]/'config.json')
        p=replace(p,initial_coupon_policy='fixed',
            initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
            simulation=replace(p.simulation,steps_per_year=12))
        with tempfile.TemporaryDirectory() as d:
            result=run(p,Path(d),n=2,outer_paths=400,
                ref_paths=256,scrambles=3,conditional_paths=256,
                curve_nodes=7,check_banks=2,check_paths=256,seed=51888)
            self.assertLessEqual(result['factorial']['pooled_conditional_price'],
                                 result['factorial']['mean']+1e-10)
            self.assertTrue((Path(d)/'AUDIT_Q_FACTORIEL_V5_18_FR.md').exists())
            self.assertEqual(result['eligible_calls']>0,True)

if __name__=='__main__': unittest.main()
