"""Fast analytical tests for V5.3 audit mechanics; no commercial-price claims."""
import unittest
import numpy as np
from research.audit_convergence_v53 import sample_hierarchy, mean_ci, gate, params


class TestV53Audit(unittest.TestCase):
    def test_jensen_nonnegative_for_mixed_signs(self):
        r=sample_hierarchy(np.array([90.,90.,110.,110.,90.,90.,110.,110.]),
                           .15,100.,[4,8])
        self.assertAlmostEqual(r[4]['intrinsic'],0.)
        self.assertAlmostEqual(r[4]['jensen_half_full'],5.)
        self.assertGreaterEqual(r[8]['jensen_half_full'],0.)

    def test_constant_path_no_jensen(self):
        r=sample_hierarchy(np.full(16,105.),.11,100.,[4,8,16])
        for n in [4,8,16]:
            self.assertAlmostEqual(r[n]['intrinsic'],5.)
            self.assertAlmostEqual(r[n]['jensen_half_full'],0.)
            self.assertAlmostEqual(r[n]['conditional_se'],0.)

    def test_inner_prefix_preserved(self):
        pv=np.r_[np.full(4,120.),np.full(4,80.)]
        r=sample_hierarchy(pv,.10,100.,[4,8])
        self.assertAlmostEqual(r[4]['intrinsic'],20.)
        self.assertAlmostEqual(r[8]['intrinsic'],0.)

    def test_odd_or_insufficient_budget_rejected(self):
        with self.assertRaises(ValueError):sample_hierarchy(np.arange(9.),.1,100.,[9])
        with self.assertRaises(ValueError):sample_hierarchy(np.arange(4.),.1,100.,[8])

    def test_ci_symmetric_and_finite(self):
        c=mean_ci([.70,.75,.80,.85])
        self.assertAlmostEqual(c['mean'],.775)
        self.assertAlmostEqual(sum(c['ci95'])/2.,.775)
        self.assertGreater(c['halfwidth'],0.)

    def test_precision_gate_refuses_large_uncertainty(self):
        stats=lambda avg,w:{'mean':avg,'halfwidth':w}
        nested={'inner_levels':[256,512],
                'prices_by_budget':{'512':{'price':stats(.8,.06),
                  'bias_proxy':stats(.02,.01),'jensen_half_full':stats(.01,.01)}},
                'paired_differences':{'512_minus_256':stats(-.015,.03)}}
        grid={'paired_differences':{'delta_96_minus_48':stats(.005,.03)}}
        g=gate(nested,grid,.02)
        self.assertFalse(g['meets_practical_gate'])
        self.assertEqual(g['target_eur_per_100'],.02)

    def test_presets_nested_and_grid(self):
        for k in ('quick','standard','high'):
            c=params(k)
            self.assertEqual(c['inner_levels'],sorted(c['inner_levels']))
            self.assertTrue(all(n%2==0 for n in c['inner_levels']))
            self.assertGreaterEqual(c['grid_reps'],3)


if __name__ == '__main__': unittest.main()
