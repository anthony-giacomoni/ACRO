import unittest
import numpy as np
from research.audit_temporal_robuste import paired_interval,temporal_verdict,GRIDS,COUPON
class TemporalAuditTests(unittest.TestCase):
    def test_policy_fixed(self):
        self.assertEqual(GRIDS,(96,192,384));self.assertEqual(COUPON,.18)
    def test_interval_constant(self):
        r=paired_interval([.2,.2,.2,.2])
        self.assertAlmostEqual(r['mean'],.2)
        self.assertAlmostEqual(r['half_width_95'],0)
    def test_interval_rejects_too_few_banks(self):
        with self.assertRaises(ValueError):paired_interval([1.,2.])
    def test_verdict_not_certified_even_if_stable(self):
        d={'ci95':[-.005,.006]}
        v=temporal_verdict(d,d)
        self.assertTrue(v['grid_stability_criterion_passed'])
        self.assertFalse(v['pricing_certified'])
    def test_verdict_rejects_unresolved_difference(self):
        d={'ci95':[-.05,.01]}; v=temporal_verdict(d,d)
        self.assertFalse(v['grid_stability_criterion_passed'])
if __name__=='__main__':unittest.main()
