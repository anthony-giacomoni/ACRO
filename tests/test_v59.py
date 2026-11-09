import unittest
import numpy as np
from acro.renewal_anchor import grid_including_anchor
from research.audit_anchor_v59 import interpolate_anchored

class TestV59(unittest.TestCase):
    def test_grid_exact_anchor_default(self):
        for count in (19,21):
            x=grid_including_anchor(-1.4,1.0,count,0.0)
            self.assertEqual(len(x),count)
            self.assertTrue(np.any(x==0.0))
            self.assertTrue(np.all(np.diff(x)>0))
            self.assertEqual(x[0],-1.4)
            self.assertEqual(x[-1],1.0)
    def test_grid_nonzero_anchor(self):
        x=grid_including_anchor(-1.4,1.0,19,-0.15)
        self.assertTrue(np.any(x==-0.15))
    def test_grid_reject_outside(self):
        with self.assertRaises(ValueError):
            grid_including_anchor(-1,1,19,2)
    def test_anchor_exact_interpolation(self):
        x=grid_including_anchor(-1.4,1.0,19,0.)
        y=100+3*x**2
        self.assertEqual(interpolate_anchored(x,y,0.,100.),100.)
    def test_anchor_missing_fails(self):
        x=np.linspace(-1.4,1.0,19)
        y=100+3*x**2
        with self.assertRaises(ValueError):
            interpolate_anchored(x,y,0.,100.)
    def test_wrong_anchor_value_fails(self):
        x=grid_including_anchor(-1.4,1.0,19,0.)
        y=100+3*x**2
        with self.assertRaises(AssertionError):
            interpolate_anchored(x,y,0.,101.)

if __name__=='__main__': unittest.main()
