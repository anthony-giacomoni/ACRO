import unittest
import numpy as np
from research.audit_precision_v54 import (ci, factor_grid, jensen_bound, note_pv_max,
                                 validate_scope, budgets)
from acro import load_project
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

class V54AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=load_project(ROOT/'config.json')
    def test_scope(self):
        validate_scope(self.p)
    def test_jensen_bound_decays(self):
        self.assertAlmostEqual(jensen_bound(200,100),10)
        self.assertAlmostEqual(jensen_bound(200,400),5)
    def test_ci_constant_replications(self):
        r=ci([1,1,1]); self.assertEqual(r['ci95'],[1.,1.])
    def test_bad_ci(self):
        with self.assertRaises(ValueError): ci([1])
    def test_factor_grid_extends_states(self):
        nodes=factor_grid([np.array([-.1,.3])],5,0.)
        self.assertGreater(nodes.max(),.3)
        self.assertLess(nodes.min(),-.1)
        self.assertTrue(np.any(abs(nodes)<1e-14))
    def test_pv_bound_positive(self):
        self.assertGreater(note_pv_max(self.p,.2),self.p.renewal_phoenix.nominal)
    def test_budget_nested_positive(self):
        for mode in ['quick','standard','high']:
            b=budgets(mode)
            self.assertEqual(b['direct_inner']%2,0)
            self.assertGreaterEqual(b['nodes'],3)
if __name__=='__main__': unittest.main()

class V54FinancialControls(unittest.TestCase):
    """Mini deterministic checks on valuation and paired quote construction."""
    @classmethod
    def setUpClass(cls):
        cls.p=load_project(ROOT/'config.json')

    def test_par_coupon_from_reference_quote(self):
        from research.audit_precision_v54 import frozen_coupon
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            c,b,a,meta=frozen_coupon(self.p,dict(quote_batches=2,quote_per_batch=150),Path(tmp))
        self.assertAlmostEqual(b+c*a, self.p.initial_phoenix.nominal, places=8)
        self.assertGreater(meta['approx_95_halfwidth'],0)

    def test_coupled_anchor_removes_independent_quote_noise(self):
        from research.audit_coupled import innovations_for_grid, quote_from_bank
        p=self.p; m=int(192*p.initial_phoenix.maturity_years)
        bank=innovations_for_grid(m,125,p.market.n_assets,1007)
        baseB,baseA,_=quote_from_bank(p,bank,192,192)
        refB,refA=83.,75.
        B,A,_=quote_from_bank(p,bank,192,192)
        self.assertAlmostEqual(refB + B-baseB, refB,places=9)
        self.assertAlmostEqual(refA + A-baseA, refA,places=9)

    def test_jensen_bound_dominates_empirical_sigma_under_bounded_pv(self):
        from research.audit_precision_v54 import note_pv_max, jensen_bound
        from acro import simulate_phoenix
        p=self.p; c=.18
        paths=simulate_phoenix(p.market,p.renewal_phoenix,256,48,1234)
        pv=paths.pv(c)
        M=note_pv_max(p,c)
        self.assertGreaterEqual(pv.min(),0.)
        self.assertLessEqual(pv.max(),M+1e-9)
        self.assertLessEqual(pv.std(ddof=0)/np.sqrt(len(pv)),jensen_bound(M,len(pv))+1e-12)

class V54CrossfitTests(unittest.TestCase):
    def test_crossfit_both_in_money(self):
        from research.audit_precision_v54 import crossfit_bounds
        self.assertEqual(crossfit_bounds(120,110,100),(15.,15.))
    def test_crossfit_both_out_money(self):
        from research.audit_precision_v54 import crossfit_bounds
        self.assertEqual(crossfit_bounds(80,90,100),(0.,0.))
    def test_crossfit_mixed_can_have_negative_lower(self):
        from research.audit_precision_v54 import crossfit_bounds
        low,up=crossfit_bounds(110,90,100)
        self.assertEqual(low,-5.)
        self.assertEqual(up,0.)
    def test_crossfit_symmetric(self):
        from research.audit_precision_v54 import crossfit_bounds
        for a,d in [(70,110),(100,102),(90,120),(150,50)]:
            self.assertEqual(crossfit_bounds(a,d,100),crossfit_bounds(d,a,100))
