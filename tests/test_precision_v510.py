import unittest
import numpy as np
from dataclasses import replace
from acro import load_project
from acro.engine import Paths
from acro.precision_bounds import CurveDraw, crossfit_payoffs, independent_curve_draw
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def curve(nodes,values):
    return CurveDraw(np.array(nodes,dtype=float),np.array(values,dtype=float),.1,100,np.array([.2]*len(nodes)))


def outer(years,factors):
    n=len(years)
    return Paths(np.zeros(n),np.zeros(n),np.array(years,float),np.array(factors,float),
                 np.full(n,np.nan),np.full(n,np.nan))

class TestPrecisionV510(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=load_project(ROOT/'config.json')

    def test_equal_curves_exact_no_jensen_gap(self):
        a=curve([-1,1],[110,90]); b=curve([-1,1],[110,90])
        lo,up=crossfit_payoffs(self.p,outer([1,1,2],[0.,.5,-.5]),a,b)
        self.assertTrue(np.allclose(lo,up))
        self.assertTrue(np.all(up>=0.))

    def test_opposite_exercise_has_strict_gap(self):
        a=curve([-1,1],[108,108]); b=curve([-1,1],[92,92])
        lo,up=crossfit_payoffs(self.p,outer([2],[0.]),a,b)
        self.assertGreater(up[0]-lo[0],0.)
        self.assertLess(lo[0],0.)

    def test_no_call_or_ineligible_zero(self):
        a=curve([-1,1],[130,130]); b=curve([-1,1],[125,125])
        lo,up=crossfit_payoffs(self.p,outer([np.nan,5.],[np.nan,0.]),a,b)
        self.assertTrue(np.allclose(lo,0.))
        self.assertTrue(np.allclose(up,0.))

    def test_out_of_grid_is_error(self):
        a=curve([-1,1],[110,90]); b=curve([-1,1],[110,90])
        with self.assertRaises(ValueError):
            crossfit_payoffs(self.p,outer([1],[2.]),a,b)

    def test_different_nodes_rejected(self):
        a=curve([-1,1],[110,90]); b=curve([-.9,1],[110,90])
        with self.assertRaises(ValueError):
            crossfit_payoffs(self.p,outer([1],[0.]),a,b)

    def test_independent_draw_reproducible(self):
        p=self.p
        nodes=np.array([-.5,0.,.5])
        a=independent_curve_draw(p,nodes,.18,12,32,64,2,7731)
        b=independent_curve_draw(p,nodes,.18,12,32,64,2,7731)
        self.assertTrue(np.array_equal(a.values,b.values))
        self.assertTrue(np.isfinite(a.reference_se))
        self.assertTrue((a.delta_node_se>=0).all())

if __name__=='__main__': unittest.main()

class TestNumbaSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p=load_project(ROOT/'config.json')

    def test_fast_matches_numpy_multiple_contracts(self):
        from acro.fast_engine import simulate_phoenix_fast
        from acro.engine import simulate_phoenix
        from acro.conditional_qmc import innovation_bank
        for model in ('gbm','stochastic_vol'):
            for memory in (False,True):
                for continuous in (False,True):
                    market=replace(self.p.market,dynamics=model)
                    t=replace(self.p.initial_phoenix,memory=memory,
                              capital_barrier_monitoring='continuous' if continuous else 'european')
                    steps=12;n=96
                    bank=innovation_bank(int(t.maturity_years*steps),n,market.n_assets,
                                         7717,'mc_bridge')
                    old=__import__('acro.engine',fromlist=['simulate_phoenix']).simulate_phoenix(
                        market,t,n,steps,0,initial_factor=.2,innovations=bank)
                    fast=simulate_phoenix_fast(market,t,n,steps,initial_factor=.2,innovations=bank)
                    for field in ('principal_pv','coupon_annuity_pv','first_call_year',
                                  'factor_at_call','final_worst','first_call_worst',
                                  'stopped_stock_martingales'):
                        a=getattr(old,field);b=getattr(fast,field)
                        self.assertTrue(np.allclose(a,b,equal_nan=True,atol=1e-10,rtol=1e-10),
                                        (model,memory,continuous,field))

    def test_fast_curve_matches_numpy_reference_exact(self):
        p=load_project(ROOT/'config.json')
        nodes=[-.5,0.,.5]
        slow=independent_curve_draw(p,nodes,.18,12,96,128,2,11732,False)
        fast=independent_curve_draw(p,nodes,.18,12,96,128,2,11732,True)
        self.assertTrue(np.allclose(slow.values,fast.values,atol=1e-10))
        self.assertTrue(np.allclose(slow.delta_node_se,fast.delta_node_se,atol=1e-10))
