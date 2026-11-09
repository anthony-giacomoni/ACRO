"""Brownian path-coupling checks for diagnostics; production default remains unchanged."""
import unittest
from dataclasses import replace
import numpy as np
from acro import Market, Phoenix, simulate_phoenix
from research.audit_coupled import innovations_for_grid, aggregate_innovations


class TestBrownianCoupling(unittest.TestCase):
    def test_brownian_coarsening_preserves_standardized_increment(self):
        eq,f=innovations_for_grid(8,23,2,100)
        e2,f2=aggregate_innovations((eq,f),8,2,1)
        np.testing.assert_allclose(e2,np.sum(eq.reshape(2,4,23,2),axis=1)/2)
        np.testing.assert_allclose(f2,np.sum(f.reshape(2,4,23),axis=1)/2)

    def test_gbm_discrete_payoffs_match_under_coupling(self):
        m=Market((100,100),(.25,.27),(.01,.02),((1.,.5),(.5,1.)),dynamics='gbm')
        t=Phoenix(maturity_years=2,observations_per_year=2,first_autocall_year=1)
        bank=innovations_for_grid(96,500,2,5123)
        series=[]
        for s in [12,24,48]:
            inn=aggregate_innovations(bank,48,s,2)
            result=simulate_phoenix(m,t,500,s,513,innovations=inn)
            series.append(result)
        for other in series[1:]:
            np.testing.assert_allclose(series[0].principal_pv,other.principal_pv,atol=1e-10)
            np.testing.assert_allclose(series[0].coupon_annuity_pv,other.coupon_annuity_pv,atol=1e-10)
            np.testing.assert_allclose(series[0].first_call_year,other.first_call_year,equal_nan=True)

    def test_wrong_bank_dimension_rejected(self):
        m=Market((100.,),(.25,),(.01,),((1.,),),dynamics='gbm')
        t=Phoenix(maturity_years=2,observations_per_year=2,first_autocall_year=1)
        with self.assertRaises(ValueError):
            simulate_phoenix(m,t,100,12,1,innovations=(np.zeros((10,100,1)),np.zeros((10,100))))


if __name__=='__main__':
    unittest.main()
