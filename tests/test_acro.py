import unittest
from dataclasses import replace
import numpy as np
from acro import Market,Phoenix,ACRO,Simulation,Project,simulate_phoenix,quote_initial,price_acro
from acro.pricing import conditional_curve,ConditionalCurve
from acro.structuring import solve_bundled_par_coupon,price_at_target_coupon


def make(dynamics='gbm',n=2,memory=False):
    m=Market(tuple(100 for _ in range(n)),tuple(.24 for _ in range(n)),tuple(.02 for _ in range(n)),
             tuple(tuple(1. if i==j else .3 for j in range(n)) for i in range(n)),dynamics=dynamics)
    t=Phoenix(maturity_years=2,observations_per_year=2,memory=memory,first_autocall_year=1)
    s=Simulation(steps_per_year=12,quote_paths=1800,outer_paths=900,conditional_paths=600,curve_nodes=5,seed=123)
    return Project(m,t,t,ACRO(last_eligible_call_year=1),s)

class TestACRO(unittest.TestCase):
    def test_par_coupon(self):
        self.assertAlmostEqual(quote_initial(make()).price,100.,places=10)
    def test_gbm_par_zero_identity(self):
        v=price_acro(make())
        self.assertTrue(v.zero_gbm_identity_applies)
        self.assertLess(abs(v.price),1e-10)
    def test_fixed_very_high_coupon_positive_premium(self):
        p=make();p=replace(p,acro=replace(p.acro,target_coupon_rate=.8))
        self.assertGreater(price_acro(p).price,0.)
    def test_no_call_zero_acro(self):
        p=make();p=replace(p,initial_phoenix=replace(p.initial_phoenix,autocall_enabled=False))
        self.assertEqual(price_acro(p).price,0.)
    def test_affine_coupon_decomposition(self):
        p=make(); s=simulate_phoenix(p.market,p.initial_phoenix,500,12,44)
        np.testing.assert_allclose(s.pv(.12)-s.pv(.07), .05*s.coupon_annuity_pv,atol=1e-11)
    def test_memory_not_less_on_same_equity_paths(self):
        p=make();x=simulate_phoenix(p.market,p.initial_phoenix,800,12,44)
        y=simulate_phoenix(p.market,replace(p.initial_phoenix,memory=True),800,12,44)
        self.assertTrue(np.all(y.coupon_annuity_pv+1e-9>=x.coupon_annuity_pv))
    def test_monitored_barrier_no_better_than_european(self):
        p=make(); t=replace(p.initial_phoenix,autocall_enabled=False,capital_barrier=.8)
        x=simulate_phoenix(p.market,t,800,12,44)
        y=simulate_phoenix(p.market,replace(t,capital_barrier_monitoring='continuous'),800,12,44)
        self.assertTrue(np.all(y.principal_pv<=x.principal_pv+1e-10))
    def test_custom_four_assets_quarterly(self):
        p=make(n=4); t=replace(p.initial_phoenix,observations_per_year=4,first_autocall_year=.5,
                               autocall_every_years=.5)
        self.assertGreater(quote_initial(replace(p,initial_phoenix=t)).coupon_rate,0)
    def test_invalid_corr(self):
        with self.assertRaises(ValueError):
            Market((100,100),(.2,.2),(.01,.01),((1.,1.3),(1.3,1.)))
    def test_invalid_schedule(self):
        with self.assertRaises(ValueError):
            replace(make().initial_phoenix,first_autocall_year=1/3).validate(12)
    def test_stochastic_vol_fixed_same_price(self):
        r=price_acro(make(dynamics='stochastic_vol'))
        self.assertAlmostEqual(r.discounted_fixed.mean(),r.price,places=10)
    def test_independent_gbm_renewal_reference(self):
        p=make()
        q=quote_initial(p)
        new=simulate_phoenix(p.market,p.renewal_phoenix,3000,12,110331)
        value=new.pv(q.coupon_rate)
        se=np.hypot(value.std(ddof=1)/np.sqrt(len(value)),
                    q.coupon_se*new.coupon_annuity_pv.mean())
        self.assertLess(abs(value.mean()-p.initial_phoenix.nominal),5*se)

    def test_inner_jensen_gap_nonnegative(self):
        p=make(dynamics='stochastic_vol')
        q=quote_initial(p)
        outer=simulate_phoenix(p.market,p.initial_phoenix,
                               p.simulation.outer_paths,12,p.simulation.seed+20)
        x=outer.factor_at_call[np.isfinite(outer.first_call_year)]
        a=conditional_curve(p,x,q,seed_offset=0)
        b=conditional_curve(p,x,q,seed_offset=930115)
        mean=ConditionalCurve(a.factors,.5*(a.principal+b.principal),
                              .5*(a.annuity+b.annuity),a.paths_per_node+b.paths_per_node)
        p1=price_acro(p,q,curve_override=a).price
        p2=price_acro(p,q,curve_override=b).price
        pm=price_acro(p,q,curve_override=mean).price
        self.assertGreaterEqual(.5*(p1+p2)-pm,-1e-10)

    def test_first_call_worst_records_triggering_level(self):
        p=make(dynamics='stochastic_vol')
        r=price_acro(p)
        observed=r.worst_at_call[r.eligible]
        if len(observed):
            self.assertTrue(np.all(np.isfinite(observed)))
            self.assertTrue(np.all(observed + 1e-10 >= p.initial_phoenix.autocall_barrier))
            self.assertTrue(np.all(np.isfinite(r.logvol_at_call[r.eligible])))

    def test_bundled_is_fair_and_lower_coupon(self):
        p=make(dynamics='stochastic_vol')
        r=price_acro(p)
        b=solve_bundled_par_coupon(p,r)
        self.assertTrue(b.feasible)
        self.assertAlmostEqual(b.combined_value,p.initial_phoenix.nominal,places=9)
        self.assertLessEqual(b.bundled_coupon,b.standalone_coupon)
        self.assertAlmostEqual(price_at_target_coupon(r,p,r.target_coupon),r.price,places=8)

    def test_bundle_fixed_target(self):
        p=make(dynamics='stochastic_vol')
        p=replace(p,acro=replace(p.acro,target_coupon_rate=.30))
        r=price_acro(p)
        b=solve_bundled_par_coupon(p,r)
        self.assertTrue(b.feasible)
        self.assertLess(b.bundled_coupon,b.standalone_coupon)
        self.assertAlmostEqual(b.combined_value,p.initial_phoenix.nominal,places=8)

    def test_riskless_non_autocall_cashflows(self):
        m=Market((100.,),(1e-8,),(0.,),((1.,),),risk_free_rate=.02,dynamics='gbm')
        t=Phoenix(maturity_years=2,observations_per_year=2,
                  autocall_enabled=False,coupon_barrier=.7,capital_barrier=.6)
        path=simulate_phoenix(m,t,20,12,73)
        expected_b=100*np.exp(-.02*2)
        expected_a=50*sum(np.exp(-.02*j/2) for j in range(1,5))
        np.testing.assert_allclose(path.principal_pv,expected_b,atol=1e-9)
        np.testing.assert_allclose(path.coupon_annuity_pv,expected_a,atol=1e-9)

    def test_stochastic_factor_zero_motion_zero_acro(self):
        p=make(dynamics='stochastic_vol')
        m=replace(p.market,factor_volatility=0.,initial_logvol_factor=0.,factor_mean=0.)
        r=price_acro(replace(p,market=m))
        self.assertAlmostEqual(r.price,0,places=10)

    def test_strike_increase_cannot_raise_acro_price(self):
        p=make(dynamics='stochastic_vol')
        q=quote_initial(p)
        low=price_acro(replace(p,acro=replace(p.acro,strike=99.)),q)
        high=price_acro(replace(p,acro=replace(p.acro,strike=101.)),q)
        self.assertGreaterEqual(low.price,high.price)

    def test_option_units_linear(self):
        p=make(dynamics='stochastic_vol')
        q=quote_initial(p)
        a=price_acro(p,q)
        b=price_acro(replace(p,acro=replace(p.acro,renewal_units=2.)),q)
        self.assertAlmostEqual(b.price,2*a.price,places=9)

    def test_no_call_graphs_dont_fail(self):
        p=make(dynamics='stochastic_vol')
        p=replace(p,initial_phoenix=replace(p.initial_phoenix,autocall_enabled=False))
        self.assertFalse(price_acro(p).eligible.any())

    def test_same_seed_repeat(self):
        p=make(dynamics='stochastic_vol')
        self.assertEqual(price_acro(p).price,price_acro(p).price)

if __name__=='__main__': unittest.main()
