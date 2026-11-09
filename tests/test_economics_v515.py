"""Cross-check economic scenario engine against unchanged original payoff engine."""
import unittest
from dataclasses import replace
from unittest.mock import patch
import numpy as np
from pathlib import Path
from acro import load_project,simulate_phoenix
from acro.pricing import ConditionalCurve
from acro.live_risk import LiveState
from acro.dynamic_hedge import run_scenario_hedge
from acro.economic_study import (simulate_full_market,contract_future_wealth,
                                 fair_fixed_payment,equal_budget_contract_shortfalls,
                                 compare_terminal_wealth,summarize_economics)

ROOT=Path(__file__).resolve().parents[1]

class TestEconomics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=load_project(ROOT/'config.json')
        p=replace(p,simulation=replace(p.simulation,steps_per_year=12,
                  quote_paths=100,conditional_paths=100,outer_paths=100),
                  initial_coupon_policy='fixed',
                  initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18))
        cls.p=p

    def test_simulated_original_payoffs_match_engine_exactly(self):
        p=self.p;n=128;seed=4123
        log,_=simulate_full_market(p,n,5.,seed)
        w=contract_future_wealth(log,p.initial_phoenix,12,p.market.risk_free_rate,
                                 np.zeros(n,dtype=int))
        engine=simulate_phoenix(p.market,p.initial_phoenix,n,12,seed)
        growth=np.exp(p.market.risk_free_rate*5.)
        np.testing.assert_allclose(w.B/growth,engine.principal_pv,atol=1e-9)
        np.testing.assert_allclose(w.A/growth,engine.coupon_annuity_pv,atol=1e-9)
        calls=np.full(n,-1,dtype=int)
        good=np.isfinite(engine.first_call_year)
        calls[good]=np.rint(engine.first_call_year[good]*12).astype(int)
        np.testing.assert_array_equal(w.call_step,calls)

    def test_memory_and_barrier_match_with_same_innovations(self):
        p=self.p
        for memory,monitoring in [(True,'european'),(False,'continuous')]:
            t=replace(p.initial_phoenix,memory=memory,capital_barrier_monitoring=monitoring)
            logs,_=simulate_full_market(p,80,5.,9305)
            v=contract_future_wealth(logs,t,12,p.market.risk_free_rate,np.zeros(80,dtype=int))
            ref=simulate_phoenix(p.market,t,80,12,9305)
            np.testing.assert_allclose(v.B*np.exp(-.025*5),ref.principal_pv,atol=1e-9)
            np.testing.assert_allclose(v.A*np.exp(-.025*5),ref.coupon_annuity_pv,atol=1e-9)

    def test_fixed_equal_q_price_and_shortfall(self):
        self.assertAlmostEqual(fair_fixed_payment(.8,.4),2.)
        y=equal_budget_contract_shortfalls(np.array([0.,1.,3.,4.]),
             np.array([False,True,True,True]),np.array([0.,1.,2.,3.]),
             .8,2.,rate=0.)
        np.testing.assert_allclose(y['acro'],np.zeros(4))
        np.testing.assert_allclose(y['fixed'],[0.,0.,1.,2.])
        np.testing.assert_allclose(y['reserve'],[0.,.2,2.2,3.2])

    def test_budget_and_reinvestment_payoffs_in_trivial_option(self):
        p=self.p
        m=replace(p.market,dynamics='gbm')
        a=replace(p.initial_phoenix,maturity_years=2,first_autocall_year=1)
        new=replace(a,coupon_rate=.18)
        p=replace(p,market=m,initial_phoenix=a,renewal_phoenix=new,
                  acro=replace(p.acro,last_eligible_call_year=1))
        # V_R(c=18%) = 82+18=100, so nobody has a positive ACRO payoff.
        cv=ConditionalCurve(np.array([-2.,2.]),np.array([82.,82.]),np.array([100.,100.]),100)
        log,x=simulate_full_market(p,60,4.,110)
        z=compare_terminal_wealth(p,log,x,cv,coupon=.18,no_call_coupon=.15,
                                  premium=0.,fixed_amount=0.,initial_note_price=100.)
        np.testing.assert_allclose(z['wealth']['acro'],z['wealth']['fixed'],atol=1e-10)
        np.testing.assert_allclose(z['wealth']['acro'],z['wealth']['reserve'],atol=1e-10)
        self.assertTrue(np.isfinite(z['wealth']['no_call']).all())
        self.assertEqual(len(summarize_economics(z,rate=.025)),4)

    def test_fair_renewal_affine_in_coupon(self):
        p=self.p
        cv=ConditionalCurve(np.array([-2.,2.]),np.array([91.,91.]),np.array([50.,50.]),100)
        log,x=simulate_full_market(p,100,10.,42)
        v=compare_terminal_wealth(p,log,x,cv,coupon=.18,no_call_coupon=.15,
                                  premium=.7,fixed_amount=1.8,initial_note_price=99.7)
        called=v['eligible']
        # Vnew(18%) = 100. Every eligible call can refuse the contract;
        # if fair coupon is also 18%, exercise is indifferent.
        np.testing.assert_allclose(v['cost'][called],0.,atol=1e-9)
        # With no exercise, ACRO note has the same reset note as fixed.
        # Fixed adds a nonnegative cash indemnity where calls are eligible.
        self.assertTrue(np.all(v['wealth']['fixed']>=v['wealth']['acro']-1e-9))

    def test_equal_budget_wealth_scales_with_nominal(self):
        p = self.p
        cv = ConditionalCurve(np.array([-2., 2.]), np.array([90., 90.]),
                              np.array([55., 55.]), 100)
        log, factors = simulate_full_market(p, 55, 10., 8742)
        base = compare_terminal_wealth(p, log, factors, cv, coupon=.18,
                     no_call_coupon=.15, premium=.7, fixed_amount=2., initial_note_price=100.)
        scale = 3.
        p300 = replace(p, initial_phoenix=replace(p.initial_phoenix, nominal=300.),
                       renewal_phoenix=replace(p.renewal_phoenix, nominal=300.),
                       acro=replace(p.acro, strike=300.))
        cv300 = ConditionalCurve(cv.factors, cv.principal*scale,
                                 cv.annuity*scale, 100)
        bigger = compare_terminal_wealth(p300, log, factors, cv300, coupon=.18,
                     no_call_coupon=.15, premium=2.1, fixed_amount=6., initial_note_price=300.)
        for strategy in ('acro', 'fixed', 'reserve', 'no_call'):
            np.testing.assert_allclose(bigger['wealth'][strategy],
                                       base['wealth'][strategy]*scale, atol=1e-8)

    def test_unsupported_terms_are_rejected_not_silently_mispriced(self):
        p=replace(self.p,acro=replace(self.p.acro,strike=105.))
        cv=ConditionalCurve(np.array([-2.,2.]),np.array([90.,90.]),np.array([50.,50.]),100)
        log,x=simulate_full_market(p,15,10.,42)
        with self.assertRaises(ValueError):
            compare_terminal_wealth(p,log,x,cv,coupon=.18,no_call_coupon=.15,
                                    premium=.7,fixed_amount=2.,initial_note_price=100.)

    def test_terminal_hedge_marks_without_unnecessary_bumps(self):
        p=replace(self.p,initial_phoenix=replace(self.p.initial_phoenix,autocall_barrier=2.))
        st=LiveState(1.,(.7,.8,.75),.05)
        curve=ConditionalCurve(np.array([-2.,2.]),np.array([90.,90.]),np.array([10.,10.]),100)
        import acro.dynamic_hedge as h
        orig=h._prices_deltas
        def fail_if_terminal(p,state,*args,**kwargs):
            if state.time_years>=2.-1e-9:raise AssertionError('Unneeded terminal delta')
            return orig(p,state,*args,**kwargs)
        with patch.object(h,'_prices_deltas',side_effect=fail_if_terminal):
            results,_,_=run_scenario_hedge(p,st,end_time=2.,scenario_seed=201,
                valuation_seed=302,curve=curve,rebalance_every_steps=3,
                n_paths=60,bump=.01,fee_bps=5.)
        self.assertEqual(len(results),9)

if __name__=='__main__':unittest.main()
