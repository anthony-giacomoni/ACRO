"""Contractual risk diagnostic tests: cash flows, optional stopping, scaling."""
import unittest
from dataclasses import replace
import numpy as np
from acro.config import load_project
from acro.engine import Paths, simulate_phoenix
from acro.financial_diagnostics import phoenix_risk_diagnostics, draw_risk_decomposition
from pathlib import Path


class FinancialDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.project=load_project(Path(__file__).resolve().parents[1]/'config.json')

    def test_loss_on_terminal_and_not_on_call(self):
        p=self.project; r=p.market.risk_free_rate; T=p.initial_phoenix.maturity_years
        N=p.initial_phoenix.nominal
        principal=np.array([N*np.exp(-r*1), N*.5*np.exp(-r*T), N*np.exp(-r*T),N*np.exp(-r*T)])
        x=Paths(principal,np.array([20.,10.,0.,15.]),np.array([1.,np.nan,np.nan,np.nan]),
                np.full(4,np.nan),np.array([np.nan,.5,.8,.75]),np.full(4,np.nan))
        d=phoenix_risk_diagnostics(x,p.market,p.initial_phoenix,.1)
        self.assertAlmostEqual(d['capital_loss_probability_Q'],.25)
        self.assertAlmostEqual(d['no_autocall_probability_Q'],.75)
        self.assertAlmostEqual(d['capital_loss_probability_given_no_autocall_Q'],1/3)
        self.assertAlmostEqual(d['mean_nominal_loss_if_loss'],50.)
        self.assertAlmostEqual(d['pv_capital_loss'],.125*N*np.exp(-r*T))
        self.assertAlmostEqual(d['decomposition_residual'],0,places=10)
        self.assertAlmostEqual(d['pv_note_uncontrolled'],principal.mean()+.1*45./4.)

    def test_real_simulation_and_scale(self):
        p=self.project
        sim=simulate_phoenix(p.market,p.initial_phoenix,256,p.simulation.steps_per_year,123)
        a=phoenix_risk_diagnostics(sim,p.market,p.initial_phoenix,.18)
        for k in ('capital_loss_probability_Q','no_autocall_probability_Q',
                  'capital_loss_probability_given_no_autocall_Q'):
            self.assertGreaterEqual(a[k],0);self.assertLessEqual(a[k],1)
        self.assertAlmostEqual(a['pv_redemption_before_loss']-a['pv_capital_loss'],
                               a['pv_actual_principal_uncontrolled'],places=8)
        terms=replace(p.initial_phoenix,nominal=300.)
        sim3=simulate_phoenix(p.market,terms,256,p.simulation.steps_per_year,123)
        b=phoenix_risk_diagnostics(sim3,p.market,terms,.18)
        self.assertEqual(a['capital_loss_probability_Q'],b['capital_loss_probability_Q'])
        for k in ('pv_redemption_before_loss','pv_capital_loss','pv_note_uncontrolled','mean_nominal_loss_if_loss'):
            self.assertAlmostEqual(b[k]/a[k],3.,places=8)

    def test_figure_from_diagnostics(self):
        import tempfile
        p=self.project
        sim=simulate_phoenix(p.market,p.initial_phoenix,128,p.simulation.steps_per_year,975)
        d=phoenix_risk_diagnostics(sim,p.market,p.initial_phoenix,.18)
        with tempfile.TemporaryDirectory() as td:
            target=Path(td)/'11_risque_capital_et_coupon.png'
            self.assertEqual(draw_risk_decomposition(d,target),target.name)
            self.assertGreater(target.stat().st_size,10_000)

    def test_no_losses(self):
        p=self.project
        n=3;N=p.initial_phoenix.nominal;T=p.initial_phoenix.maturity_years;r=p.market.risk_free_rate
        x=Paths(np.full(n,N*np.exp(-r*T)),np.zeros(n),np.full(n,np.nan),
                np.full(n,np.nan),np.full(n,.8),np.full(n,np.nan))
        d=phoenix_risk_diagnostics(x,p.market,p.initial_phoenix,.12)
        self.assertEqual(d['capital_loss_probability_Q'],0.)
        self.assertIsNone(d['mean_nominal_loss_if_loss'])

if __name__=='__main__':unittest.main()
