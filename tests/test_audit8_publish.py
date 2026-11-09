"""Regression checks for external 2026-11-08 audit findings.

These tests use small simulation budgets and do not certify the market price,
Monte Carlo inner uncertainty or continuous-time convergence.
"""
from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

from acro.config import load_project
from acro.analysis import factor_correlation_scenarios
from acro.controlled_pricing import price_acro_controlled
from run_demo import run

ROOT=Path(__file__).resolve().parents[1]


class Audit8PublicationFixes(unittest.TestCase):
    def test_reference_rho_uses_same_controlled_quote(self):
        p=load_project(ROOT/'config.json')
        p=replace(p,simulation=replace(p.simulation,quote_paths=128,outer_paths=128,
                                       conditional_paths=64,curve_nodes=5))
        r,_=price_acro_controlled(p,ref_paths=128,scrambles=3)
        rows=factor_correlation_scenarios(p,r.quote.coupon_rate,values=(-0.55,),
                     pricing_method='gbm_cv',ref_paths=128,scrambles=3,base_result=r)
        base=[row for row in rows if row['rho']==p.market.equity_factor_correlation]
        self.assertEqual(len(base),1)
        self.assertAlmostEqual(base[0]['premium'],r.price,places=12)
        self.assertEqual(base[0]['method'],'gbm_cv')
        self.assertEqual(len(rows),2)
        self.assertTrue(all(row['premium']>=0 for row in rows))

    def test_rho_reference_recomputed_without_injected_result(self):
        # Catches accidental reversion to the old legacy pricer.
        p = load_project(ROOT/'config.json')
        p = replace(p, simulation=replace(p.simulation, quote_paths=256,
                    outer_paths=256, conditional_paths=128, curve_nodes=5))
        r, _ = price_acro_controlled(p, ref_paths=128, scrambles=3)
        rows = factor_correlation_scenarios(
            p, r.quote.coupon_rate, values=(-.55,), pricing_method='gbm_cv',
            ref_paths=128, scrambles=3, base_result=None)
        at_base = [z for z in rows if abs(z['rho'] - p.market.equity_factor_correlation) < 1e-12]
        self.assertEqual(len(at_base), 1)
        self.assertAlmostEqual(at_base[0]['premium'], r.price, places=9)

    def test_gui_monetary_comparison_requires_same_nominal_and_policy(self):
        from acro_gui import comparable_run_snapshots
        base = {'initial_nominal':100.,'initial_coupon_policy':'fair','acro_price':0.8}
        same = dict(base,acro_price=0.9)
        bigger = dict(base,initial_nominal=300.,acro_price=2.4)
        fixed = dict(base,initial_coupon_policy='fixed')
        self.assertTrue(comparable_run_snapshots(same,base))
        self.assertFalse(comparable_run_snapshots(bigger,base))
        self.assertFalse(comparable_run_snapshots(fixed,base))
        self.assertFalse(comparable_run_snapshots(base,None))

    def test_scaled_contract_uses_actual_nominal_in_report(self):
        cfg=json.loads((ROOT/'config.json').read_text(encoding='utf8'))
        cfg['initial_phoenix']['nominal']=300
        cfg['renewal_phoenix']['nominal']=300
        cfg['acro']['strike']=300
        with tempfile.TemporaryDirectory() as temp:
            tc=Path(temp)/'config.json';tc.write_text(json.dumps(cfg),encoding='utf8')
            out=Path(temp)/'results'
            result=run(config=tc,output=out,quick=True,sensitivity=False,pricing_method='auto')
            self.assertEqual(result['initial_nominal'],300)
            self.assertIn('initial nominal 300 €',(out/'summary.md').read_text())
            # 9 legacy plots + the financial risk waterfall (figure 11).
            self.assertEqual(len(list(out.glob('*.png'))),10)
            self.assertTrue((out/'11_risque_capital_et_coupon.png').exists())
            self.assertGreater(result['acro_price'],2.)

    def test_cli_defaults_to_same_pricer_as_spyder(self):
        from inspect import signature
        self.assertEqual(signature(run).parameters['pricing_method'].default,'auto')
        self.assertEqual(signature(__import__('run_demo')._run_unstaged).parameters['pricing_method'].default,'auto')

if __name__=='__main__':
    unittest.main()
