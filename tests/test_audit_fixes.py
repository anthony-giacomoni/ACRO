"""Regression tests for defects confirmed by the independent portfolio audit."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from acro import load_project
from acro.config import Market, Simulation
from acro.controlled_pricing import controlled_quote
from acro.pricing import ConditionalCurve, price_acro
from acro.dynamic_hedge import run_scenario_hedge
from acro.live_risk import LiveState
from run_demo import run

ROOT=Path(__file__).resolve().parents[1]

class AuditCorrections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=load_project(ROOT/'config.json')
        cls.p=replace(p,initial_coupon_policy='fixed',
                      initial_phoenix=replace(p.initial_phoenix,coupon_rate=.18),
                      simulation=replace(p.simulation,steps_per_year=12,
                            quote_paths=128,outer_paths=750,conditional_paths=128,curve_nodes=5))

    def test_par_coupon_vs_exercise_threshold_different_strike(self):
        p=replace(self.p,acro=replace(self.p.acro,strike=90))
        crv=ConditionalCurve(np.array([-10.,10.]),np.array([93.,93.]),
                             np.array([10.,10.]),128)
        r=price_acro(p,curve_override=crv)
        self.assertTrue(np.any(r.eligible))
        self.assertTrue(np.allclose(r.fair_renewal_coupon[r.eligible],.7))
        self.assertTrue(np.allclose(r.exercise_coupon_threshold[r.eligible],-.3))
        self.assertTrue(np.allclose(r.renewal_value[r.eligible],94.8))

    def test_nan_and_inf_rejected(self):
        m=self.p.market
        for changed in [dict(risk_free_rate=float('nan')),
                        dict(dividends=(.01,float('inf'),.02)),
                        dict(volatilities=(.2,float('nan'),.3)),
                        dict(correlations=((1.,.2,float('nan')),(.2,1.,.2),(.2,.2,1.)) )]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                replace(m,**changed)
        with self.assertRaises(ValueError):
            replace(self.p.acro,strike=float('inf')).validate()
        with self.assertRaises(ValueError):
            replace(self.p.initial_phoenix,coupon_barrier=float('nan')).validate(12)
        with self.assertRaises(ValueError):
            Simulation(seed=-1).validate()

    def test_failure_keeps_previous_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp)
            cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
            cfg['market']['dynamics']='gbm'
            source=t/'gbm.json';source.write_text(json.dumps(cfg),encoding='utf-8')
            output=t/'results';output.mkdir()
            keep=output/'old_figure.png';keep.write_bytes(b'original-PNG-bytes')
            with self.assertRaises(ValueError):
                run(config=source,output=output,quick=True,sensitivity=False,pricing_method='gbm_cv')
            self.assertEqual(keep.read_bytes(),b'original-PNG-bytes')
            self.assertEqual(set(x.name for x in output.iterdir()),{'old_figure.png'})

    def test_auto_routes_gbm_and_preserves_unrelated_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            t=Path(tmp)
            cfg=json.loads((ROOT/'config.json').read_text(encoding='utf-8'))
            cfg['market']['dynamics']='gbm'
            source=t/'gbm.json';source.write_text(json.dumps(cfg),encoding='utf-8')
            output=t/'results';output.mkdir()
            (output/'personal_notes.txt').write_text('keep me',encoding='utf-8')
            r=run(config=source,output=output,quick=True,
                  sensitivity=False,pricing_method='auto')
            self.assertEqual(r['pricing_method'],'legacy')
            self.assertTrue((output/'metrics.json').exists())
            self.assertEqual((output/'personal_notes.txt').read_text(),'keep me')
            effective=json.loads((output/'effective_config.json').read_text())
            self.assertEqual(effective['simulation']['quote_paths'],2500)
            self.assertEqual(effective['simulation']['outer_paths'],1800)
            self.assertEqual(len(list(output.glob('0[1-7]_*.png'))),6)

    def test_both_seeds_control_initial_quote(self):
        a=replace(self.p,simulation=replace(self.p.simulation,seed=19701))
        b=replace(self.p,simulation=replace(self.p.simulation,seed=19702))
        qa,_,_=controlled_quote(a,ref_paths=128,scrambles=2)
        qb,_,_=controlled_quote(b,ref_paths=128,scrambles=2)
        qa2,_,_=controlled_quote(a,ref_paths=128,scrambles=2)
        self.assertEqual((qa.principal,qa.annuity),(qa2.principal,qa2.annuity))
        self.assertNotEqual((qa.principal,qa.annuity),(qb.principal,qb.annuity))

    def test_surviving_terminal_mark_uses_independent_curve(self):
        c=ConditionalCurve(np.array([-4.,4.]),np.array([105.,105.]),
                           np.array([10.,10.]),128)
        c1=ConditionalCurve(np.array([-4.,4.]),np.array([90.,90.]),
                           np.array([10.,10.]),128)
        c2=ConditionalCurve(np.array([-4.,4.]),np.array([128.,128.]),
                           np.array([10.,10.]),128)
        kwargs=dict(p=self.p,initial=LiveState(1.,(.88,.92,.84),0.),end_time=1.25,
                    scenario_seed=4096,valuation_seed=3044,curve=c,
                    rebalance_every_steps=3,n_paths=100,bump=.01,fee_bps=5.)
        r1,t1,h1=run_scenario_hedge(**kwargs,settlement_curve=c1)
        r2,t2,h2=run_scenario_hedge(**kwargs,settlement_curve=c2)
        self.assertTrue(h1[-1].alive)
        self.assertEqual(t1,t2)
        self.assertEqual(h1,h2)
        def error(rows,component):
            return next(r['pnl_excess'] for r in rows if r['component']==component and r['method']=='dynamic')
        self.assertNotAlmostEqual(error(r1,'package'),error(r2,'package'))
        self.assertAlmostEqual(error(r1,'phoenix'),error(r2,'phoenix'))

if __name__=='__main__':unittest.main()
