"""Targeted invariants for optional static-factor ACRO stress-test research."""
import unittest
from dataclasses import replace
from pathlib import Path
import numpy as np
from acro.config import load_project
from acro.controlled_pricing import price_acro_controlled
from acro.pricing import price_acro
from study_acro_market_risks import BASE_COUPON,scenarios,summarize,comparisons

ROOT=Path(__file__).resolve().parents[1]

class MarketSensitivityStudyTests(unittest.TestCase):
    def setUp(self):
        self.p=load_project(ROOT/'config.json')

    def test_correlations_are_positive_definite_and_keep_diagonal(self):
        for name,family,market in scenarios(self.p.market):
            self.assertEqual(self.p.market.n_assets,market.n_assets)
            np.linalg.cholesky(np.asarray(market.correlations,dtype=float))
            np.testing.assert_allclose(np.diag(market.correlations),1.)

    def test_each_shock_changes_only_one_parameter_family(self):
        baseline=self.p.market
        for name,family,m in scenarios(baseline):
            for field,group in [('volatilities','vol'),('correlations','corr'),
                                ('dividends','div'),('risk_free_rate','rate')]:
                if group!=family:
                    self.assertEqual(getattr(m,field),getattr(baseline,field))
            if family=='base': self.assertEqual(m,baseline)

    def test_aggregate_and_paired_differences(self):
        rows=[]
        for seed in (1,2):
            rows.append({'seed':seed,'name':'Référence','family':'base','fair_coupon_pct':18.,
                         'fair_option_eur':0.8,'fixed_option_eur':0.7,'fixed_initial_note_mtm_eur':100.,
                         'call_probability_pct':40.})
            rows.append({'seed':seed,'name':'Corr +','family':'corr','fair_coupon_pct':17.,
                         'fair_option_eur':0.9,'fixed_option_eur':1.5,'fixed_initial_note_mtm_eur':102.,
                         'call_probability_pct':45.})
        a=summarize(rows)
        d=comparisons(rows,a)
        self.assertEqual(d[0]['fixed_option_eur_vs_base'],0.8)
        self.assertAlmostEqual(d[0]['fair_option_eur_vs_base'],0.1)
        self.assertEqual(d[0]['fixed_option_eur_diff_se'],0.)

    def test_fixed_coupon_reprices_payoff_without_reusing_fair_coupon(self):
        s=replace(self.p.simulation,quote_paths=400,outer_paths=400,curve_nodes=5,conditional_paths=200)
        p=replace(self.p,simulation=s,initial_coupon_policy='fair')
        result,_=price_acro_controlled(p,ref_paths=256,scrambles=2)
        q=result.quote
        qp=replace(q,coupon_rate=BASE_COUPON,price=q.principal+BASE_COUPON*q.annuity)
        fixedp=replace(p,initial_coupon_policy='fixed',initial_phoenix=replace(p.initial_phoenix,coupon_rate=BASE_COUPON))
        fixed=price_acro(fixedp,quote=qp,curve_override=result.curve)
        self.assertAlmostEqual(fixed.quote.coupon_rate,BASE_COUPON)
        self.assertAlmostEqual(fixed.target_coupon,BASE_COUPON)
        self.assertAlmostEqual(fixed.quote.price,q.principal+BASE_COUPON*q.annuity)
        self.assertTrue(np.isfinite(fixed.price))
        self.assertTrue(np.all(fixed.discounted_payoff>=0.))

if __name__=='__main__': unittest.main()
