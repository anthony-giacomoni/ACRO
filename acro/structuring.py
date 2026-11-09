"""ACRO structuring economics, using already-simulated paths and curve.

A packaged note sold at par contains *two* assets for the investor:
A(c) + ACRO(c). Its coupon therefore has to be lower than the standalone
Phoenix coupon. Pricing here is a model-based fair-value decomposition,
not a bank quotation, and makes no claim of commercial profitability.
"""
from dataclasses import dataclass
from scipy.optimize import brentq
import numpy as np
from .config import Project
from .pricing import ACROResult


@dataclass(frozen=True)
class BundledQuote:
    standalone_coupon: float
    bundled_coupon: float
    coupon_haircut_bp: float
    acro_premium_at_standalone_coupon: float
    bundled_acro_premium: float
    bundled_initial_note_value: float
    combined_value: float
    standalone_note_annuity: float
    feasible: bool


def price_at_target_coupon(result: ACROResult, project: Project, coupon_rate: float) -> float:
    """Reprice ACRO payoff with same market states and renewed-note curve.

    This is exact within the precomputed conditional-curve approximation,
    provided the renewal coupon does NOT change event times/barriers.
    """
    r = result
    b = np.zeros_like(r.discounted_payoff)
    eligible = r.eligible
    if eligible.any():
        B, A = r.curve.at(r.logvol_at_call[eligible])
        c = (project.acro.target_coupon_rate if project.acro.target_coupon_rate is not None
             else coupon_rate)
        intrinsic = project.acro.renewal_units * np.maximum(B + c*A - project.acro.strike,0)
        b[eligible] = np.exp(-project.market.risk_free_rate*r.call_years[eligible])*intrinsic
    return float(b.mean())


def solve_bundled_par_coupon(project: Project, result: ACROResult) -> BundledQuote:
    """Solve initial note + option = issue price, on the SAME simulated paths.

    Only a non-negative bundled coupon is considered. When the extra right is
    too expensive for that funding budget, mark as infeasible; do not invent
    a negative coupon or an arbitrarily adjusted price.
    """
    q=result.quote
    nominal=project.initial_phoenix.nominal
    if q.annuity <= 0:
        raise ValueError('Initial note has no coupon annuity')
    fair_c=(nominal-q.principal)/q.annuity
    premium_at_fair=price_at_target_coupon(result,project,fair_c)
    def excess(c):
        return q.principal+c*q.annuity+price_at_target_coupon(result,project,c)-nominal
    if fair_c < 0 or excess(0)>1e-9:
        return BundledQuote(fair_c, float('nan'),float('nan'),premium_at_fair,
                            float('nan'),float('nan'),float('nan'),q.annuity,False)
    bundled=float(brentq(excess,0,max(fair_c,0)+1e-12,xtol=1e-12))
    option=price_at_target_coupon(result,project,bundled)
    principal_value=q.principal+bundled*q.annuity
    return BundledQuote(float(fair_c),bundled,1e4*(fair_c-bundled),
                        premium_at_fair,option,float(principal_value),
                        float(principal_value+option),q.annuity,True)
