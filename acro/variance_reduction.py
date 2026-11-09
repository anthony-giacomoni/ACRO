"""Exact-mean discounted stopped-stock martingale control variates.

The risk-neutral stock process is a martingale after normalization by
exp((r-q)t), including at any bounded stopping time. Controls therefore have
known expectation 1 and can reduce both note quoting and conditional renewal
MC error without modifying any product payoff. Coefficients MUST be trained on
independent paths or cross-fitted to avoid in-sample fitting bias.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .engine import Paths

@dataclass
class ControlResult:
    estimate: float
    std_error: float
    naive: float
    naive_std_error: float
    coefficients: np.ndarray
    variance_ratio: float
    training_paths: int
    evaluation_paths: int


def estimate_coefficients(y, controls, ridge=1e-8):
    y=np.asarray(y,float); x=np.asarray(controls,float)
    if x.ndim!=2 or y.ndim!=1 or len(y)!=len(x) or len(y)<3:
        raise ValueError('Training y/controls dimensions invalid')
    centered=x-1.0
    # Train centered regression with an intercept. The intercept is NOT used
    # to adjust the known control mean; only the coefficients are transferred.
    centered=centered-centered.mean(axis=0,keepdims=True)
    yc=y-y.mean()
    return np.linalg.solve(centered.T@centered+ridge*np.eye(x.shape[1]),centered.T@yc)


def controlled_mean(y_train, m_train, y_eval, m_eval):
    """Unbiased for E[y] conditional on independent regression coefficients.

    Works for a note with coupon rate fixed contractually, a par coupon's
    principal/annuity components, or any payoff with controls of known mean 1.
    """
    a=np.asarray(y_eval,float); x=np.asarray(m_eval,float)
    b=estimate_coefficients(y_train,m_train)
    if x.ndim!=2 or len(a)<2 or x.shape!=(len(a),len(b)):
        raise ValueError('Evaluation dimensions invalid')
    adjusted=a-(x-1.0)@b
    se=float(adjusted.std(ddof=1)/np.sqrt(len(adjusted)))
    old=float(a.std(ddof=1)/np.sqrt(len(a)))
    return ControlResult(float(adjusted.mean()),se,float(a.mean()),old,b,
                         (se/old)**2 if old>0 else 0.,len(y_train),len(a))


def control_phoenix(paths_train:Paths,paths_eval:Paths,coupon:float):
    if paths_train.stopped_stock_martingales is None or paths_eval.stopped_stock_martingales is None:
        raise ValueError('Simulation must provide stopped stock controls')
    return controlled_mean(paths_train.pv(coupon),paths_train.stopped_stock_martingales,
                           paths_eval.pv(coupon),paths_eval.stopped_stock_martingales)
