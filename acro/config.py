"""ACRO research inputs. Rates and barrier levels are decimal fractions."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np


def _all_finite(*arrays):
    """Reject NaN/Infinity before propagating nonfinancial simulation outputs."""
    if any(not np.all(np.isfinite(np.asarray(a,dtype=float))) for a in arrays):
        raise ValueError("All numerical market and contract inputs must be finite")


def grid_step(years, per_year):
    if not np.isfinite(float(years)) or not np.isfinite(float(per_year)):
        raise ValueError("Years and grid steps must be finite")
    x = float(years) * per_year
    if not np.isclose(x, round(x), atol=1e-9):
        raise ValueError(f'{years} years is not on the {per_year}/year simulation grid')
    return int(round(x))


@dataclass(frozen=True)
class Market:
    initial_spots: tuple
    volatilities: tuple
    dividends: tuple
    correlations: tuple
    risk_free_rate: float = 0.025
    dynamics: str = 'stochastic_vol'  # or 'gbm'
    initial_logvol_factor: float = 0.0
    factor_mean: float = 0.0
    factor_reversion: float = 1.4
    factor_volatility: float = 0.30
    equity_factor_correlation: float = -0.65

    def __post_init__(self):
        n = len(self.initial_spots)
        if not n or len(self.volatilities) != n or len(self.dividends) != n:
            raise ValueError('Spot, vol and dividend arrays must have equal positive size.')
        _all_finite(self.initial_spots, self.volatilities, self.dividends,
                    self.correlations, (self.risk_free_rate, self.initial_logvol_factor,
                    self.factor_mean, self.factor_reversion, self.factor_volatility,
                    self.equity_factor_correlation))
        if min(self.initial_spots) <= 0 or min(self.volatilities) <= 0:
            raise ValueError('Spots and volatilities must be > 0.')
        if self.dynamics not in ('gbm', 'stochastic_vol'):
            raise ValueError('Unknown dynamics.')
        if self.factor_reversion < 0 or self.factor_volatility < 0 or abs(self.equity_factor_correlation) > 1:
            raise ValueError('Invalid volatility factor parameters.')
        a = np.asarray(self.correlations, dtype=float)
        if a.shape != (n,n) or not np.allclose(a,a.T) or not np.allclose(np.diag(a),1):
            raise ValueError('Correlation matrix must be NxN, symmetric, diagonal 1.')
        try: np.linalg.cholesky(a)
        except np.linalg.LinAlgError as exc: raise ValueError('Correlation matrix not positive definite.') from exc

    @property
    def n_assets(self): return len(self.initial_spots)


@dataclass(frozen=True)
class Phoenix:
    maturity_years: float = 5
    nominal: float = 100
    observations_per_year: int = 2
    coupon_rate: float = 0.12
    coupon_barrier: float = 0.70
    memory: bool = False
    autocall_enabled: bool = True
    first_autocall_year: float = 1
    autocall_every_years: float = 1
    autocall_barrier: float = 1
    capital_barrier: float = .60
    capital_barrier_monitoring: str = 'european'  # 'continuous' = time grid approximate
    basket_rule: str = 'worst_of'  # worst_of, best_of, equal_weight

    def validate(self, per_year):
        _all_finite((self.maturity_years, self.nominal, self.coupon_rate,
                     self.coupon_barrier, self.first_autocall_year,
                     self.autocall_every_years, self.autocall_barrier,
                     self.capital_barrier))
        if self.nominal <= 0 or self.maturity_years <= 0 or self.coupon_rate < 0:
            raise ValueError('Nominal, maturity, coupon invalid.')
        if not isinstance(self.observations_per_year,int) or self.observations_per_year < 1:
            raise ValueError('Coupon frequency must be positive integer.')
        if per_year % self.observations_per_year:
            raise ValueError('Simulation steps/year must be multiple of coupon frequency.')
        maturity_step = grid_step(self.maturity_years,per_year)
        coupon_step = per_year//self.observations_per_year
        if maturity_step % coupon_step:
            raise ValueError('Maturity must coincide with coupon observation.')
        if self.autocall_enabled:
            first = grid_step(self.first_autocall_year,per_year)
            spacing = grid_step(self.autocall_every_years,per_year)
            if first < 1 or spacing < 1 or first % coupon_step or spacing % coupon_step:
                raise ValueError('Autocall dates must coincide with coupon dates.')
        if not (0 <= self.coupon_barrier <= 2 and 0 <= self.capital_barrier <= 1 and
                0 < self.autocall_barrier <= 2):
            raise ValueError('Barriers out of range.')
        if self.capital_barrier_monitoring not in ('european','continuous'):
            raise ValueError('Unknown capital barrier monitoring.')
        if self.basket_rule not in ('worst_of', 'best_of', 'equal_weight'):
            raise ValueError('Unknown basket_rule: worst_of, best_of, equal_weight.')

    def call_steps(self, per_year):
        self.validate(per_year)
        if not self.autocall_enabled: return set()
        return set(range(grid_step(self.first_autocall_year,per_year),
                         grid_step(self.maturity_years,per_year),
                         grid_step(self.autocall_every_years,per_year)))


@dataclass(frozen=True)
class ACRO:
    strike: float = 100.0
    renewal_units: float = 1.0
    last_eligible_call_year: float | None = 4.0
    target_coupon_rate: float | None = None  # None = initial product's coupon
    def validate(self):
        _all_finite((self.strike, self.renewal_units))
        if self.last_eligible_call_year is not None:
            _all_finite((self.last_eligible_call_year,))
        if self.target_coupon_rate is not None:
            _all_finite((self.target_coupon_rate,))
        if self.strike <= 0 or self.renewal_units <= 0:
            raise ValueError('Strike and renewal quantity must be >0.')
        if self.last_eligible_call_year is not None and self.last_eligible_call_year < 0:
            raise ValueError('Invalid option expiry.')
        if self.target_coupon_rate is not None and self.target_coupon_rate < 0:
            raise ValueError('Target coupon cannot be negative.')


@dataclass(frozen=True)
class Simulation:
    steps_per_year: int = 12
    quote_paths: int = 12000
    outer_paths: int = 6000
    conditional_paths: int = 4500
    curve_nodes: int = 11
    seed: int = 20261105
    def validate(self):
        sizes=(self.steps_per_year,self.quote_paths,self.outer_paths,
               self.conditional_paths,self.curve_nodes,self.seed)
        if any(not isinstance(x,(int,np.integer)) or isinstance(x,bool) for x in sizes):
            raise ValueError("Simulation grids, sample counts, and seeds must be integers")
        if self.seed < 0: raise ValueError("Random seed must be non-negative")
        if self.steps_per_year < 1 or min(self.quote_paths, self.outer_paths, self.conditional_paths) < 2 or self.curve_nodes < 3:
            raise ValueError('Simulation sample sizes or curve nodes too small.')


@dataclass(frozen=True)
class Project:
    market: Market
    initial_phoenix: Phoenix
    renewal_phoenix: Phoenix
    acro: ACRO
    simulation: Simulation
    initial_coupon_policy: str = 'fair'  # 'fair': solve coupon for issue at par; or 'fixed'
    def validate(self):
        self.simulation.validate()
        self.initial_phoenix.validate(self.simulation.steps_per_year)
        self.renewal_phoenix.validate(self.simulation.steps_per_year)
        self.acro.validate()
        if self.initial_coupon_policy not in ('fair','fixed'):
            raise ValueError('Choose fair or fixed coupon.')


def load_project(path: str | Path) -> Project:
    with open(path,encoding='utf-8') as f: d=json.load(f)
    m=d['market'].copy()
    for k in ('initial_spots','volatilities','dividends'): m[k]=tuple(m[k])
    m['correlations']=tuple(tuple(row) for row in m['correlations'])
    p=Project(Market(**m),Phoenix(**d['initial_phoenix']),Phoenix(**d['renewal_phoenix']),
              ACRO(**d['acro']),Simulation(**d['simulation']),d.get('initial_coupon_policy','fair'))
    p.validate()
    return p
