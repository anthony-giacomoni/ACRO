# Reproducibility and numerical checks

The project is a **synthetic, uncalibrated research pricer**, not a trading quote. The main branch runs its tests and lightweight pricing checks through GitHub Actions: [Python tests and pricing smoke checks](../.github/workflows/tests.yml). A passing check supports reproducibility of the tested implementation, **not market-price accuracy**.

## Test and demo

From the repository root, with Python 3.11 and the packages in `requirements.txt`:

```bash
python -m pytest -q
python run_demo.py --quick --no-sensitivity --output results_local
```

The local suite passed 247 tests plus four subtests before publication of the validation additions. CI runs the suite from a fresh environment and checks that the headless pricer generates finite metrics and output files. It does not launch the Tkinter GUI.

## Reproduce a premium diagnostic

```bash
python -m research.reproduce_premium --quick --output results_premium_validation
# For larger, more expensive banks:
python -m research.reproduce_premium --standard --output results_premium_validation
```

The protocol writes `summary.json`, per-bank CSVs and a short `README.md` explaining its numbers. Each bank has an independently seeded estimator.

The script performs **two different experiments**:

1. **Fair-coupon controlled pricing:** re-estimates the original Phoenix par coupon in every independent bank, then values ACRO with a conditional renewal-value curve. It compares two simulation budgets.
2. **Fixed-18%-coupon direct checks:** holds the coupon at 18% and uses independently estimated direct renewal values at selected autocall states (exercise-policy lower/upper diagnostics). It compares 12 and 48 steps/year in quick mode, and adds 96 steps/year in standard mode.

**Do not compare the premiums of those two experiments as if they were the same contract.** The first changes the coupon with the simulated model quote; the second fixes it. The direct cross-grid experiments are not Brownian-coupled. Lower/upper labels refer to the *expected* exercise-policy bounds under the discrete-model assumptions, not guaranteed sample-by-sample or confidence bounds.

### Illustrative quick-mode execution (9 November 2026)

These are low-budget, deliberately noisy results from the synthetic model, **not target values for test assertions**:

| Experiment | Budget / grid | Mean premium (EUR/100) | Approximate interval |
|---|---|---:|---|
| Fair coupon | Small | 0.799 | [0.686, 0.912] |
| Fair coupon | Large | 0.784 | [0.686, 0.881] |
| Fixed 18% | 12 steps/year | midpoint 0.665 | [0.121, 1.133] |
| Fixed 18% | 48 steps/year | midpoint 0.621 | [0.356, 0.835] |

The fair-coupon means here use four independently seeded banks per budget; the fixed-coupon intervals combine estimated lower and upper bank intervals, also with four independent banks per grid. The broad fixed-coupon intervals show that the **quick protocol cannot establish a narrow grid error**. These figures are *not* evidence of pricing stability to one or two cents.

## What is checked—and what is not

Tests exercise Phoenix coupon and redemption logic, autocall conditions, basket rules for one to six equities, admissible correlations, option invariants, paired simulation/reference identities and regression behaviour. The reproducible premium protocol additionally illustrates sensitivity to simulation budget and direct evaluation of the exercise decision.

The between-bank intervals do not bound **all** inner Monte Carlo noise, conditional interpolation error, time discretization or model specification. No live volatility surface, correlation calibration, issuer credit, funding or hedgeability validation is claimed. The annual-correlation exercise remains experimental. The historical fixed-coupon benchmark near EUR 0.70/100 and the fair-coupon demo near EUR 0.78/100 are **not interchangeable**.
