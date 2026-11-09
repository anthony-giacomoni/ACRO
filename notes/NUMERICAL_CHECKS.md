# Reproducibility and checks

The repository includes a numerical/regression test suite. The latest release tests passed with `pytest` (247 tests plus four subtests). A passing suite means the implementation matches the tested assumptions and regression cases; it does not imply a validated market price.

To run the checks from the project directory:

```bash
python -m pytest -q
```

For a quick end-to-end run with the reference configuration:

```bash
python run_demo.py --quick --no-sensitivity --output results_local
```

This generates `metrics.json`, `run_config.json`, a Markdown summary, exported CSVs and PNG figures. The output directory is ignored by Git. The quick mode reduces the number of simulated paths and is noisier than larger numerical studies.

What the checks cover includes Phoenix cash-flow and coupon logic, autocall events, basket rules for one to six equities, admissible correlations, paired simulation/reference identities, Monte Carlo estimators, numerical diagnostics, and GUI/run-output behaviour.

Some more computationally expensive and exploratory scripts are kept in `research/`. They are useful for investigating model behaviour but are not required to launch the application. The annual-correlation exercise should be considered experimental and its results should not be substituted for the reference model quote.

A good next step would be full market calibration followed by independent benchmark valuations and more systematic numerical-error budgets. Those are not claimed in this release.
