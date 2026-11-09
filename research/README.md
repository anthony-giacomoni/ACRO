# Research scripts

I kept the longer numerical studies separate from the GUI and the core pricer. Some filenames have version numbers because these experiments were added at different stages of development.

They cover benchmark valuations, time-grid and interpolation checks, control variates, Greeks, economic scenarios, hedging experiments and an exploratory annual-correlation model. Not every experiment is part of the standard ACRO quote.

Start with `run_demo.py` at the repository root. For the regular tests, use `python -m pytest -q`. If you want to inspect a study script, run it as a module from the repository root, for example:

```bash
python -m research.study_annual_correlations --help
```

These are research diagnostics, not independent verification or market calibration.
