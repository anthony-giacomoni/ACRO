"""Small smoke/invariant tests for the final research release itself."""
import unittest
import numpy as np
from research.audit_final_grids import quantile_summary
from acro import load_project
from pathlib import Path

class ReleaseFinalTests(unittest.TestCase):
    def test_bank_interval_contains_sample_mean(self):
        mean,se,ci=quantile_summary([0.1,0.2,0.3,0.2])
        self.assertAlmostEqual(mean,0.2)
        self.assertGreater(se,0)
        self.assertLess(ci[0],mean)
        self.assertGreater(ci[1],mean)

    def test_project_configuration_valid(self):
        p=load_project(Path(__file__).resolve().parents[1]/'config.json')
        p.validate()
        self.assertEqual(p.market.n_assets,3)

    def test_public_release_files_are_present(self):
        root = Path(__file__).resolve().parents[1]
        for file in ['README.md', 'LICENSE', 'config.json', 'START_HERE.py',
                     'notes/MODEL_NOTES.md', 'notes/NUMERICAL_CHECKS.md',
                     'docs/ACRO_interface.png']:
            self.assertTrue((root / file).is_file(), file)

if __name__=='__main__': unittest.main()
