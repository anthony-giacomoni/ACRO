"""UI exercise options: independent educational sample size and valid correlations."""
import tempfile
import unittest
from pathlib import Path

from acro_gui import (ILLUSTRATIVE_PATH_CHOICES, check_correlation_triple,
                      correlation_interval, correlation_limits)


class CorrelationUISupportTests(unittest.TestCase):
    def test_third_interval_from_two(self):
        low, high = correlation_interval(.50, .35)
        self.assertAlmostEqual(low, .50*.35-(.75*.8775)**.5)
        self.assertAlmostEqual(high, .50*.35+(.75*.8775)**.5)
        self.assertTrue(low < .45 < high)

    def test_all_three_visible_bounds(self):
        bounds = correlation_limits(.5, .35, .45)
        for pair, rng in bounds.items():
            self.assertLess(rng[0], rng[1])
        self.assertEqual(len(bounds), 3)
        self.assertIsInstance(check_correlation_triple(.5, .35, .45), dict)

    def test_bad_combination_fails_with_a_range(self):
        with self.assertRaisesRegex(ValueError, 'Plage autorisée'):
            check_correlation_triple(.9, .9, -.9)
        with self.assertRaisesRegex(ValueError, 'strictement'):
            check_correlation_triple(1, .25, .25)
        with self.assertRaises(ValueError):
            correlation_interval(float('nan'), .2)

    def test_allowed_visual_counts_include_all_requested(self):
        self.assertEqual(ILLUSTRATIVE_PATH_CHOICES, (48, 100, 500, 1000))


if __name__ == '__main__':
    unittest.main()
