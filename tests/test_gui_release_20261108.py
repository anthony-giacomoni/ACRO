"""Portable regressions for public GUI release; no Tk display required."""
import tempfile
import unittest
from pathlib import Path
from acro_gui import validate_custom_coupon_selection, available_figures


class FinalGUIReleaseRegressions(unittest.TestCase):
    def test_checkbox_cannot_enable_empty_target_coupon(self):
        with self.assertRaisesRegex(ValueError, 'Coupon ACRO cible activé'):
            validate_custom_coupon_selection(True, '')
        with self.assertRaisesRegex(ValueError, 'Coupon ACRO cible activé'):
            validate_custom_coupon_selection(True, '   ')
        self.assertIsNone(validate_custom_coupon_selection(True, '19.5'))
        self.assertIsNone(validate_custom_coupon_selection(False, ''))

    def test_optional_rho_chart_appears_only_if_exported(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td)
            for name in ('01_carte_payoff_worstof_vol.png',
                         '08_trajectoires_monte_carlo.png'):
                (folder / name).touch()
            names = [name for name, _ in available_figures(folder)]
            self.assertEqual(len(names), 2)
            (folder / '06_sensibilite_dependance.png').touch()
            names = [name for name, _ in available_figures(folder)]
            self.assertEqual(len(names), 3)
            self.assertIn('06_sensibilite_dependance.png', names)
