"""Regression tests for audit findings of the portfolio UI release."""
import csv
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

from acro_gui import run_cancelable_process


class FakeChild:
    def __init__(self):
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def communicate(self):
        if self.returncode is None:
            self.returncode = 0
        return ('model stdout', '')


class PublicationAuditTests(unittest.TestCase):
    def test_cancel_before_popen_never_starts(self):
        event=threading.Event();event.set()
        def forbidden(*args, **kwargs):
            raise AssertionError('Popen called despite early cancellation')
        state, output = run_cancelable_process(['x'], '.', event, popen_factory=forbidden)
        self.assertEqual(state,'cancelled')

    def test_cancel_in_popen_window_is_honored(self):
        event=threading.Event();made=[]
        def delayed(*args, **kwargs):
            event.set()  # User presses Stop while Popen itself is not yet returned.
            child=FakeChild();made.append(child)
            return child
        state,_=run_cancelable_process(['x'],'.',event,popen_factory=delayed)
        self.assertEqual(state,'cancelled')
        self.assertTrue(made[0].terminated)

    def test_cancel_after_process_spawn_is_honored(self):
        event=threading.Event();made=[]
        def build(*args, **kwargs):
            child=FakeChild();made.append(child)
            return child
        def on_started(child):
            event.set()
        state,_=run_cancelable_process(['x'],'.',event,popen_factory=build,on_started=on_started)
        self.assertEqual(state,'cancelled')
        self.assertTrue(made[0].terminated)

    def test_unchanged_exit_stays_successful(self):
        event=threading.Event()
        state,output=run_cancelable_process(['x'],'.',event,popen_factory=lambda *a,**k: FakeChild())
        self.assertEqual((state,output),('success','model stdout'))

    def test_report_label_is_conditional(self):
        text=(Path(__file__).resolve().parents[1]/'run_demo.py').read_text()
        self.assertIn("if gbm.zero_gbm_identity_applies",text)
        self.assertIn('zero identity not applicable',text)

    def test_illustrative_export_metadata(self):
        text=(Path(__file__).resolve().parents[1]/'acro'/'portfolio_visuals.py').read_text()
        for name in ['logvol_factor_at_call','renewal_value_approx','curve_factor_clipped']:
            self.assertIn(name,text)

    def test_public_readme_has_no_local_path(self):
        text = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
        self.assertNotRegex(text, r'/Users/[^/\s]+/')
        self.assertIn('LAUNCH_ACRO_GUI.py', text)
        self.assertIn('results_local/', text)

if __name__=='__main__':
    unittest.main()
