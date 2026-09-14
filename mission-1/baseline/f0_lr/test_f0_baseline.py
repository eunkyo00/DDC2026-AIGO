"""Small deterministic checks for F0 filtering and Train-only preprocessing."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

MODULE = Path(__file__).with_name("run_f0_baseline.py")
SPEC = importlib.util.spec_from_file_location("run_f0_baseline", MODULE)
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


class TestFeatures(unittest.TestCase):
    def test_speech_range_tone_has_f0(self):
        sr = 8000
        t = np.arange(sr) / sr
        x = 0.2 * np.sin(2 * np.pi * 180 * t)
        row = BASELINE.extract_segment(x, sr)
        self.assertTrue(row["raw_f0_success"])
        self.assertAlmostEqual(row["f0_median_hz"], 180, delta=3)

    def test_660_hz_tone_is_not_a_reliable_speech_f0(self):
        sr = 8000
        t = np.arange(sr) / sr
        x = 0.2 * np.sin(2 * np.pi * 660 * t)
        row = BASELINE.extract_segment(x, sr)
        self.assertTrue(row["raw_f0_success"])
        self.assertGreater(row["raw_high_frames"], 0)
        self.assertFalse(row["reliable_f0"])

    def test_silence_fails_pitch(self):
        row = BASELINE.extract_segment(np.zeros(8000), 8000)
        self.assertFalse(row["raw_f0_success"])
        self.assertFalse(row["reliable_f0"])


if __name__ == "__main__":
    unittest.main()
