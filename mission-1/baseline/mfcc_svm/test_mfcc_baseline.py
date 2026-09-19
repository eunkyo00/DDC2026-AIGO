"""Synthetic tests for fixed-length MFCC extraction and call aggregation."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np
from scipy.fft import dct

MODULE = Path(__file__).with_name("run_mfcc_baseline.py")
SPEC = importlib.util.spec_from_file_location("run_mfcc_baseline", MODULE)
BASELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BASELINE)


class TestMfcc(unittest.TestCase):
    def test_one_second_signal_has_finite_26d_feature(self):
        t = np.arange(BASELINE.SAMPLE_RATE) / BASELINE.SAMPLE_RATE
        samples = 0.2 * np.sin(2 * np.pi * 180 * t)
        feature, quality = BASELINE.extract_segment(samples)
        self.assertEqual(feature.shape, (26,))
        self.assertTrue(np.isfinite(feature).all())
        self.assertFalse(quality["short_segment"])

    def test_sub_window_segment_is_retained_and_flagged(self):
        samples = np.linspace(-0.1, 0.1, 80)
        feature, quality = BASELINE.extract_segment(samples)
        self.assertEqual(feature.shape, (26,))
        self.assertTrue(np.isfinite(feature).all())
        self.assertTrue(quality["short_segment"])

    def test_silence_proxy(self):
        feature, quality = BASELINE.extract_segment(np.zeros(800))
        self.assertTrue(np.isfinite(feature).all())
        self.assertEqual(quality["silence_ratio"], 1.0)
        self.assertTrue(quality["high_silence"])

    def test_precomputed_dct_matches_scipy(self):
        rng = np.random.default_rng(42)
        log_mel = rng.normal(size=(7, BASELINE.N_MELS))
        _, _, basis = BASELINE.analysis_constants()
        expected = dct(log_mel, type=2, axis=1, norm="ortho")[:, :BASELINE.N_MFCC]
        np.testing.assert_allclose(log_mel @ basis.T, expected, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
