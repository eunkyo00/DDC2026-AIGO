"""Fast tests that do not download or load the pretrained checkpoint."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import torch

MODULE = Path(__file__).with_name("run_wav2vec2_frozen.py")
SPEC = importlib.util.spec_from_file_location("wav2vec2_frozen", MODULE)
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)


class FrozenWav2Vec2Tests(unittest.TestCase):
    def test_frozen_split_is_reused_without_leakage(self):
        root = Path(__file__).resolve().parents[3]
        jobs, checksum = baseline.load_jobs(
            root / "mission-1/validation/split_assignments.csv",
            root / "mission-1/validation/manifests/calls.csv",
            root / "mission-1/eda/eda_outputs/segments.csv")
        self.assertEqual(checksum, baseline.EXPECTED_SPLIT_SHA256)
        self.assertEqual(len(jobs), 27_985)
        self.assertEqual(sum(j["partition"] == "train" for j in jobs), 22_388)
        self.assertEqual(sum(j["partition"] == "internal_validation" for j in jobs), 5_597)
        train = {j["call_id"] for j in jobs if j["partition"] == "train"}
        valid = {j["call_id"] for j in jobs if j["partition"] == "internal_validation"}
        self.assertFalse(train & valid)
        self.assertEqual(sum(j["n_segments"] for j in jobs), 442_639)

    def test_caller_crop_uses_rounded_sample_boundaries(self):
        audio = np.arange(80, dtype=np.float32)
        cropped = baseline.crop_segment(audio, 8_000, 0.002, 0.006)
        np.testing.assert_array_equal(cropped, audio[16:48])
        with self.assertRaises(ValueError):
            baseline.crop_segment(audio, 8_000, -1, 0.001)

    def test_polyphase_resampling_doubles_length_and_is_finite(self):
        t = np.arange(800, dtype=np.float32) / 8_000
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        result = baseline.resample_8k_to_16k(audio)
        self.assertEqual(len(result), 1_600)
        self.assertTrue(np.isfinite(result).all())

    def test_masked_pooling_excludes_padding(self):
        hidden = torch.tensor([[[1., 2.], [3., 4.], [100., 200.]]])
        mask = torch.tensor([[1, 1, 0]])
        result = baseline.masked_temporal_mean(hidden, mask)
        torch.testing.assert_close(result, torch.tensor([[2., 3.]]))

    def test_equal_segment_mean_and_finiteness(self):
        one = np.ones(baseline.HIDDEN_DIM, dtype=np.float32)
        three = np.full(baseline.HIDDEN_DIM, 3, dtype=np.float32)
        result = baseline.aggregate_segment_embeddings([one, three])
        self.assertEqual(result.shape, (baseline.HIDDEN_DIM,))
        np.testing.assert_array_equal(result, np.full(baseline.HIDDEN_DIM, 2, dtype=np.float32))
        self.assertTrue(np.isfinite(result).all())

    def test_freeze_encoder_sets_trainable_parameters_to_zero(self):
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Linear(2, 1))
        total, trainable = baseline.freeze_encoder(model)
        self.assertGreater(total, 0)
        self.assertEqual(trainable, 0)
        self.assertFalse(model.training)

    def test_cache_commit_is_resume_safe_and_validated(self):
        jobs = [{"call_id": "a"}, {"call_id": "b"}]
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            embeddings, status, metadata = baseline.initialize_cache(cache, jobs, "split")
            embeddings[0] = np.arange(baseline.HIDDEN_DIM, dtype=np.float32)
            baseline.durable_memmap_flush(embeddings, cache / "call_embeddings.npy")
            baseline.append_jsonl_durable(cache / "quality.jsonl", {
                "call_id": "a", "successful_segments": 1, "failed_segments": 0,
                "too_short_segments": 0, "model_chunks": 1})
            status[0] = 1
            baseline.durable_memmap_flush(status, cache / "status.npy")

            reopened_embeddings, reopened_status, reopened_metadata = baseline.initialize_cache(
                cache, jobs, "split")
            quality = baseline.load_quality_rows(cache / "quality.jsonl")
            baseline.validate_completed_cache(
                cache, jobs, reopened_embeddings, reopened_status, quality)
            self.assertEqual(metadata, reopened_metadata)
            self.assertEqual(reopened_status.tolist(), [1, 0])
            self.assertEqual(baseline.cache_progress(cache, 2)["completed_calls"], 1)

            reopened_embeddings[0, 0] = np.nan
            baseline.durable_memmap_flush(reopened_embeddings, cache / "call_embeddings.npy")
            with self.assertRaisesRegex(ValueError, "NaN/inf"):
                baseline.validate_completed_cache(
                    cache, jobs, reopened_embeddings, reopened_status, quality)

    def test_io_failure_is_durable_fail_fast_and_same_cache_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(cache_dir=root / "cache", results_dir=root / "results",
                                   progress_every=1)
            jobs = [{"call_id": "call-a", "wav_path": str(root / "missing.wav")}]
            with self.assertRaisesRegex(RuntimeError, "stopped immediately"):
                baseline.extract_all(args, jobs, "split", None, None, None, {})
            status = np.load(args.cache_dir / "status.npy")
            self.assertEqual(status.tolist(), [2])
            failures = (args.cache_dir / "failures.jsonl").read_text(encoding="utf-8")
            self.assertIn("FileNotFoundError", failures)
            self.assertEqual(baseline.cache_progress(args.cache_dir, 1)["failed_calls"], 1)

            quality = {"successful_segments": 1, "failed_segments": 0,
                       "too_short_segments": 0, "model_chunks": 1}
            with mock.patch.object(baseline, "require_storylink"), mock.patch.object(
                    baseline, "encode_call", return_value=(
                        np.ones(baseline.HIDDEN_DIM, dtype=np.float32), quality)):
                baseline.extract_all(args, jobs, "split", None, None, None, {})
            resumed = np.load(args.cache_dir / "status.npy")
            self.assertEqual(resumed.tolist(), [1])
            self.assertEqual(baseline.cache_progress(args.cache_dir, 1)["failed_calls"], 0)
            self.assertTrue((args.results_dir / "extraction_summary.json").exists())

    def test_extract_skips_a_valid_completed_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(cache_dir=root / "cache", results_dir=root / "results",
                                   progress_every=1)
            jobs = [{"call_id": "done", "wav_path": "unused"},
                    {"call_id": "pending", "wav_path": "unused"}]
            embeddings, status, _ = baseline.initialize_cache(args.cache_dir, jobs, "split")
            embeddings[0] = np.ones(baseline.HIDDEN_DIM, dtype=np.float32)
            baseline.durable_memmap_flush(embeddings, args.cache_dir / "call_embeddings.npy")
            quality = {"successful_segments": 1, "failed_segments": 0,
                       "too_short_segments": 0, "model_chunks": 1}
            baseline.append_jsonl_durable(args.cache_dir / "quality.jsonl",
                                           {"call_id": "done", **quality})
            status[0] = 1
            baseline.durable_memmap_flush(status, args.cache_dir / "status.npy")
            fake_encode = mock.Mock(return_value=(
                np.full(baseline.HIDDEN_DIM, 2, dtype=np.float32), quality))
            with mock.patch.object(baseline, "require_storylink"), mock.patch.object(
                    baseline, "encode_call", fake_encode):
                baseline.extract_all(args, jobs, "split", None, None, None, {})
            self.assertEqual(fake_encode.call_count, 1)
            self.assertEqual(fake_encode.call_args.args[0]["call_id"], "pending")
            self.assertEqual(np.load(args.cache_dir / "status.npy").tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
