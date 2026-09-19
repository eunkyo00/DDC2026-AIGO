#!/usr/bin/env python3
"""기존 float16 / 신규 uint8 shard의 로딩 정밀도를 검사한다."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
    from feature_dataset import PrecomputedFeatureDataset
    DEPENDENCY_ERROR = None
except (ModuleNotFoundError, RuntimeError) as exc:
    DEPENDENCY_ERROR = exc


@unittest.skipIf(DEPENDENCY_ERROR is not None, str(DEPENDENCY_ERROR))
class FeatureDatasetTest(unittest.TestCase):
    def test_uint8_view_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "Validation"
            folder.mkdir()
            np.save(folder / "features_0000.npy", np.full((2, 64, 48), 127, dtype=np.uint8))
            np.save(folder / "labels_0000.npy", np.array([0, 1], dtype=np.int8))
            (root / "metadata.json").write_text(json.dumps({
                "format_version": 1,
                "feature_variant": "first_plus_whole",
                "splits": {"Validation": {
                    "complete": True, "samples": 2,
                    "shards": [{"features": "features_0000.npy", "labels": "labels_0000.npy", "samples": 2}],
                }},
            }), encoding="utf-8")
            dataset = PrecomputedFeatureDataset(root, "Validation")
            x, y = dataset[1]
            self.assertEqual(x.shape, (1, 64, 48))
            self.assertEqual(x.dtype, torch.float32)
            self.assertAlmostEqual(float(x.mean()), 127 / 255, places=6)
            self.assertEqual(int(y), 1)


if __name__ == "__main__":
    unittest.main()
