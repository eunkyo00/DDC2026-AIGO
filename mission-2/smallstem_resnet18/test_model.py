#!/usr/bin/env python3

import unittest

try:
    import torch

    from model import build_model

    DEPENDENCY_ERROR = None
except (ModuleNotFoundError, RuntimeError) as exc:
    DEPENDENCY_ERROR = exc


@unittest.skipIf(DEPENDENCY_ERROR is not None, f"optional model dependency missing: {DEPENDENCY_ERROR}")
class Mission2ModelTest(unittest.TestCase):
    def test_forward_shape(self):
        for model_name in ("resnet18", "resnet18_smallstem", "hybrid_resnet18_tdnn"):
            model = build_model(pretrained=False, spec_augment=False, model_name=model_name).eval()
            for frames in (24, 48):
                with self.subTest(model_name=model_name, frames=frames):
                    with torch.inference_mode():
                        output = model(torch.rand(2, 1, 64, frames))
                    self.assertEqual(tuple(output.shape), (2, 2))


if __name__ == "__main__":
    unittest.main()
