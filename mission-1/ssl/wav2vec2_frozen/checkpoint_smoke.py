"""Checkpoint/MPS diagnostic using generated audio; this is not a dataset smoke test."""
import argparse
import importlib.util
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("wav2vec2_frozen", HERE / "run_wav2vec2_frozen.py")
baseline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(baseline)

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", required=True)
parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
parser.add_argument("--out", type=Path, default=HERE / "results/checkpoint_smoke.json")
args = parser.parse_args()
device = baseline.resolve_device(args.device)
extractor, model, total, trainable, loading = baseline.load_encoder(
    args.model_path, Path("/private/tmp/ddc-hf-cache"), device)
parameter_versions = [p._version for p in model.parameters()]
results = []
started = time.perf_counter()
for duration_s in (0.02, 1.0, 16.0):
    t = np.arange(round(duration_s * baseline.SOURCE_SAMPLE_RATE), dtype=np.float32) / baseline.SOURCE_SAMPLE_RATE
    source = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    resampled = baseline.resample_8k_to_16k(source)
    embedding, info = baseline.encode_segment(resampled, extractor, model, device)
    results.append({"source_duration_s": duration_s, "source_samples": len(source),
                    "resampled_samples": len(resampled), "embedding_shape": list(embedding.shape),
                    "finite": bool(np.isfinite(embedding).all()), **info})
if device.type == "mps":
    torch.mps.synchronize()
payload = {"scope": "generated-waveform checkpoint/MPS diagnostic; not a real-data smoke test",
           "checkpoint": baseline.CHECKPOINT, "revision": baseline.REVISION,
           "device": str(device), "os": platform.platform(), "versions": baseline.versions(),
           "total_encoder_parameters": total, "trainable_encoder_parameters": trainable,
           "frozen_parameter_versions_unchanged": parameter_versions == [p._version for p in model.parameters()],
           "elapsed_seconds": time.perf_counter() - started,
           "memory": baseline.memory_snapshot(device), "cases": results,
           "loading_info": {"missing_keys": sorted(loading.get("missing_keys", [])),
                            "unexpected_keys": sorted(loading.get("unexpected_keys", []))}}
baseline.write_json(args.out, payload)
print(json.dumps(payload, indent=2))
