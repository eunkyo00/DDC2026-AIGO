#!/usr/bin/env python3

import json
import tempfile
import unittest
import wave
from pathlib import Path

from run_eda import analyze_split


class Mission2EdaTest(unittest.TestCase):
    def test_synthetic_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            split_dir = Path(tmp) / "Training"
            wav_dir = split_dir / "1.원천데이터" / "TS_서울_구급"
            json_dir = split_dir / "2.라벨링데이터" / "TL_서울_구급"
            wav_dir.mkdir(parents=True)
            json_dir.mkdir(parents=True)

            with wave.open(str(wav_dir / "sample.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(8000)
                wav.writeframes(b"\x00\x00" * 16000)

            label = {
                "utterances": [
                    {"startAt": 0, "endAt": 500, "speaker": 0, "text": "ignored"},
                    {"startAt": 500, "endAt": 1500, "speaker": 1, "text": "ignored"},
                ]
            }
            with (json_dir / "sample.json").open("w", encoding="utf-8") as handle:
                json.dump(label, handle, ensure_ascii=False)

            summary, calls, counts, bins, issues = analyze_split("Training", split_dir, 0, False)
            self.assertEqual(summary["calls_scanned"], 1)
            self.assertEqual(counts[("Training", 0)], 1)
            self.assertEqual(counts[("Training", 1)], 1)
            self.assertEqual(calls[0]["sample_rate"], 8000)
            self.assertFalse(issues)
            self.assertNotIn("text", calls[0])


if __name__ == "__main__":
    unittest.main()
