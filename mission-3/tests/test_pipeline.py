import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import head_tail, parse_call, fresh_dir

spec = importlib.util.spec_from_file_location("prepare", Path(__file__).resolve().parents[1] / "data/prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PipelineTests(unittest.TestCase):
    def test_text_only_and_empty_targets(self):
        row = parse_call({"utterances": [{"text": "속이 불편해요", "speaker": "SECRET"}],
                          "recordId": "SECRET", "symptom": ["기타"]}, "a.json")
        self.assertEqual(row["text"], "속이 불편해요")
        self.assertNotIn("SECRET", json.dumps(row))
        self.assertEqual(row["오심"], 0)

    def test_multilabel(self):
        row = parse_call({"utterances": [], "symptom": ["구토", "오심", "기타"]}, "a.json")
        self.assertEqual((row["구토"], row["오심"], row["열상"]), (1, 1, 0))

    def test_head_tail(self):
        self.assertEqual(head_tail(list(range(510))), list(range(510)))
        self.assertEqual(head_tail(list(range(600))), list(range(255)) + list(range(345, 600)))

    def test_zip_filter_and_traversal_not_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            source = tmp / "sample.zip"
            data = json.dumps({"utterances": [{"text": "test"}], "symptom": []})
            with zipfile.ZipFile(source, "w") as z:
                z.writestr("labels/a.json", data)
                z.writestr("__MACOSX/._a.json", "invalid")
                z.writestr("labels/._b.json", "invalid")
                z.writestr("../escaped.json", data)
            self.assertEqual(prepare.prepare(source, tmp / "calls.jsonl"), 2)
            self.assertFalse((tmp.parent / "escaped.json").exists())

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "existing").touch()
            with self.assertRaises(FileExistsError):
                fresh_dir(tmp)


if __name__ == "__main__":
    unittest.main()
