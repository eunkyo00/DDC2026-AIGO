"""Read ZIP members without extracting audio or Mac resource files."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core import ROOT, fresh_dir, parse_call, write_json, file_sha256


def prepare(source, output):
    seen = set()
    with zipfile.ZipFile(source) as archive, output.open("w", encoding="utf-8") as stream:
        for name in sorted(archive.namelist()):
            path = Path(name)
            if path.suffix.lower() != ".json" or "__MACOSX" in path.parts or path.name.startswith("._"):
                continue
            if path.name in seen:
                raise ValueError(f"Duplicate basename: {path.name}")
            seen.add(path.name)
            row = parse_call(json.loads(archive.read(name).decode("utf-8-sig")), path.name)
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(seen)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-zip", type=Path, required=True)
    parser.add_argument("--validation-zip", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/processed/calls")
    args = parser.parse_args()
    out = fresh_dir(args.out_dir)
    counts = {"training": prepare(args.train_zip, out / "training.jsonl"),
              "validation": prepare(args.validation_zip, out / "validation.jsonl")}
    write_json(out / "metadata.json", {"counts": counts, "input": "utterances[].text only",
        "sha256": {"train_zip": file_sha256(args.train_zip), "validation_zip": file_sha256(args.validation_zip)}})
    print(counts)
