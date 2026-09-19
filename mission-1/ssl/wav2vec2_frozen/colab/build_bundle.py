"""Build a code-only Colab handoff bundle; never include WAVs or extraction cache."""
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent
files = [BASE / 'run_wav2vec2_frozen.py', BASE / 'test_wav2vec2_frozen.py',
         BASE / 'results/smoke_calls.csv', BASE / 'results/benchmark_calls.csv',
         BASE / 'results/benchmark_results.json',
         ROOT / 'mission-1/validation/split_assignments.csv']
files += [p for p in HERE.iterdir() if p.suffix in {'.py', '.md', '.txt', '.ipynb'}]
output = HERE / 'artifacts/colab_code_bundle.zip'
output.parent.mkdir(exist_ok=True)
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for path in sorted(files):
        z.write(path, path.relative_to(ROOT))
with zipfile.ZipFile(output) as z:
    assert z.testzip() is None
print(output)
