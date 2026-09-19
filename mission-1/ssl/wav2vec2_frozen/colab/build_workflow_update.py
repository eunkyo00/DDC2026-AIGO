"""Small update for an already prepared Colab checkout; no data or model included."""
import hashlib
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
output = HERE / 'artifacts/colab_workflow_update.zip'
output.parent.mkdir(exist_ok=True)
names = ['colab_runner.py', 'colab_session.py', 'README.md',
         'Frozen_Wav2Vec2_Colab.ipynb', 'test_colab_runner.py', 'build_workflow_update.py']
with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
    for name in names:
        path = HERE / name
        archive.write(path, path.relative_to(ROOT))
with zipfile.ZipFile(output) as archive:
    assert archive.testzip() is None
print(output)
print('Bytes:', output.stat().st_size)
print('SHA256:', hashlib.sha256(output.read_bytes()).hexdigest())
