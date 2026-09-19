"""Interactive, extraction-only preparation. No method starts a full extraction."""
import json
import importlib.metadata
import sys
from pathlib import Path

if ('numpy' in sys.modules
        and sys.modules['numpy'].__version__ != importlib.metadata.version('numpy')):
    raise RuntimeError('NumPy was imported before dependency installation. '
                       'Restart the Colab Python session and rerun this cell; keep files/config.')
import colab_runner as runner


class ColabSession:
    """Load the validated indices/model once while retaining manual stage boundaries."""

    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = runner.load_config(self.config_path)
        self.device, self.gpu = runner.cuda_info()
        self.output = Path(self.config['OUTPUT_DIR'])
        self.output.mkdir(parents=True, exist_ok=True)
        print('Loading fixed indices once for this session...', flush=True)
        self.jobs, self.identity = runner.load_inputs(self.config)
        self.identity['execution_environment'] = runner.execution_identity(self.gpu)
        self.model = None
        print(f'Indices ready: {len(self.jobs)} calls; '
              f'{sum(j["n_segments"] for j in self.jobs)} caller segments', flush=True)

    def _unchanged(self):
        """Cheap content checks retain safety without reparsing one million CSV rows."""
        if runner.load_config(self.config_path) != self.config:
            raise RuntimeError('Config changed; restart ColabSession')
        files = {'split_sha256': self.config['SPLIT_CSV'],
                 'manifest_sha256': self.config['MANIFEST_CSV'],
                 'segments_sha256': self.config['SEGMENTS_CSV'],
                 'baseline_code_sha256': runner.base.__file__,
                 'adapter_code_sha256': runner.__file__}
        for key, path in files.items():
            if runner.base.sha256_file(Path(path)) != self.identity[key]:
                raise RuntimeError(f'{key} changed; restart runtime/session before continuing')

    def preflight(self):
        self._unchanged()
        selected = {j['call_id']: j for mode in ('smoke', 'benchmark')
                    for j in runner.saved_subset(self.jobs, mode)}
        runner.preflight(self.jobs, self.output, self.identity, list(selected.values()))
        report = json.loads((self.output / 'preflight.json').read_text())
        if report['selected_wav_calls'] != 22 or report['checked_calls'] != 22:
            raise RuntimeError('Expected exactly 22 unique smoke/benchmark WAVs')
        print('MINIMAL PREFLIGHT PASS: 22 WAVs', flush=True)
        return report

    def _subset(self, stage):
        self._unchanged()
        runner.gate(self.output, 'preflight', self.identity)
        if stage == 'benchmark':
            runner.gate(self.output, 'smoke', self.identity)
        if self.model is None:
            print('Loading pinned FP32 encoder; first download can take several minutes...', flush=True)
            self.model = runner.load_model(self.device, self.output)
        else:
            print('Reusing the same frozen CUDA encoder.', flush=True)
        model = self.model[1]
        if (model.training or any(p.requires_grad or p.dtype != runner.torch.float32
                                 or p.device.type != 'cuda' for p in model.parameters())):
            raise RuntimeError('Retained encoder must remain eval/frozen/FP32/CUDA')
        return runner.subset_run(self.jobs, self.output, self.identity, stage,
                                 self.device, self.gpu, loaded_model=self.model)

    def smoke(self):
        return self._subset('smoke')

    def benchmark(self):
        return self._subset('benchmark')

    def release_model(self):
        """Use before a later, explicitly approved standalone extraction subprocess."""
        import gc
        self.model = None
        gc.collect()
        runner.torch.cuda.empty_cache()
        print('Notebook encoder released.', flush=True)
