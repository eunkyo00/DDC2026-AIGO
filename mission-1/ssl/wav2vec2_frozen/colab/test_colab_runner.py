"""Offline integration tests: no pretrained model, real WAV dataset or CUDA run."""
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location('colab_runner', Path(__file__).with_name('colab_runner.py'))
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)
session_spec = importlib.util.spec_from_file_location(
    'colab_session_test', Path(__file__).with_name('colab_session.py'))
session_module = importlib.util.module_from_spec(session_spec)
with mock.patch.dict(sys.modules, {'colab_runner': c}):
    session_spec.loader.exec_module(session_module)


class ColabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = c.BASE_DIR.parents[2]
        cls.jobs, _ = c.base.load_jobs(
            root / 'mission-1/validation/split_assignments.csv',
            root / 'mission-1/validation/manifests/calls.csv',
            root / 'mission-1/eda/eda_outputs/segments.csv')

    def test_saved_real_subsets(self):
        smoke = c.saved_subset(self.jobs, 'smoke')
        bench = c.saved_subset(self.jobs, 'benchmark')
        self.assertEqual([j['call_id'] for j in smoke], [j['call_id'] for j in c.base.select_smoke(self.jobs)])
        self.assertEqual([j['call_id'] for j in bench], [j['call_id'] for j in c.base.select_representative(self.jobs, 5)])
        self.assertEqual(sum(j['n_segments'] for j in smoke), 142)
        self.assertEqual(sum(j['n_segments'] for j in bench), 336)
        self.assertAlmostEqual(sum(j['total_duration_s'] for j in bench), 655.775)
        self.assertEqual(len({j['call_id'] for j in smoke + bench}), 22)

    def test_cuda_unavailable_stops(self):
        with mock.patch.object(c.torch.cuda, 'is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                c.cuda_info()

    def test_transformers_loading_sets_are_json_compatible(self):
        loading = {'missing_keys': set(), 'unexpected_keys': {'b', 'a'},
                   'nested': ({'x'},)}
        converted = c.json_compatible(loading)
        self.assertEqual(converted['unexpected_keys'], ['a', 'b'])
        self.assertEqual(converted['nested'], [['x']])
        json.dumps(converted)

    def test_full_guard_before_any_work(self):
        with self.assertRaisesRegex(RuntimeError, 'explicit'):
            c.extract_future([], Path('/unused'), {}, None)

    def test_success_gate_rejects_changed_inputs_or_failure(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            identity = {'split': 'one'}
            c.save_json(root / 'smoke.json', {'status': 'pass', 'identity': identity})
            c.gate(root, 'smoke', identity)
            with self.assertRaises(RuntimeError):
                c.gate(root, 'smoke', {'split': 'two'})
            c.save_json(root / 'smoke.json', {'status': 'failed', 'identity': identity})
            with self.assertRaises(RuntimeError):
                c.gate(root, 'smoke', identity)

    def test_full_wav_read_preflight_and_failure_report(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            wav = root / 'a.wav'
            c.sf.write(wav, c.np.zeros(8000, dtype='float32'), 8000)
            job = {'call_id': 'a', 'wav_path': str(wav), 'n_segments': 1, 'intervals': [(0, 1)]}
            c.preflight([job], root, {})
            self.assertEqual(json.loads((root / 'preflight.json').read_text())['status'], 'pass')
            c.sf.write(wav, c.np.zeros((8000, 2), dtype='float32'), 8000)
            with self.assertRaises(RuntimeError):
                c.preflight([job], root, {})
            self.assertEqual(json.loads((root / 'preflight.json').read_text())['invalid_audio'], 1)
            wav.unlink()
            with self.assertRaises(RuntimeError):
                c.preflight([job], root, {})
            report = json.loads((root / 'preflight.json').read_text())
            self.assertEqual((report['missing_wav'], report['io_errors']), (1, 1))

    def test_segment_failure_has_exact_interval_and_restores_functions(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            wav = root / 'a.wav'
            c.sf.write(wav, c.np.zeros(8000, dtype='float32'), 8000)
            job = {'call_id': 'a', 'wav_path': str(wav), 'intervals': [(0, .1), (.1, .2)]}
            original_call, original_crop = c.base.encode_call, c.base.crop_segment
            info = {'too_short': False, 'chunks': 1}
            with mock.patch.object(c.base, 'encode_segment', side_effect=[(c.np.zeros(768), info), RuntimeError('GPU error')]):
                with self.assertRaises(RuntimeError):
                    with c.trace_failures(root, 'smoke'):
                        c.base.encode_call(job, None, None, None)
            failure = json.loads(next((root / 'failures').glob('*.json')).read_text())
            self.assertEqual(failure['segment_index_0based'], 1)
            self.assertEqual(failure['interval_seconds'], [.1, .2])
            self.assertEqual(failure['failed_segments'], 1)
            self.assertIs(c.base.encode_call, original_call)
            self.assertIs(c.base.crop_segment, original_crop)

    def test_resume_after_progress_write_crash_skips_committed_call(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = [{'call_id': 'a', 'n_segments': 1, 'wav_path': 'mock-a.wav', 'gender': 'F', 'partition': 'train'},
                    {'call_id': 'b', 'n_segments': 1, 'wav_path': 'mock-b.wav', 'gender': 'M', 'partition': 'internal_validation'}]
            identity = {'test': True}
            for stage in ('preflight', 'smoke', 'benchmark'):
                c.save_json(root / f'{stage}.json', {'status': 'pass', 'identity': identity})
            quality = {'successful_segments': 1, 'failed_segments': 0}
            save = c.save_json
            def crash(path, value):
                if path.name == 'progress.json' and value['completed_calls'] == 1:
                    raise OSError('Session interrupted after commit')
                save(path, value)
            with mock.patch.object(c, 'load_model', return_value=(None, None, {})), \
                 mock.patch.object(c.base, 'encode_call', return_value=(c.np.ones(768, dtype='float32'), quality)), \
                 mock.patch.object(c, 'save_json', side_effect=crash):
                with self.assertRaises(OSError):
                    c.extract_future(jobs, root, identity, None, True)
            self.assertTrue((root / 'call_cache/a/a.npz').exists())
            processed = []
            def encode(job, *args):
                processed.append(job['call_id'])
                return c.np.ones(768, dtype='float32'), quality
            with mock.patch.object(c, 'load_model', return_value=(None, None, {})), \
                 mock.patch.object(c.base, 'encode_call', side_effect=encode):
                c.extract_future(jobs, root, identity, None, True)
            self.assertEqual(processed, ['b'])
            self.assertEqual(json.loads((root / 'progress.json').read_text())['completed_calls'], 2)
            summary = json.loads((root / 'extraction_summary.json').read_text())
            self.assertEqual(summary['embedding_shape'], [2, 768])
            self.assertEqual(summary['nan_values'], 0)
            self.assertEqual(summary['failed_segments'], 0)
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                c.read_completed(root / 'call_cache/a/a.npz', jobs[0], 'other-environment')
            (root / 'call_cache/a/a.npz').write_bytes(b'broken')
            with self.assertRaises(Exception):
                c.read_completed(root / 'call_cache/a/a.npz', jobs[0])

    def test_minimum_preflight_does_not_claim_unread_wavs(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            wav = root / 'a.wav'
            c.sf.write(wav, c.np.zeros(8000, dtype='float32'), 8000)
            selected = {'call_id': 'a', 'wav_path': str(wav), 'n_segments': 1, 'intervals': [(0, 1)]}
            unselected = {**selected, 'call_id': 'b', 'wav_path': str(root / 'unavailable.wav')}
            c.preflight([selected, unselected], root, {}, audio_jobs=[selected])
            report = json.loads((root / 'preflight.json').read_text())
            self.assertEqual(report['calls'], 2)
            self.assertEqual(report['checked_calls'], 1)
            self.assertEqual(report['checked_call_ids'], ['a'])
            self.assertIn('not a full WAV audit', report['wav_check_scope'])

    def test_extraction_subset_does_not_fit_classifier(self):
        jobs = [{'call_id': 'a', 'n_segments': 1, 'total_duration_s': 1,
                 'gender': 'F', 'partition': 'train'}]
        model = c.torch.nn.Linear(1, 1)
        with mock.patch.object(c.base, 'encode_call', return_value=(c.np.zeros(768, dtype='float32'), {})), \
             mock.patch.object(c.base, 'LogisticRegression', side_effect=AssertionError('No Colab classifier')):
            _, matrix, report = c.extract_subset(jobs, None, model, c.torch.device('cpu'), 'smoke')
        self.assertEqual(matrix.shape, (1, 768))
        self.assertIn('not run', report['classifier'])

    def test_gpu_environment_identity_excludes_free_memory(self):
        gpu = {'device': 'cuda:0', 'gpu_name': 'GPU-A', 'compute_capability': [7, 5],
               'CUDA_build': '12', 'cudnn_version': 1, 'nvidia_driver': '1', 'free_gpu_memory_bytes': 100}
        first = c.execution_identity(gpu)
        gpu['free_gpu_memory_bytes'] = 200
        self.assertEqual(first, c.execution_identity(gpu))
        gpu['gpu_name'] = 'GPU-B'
        self.assertNotEqual(first, c.execution_identity(gpu))

    def test_notebook_no_extract_execution(self):
        nb = json.loads(Path(__file__).with_name('Frozen_Wav2Vec2_Colab.ipynb').read_text())
        for cell in nb['cells']:
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                compile(source, '<notebook-cell>', 'exec')
                self.assertNotIn('--confirm-full-extraction', source)
                self.assertNotIn('run_stage("extract")', source)

    def test_timing_restores_reader_after_encoding_error(self):
        original = c.sf.read
        model = c.torch.nn.Linear(1, 1)
        with mock.patch.object(c.base, 'encode_call', side_effect=OSError('test read failure')):
            with self.assertRaises(OSError):
                c.extract_subset([{}], None, model, c.torch.device('cpu'), 'smoke')
        self.assertIs(c.sf.read, original)

    def test_interactive_session_reuses_model_but_requires_smoke_gate(self):
        session = session_module.ColabSession.__new__(session_module.ColabSession)
        session._unchanged = mock.Mock()
        session.output, session.identity, session.jobs = Path('/unused'), {}, []
        session.device, session.gpu, session.model = None, {}, None
        parameter = SimpleNamespace(requires_grad=False, dtype=c.torch.float32,
                                    device=SimpleNamespace(type='cuda'))
        model = SimpleNamespace(training=False, parameters=lambda: [parameter])
        with mock.patch.object(c, 'gate') as gate, \
             mock.patch.object(c, 'load_model', return_value=(None, model, {})) as load, \
             mock.patch.object(c, 'subset_run') as subset:
            session.smoke()
            session.benchmark()
            self.assertEqual(load.call_count, 1)
            self.assertEqual(subset.call_count, 2)
            self.assertEqual(gate.call_args_list[-1].args[1], 'smoke')
        with mock.patch.object(c, 'gate', side_effect=RuntimeError('failed smoke')), \
             mock.patch.object(c, 'subset_run') as subset:
            with self.assertRaises(RuntimeError):
                session.benchmark()
            subset.assert_not_called()
        self.assertFalse(hasattr(session, 'extract'))

    def test_benchmark_storage_probe_is_separate_from_full_cache(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            job = {'call_id': 'aa', 'n_segments': 1, 'total_duration_s': 1.0}
            quality = {'successful_segments': 1, 'failed_segments': 0,
                       'too_short_segments': 0, 'model_chunks': 1}
            measurement = {'elapsed_seconds': 1., 'seconds_per_call': 1.,
                           'seconds_per_segment': 1., 'audio_realtime_factor': 1., 'memory': {}}
            with mock.patch.object(c, 'gate'), \
                 mock.patch.object(c, 'saved_subset', return_value=[job]), \
                 mock.patch.object(c, 'load_model', side_effect=AssertionError('must reuse model')), \
                 mock.patch.object(c, 'extract_subset', return_value=(
                     [quality], c.np.zeros((1, 768), dtype='float32'), measurement)), \
                 mock.patch.object(c.torch.cuda, 'synchronize'), \
                 mock.patch.object(c.torch.cuda, 'reset_peak_memory_stats'), \
                 mock.patch.object(c.torch.cuda, 'max_memory_allocated', return_value=0), \
                 mock.patch.object(c.torch.cuda, 'max_memory_reserved', return_value=0):
                report = c.subset_run([job], root, {}, 'benchmark', None, {}, (None, None, {}))
            self.assertEqual(report['status'], 'pass')
            self.assertGreaterEqual(report['projection']['audio_plus_cache_hours'],
                                    report['projection']['audio_scaled_hours'])
            self.assertFalse((root / 'call_cache').exists())
            self.assertEqual(len(list((root / 'benchmark_storage_probe').glob('*/*/*.npz'))), 1)


if __name__ == '__main__':
    unittest.main()
