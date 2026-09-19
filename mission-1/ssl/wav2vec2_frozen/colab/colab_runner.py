"""CUDA-only adapter. Default workflow stops after benchmark; baseline math is imported."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import uuid

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
import run_wav2vec2_frozen as base

np, torch, sf = base.np, base.torch, base.sf


def save_json(path, value):
    """Close temp file before rename; no directory fsync requirement on Drive FUSE."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{uuid.uuid4().hex}')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def require_drive(path):
    path = Path(path).expanduser().resolve()
    mount = Path('/content/drive')
    if not os.path.ismount(mount) or not path.is_relative_to(mount):
        raise ValueError(f'Mounted Google Drive path required: {path}')
    return path


def cuda_info():
    available = torch.cuda.is_available()
    info = {'torch.cuda.is_available()': available, 'PyTorch': torch.__version__,
            'Transformers': base.versions()['transformers'], 'CUDA_build': torch.version.cuda}
    print(json.dumps(info, indent=2), flush=True)
    if not available:
        raise RuntimeError('STOP: choose a GPU runtime in Colab; CUDA is unavailable.')
    device = torch.device('cuda:0')
    props = torch.cuda.get_device_properties(device)
    free, total = torch.cuda.mem_get_info(device)
    info.update(device=str(device), gpu_name=props.name,
                gpu_memory_bytes=props.total_memory, free_gpu_memory_bytes=free,
                total_gpu_memory_bytes=total, compute_capability=list(torch.cuda.get_device_capability(device)),
                cudnn_version=torch.backends.cudnn.version())
    info['nvidia_driver'] = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], text=True).splitlines()[0].strip()
    print(json.dumps(info, indent=2), flush=True)
    return device, info


def execution_identity(gpu):
    # Stable across sessions; excludes free memory and UUID so same GPU type can resume.
    return {key: gpu[key] for key in ('device', 'gpu_name', 'compute_capability',
                                      'CUDA_build', 'cudnn_version', 'nvidia_driver')}


def identity_digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


def json_compatible(value):
    """Normalize Transformers diagnostics without changing model state or inference."""
    if isinstance(value, dict):
        return {key: json_compatible(item) for key, item in value.items()}
    if isinstance(value, set):
        return [json_compatible(item) for item in sorted(value)]
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    return value


def load_config(path):
    config = json.loads(Path(path).read_text())
    for key in ('DATA_ROOT', 'SEGMENTS_CSV', 'MANIFEST_CSV', 'SPLIT_CSV', 'OUTPUT_DIR'):
        if not config.get(key):
            raise ValueError(f'Set {key} in config first')
    config['DATA_ROOT'] = str(require_drive(config['DATA_ROOT']))
    config['OUTPUT_DIR'] = str(require_drive(config['OUTPUT_DIR']))
    return config


def load_inputs(config):
    split, manifest, segments = [Path(config[k]) for k in ('SPLIT_CSV', 'MANIFEST_CSV', 'SEGMENTS_CSV')]
    # The original loader validates split checksum, coverage, labels and segment counts.
    jobs, split_hash = base.load_jobs(split, manifest, segments)
    rows = list(base.read_csv(manifest))
    if len(rows) != len(jobs) or len({r['call_id'] for r in rows}) != len(jobs):
        raise ValueError('Duplicate or extra manifest rows')
    root = Path(config['DATA_ROOT'])
    for job in jobs:
        relative = Path(job['wav_relative_path'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe relative WAV path: {relative}')
    base.remap_wav_paths(jobs, root)
    identity = {'split_sha256': split_hash,
                'manifest_sha256': base.sha256_file(manifest),
                'segments_sha256': base.sha256_file(segments),
                'baseline_code_sha256': base.sha256_file(Path(base.__file__)),
                'adapter_code_sha256': base.sha256_file(Path(__file__)),
                'data_root': str(root), 'checkpoint': base.CHECKPOINT,
                'revision': base.REVISION, 'versions': base.versions(),
                'dtype': 'float32', 'batch_size': 1, 'autocast': False, 'tf32': False}
    return jobs, identity


def gate(output, stage, identity):
    record = json.loads((output / f'{stage}.json').read_text())
    if record.get('status') != 'pass' or record.get('identity') != identity:
        raise RuntimeError(f'Rerun {stage}: missing success or inputs/code/environment changed')
    return record


def preflight(jobs, output, identity, audio_jobs=None):
    audio_jobs = jobs if audio_jobs is None else audio_jobs
    report = {'status': 'running', 'identity': identity, 'calls': len(jobs),
              'wav_check_scope': 'selected smoke/benchmark WAVs; not a full WAV audit',
              'selected_wav_calls': len(audio_jobs),
              'checked_call_ids': [j['call_id'] for j in audio_jobs],
              'segments': sum(j['n_segments'] for j in jobs), 'checked_calls': 0,
              'missing_wav': 0, 'io_errors': 0, 'invalid_audio': 0, 'failures': []}
    save_json(output / 'preflight.json', report)
    for i, job in enumerate(audio_jobs):
        print(f'Preflight WAV {i + 1}/{len(audio_jobs)}: {job["call_id"]}', flush=True)
        phase = 'open_read'
        try:
            if not Path(job['wav_path']).is_file():
                report['missing_wav'] += 1
                raise FileNotFoundError(job['wav_path'])
            with sf.SoundFile(job['wav_path']) as wav:
                if wav.samplerate != 8000 or wav.channels != 1:
                    phase = 'format'
                    raise ValueError(f'Expected 8000 Hz mono; got {wav.samplerate}, {wav.channels}')
                phase = 'crop_bounds'
                for start, end in job['intervals']:
                    if not (math.isfinite(start) and math.isfinite(end)
                            and 0 <= round(start * 8000) < round(end * 8000) <= wav.frames):
                        raise ValueError(f'Invalid caller bounds: {start}, {end}')
                # Read each selected WAV in bounded blocks; remaining WAVs are checked during extraction.
                phase = 'open_read'
                count = 0
                for block in wav.blocks(blocksize=65536, dtype='float32'):
                    count += len(block)
                    if not np.isfinite(block).all():
                        phase = 'format'
                        raise ValueError('Non-finite WAV samples')
                if count != wav.frames:
                    raise OSError(f'Short WAV read: {count}/{wav.frames}')
            report['checked_calls'] += 1
        except Exception as error:
            if phase == 'open_read':
                report['io_errors'] += 1
            else:
                report['invalid_audio'] += 1
            report['failures'].append({'call_id': job['call_id'], 'wav_path': job['wav_path'],
                                       'phase': phase, 'error': repr(error)})
        if (i + 1) % 250 == 0:
            print(f'Preflight {i + 1}/{len(audio_jobs)}; failures={len(report["failures"])}', flush=True)
            save_json(output / 'preflight.json', report)
    report['status'] = 'pass' if not report['failures'] else 'failed'
    save_json(output / 'preflight.json', report)
    print(json.dumps({k: v for k, v in report.items() if k != 'failures'}, indent=2))
    if report['status'] != 'pass':
        raise RuntimeError('Preflight failed; see preflight.json. Smoke/benchmark blocked.')


def saved_subset(jobs, mode):
    records = list(base.read_csv(BASE_DIR / 'results' / f'{mode}_calls.csv'))
    by_id = {j['call_id']: j for j in jobs}
    expected = 6 if mode == 'smoke' else 20
    if len(records) != expected or len({r['call_id'] for r in records}) != expected:
        raise ValueError('Invalid original subset fixture')
    subset = []
    for r in records:
        j = by_id[r['call_id']]
        if (j['gender'] != r['gender'] or j['partition'] != r['partition']
                or j['n_segments'] != int(r['n_segments'])
                or not math.isclose(j['total_duration_s'], float(r['total_duration_s']), abs_tol=1e-8)):
            raise ValueError('Subset no longer matches original Mac benchmark')
        subset.append(j)
    return subset


@contextmanager
def trace_failures(output, mode):
    """Observe original functions without changing crop/resampling/forward math."""
    original_call, original_crop = base.encode_call, base.crop_segment
    state = {}
    def crop(*args, **kwargs):
        state['segment_index_0based'] += 1
        state['interval_seconds'] = [args[2], args[3]]
        return original_crop(*args, **kwargs)
    def call(job, *args, **kwargs):
        state.clear()
        state.update(call_id=job['call_id'], wav_path=job['wav_path'], segment_index_0based=-1)
        try:
            return original_call(job, *args, **kwargs)
        except Exception as error:
            failure = {**state, 'mode': mode, 'timestamp': base.utc_now(),
                       'failed_calls': 1,
                       'failed_segments': int(state['segment_index_0based'] >= 0),
                       'io_errors': int(isinstance(error, (OSError, sf.LibsndfileError))),
                       'error': repr(error)}
            save_json(output / 'failures' / f'{mode}-{uuid.uuid4().hex}.json', failure)
            print(json.dumps(failure, ensure_ascii=False), flush=True)
            raise
    base.encode_call, base.crop_segment = call, crop
    try:
        yield
    finally:
        base.encode_call, base.crop_segment = original_call, original_crop


def load_model(device, output):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    extractor, model, total, trainable, loading = base.load_encoder(
        None, Path('/content/wav2vec2_hf_cache'), device)
    actual = next(model.parameters()).device
    loading_info = json_compatible(loading)
    loading_info = json_compatible(loading)
    loading_info = json_compatible(loading)
    info = {'checkpoint': base.CHECKPOINT, 'revision': base.REVISION,
            'actual_model_device': str(actual), 'total_parameters': total,
            'trainable_encoder_parameters': trainable, 'hidden_dimension': model.config.hidden_size,
            'dtype': str(next(model.parameters()).dtype), 'loading_info': loading_info}
    print(json.dumps(info, indent=2), flush=True)
    if (actual.type != 'cuda' or trainable != 0 or model.training
            or any(p.dtype != torch.float32 for p in model.parameters())
            or loading_info.get('missing_keys') or loading_info.get('mismatched_keys')):
        raise RuntimeError('CUDA/frozen/FP32/checkpoint invariant failed')
    return extractor, model, info


def extract_subset(jobs, extractor, model, device, mode):
    """Same original encode_call; extraction only, no classifier fit in Colab."""
    parameter_versions = [p._version for p in model.parameters()]
    started = time.perf_counter()
    rows, embeddings = [], []
    read_seconds = 0.0
    original_read = sf.read
    def timed_read(*args, **kwargs):
        nonlocal read_seconds
        read_started = time.perf_counter()
        try:
            return original_read(*args, **kwargs)
        finally:
            read_seconds += time.perf_counter() - read_started
    sf.read = timed_read
    try:
        for index, job in enumerate(jobs):
            embedding, quality = base.encode_call(job, extractor, model, device)
            embeddings.append(embedding)
            rows.append({**{key: job[key] for key in ('call_id', 'gender', 'partition',
                                                     'n_segments', 'total_duration_s')}, **quality})
            print(f'{mode}: {index + 1}/{len(jobs)} calls complete', flush=True)
    finally:
        sf.read = original_read
    # encode_call copies results to CPU, which also waits for each CUDA forward.
    elapsed = time.perf_counter() - started
    if parameter_versions != [p._version for p in model.parameters()]:
        raise RuntimeError('Frozen encoder parameter version changed')
    matrix = np.stack(embeddings)
    if matrix.shape != (len(jobs), 768) or not np.isfinite(matrix).all():
        raise ValueError('Subset embedding shape/finiteness check failed')
    segments = sum(j['n_segments'] for j in jobs)
    audio_seconds = sum(j['total_duration_s'] for j in jobs)
    return rows, matrix, {
        'mode': mode, 'calls': len(jobs), 'segments': segments,
        'caller_audio_seconds': audio_seconds, 'elapsed_seconds': elapsed,
        'seconds_per_call': elapsed / len(jobs), 'seconds_per_segment': elapsed / segments,
        'audio_realtime_factor': elapsed / audio_seconds, 'embedding_shape': list(matrix.shape),
        'finite': True, 'frozen_parameter_versions_unchanged': True,
        'wav_open_read_seconds': read_seconds,
        'non_read_seconds': elapsed - read_seconds,
        'classifier': 'not run; train/evaluate locally after complete CUDA extraction',
        'memory': base.memory_snapshot(device)}


def subset_run(jobs, output, identity, mode, device, gpu, loaded_model=None):
    gate(output, 'preflight', identity)
    if mode == 'benchmark':
        gate(output, 'smoke', identity)
    subset = saved_subset(jobs, mode)
    report = {'status': 'running', 'identity': identity, 'gpu': gpu}
    save_json(output / f'{mode}.json', report)
    try:
        extractor, model, encoder = loaded_model if loaded_model is not None else load_model(device, output)
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        with trace_failures(output, mode):
            rows, matrix, measurement = extract_subset(subset, extractor, model, device, mode)
        torch.cuda.synchronize(device)
        measurement['memory'].update(
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device))
        measurement.update(failed_calls=0, failed_segments=0, io_errors=0)
        # Original encode_call moves each result to CPU, synchronizing every CUDA forward.
        # Its elapsed_seconds includes WAV reads + extraction, excludes model load/classifier.
        report.update(status='pass', encoder=encoder, measurement=measurement)
        base.write_rows(output / f'{mode}_calls.csv', rows)
        np.save(output / f'{mode}_embeddings.npy', matrix)
        if mode == 'benchmark':
            # Measure the real per-call Drive commit separately from Mac-comparable extraction.
            # Diagnostic files never become full-extraction completion markers.
            storage_started = time.perf_counter()
            storage_dir = output / 'benchmark_storage_probe' / uuid.uuid4().hex
            fingerprint = identity_digest(identity)
            for job, embedding, row in zip(subset, matrix, rows):
                quality = {k: row[k] for k in ('successful_segments', 'failed_segments',
                                               'too_short_segments', 'model_chunks')}
                commit_call(call_cache_path(storage_dir, job), job, embedding, quality, fingerprint)
                save_json(storage_dir / 'progress.json', {'last_call_id': job['call_id']})
            storage_seconds = time.perf_counter() - storage_started
            measurement['drive_cache_probe_seconds'] = storage_seconds
            measurement['drive_cache_seconds_per_call'] = storage_seconds / len(subset)
            mac = json.loads((BASE_DIR / 'results/benchmark_results.json').read_text())
            total_audio = sum(j['total_duration_s'] for j in jobs)
            report['projection'] = {
                'dataset_calls': len(jobs), 'dataset_caller_audio_seconds': total_audio,
                'audio_scaled_hours': measurement['audio_realtime_factor'] * total_audio / 3600,
                'call_scaled_hours': measurement['seconds_per_call'] * len(jobs) / 3600,
                'segment_scaled_hours': measurement['seconds_per_segment'] * base.EXPECTED_SEGMENTS / 3600,
                'estimated_cache_write_hours': storage_seconds / len(subset) * len(jobs) / 3600,
                'audio_plus_cache_hours': (measurement['audio_realtime_factor'] * total_audio
                                           + storage_seconds / len(subset) * len(jobs)) / 3600,
                'note': 'Estimate only; Drive latency/session limits/cache writes can increase runtime.',
                'full_extraction_started': False}
            report['mac_reference'] = {'measurement': mac['measurement'], 'projection': mac['projection']}
            report['speedup_vs_mac'] = mac['measurement']['elapsed_seconds'] / measurement['elapsed_seconds']
    except Exception as error:
        report.update(status='failed', error=repr(error))
        save_json(output / f'{mode}.json', report)
        raise
    save_json(output / f'{mode}.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if mode == 'benchmark':
        print('STOP: benchmark complete. Full extraction was NOT started.', flush=True)
    return report


def call_cache_path(cache, job):
    # Shard to avoid tens of thousands of files in one Drive directory.
    return cache / job['call_id'][:2] / f"{job['call_id']}.npz"


def read_completed(path, job, fingerprint=None):
    """A closed, renamed NPZ is the per-call commit; validate before skipping."""
    with np.load(path, allow_pickle=False) as data:
        if fingerprint is not None and str(data['identity_sha256'].item()) != fingerprint:
            raise ValueError(f'Call cache environment fingerprint differs: {path}')
        embedding = data['embedding']
        quality = json.loads(str(data['quality'].item()))
        if (embedding.shape != (768,) or embedding.dtype != np.float32
                or not np.isfinite(embedding).all() or str(data['call_id'].item()) != job['call_id']
                or quality['successful_segments'] != job['n_segments'] or quality['failed_segments'] != 0):
            raise ValueError(f'Invalid completed call cache: {path}')
    return embedding, quality


def commit_call(path, job, embedding, quality, fingerprint=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp-{uuid.uuid4().hex}')
    with tmp.open('wb') as f:
        np.savez(f, embedding=embedding, call_id=job['call_id'], quality=json.dumps(quality),
                 identity_sha256=fingerprint or '')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    read_completed(path, job, fingerprint)


def audit_full_cache(jobs, output, identity):
    """Read all committed embeddings without a GPU or stacking them in RAM."""
    cache = output / 'call_cache'
    fingerprint = identity_digest(identity)
    if json.loads((cache / 'identity.json').read_text()) != identity:
        raise ValueError('Cache identity differs during full audit')
    expected = {call_cache_path(cache, j) for j in jobs}
    actual = set(cache.glob('*/*.npz'))
    if actual != expected or len({j['call_id'] for j in jobs}) != len(jobs):
        raise ValueError('Full cache call coverage mismatch (missing/extra/duplicate calls)')
    total_segments = 0
    for job in jobs:
        _, quality = read_completed(call_cache_path(cache, job), job, fingerprint)
        total_segments += quality['successful_segments']
    # Failed attempts remain in separate logs, including errors recovered by resume.
    failures = []
    for path in (output / 'failures').glob('extract-*.json'):
        failures.append(json.loads(path.read_text()))
    summary = {
        'status': 'complete', 'identity': identity, 'completed_calls': len(jobs),
        'expected_calls': len(jobs), 'pending_calls': 0, 'failed_calls': 0,
        'successful_segments': total_segments, 'failed_segments': 0,
        'embedding_shape': [len(jobs), 768], 'dtype': 'float32',
        'nan_values': 0, 'inf_values': 0, 'call_id_coverage_exact': True,
        'historical_failed_call_attempts': len(failures),
        'historical_failed_segment_attempts': sum(r['failed_segments'] for r in failures),
        'historical_io_errors': sum(r['io_errors'] for r in failures),
        'call_order_sha256': hashlib.sha256('\n'.join(j['call_id'] for j in jobs).encode()).hexdigest(),
        'verified_utc': base.utc_now(), 'classifier_evaluation': 'not run'}
    base.write_rows(output / 'embedding_index.csv', [
        {**{k: j[k] for k in ('call_id', 'gender', 'partition', 'n_segments')},
         'row_index': i, 'cache_relative_path': str(call_cache_path(cache, j).relative_to(output))}
        for i, j in enumerate(jobs)])
    save_json(output / 'extraction_summary.json', summary)
    return summary


def extract_future(jobs, output, identity, device, confirmed=False):
    # Deliberately never called by the notebook, even by Run all.
    if not confirmed:
        raise RuntimeError('Full extraction requires explicit --confirm-full-extraction')
    for stage in ('preflight', 'smoke', 'benchmark'):
        gate(output, stage, identity)
    cache = output / 'call_cache'
    fingerprint = identity_digest(identity)
    cache.mkdir(exist_ok=True)
    metadata = cache / 'identity.json'
    if metadata.exists():
        if json.loads(metadata.read_text()) != identity:
            raise ValueError('Cache identity differs; do not mix environments/data/code')
    else:
        if any(cache.iterdir()):
            raise ValueError('Nonempty cache without identity')
        save_json(metadata, identity)
    # Exclusive file, not flock (which is not a cross-session lock on Drive).
    lock = cache / 'ACTIVE_SESSION.lock'
    with lock.open('x') as f:
        f.write(json.dumps({'created': base.utc_now(), 'pid': os.getpid()}))
    try:
        save_json(output / 'extraction_summary.json', {'status': 'in_progress', 'identity': identity})
        done = set()
        for job in jobs:
            file = call_cache_path(cache, job)
            if file.exists():
                read_completed(file, job, fingerprint)
                done.add(job['call_id'])
        initial = len(done)
        started = time.perf_counter()
        def progress(last=None, failed=None):
            save_json(output / 'progress.json', {
                'completed_calls': len(done), 'total_calls': len(jobs),
                'pending_or_retry_calls': len(jobs) - len(done),
                'completed_this_session': len(done) - initial,
                'session_elapsed_seconds': time.perf_counter() - started,
                'last_call_id': last, 'failed_call': failed, 'failed_calls': int(failed is not None),
                'embedding_shape_completed': [len(done), 768], 'updated_utc': base.utc_now()})
        progress()
        extractor, model, _ = load_model(device, output)
        with trace_failures(output, 'extract'):
            for job in jobs:
                if job['call_id'] in done:
                    continue
                try:
                    embedding, quality = base.encode_call(job, extractor, model, device)
                    commit_call(call_cache_path(cache, job), job, embedding, quality, fingerprint)
                except Exception as error:
                    save_json(output / 'failures' / f'cache-or-call-{uuid.uuid4().hex}.json',
                              {'call_id': job['call_id'], 'error': repr(error)})
                    progress(failed=job['call_id'])
                    raise
                done.add(job['call_id'])
                progress(last=job['call_id'])
                if len(done) % 10 == 0:
                    print(f'Completed {len(done)}/{len(jobs)}', flush=True)
        summary = audit_full_cache(jobs, output, identity)
        print(json.dumps(summary, indent=2), flush=True)
        print('Extraction complete; take the cache/index/identity/summary/logs to local evaluation.')
    finally:
        lock.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', required=True, choices=['cuda', 'preflight', 'smoke', 'benchmark', 'extract'])
    parser.add_argument('--confirm-full-extraction', action='store_true')
    args = parser.parse_args()
    if args.stage == 'extract' and not args.confirm_full_extraction:
        parser.error('STOP: full extraction requires --confirm-full-extraction')
    config = load_config(args.config)
    device, gpu = cuda_info()
    if args.stage == 'cuda':
        return
    output = Path(config['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    jobs, identity = load_inputs(config)
    identity['execution_environment'] = execution_identity(gpu)
    if args.stage == 'preflight':
        selected = {j['call_id']: j for mode in ('smoke', 'benchmark') for j in saved_subset(jobs, mode)}
        preflight(jobs, output, identity, audio_jobs=list(selected.values()))
    elif args.stage in ('smoke', 'benchmark'):
        subset_run(jobs, output, identity, args.stage, device, gpu)
    else:
        extract_future(jobs, output, identity, device, args.confirm_full_extraction)


if __name__ == '__main__':
    main()
