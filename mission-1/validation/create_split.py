"""Frozen call-level split using EDA indexes only. Python standard library; no audio/model imports."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import unicodedata

SEED = 42
EXPECTED_CALLS = 27985
EXPECTED_GENDERS = {'M': 13100, 'F': 14885}
ALGORITHM = 'gender-stratified SHA256 seeded rank v1; validation quota 1/5 per gender'
BINS = ['<0.5s', '0.5–1s', '1–2s', '2–4s', '>=4s']
SOURCES = ['calls.csv', 'segments.csv', 'audio_headers.csv', 'leakage.json', 'inventory.json', 'run_metadata.json']
FIELDS = ['call_id', 'gender', 'wav_path', 'wav_relative_path', 'n_segments', 'total_duration',
          'reporter_union_duration', 'n_all_segments', 'all_segment_duration', 'partition']


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, fields, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: row[k] for k in fields} for row in rows)


def validate_calls(rows, expected=None):
    seen = {}
    for r in rows:
        key = r['call_id']
        require(key and len(key)==64 and all(c in '0123456789abcdef' for c in key), 'Invalid call_id')
        require(r['gender'] in ('M', 'F'), f'Missing/invalid gender: {key}')
        if key in seen:
            require(seen[key]==r['gender'], f'Conflicting genders for call_id: {key}')
            raise ValueError(f'Duplicate call_id: {key}')
        seen[key] = r['gender']
    if expected is not None:
        require(len(seen)==expected, f'Expected {expected} calls, got {len(seen)}')


def assign(rows, seed=SEED):
    """Hash-based pseudorandom order, stable across Python versions and input row order."""
    validate_calls(rows)
    result = {}
    for gender in ('F', 'M'):
        group = [r['call_id'] for r in rows if r['gender']==gender]
        require(len(group)>=5 and len(group)%5==0, 'v1 requires each gender count divisible by 5')
        order = sorted(group, key=lambda key: (sha(f'{seed}\0{gender}\0{key}'), key))
        n_val = len(order)//5
        result.update({key: 'internal_validation' if i<n_val else 'train' for i,key in enumerate(order)})
    return result


def describe(values):
    v = list(values)
    return {'mean': statistics.mean(v), 'median': statistics.median(v), 'min': min(v), 'max': max(v)}


def distribution(rows):
    counts = Counter(r['gender'] for r in rows)
    bins = Counter()
    for r in rows:
        bins.update(r['_bins'])
    n_segments = sum(r['n_segments'] for r in rows)
    return {'calls': len(rows), 'gender_counts': dict(counts),
            'gender_ratios': {g: counts[g]/len(rows) for g in ('M','F')},
            'reporter_segments': n_segments, 'all_speaker_segments': sum(r['n_all_segments'] for r in rows),
            'segments_per_call': describe(r['n_segments'] for r in rows),
            'total_reporter_duration_s': sum(r['_ms'] for r in rows)/1000,
            'total_reporter_union_duration_s': round(sum(r['reporter_union_duration'] for r in rows),3),
            'duration_per_call_s': describe(r['total_duration'] for r in rows),
            'duration_bins_counts': {b: bins[b] for b in BINS},
            'duration_bins_ratios': {b: bins[b]/n_segments for b in BINS}}


def check_partitions(rows, assignment):
    ids = {r['call_id'] for r in rows}
    require(set(assignment)==ids, 'Split membership does not exactly cover eligible calls')
    require(set(assignment.values())=={'train','internal_validation'}, 'Invalid partition names')
    train = [r for r in rows if assignment[r['call_id']]=='train']
    valid = [r for r in rows if assignment[r['call_id']]=='internal_validation']
    tests = {}
    for field, label in [('call_id','call_id_overlap'),('wav_path','wav_path_overlap'),
                         ('wav_relative_path','relative_wav_path_overlap'),('_prefix','filename_prefix_overlap')]:
        count = len({r[field] for r in train} & {r[field] for r in valid})
        tests[label] = count
        require(count==0, f'{label}: {count}')
    require(len({r['wav_path'] for r in rows})==len(rows), 'Repeated wav_path in dataset')
    return train, valid, tests


def build(eda, data_root, frozen=None):
    rows = []
    for r in read_rows(eda/'calls.csv'):
        # Official Validation is excluded before eligibility or sampling decisions.
        if r['split']!='Training' or r['matched']!='True':
            continue
        require(r['seoul_address']=='True', 'Non-Seoul call in eligible Training')
        require(int(r['invalid_segments'])==0, 'Invalid segments require a new reviewed dataset version')
        rows.append({'call_id':r['file_ref'], 'gender':r['gender'],
                     'n_segments':int(r['n_valid_reporter_segments']), 'total_duration':float(r['reporter_sum_s']),
                     'reporter_union_duration':round(float(r['reporter_union_s']),3),
                     'n_all_segments':int(r['n_segments']), '_bins':Counter(), '_ms':0, '_all_ms':0,
                     '_observed':0, '_all_observed':0})
    validate_calls(rows, EXPECTED_CALLS)
    require(dict(Counter(r['gender'] for r in rows))==EXPECTED_GENDERS, 'Gender counts differ from frozen dataset specification')
    by_id = {r['call_id']:r for r in rows}
    # Directory listing only. No WAV/JSON contents, audio decoders or feature computation.
    wav_map = defaultdict(list)
    for directory, dirs, files in os.walk(data_root/'Training'):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.') and d!='__MACOSX')
        for name in sorted(files):
            if name.startswith('._') or not name.lower().endswith('.wav'):
                continue
            stem = unicodedata.normalize('NFC', Path(name).stem)
            wav_map[sha(stem)].append((Path(directory)/name, stem))
    for r in rows:
        entries = wav_map[r['call_id']]
        require(len(entries)==1, f'Missing or ambiguous WAV filename: {r["call_id"]}')
        path, stem = entries[0]
        r.update(wav_path=unicodedata.normalize('NFC',str(path)),
                 wav_relative_path=unicodedata.normalize('NFC',str(path.relative_to(data_root))),
                 _prefix=sha(stem.split('_')[0]))
    assignment = frozen if frozen is not None else assign(rows)
    train, valid, checks = check_partitions(rows, assignment)
    header_ids = set()
    for r in read_rows(eda/'audio_headers.csv'):
        if r['split']=='Training' and r['file_ref'] in by_id:
            require(r['riff_status']=='ok', 'EDA WAV structure check failed')
            require(r['file_ref'] not in header_ids, 'Duplicate EDA audio header reference')
            header_ids.add(r['file_ref'])
    require(header_ids==set(by_id), 'Missing EDA audio header coverage')
    segment_assignments = Counter()
    for seg in read_rows(eda/'segments.csv'):
        if seg['split']!='Training' or seg['file_ref'] not in by_id:
            continue
        require(seg['matched']=='True' and seg['exceeds_wav']=='False', 'Invalid eligible segment flags')
        r = by_id[seg['file_ref']]
        require(seg['speaker'] in ('0','1'), 'Unexpected speaker role')
        start_ms, end_ms = round(float(seg['start_s'])*1000), round(float(seg['end_s'])*1000)
        duration_ms = end_ms-start_ms
        require(start_ms>=0 and duration_ms>0, 'Invalid segment timing')
        require(abs(float(seg['duration_s'])-duration_ms/1000)<1e-8, 'Segment duration inconsistent with millisecond boundaries')
        r['_all_observed'] += 1
        r['_all_ms'] += duration_ms
        # One parent lookup for every segment, including dispatcher segments.
        partition = assignment[seg['file_ref']]
        segment_assignments[partition] += 1
        if seg['speaker']=='1':
            require(seg['gender']==r['gender'], f'Conflicting segment gender: {r["call_id"]}')
            r['_observed'] += 1
            r['_ms'] += duration_ms
            index = 0 if duration_ms<500 else 1 if duration_ms<1000 else 2 if duration_ms<2000 else 3 if duration_ms<4000 else 4
            r['_bins'][BINS[index]] += 1
    for r in rows:
        require(r['_observed']==r['n_segments'] and r['_all_observed']==r['n_all_segments'], 'EDA call/segment counts do not reconcile')
        require(abs(r['total_duration']-r['_ms']/1000)<1e-7, 'EDA call/segment durations do not reconcile')
        require(r['n_segments']>0, 'No reporter segments')
        r['total_duration'] = r['_ms']/1000
        r['all_segment_duration'] = r['_all_ms']/1000
        r['partition'] = assignment[r['call_id']]
    checks.update(segment_leakage_calls=0, unassigned_segments=0, duplicate_call_ids=0,
                  missing_gender=0, conflicting_gender=0, official_validation_calls_in_split=0,
                  all_segment_partition_counts=dict(segment_assignments),
                  reporter_segments_checked=sum(r['n_segments'] for r in rows),
                  segment_rule='Each existing EDA row inherits exactly one partition from its unique parent call_id; no segment index copied')
    legacy = json.loads((eda/'leakage.json').read_text())
    additional = {}
    for key in ('_id','recordId','audioPath'):
        evidence = legacy[key]
        require(evidence['unique_values']>=len(rows) and evidence['duplicate_values']==0,
                f'EDA {key} evidence is insufficient; require per-call group audit')
        additional[key] = {'internal_overlap':0,
                           'basis':'Inherited EDA global uniqueness among parsed calls: a subset cannot introduce a duplicate. Raw IDs not reread.',
                           'eda_unique_values':evidence['unique_values']}
    checks['additional_ids'] = additional
    checks['same_person'] = '확인 필요: speaker is a role, not a persistent person identifier'
    checks['edited_original_audio'] = '확인 필요: edited/re-encoded audio and event identity not established'
    checks['audio_bytes'] = 'Prior EDA found no exact-byte duplicates; WAV contents were not rehashed in this step'
    stats = {'train':distribution(train), 'internal_validation':distribution(valid)}
    quality = {}
    for field in ('n_segments','total_duration'):
        x,y = [r[field] for r in train], [r[field] for r in valid]
        pooled = math.sqrt((statistics.variance(x)+statistics.variance(y))/2)
        smd = (statistics.mean(y)-statistics.mean(x))/pooled if pooled else 0.0
        quality[field+'_standardized_mean_difference'] = smd
    quality['max_duration_bin_difference_pp'] = max(abs(stats['train']['duration_bins_ratios'][b]-stats['internal_validation']['duration_bins_ratios'][b])*100 for b in BINS)
    quality['gender_difference_pp'] = abs(stats['train']['gender_ratios']['F']-stats['internal_validation']['gender_ratios']['F'])*100
    quality['review_thresholds'] = 'Descriptive flag: |SMD| > 0.1 or max bin difference > 2 percentage points; never search seeds to improve this result'
    quality['review_flag'] = any(abs(quality[k])>.1 for k in quality if k.endswith('standardized_mean_difference')) or quality['max_duration_bin_difference_pp']>2
    majority = max(('M','F'), key=lambda g:(stats['train']['gender_counts'][g],g))
    correct = stats['internal_validation']['gender_counts'][majority]
    summary = {'dataset_version':'m1_internal_call_v1','source_split':'Training only', 'seed':SEED,
               'algorithm':ALGORITHM,'ratio':{'train':.8,'internal_validation':.2},'stratify':'gender',
               'total_calls':len(rows), 'statistics':stats,'quality':quality,
               'majority_baseline':{'selected_from':'internal train counts only','class':majority,
                                    'correct_calls':correct,'evaluated_calls':len(valid),'accuracy':correct/len(valid)},
               'ready_for_step4_split':not quality['review_flag'],
               'column_semantics':{'call_id':'EDA file_ref = SHA256(NFC WAV stem)',
                   'n_segments':'reporter segments only (speaker=1)',
                   'total_duration':'reporter segment duration sum, seconds; overlaps counted as in EDA',
                   'reporter_union_duration':'EDA reporter interval union seconds; no audio aggregation performed',
                   'n_all_segments':'all speaker roles; each follows the same parent call partition'}}
    return sorted(rows,key=lambda r:r['call_id']), assignment, summary, checks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--eda-dir',type=Path,default=Path(__file__).resolve().parents[1]/'eda/eda_outputs')
    ap.add_argument('--data-root',type=Path)
    ap.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--verify',action='store_true',help='Read-only verification of existing frozen assignments and source/output checksums')
    args = ap.parse_args()
    eda,out = args.eda_dir.resolve(),args.out_dir.resolve()
    data_root = (args.data_root or Path(json.loads((eda/'run_metadata.json').read_text())['data_root'])).resolve()
    require((data_root/'Training').is_dir(),'Training data directory unavailable')
    require(out!=data_root and data_root not in out.parents, 'Output must not be inside original data')
    require(out!=eda and eda not in out.parents, 'Output must not be inside EDA artifacts')
    outputs = ['split_assignments.csv','split_summary.json','validation_checks.json','split_metadata.json','REPORT.md',
               'manifests/calls.csv','manifests/train_calls.csv','manifests/val_calls.csv']
    hashes = {name:file_hash(eda/name) for name in SOURCES}
    frozen = None
    if args.verify:
        meta = json.loads((out/'split_metadata.json').read_text())
        require(meta['source_hashes']==hashes,'EDA inputs changed; do not silently redefine v1')
        for name,h in meta['output_hashes'].items():
            require(file_hash(out/name)==h,f'Frozen artifact modified: {name}')
        frozen_rows = list(read_rows(out/'split_assignments.csv'))
        validate_calls(frozen_rows,EXPECTED_CALLS)
        frozen = {r['call_id']:r['partition'] for r in frozen_rows}
    else:
        require(not any((out/name).exists() for name in outputs),'Frozen outputs already exist: use --verify; no overwrite option')
    rows,assignment,summary,checks = build(eda,data_root,frozen)
    require(assignment==assign(list(reversed(rows))), 'Assignment not reproducible with reversed input order')
    checks['seed_reproduction_passed'] = True
    if args.verify:
        require(summary==json.loads((out/'split_summary.json').read_text()),'Recomputed summary differs')
        require(checks==json.loads((out/'validation_checks.json').read_text()),'Recomputed validation differs')
        next_gender = {r['call_id']:r['gender'] for r in rows}
        for r in frozen_rows:
            require(r['gender']==next_gender[r['call_id']], 'Frozen gender mismatch')
        for name,part in [('calls',None),('train_calls','train'),('val_calls','internal_validation')]:
            actual = list(read_rows(out/f'manifests/{name}.csv'))
            expected = [{k:str(r[k]) for k in FIELDS} for r in rows if part is None or r['partition']==part]
            require(actual==expected,f'Local manifest differs from reconstructed eligible calls: {name}')
        print('PASS: frozen file hashes, source hashes, deterministic membership, distributions and leakage checks')
        return
    out.mkdir(parents=True,exist_ok=True)
    (out/'manifests').mkdir(exist_ok=True)
    write_csv(out/'split_assignments.csv',['call_id','gender','partition'],rows)
    write_csv(out/'manifests/calls.csv',FIELDS,rows)
    write_csv(out/'manifests/train_calls.csv',FIELDS,[r for r in rows if r['partition']=='train'])
    write_csv(out/'manifests/val_calls.csv',FIELDS,[r for r in rows if r['partition']=='internal_validation'])
    write_json(out/'split_summary.json',summary)
    write_json(out/'validation_checks.json',checks)
    from report import render
    (out/'REPORT.md').write_text(render(summary,checks),encoding='utf-8')
    write_json(out/'split_metadata.json',{'created_utc':datetime.now(timezone.utc).isoformat(),
        'python':sys.version,'seed':SEED,'algorithm':ALGORITHM,'expected_gender_counts':EXPECTED_GENDERS,
        'source_hashes':hashes,'generator_sha256':file_hash(Path(__file__)),
        'output_hashes':{name:file_hash(out/name) for name in outputs if name!='split_metadata.json'},
        'frozen_policy':'v1 membership must be reused by all models. Verification never writes. Changed cohort requires a new explicitly reviewed version.',
        'source_paths_note':'EDA paths configured via --eda-dir; WAV root from EDA run_metadata or --data-root; no official Validation members used'})
    print(json.dumps({'calls':summary['total_calls'],'partitions':{k:v['calls'] for k,v in summary['statistics'].items()},
                      'majority':summary['majority_baseline'],'quality':summary['quality']},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
