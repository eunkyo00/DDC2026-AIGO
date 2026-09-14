"""Direct JSON audit for the frozen cohort; never recreates or edits its split."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import unicodedata


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def sha(text):return hashlib.sha256(text.encode()).hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig',newline='') as f:yield from csv.DictReader(f)


def read_one(item):
    key,path=item
    raw=path.read_bytes()
    return key,json.loads(raw.decode('utf-8-sig')),hashlib.sha256(raw).hexdigest()


def keys(obj,prefix='$'):
    yield prefix,type(obj).__name__
    if isinstance(obj,dict):
        for k,v in obj.items():yield from keys(v,prefix+'.'+k)
    elif isinstance(obj,list):
        for v in obj:yield from keys(v,prefix+'[]')


def main():
    p=argparse.ArgumentParser();p.add_argument('--validation-dir',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--eda-dir',type=Path,default=Path(__file__).resolve().parents[2]/'eda/eda_outputs')
    p.add_argument('--out-dir',type=Path,default=Path(__file__).resolve().parent/'results')
    a=p.parse_args();out=a.out_dir
    if out.exists():raise ValueError('Result directory exists; choose a new audit directory')
    meta=json.loads((a.validation_dir/'split_metadata.json').read_text())
    for n,h in meta['output_hashes'].items():assert digest(a.validation_dir/n)==h,n
    for n,h in meta['source_hashes'].items():assert digest(a.eda_dir/n)==h,n
    frozen={r['call_id']:r for r in rows(a.validation_dir/'split_assignments.csv')}
    assert len(frozen)==27985
    root=Path(json.loads((a.eda_dir/'run_metadata.json').read_text())['data_root'])
    paths=defaultdict(list)
    for directory,dirs,names in os.walk(root/'Training'):
        dirs[:]=[d for d in dirs if d!='__MACOSX' and not d.startswith('.')]
        for n in names:
            if n.lower().endswith('.json') and not n.startswith('._'):
                paths[sha(unicodedata.normalize('NFC',Path(n).stem))].append(Path(directory)/n)
    expected=defaultdict(list)
    for r in rows(a.eda_dir/'segments.csv'):
        if r['split']=='Training' and r['file_ref'] in frozen:
            expected[r['file_ref']].append((round(float(r['start_s'])*1000),round(float(r['end_s'])*1000),int(r['speaker'])))
    items=[]
    for key in sorted(frozen):
        assert len(paths[key])==1,key
        items.append((key,paths[key][0]))
    id_counts={k:Counter() for k in ('_id','recordId','audioPath')}
    partitions={k:defaultdict(set) for k in id_counts};missing=Counter();schema=defaultdict(Counter);fingerprints=[]
    success=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        it=iter(items)
        while batch:=list(islice(it,32)):
            for key,d,h in pool.map(read_one,batch):
                assert d['gender']==frozen[key]['gender'],'Gender changed since frozen EDA'
                actual=[(u['startAt'],u['endAt'],u['speaker']) for u in d['utterances']]
                assert actual==expected[key],'Segment boundaries/roles changed since EDA'
                for name in id_counts:
                    value=d.get(name)
                    if not isinstance(value,str) or not value:missing[name]+=1;continue
                    hashed=sha(value);id_counts[name][hashed]+=1;partitions[name][frozen[key]['partition']].add(hashed)
                for path,typ in keys(d):schema[path][typ]+=1
                fingerprints.append({'call_id':key,'json_sha256':h})
                success+=1
            if success%2048==0:print('Direct JSON audit',success,'/',len(items),flush=True)
    result={'cohort_calls':len(frozen),'json_parsed':success,'gender_mismatches':0,'segment_index_mismatches':0,
            'split_assignment_sha256':digest(a.validation_dir/'split_assignments.csv'),
            'stratification':{'sklearn_train_test_split_used':False,'seed':42,'algorithm':'gender-wise SHA256 seeded ranking',
                'rounding':'len(gender_group)//5 for internal validation; remainders forbidden by v1 assertion',
                'why_ratios_identical':'13100 and 14885 are exactly divisible by 5; quotas are 2620 and 2977',
                'split_recreated':False},'ids':{},'schema':schema,
            'speaker_level_risk':'Call-level leakage는 검증 완료했지만 서로 다른 call_id에 동일 신고자가 반복 등장하는 speaker-level leakage 여부는 현재 메타데이터만으로 확인할 수 없다.',
            'risk_status':'known unresolved risk','method':'All selected Training JSON reread directly; no quoted EDA uniqueness substitution'}
    for name,c in id_counts.items():
        sets=partitions[name]
        result['ids'][name]={'total_rows':len(frozen),'nonempty_rows':sum(c.values()),'unique_ids':len(c),
            'missing_rows':missing[name],'duplicate_excess_rows':sum(v-1 for v in c.values()),
            'duplicate_values':sum(v>1 for v in c.values()),
            'train_unique_ids':len(sets['train']),'internal_validation_unique_ids':len(sets['internal_validation']),
            'internal_overlap':len(sets['train']&sets['internal_validation'])}
    result['person_identifier_key_candidates']=[k for k in schema if re.search(r'caller|person|phone|telephone|anonym|customer|speaker',k,re.I)]
    result['person_identifier_assessment']='Only utterances[].speaker role field found; _id/recordId/audioPath are call/source identifiers without evidence of stable person linkage. utterances[].id identifies an utterance.'
    assert all(v['missing_rows']==v['duplicate_excess_rows']==v['internal_overlap']==0 for v in result['ids'].values())
    for n,h in meta['output_hashes'].items():assert digest(a.validation_dir/n)==h,'Frozen artifact modified'
    out.mkdir(parents=True)
    (out/'sanity_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (out/'json_fingerprints.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['call_id','json_sha256']);w.writeheader();w.writerows(fingerprints)
    lines=['# STEP 3 추가 sanity check','',
        '고정 split과 EDA 파일은 수정하거나 다시 만들지 않았다. 대상 Training JSON 27,985개를 직접 읽었다.','',
        '## Stratification','',
        '`sklearn.train_test_split`은 사용하지 않았다. seed 42와 gender/call_id로 SHA256 순서를 만들고 gender별 앞 n//5를 Internal Validation으로 배정했다. '
        '나머지가 있으면 오류를 내므로 반올림 손실은 없다. 남 13,100과 여 14,885가 모두 5의 배수이므로 양쪽 비율이 동일하다. 기존 구현에 문제는 없으며 split은 유지했다.','',
        '| ID | row | unique | duplicate 초과 row | Train/Validation overlap |','|---|---:|---:|---:|---:|']
    for name,v in result['ids'].items():lines.append(f"| {name} | {v['total_rows']} | {v['unique_ids']} | {v['duplicate_excess_rows']} | {v['internal_overlap']} |")
    lines+=['','gender와 전체 발화 경계/역할을 기존 EDA 인덱스에 대조했고 불일치 0건이다. JSON fingerprint는 가명 call_id와 SHA256만 저장했다. 원본 ID·주소·전사 원문은 내보내지 않았다.','',
        '## Speaker-level leakage — known unresolved risk','',result['speaker_level_risk'],'',
        '신고자 고유 ID, anonymized phone hash, caller/person ID에 해당하는 별도 필드는 발견하지 못했다. '
        '`speaker`는 0/1 역할이며 개인 식별자가 아니다. `_id`, `recordId`, `audioPath`의 고유성은 사람 단위 분리 근거가 아니다. '
        'ID 인코딩에 숨겨진 업무 의미나 외부 연결표의 존재도 확인되지 않았으며, 억지로 추정하지 않았다.','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'json_parsed':success,'ids':result['ids'],'person_candidates':result['person_identifier_key_candidates']},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
