"""Render the human-readable EDA report from completed local audit outputs."""
import argparse
import json
from pathlib import Path
import pandas as pd


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"]*len(headers)) + " |"] +
                     ["| " + " | ".join(map(str,row)) + " |" for row in rows])


def f(x):
    return "확인 불가" if x is None or pd.isna(x) else f"{x:,.3f}"


def pct(x):
    return "확인 불가" if x is None or pd.isna(x) else f"{100*x:.2f}%"


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results",type=Path,default=Path(__file__).parent/"eda_outputs")
    ap.add_argument("--out",type=Path,default=Path(__file__).parent/"REPORT.md")
    args=ap.parse_args()
    root=args.results
    s=json.loads((root/"summary.json").read_text())
    run=json.loads((root/"run_metadata.json").read_text())
    schema=json.loads((root/"schema.json").read_text())
    examples=json.loads((root/"schema_examples_redacted.json").read_text())
    calls=pd.read_csv(root/"calls.csv")
    segments=pd.read_csv(root/"segments.csv")
    acoustic=pd.read_csv(root/"sample_acoustics.csv") if (root/"sample_acoustics.csv").exists() and not run['arguments']['skip_acoustics'] else pd.DataFrame()
    report=["# DDC Mission 1 — 데이터 구조와 EDA", "",
        "이번 결과는 외장 드라이브의 실제 파일을 읽어 계산했다. 기존 README 수치를 실측값으로 대신하지 않았다. "
        "모델 구현·학습·성별 예측은 하지 않았다. 원본 데이터와 기존 팀원 파일은 수정하지 않았고, "
        "`mission1/eunkyo` 브랜치의 `mission-1/eda/`에만 작업 산출물을 추가했다.", "",
        f"분석 경로: `{run['data_root']}`. 완료 실행 소요 시간: {run['elapsed_seconds']/60:.1f}분 "
        "(앞선 I/O 방식 점검·중단 실행 시간 제외). 원본은 읽기 전용으로 취급했다.", "",
        "## 1. 데이터 구조", "",
        "저장소는 `common/`(공통 EDA·전처리), `mission-1/`(미션별 코드), `0.중요/`(규칙) 구조다. "
        "기존 기본 데이터 경로는 존재하지 않고 환경변수 override도 없어 실제 외장 드라이브를 지정했다.", "",
        "```text\n대학부 데이터/\n├── Training/\n│   ├── 1.원천데이터/TS_서울_구급/*.wav\n│   └── 2.라벨링데이터/TL_서울_구급/*.json\n└── Validation/\n    ├── 1.원천데이터/VS_서울_구급/*.wav\n    └── 2.라벨링데이터/VL_서울_구급/*.json\n```", "",
        "split 안에서 확장자를 제외한 stem을 NFC 정규화해 매칭한다. 파일명 예시는 "
        "`<24자리 ID>_<8자리 날짜 형태 토큰>.wav/json`이며 실제 ID는 공개하지 않는다. "
        "날짜 형태 토큰의 정확한 업무 의미는 확인 필요다. `audioPath`는 로컬 경로 매칭에 사용하지 않았다.", "",
        "전체 JSON에서 확인한 **모든 키 경로와 타입**은 다음과 같다. 개수는 그 경로의 값 등장 횟수다.", "",
        table(["키 경로","타입: 횟수"],[[k,", ".join(f"{t}: {n:,}" for t,n in v.items())] for k,v in schema.items()]), "",
        "최상위 `startAt/endAt`은 통화 범위, `utterances[].startAt/endAt`은 발화 경계다. "
        "단위는 밀리초이며 WAV 초 단위 길이와 비교했다. 실제 화자 키는 **소문자 `speaker`**다. "
        "역할 매핑 `0=119대원/상담원`, `1=신고자`는 저장소 `common/config.py`의 정의를 사용했다. "
        "이는 화자 역할이며 고유한 사람 식별자가 아니다. 라벨만으로 통화당 실제 사람 수를 확정할 수 없다.", "",
        "`gender`는 최상위의 M(남)/F(여) 라벨로 신고자 통화에 적용했다. 상담원 segment에 해당 라벨을 복사하지 않았다. "
        "동일 역할에 복수 실제 사람이 포함되는 통화의 처리, 성별 라벨 작성 기준은 공식 데이터 명세 확인 필요다.", "",
        "실제 JSON 예시 하나의 전체 키 구조(발화는 앞 3개, 비허용/개인정보 값은 마스킹):", "",
        "```json\n"+json.dumps(examples[0]['example'],ensure_ascii=False,indent=2)+"\n```", "",
        "실제로 읽은 다른 JSON 예시(전체 비식별 예시 6개는 `schema_examples_redacted.json`):", "",
        table(["split/예시","gender","통화 endAt(ms)","발화 수","앞 3개 발화 (startAt, endAt, speaker)"],
              [[f"{e['split']}/{i+1}",e['example']['gender'],e['example']['endAt'],e['total_utterances'],
                "; ".join(str((u.get('startAt'),u.get('endAt'),u.get('speaker'))) for u in e['example']['utterances'])]
               for i,e in enumerate(examples)]), "", "## 2. 파일 규모와 성별", ""]
    inv=s['inventory']
    report[6:6]=[
        f"- Training: WAV {inv['Training']['wav_files']:,}개, JSON {inv['Training']['json_files']:,}개, "
        f"WAV 누락 {inv['Training']['json_only_stems']:,}개. 정상 JSON과 WAV가 연결된 통화 "
        f"{int(((calls.split=='Training')&calls.matched).sum()):,}개.",
        f"- JSON 파싱/스키마 실패: {sum(v['json_files'] for v in inv.values())-len(calls):,}개. "
        "실패 파일을 압축본 라벨로 대체하지 않았다.",
        f"- 정상 파싱 통화의 {pct((calls.reporter_dispatcher_overlap_s>0).mean())}에 신고자·상담원 라벨 구간 겹침이 있다. "
        "신고자 crop을 순수한 한 화자 음성으로 단정할 수 없다.",
        "- 고유 인물/사건 단위 분리와 편집된 원본 중복은 확인 필요다. F0 단독 충분성이나 분류 성능은 평가하지 않았다.", ""]
    report += [table(["split","WAV","JSON","1:1 매칭","JSON만 존재","WAV만 존재","중복 stem WAV/JSON"],
        [[k,v['wav_files'],v['json_files'],v['matched_unique_pairs'],v['json_only_stems'],v['wav_only_stems'],
          f"{v['duplicate_wav_stems']}/{v['duplicate_json_stems']}"] for k,v in inv.items()]), "",
        "위 수는 macOS `._*` 보조 파일을 제외한 값이다. 보조 파일 제외 개수: "+
        "; ".join(f"{k}: {v['ignored_macos_sidecars']}" for k,v in inv.items())+".", "",
        "Training의 현재 WAV 수는 기존 README의 29,200개와 다르다. **누락 WAV를 확보하기 전에는 "
        "전체 라벨 수를 학습 가능한 통화 수로 쓰면 안 된다.** Downloads의 부분 추출 폴더는 합치지 않았다.", ""]
    report += ["파일명 매칭 성공과 JSON 파싱 성공은 구분해야 한다.", "",
        table(['split','JSON 파일 수','JSON 파싱 성공','파싱/스키마 실패','파일명 매칭 + 파싱 성공'],
              [[k,v['json_files'],int((calls.split==k).sum()),v['json_files']-int((calls.split==k).sum()),
                int(((calls.split==k)&calls.matched).sum())] for k,v in inv.items()]), ""]
    if (root/'json_failure_audit.json').exists():
        failed=json.loads((root/'json_failure_audit.json').read_text())
        report += ["실패 JSON은 원본을 수정하지 않고 압축본의 같은 이름 파일과 별도로 비교했다.", "",
            table(['가명 참조 앞 12자리','로컬 해독/파싱','로컬/압축본 bytes','바이트 일치','압축본 파싱','압축본 gender'],
                  [[r['file_ref'][:12],r.get('local_parse','확인 필요'),f"{r.get('local_bytes')}/{r.get('archive_bytes')}",
                    r.get('bytes_equal','확인 필요'),r.get('archive_parse','확인 필요'),r.get('archive_gender','확인 필요')]
                   for r in failed]), "",
            "파일 크기가 같아도 내용 무결성을 보장하지 않는다. 압축본과 다른 바이트가 확인된 로컬 JSON은 "
            "손상/변경 원인을 확인하고 검증된 사본으로 복구할지 결정해야 한다. 이번에는 교체하지 않았고 "
            "실패 JSON의 통화·segment를 실측 파싱 통계에서 제외했다. 압축본 라벨로 조용히 대체하지 않았다.", ""]
    if (root/'archive_crosscheck.json').exists():
        archive=json.loads((root/'archive_crosscheck.json').read_text())
        same = all(archive[k]['wav_count']==v['wav_files'] and archive[k]['json_count']==v['json_files']
                   and all(archive[k]['genders'].get(g,0)==s['groups'][k+'/all_labels']['genders'][g]['calls'] for g in ('M','F'))
                   for k,v in inv.items())
        report += ["Downloads `대학부 데이터.zip`의 중앙 디렉터리 및 전체 JSON 교차 검사와 이번 결과의 "
          f"WAV/JSON·성별 개수 일치 여부: **{'일치' if same else '불일치 — 원본/압축본 범위 확인 필요'}**. "
          "최초 작업 때 별도로 조사한 `Training.zip` 목록은 WAV 27,987 / JSON 29,200이었다. "
          "압축본과 외장 드라이브 전체 파일의 바이트 동일성을 검증한 것은 아니다. "
          "따라서 외장 드라이브 복사 과정만의 문제라고 단정하지 않는다.", "",
          "압축본에서 WAV가 없는 Training JSON의 성별: "+str(archive['Training']['missing_wav_gender'])+". "
          "파일명의 날짜 형태 토큰별 누락 수: "+str(archive['Training']['missing_wav_date_tokens'])+". "
          "누락이 특정 토큰에 집중되므로 확보 후 분포를 다시 계산해야 한다.", ""]
    demographics=[]
    for key,g in s['groups'].items():
        total_seg=sum(v['reporter_segments'] for v in g['genders'].values())
        for gender,v in g['genders'].items():
            if not v['calls']: continue
            demographics.append([key,gender,v['calls'],pct(v['call_ratio']),v['reporter_segments'],
                                 pct(v['reporter_segments']/total_seg) if total_seg else '-'])
    report += [table(["범위","gender","통화 수","통화 비율","유효 신고자 segment","segment 비율"],demographics), "",
        "`all_labels`는 모든 파싱 가능한 JSON, `matched`는 WAV·JSON이 각각 하나씩 있는 통화다. "
        "통화별 신고자 표본 수는 위 통화 수와 대응하지만, **중복 제거된 실제 신고자 인원수는 확인 필요**다. "
        "segment 비율은 발화가 많은 통화에 더 큰 가중치가 생긴 분포이며 독립적인 사람 표본 분포가 아니다.", "",
        "![Gender distribution](eda_outputs/gender_distribution.png)", "", "## 3. 발화 길이 및 통화별 발화량", ""]
    duration_rows=[]
    for key,g in s['groups'].items():
        if not key.endswith('all_labels'): continue
        for label,v in [('all speakers',g['all_segment_duration_s'])]+[(k,d['segment_duration_s']) for k,d in g['genders'].items() if d['calls']]:
            duration_rows.append([key.split('/')[0],label,v['n']]+[f(v.get(t)) for t in ('mean','median','min','max','q25','q75','q90','q95')])
    report += [table(["split","대상","유효 segment 수","평균 s","중앙값","최소","최대","Q25","Q75","Q90","Q95"],duration_rows), "",
        "M/F 행은 신고자만 포함한다. 시간 누락·비수치·음수 시작·end≤start는 길이 집계에서 제외하고 문제 목록에 남긴다. "
        "양의 길이지만 WAV 경계를 넘는 구간은 라벨 통계에 남겨 두고 별도로 표시한다. "
        "이 표는 전체 라벨 기준이며 WAV가 매칭된 부분집합의 같은 통계도 `summary.json`에 있다.", ""]
    short=[]
    for key,g in s['groups'].items():
        if not key.endswith('all_labels'):continue
        for label,counts,n in [('all speakers',g['all_short_segments'],g['all_segments'])]+[(k,v['short_segments'],v['reporter_segments']) for k,v in g['genders'].items() if v['calls']]:
            short.append([key.split('/')[0],label]+[f"{counts[str(t)]:,} ({pct(counts[str(t)]/n)})" for t in (.5,1,2)])
    report += [table(["split","대상","<0.5s","<1s","<2s"],short), "",
        "길이 임계값은 엄격한 미만(<)이며 서로 중첩된 집계다. 짧은 발화를 제외하면 사용할 수 있는 "
        "신고자 정보와 통화별 가중치가 달라지므로, 지금 임의로 삭제하지 않았다.", "",
        table(["split","파일당 전체 segment 평균/중앙값/최대","speaker 값 개수 min/max","신고자 0개 통화"],
              [[k.split('/')[0],'/'.join(f(g['segments_per_call'][t]) for t in ('mean','median','max')),
                '/'.join(f(g['speaker_values_per_call'][t]) for t in ('min','max')),g['zero_reporter_calls']]
               for k,g in s['groups'].items() if k.endswith('all_labels')]), ""]
    percall=[]
    for key,g in s['groups'].items():
        if not key.endswith('all_labels'):continue
        for gender,v in g['genders'].items():
            if not v['calls']:continue
            percall.append([key.split('/')[0],gender,f(v['reporter_segments_per_call']['mean']),f(v['reporter_segments_per_call']['median']),
                            f(v['reporter_sum_s']['mean']),f(v['reporter_sum_s']['median']),
                            f(v['reporter_union_s']['mean']),f(v['reporter_union_s']['median']),pct(v['multiple_reporter_segments_ratio'])])
    report += [table(["split","gender","신고자 segment 평균","중앙값","단순 합 평균 s","중앙값","합집합 평균 s","중앙값","다중 segment 통화 비율"],percall), "",
        "발화 시간의 단순 합과 합집합은 다를 수 있다. 같은 역할의 겹친 구간을 그대로 이어 붙이면 "
        "동일 시간대 음성이 반복될 수 있다. 상담원·신고자 annotation 교집합은 다음과 같다.", "",
        table(["split","겹침 있는 통화/전체","비율","통화별 신고자 시간 중 겹침 비율 중앙값","Q95"],
              [[split,f"{int((c.reporter_dispatcher_overlap_s>0).sum()):,}/{len(c):,}",pct((c.reporter_dispatcher_overlap_s>0).mean()),
                pct(c.reporter_overlap_ratio.median()),pct(c.reporter_overlap_ratio.quantile(.95))]
               for split,c in calls.groupby('split')]), "",
        "구간 교집합은 라벨 시간 경계의 겹침을 뜻하며 실제 동시 발성 여부는 청취 확인 필요다. "
        "다만 신고자 crop이 순수한 한 사람 음성만 포함한다고 가정하기는 어렵다.", "",
        "![Segment duration](eda_outputs/segment_duration_distribution.png)", "",
        "![Call duration by gender](eda_outputs/gender_duration_comparison.png)", "",
        "히스토그램 x축은 로그 초, boxplot은 이상치 점만 숨겼으며 통계에서는 제외하지 않았다.", "",
        "## 4. WAV와 음향 표본", "",
        table(["split","헤더 성공","sample rate:수","channel:수","subtype:수","파일 크기 GB"],
              [[k,v['headers_read'],v['sample_rates'],v['channels'],v['subtypes'],f(v['total_bytes']/1e9)] for k,v in s['audio'].items()]), "",
        table(["split","WAV 길이 평균 s","중앙값","최소","최대","Q25","Q75","Q90","Q95"],
              [[k]+[f(v['duration_s'].get(t)) for t in ('mean','median','min','max','q25','q75','q90','q95')] for k,v in s['audio'].items()]), "",
        table(["split","RIFF 상태:수","JSON endAt/1000 − WAV 길이 평균 s","최소","최대"],
              [[k,v['structure_status']]+[f(v['json_end_minus_wav_s'].get(t)) for t in ('mean','min','max')] for k,v in s['audio'].items()]), "",
        "헤더·RIFF chunk 경계는 모든 존재 WAV를 검사했다. **헤더/크기 검사가 통과해도 전 파형 무손상을 "
        "보장하지 않는다.** 표본 외 WAV의 디코딩·청취 이상은 확인 필요다. "
        "segment 범위 초과 허용오차는 1ms, 통화 endAt 불일치 플래그는 절댓값 20ms 초과다.", ""]
    differences=[]
    report += [table(['split','WAV 헤더 실패','RIFF 이상','WAV 초과 segment','통화 endAt 차이 >20ms','표본 디코딩/특징 실패'],
        [[split]+[s['issue_counts_by_split'][split].get(k,0) for k in
                  ('wav_header_failure','wav_structure','segment_exceeds_wav','json_end_wav_mismatch_gt_20ms','sample_audio_or_features_failure')]
         for split in ('Training','Validation')]), "",
        "WAV와 JSON 길이 비교는 실제 WAV가 존재하고 JSON이 정상 파싱된 통화에서만 가능하다.", ""]
    for split in ('Training','Validation'):
        g=s['groups'][split+'/matched']['genders']
        if g['M']['calls'] and g['F']['calls']:
            differences.append([split]+[f(g['F'][key]['mean']/g['M'][key]['mean'])
                                       if g['M'][key].get('mean',0)>0 else '확인 불가'
                                       for key in ('segment_duration_s','reporter_segments_per_call','reporter_union_s')])
    report += ["실제 WAV가 매칭된 통화에서 **여성 평균 / 남성 평균**의 비는 다음과 같다. "
               "1보다 크면 여성 라벨 그룹의 평균이 더 크다. 통화·segment 구조와 함께 해석해야 하며 "
               "성별 자체의 인과효과나 예측 성능을 뜻하지 않는다.", "",
               table(['split','segment 길이 평균 비','통화별 segment 수 평균 비','총 신고자 합집합 시간 평균 비'],differences), ""]
    if not acoustic.empty:
        report += [f"완전 디코딩 및 음향 분석 성공 표본은 {len(acoustic)}통화다. "
            "split×gender×신고자 발화 합집합 시간 사분위에서 각 5개를 기본으로 추출했다. "
            "seed=42이며 실제 옵션은 run_metadata.json을 따른다. 동일 통화를 중복 추출하지 않는다. "
            "길이 사분위별 균형 표본이므로 아래 수치를 모집단 비율이나 독립 segment 표본의 결과로 해석하지 않는다.", "",
            table(["split/gender","통화 n","통화 RMS 중앙값","신고자 RMS 중앙값","무음 대용 비율 중앙값","near-full-scale 있는 통화"],
                  [[f"{sp}/{g}",len(a),f(a.rms.median()),f(a.reporter_rms.median()),pct(a.silence_frame_ratio.median()),
                    int((a.near_full_scale_sample_ratio>0).sum())] for (sp,g),a in acoustic.groupby(['split','gender'])]), "",
            "무음 대용치는 20ms 프레임 RMS <0.001(-60dBFS), 클리핑 대용치는 |sample|≥0.999이다. "
            "VAD나 완전한 왜곡 검출기가 아니다. 낮은 RMS의 음성과 잡음, 재스케일된 클리핑을 잘못 판단할 수 있다.", "",
            table(["split/gender","통화별 F0 중앙값의 중앙값 Hz","F0 min–max Hz","pYIN voiced ratio 중앙값","centroid 중앙값 Hz","329–330Hz 중앙값 통화 n"],
                  [[f"{sp}/{g}",f(a.f0_median_hz.median()),f(a.f0_median_hz.min())+'–'+f(a.f0_median_hz.max()),
                    pct(a.pitch_voiced_ratio.median()),f(a.centroid_mean_hz.median()),int(a.f0_median_hz.between(329,330).sum())]
                   for (sp,g),a in acoustic.groupby(['split','gender'])]), "",
            "F0는 pYIN 65–650Hz, 512 sample frame, 160 sample hop을 사용했다. 통화당 신고자 segment 최대 3개, "
            "각 최대 3초의 무작위 창이다. 0.5초 미만 segment는 pitch 표본에서 제외했다. "
            "F0와 voiced ratio는 알고리즘 추정치로, 울음·비명·긴장·잡음·음성 겹침에서 오류와 octave jump가 가능하다. "
            "추정 범위 밖의 음성도 있을 수 있다. 관측된 성별 차이만으로 **F0 단독으로 충분하다는 결론을 내리지 않는다.**", "",
            "같은 창에서 MFCC 13개 각각의 mean/std를 저장했다. 아래는 일부 계수의 통화별 mean에 대한 중앙값이다. "
            "MFCC는 신호 크기·채널·잡음의 영향도 받으므로 숫자의 크기를 성별 근거로 바로 해석하지 않는다.", "",
            table(["split/gender","MFCC1","MFCC2","MFCC3"],
                  [[f"{sp}/{g}"]+[f(a[f'mfcc_{i}_mean'].median()) for i in (1,2,3)] for (sp,g),a in acoustic.groupby(['split','gender'])]), "",
            "![Sample acoustics](eda_outputs/sample_acoustic_features.png)", "",
            "울음·비명·긴장 상태와 잡음 종류는 자동 수치만으로 확정하지 않았다. 이번 분석에서 직접 청취·수작업 "
            "감정 라벨링은 수행하지 않았으며 **확인 필요**다. RMS/무음/peak/voiced ratio 극단 표본과 "
            "화자 구간 겹침이 큰 통화를 우선 청취 대상으로 삼을 수 있다.", ""]
    if (root/'pitch_peak_audit.json').exists():
        tone=json.loads((root/'pitch_peak_audit.json').read_text())
        index=report.index('## 1. 데이터 구조')
        report[index:index]=["- 음향 표본의 반복된 약 329.48Hz 추정값을 추적해 남녀 진단 구간에서 "
                            "**약 660Hz 단일 톤**을 확인했다. 아래 F0 분포는 순수한 사람 음성 F0 분포가 아니다.", ""]
        report += ["### F0 반복값의 실제 원인 점검", "",
            f"80통화 표본 중 {tone['repeated_bin_calls']}통화에서 통화별 F0 중앙값이 약 329.48Hz였다. "
            "이 가운데 남녀 각 1통화의 진단 창과 낮은 F0 남성 대조 창을 조사했다. "
            "남녀 진단 창 모두에서 약 660Hz의 안정적인 협대역 톤이 관측됐다. "
            "이것은 20통화 모두를 톤으로 확정하거나 모집단 톤 비율을 추정한 결과가 아니다.", "",
            table(['진단 창','스펙트럼 최고점 Hz','pYIN fmax650 Hz','pYIN fmax1000 Hz','650–670Hz 에너지 비율'],
                  [[f"{d['split']}/{d['gender']}",f(d['plotted_window']['peak_spectral_hz']),
                    f(d['plotted_window']['f0_median_hz']),f(d['plotted_window'].get('f0_median_hz_fmax1000')),
                    pct(d['plotted_window'].get('power_650_670_ratio'))] for d in tone['diagnostics']]), "",
            "알려진 합성 660Hz 정현파에서도 fmax=650일 때 약 329.48Hz, fmax=1000일 때 약 662.77Hz를 "
            "추정했다(양쪽 voiced ratio 1.0). 즉, 제한된 탐색 범위에서 반 주파수 추정이 발생하며 "
            "**높은 voiced ratio가 사람 음성임을 보장하지 않는다.** 110/220Hz 합성 신호 검증이 통과해도 "
            "실제 데이터의 톤 오염 문제는 별도로 남는다.", "",
            "톤의 생성 목적(원본 신호, 가공 이력 등)은 확인 필요다. 임의로 마스킹 신호라고 확정하거나 "
            "제거하지 않았다. 톤 구간 처리와 음성 여부 확인을 결정한 후 음향 특징을 재검토해야 한다. "
            "fmax를 높이는 것만으로 톤이 사람 음성으로 바뀌는 것은 아니므로 그것을 해결책으로 단정하지 않는다.", "",
            "![Tonal F0 diagnostics](eda_outputs/pitch_peak_diagnostics.png)", ""]
    report += ["## 5. 누수 검사", "",
        "아래 ID·경로·stem 비교는 정상 파싱된 JSON 기준이다. 파싱 실패 JSON은 제외되며, "
        "압축본 전체 JSON의 별도 ID 교차 검사는 archive_crosscheck.json에 있다.", "",
        table(["비교 기준","고유값 수","중복 값 수(전체)","Training–Validation 교차 값 수"],
              [[k,v['unique_values'],v['duplicate_values'],v['cross_split_values']] for k,v in s['leakage'].items() if k!='exact_wav_bytes']), ""]
    ex=s['leakage']['exact_wav_bytes']
    report += [f"정확한 WAV 바이트 중복: 헤더를 읽은 {ex['header_readable_files']:,}개 대상. "
        f"같은 크기 후보 {ex['size_prefilter_files']:,}개를 앞뒤 4KiB로 비교하고, "
        f"남은 {ex['full_hashed_files']:,}개에 전체 SHA256을 계산했다. "
        f"동일 바이트 그룹 {len(ex['duplicate_groups']):,}개, split 간 그룹 {ex['cross_split_groups']:,}개, "
        f"읽기 실패 {ex['read_failures']:,}개다.", "",
        "동일 바이트면 크기·앞뒤 바이트도 같으므로 이 절차는 헤더가 읽힌 파일의 정확한 파일 중복을 검사한다. "
        "헤더 실패 파일, 헤더만 다른 동일 PCM, 재인코딩·부분 잘라내기·동일 인물 재신고는 별개다.", "",
        table(["질문","판정"],[
            ["동일 ID/경로/파일명의 split 교차", "위 실측 교차 값 수 참고"],
            ["동일 사건을 파일명/recordId만으로 확정", "확인 필요: 업무상 ID 정의와 원본-가공본 계보가 없음"],
            ["편집·재인코딩된 같은 원본 음성", "확인 필요: 음향 fingerprint/원본 매핑 미실시"],
            ["동일한 실제 사람이 여러 통화로 나뉨", "확인 필요: 지속적인 개인 speaker ID가 없음"],
            ["통화 내 segment를 임의 train/valid 분할", "같은 신고자 정보가 양쪽으로 갈 수 있음; 통화/원본 그룹 단위 분할 필요"],
            ["gender 메타데이터의 모델 입력 유입", "직접 정답 누수 위험; gender는 target에만 사용"],
            ["text/address/sentiment/recordId/audioPath/파일명 입력", "허용 정보 밖의 입력 및 지름길 위험; 모델 feature allowlist에서 제외"]]), "",
        "이후 입력은 WAV 및 허용된 startAt/endAt/speaker로 제한하고, 출력 라벨은 별도 경로로 관리해야 한다. "
        "EDA CSV 자체를 통째로 모델에 입력하지 않는다. Validation은 학습·스케일러 fit에 섞지 않는다.", "",
        "## 6. 발견한 문제와 모델링 전 결정 사항", "",
        table(["문제 종류","Training","Validation"],
              [[kind,s['issue_counts_by_split']['Training'].get(kind,0),s['issue_counts_by_split']['Validation'].get(kind,0)]
               for kind in sorted(s['issues'])]), "",
        "문제 수의 단위: missing_wav·reporter_self_overlap·통화 endAt 불일치는 통화/파일, "
        "invalid_segment_time·segment_exceeds_wav는 segment, 읽기 실패는 해당 검사 항목이다. "
        "한 통화가 여러 문제에 중복 집계될 수 있다. 상세 가명 참조는 issues.csv에 있다.", "",
        "1. 누락 WAV의 원본 확보 여부와 최종 분석 모집단을 확정한다. 확보 후 날짜 토큰·성별·발화량 분포를 재검사한다.",
        "2. 공식 스키마의 역할 매핑, gender 라벨 대상, 동일 통화 내 복수 신고자 처리 기준을 확인한다.",
        "3. 출력/평가 단위를 통화로 확정하고 여러 segment의 결합 방식과 최소 유효 발화량을 결정한다. 지금은 삭제·trim·padding하지 않았다.",
        "4. 구간 경계 오류·같은 역할 겹침·상대 화자 혼입의 처리 방침을 정한다. 구간 합집합과 단순 이어붙이기를 동일시하지 않는다.",
        "5. 원본/사건/인물 식별 범위를 확인하고 통화 단위 분할을 유지한다. ID 교차 0을 사람 단위 누수 0으로 해석하지 않는다.",
        "6. 짧은 발화와 극단 음향 표본을 청취하고 pitch 오류·왜곡·잡음을 확인한다. 발견된 660Hz 톤의 출처·처리 기준을 확인한다. F0 외 에너지·스펙트럼 특징도 비교 가능한 상태로 남겼다.",
        "7. 이후 모든 학습 전처리의 fit은 Training에 한정하고, 허용 입력만 통과시키는 구조를 구현한다.", "",
        "이번 단계의 관찰은 데이터 품질과 평가 단위에 관한 것이다. 특정 모델 선택이나 F0 단독 분류의 충분성, "
        "Accuracy 성능을 판단한 결과가 아니다.", "", "## 7. 재현과 검증", "",
        "`README.md`에 실행 명령·의존성·집계 정의·한계를 기록했다. `run_eda.py` 실행 후 "
        "`build_report.py`를 실행하면 이 보고서를 새 결과로 다시 생성한다. "
        "`test_eda.py`는 합성 데이터로 누락, 시간 오류, 구간 교집합/합집합, WAV 범위 초과, "
        "split 간 바이트 중복, 비식별 출력을 검증한다. 모델 테스트가 아니다.", "",
        "실제 실행 버전: "+", ".join(f"{k} {v}" for k,v in run['versions'].items())+". "
        "세부 CSV/JSON/그래프는 eda_outputs에 있으며 기존 gitignore 규칙에 따라 Git에 추가되지 않는다.", ""]
    rendered='\n'.join(report).replace('(eda_outputs/', '('+str(root.resolve())+'/')
    args.out.write_text(rendered,encoding='utf-8')
    print(args.out)


if __name__=='__main__':
    main()
