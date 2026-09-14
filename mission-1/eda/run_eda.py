"""Read-only Mission 1 EDA. No training, source edits, or raw text exports.

Run from any directory with --data-root pointing to a parent of Training/Validation.
Outputs are audit data, NEVER model input. File references use SHA256 pseudonyms.
"""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import struct
import sys
import time
import unicodedata
import wave
from types import SimpleNamespace
from itertools import islice

os.environ.setdefault("MPLCONFIGDIR", "/tmp/ddc-m1-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/ddc-m1-numba")
import numpy as np
import pandas as pd
import soundfile as sf


def token(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def stats(values):
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if not len(x):
        return {"n": 0}
    return dict(n=len(x), mean=float(x.mean()), median=float(np.median(x)),
                min=float(x.min()), max=float(x.max()),
                **{f"q{q}": float(np.percentile(x, q)) for q in (25, 75, 90, 95)})


def schema(obj, result, prefix="$"):
    result[prefix][type(obj).__name__] += 1
    if isinstance(obj, dict):
        for k, v in obj.items():
            schema(v, result, prefix + "." + k)
    elif isinstance(obj, list):
        for v in obj:
            schema(v, result, prefix + "[]")


def redact(obj, key=""):
    if isinstance(obj, dict):
        return {k: redact(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v, key) for v in obj[:3]]
    if key in ("gender", "startAt", "endAt", "speaker", "Speaker"):
        return obj
    return f"<{type(obj).__name__}:redacted>"


def merged(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1][1] = max(b, out[-1][1])
        else:
            out.append([a, b])
    return out


def overlap_seconds(left, right):
    a, b = merged(left), merged(right)
    i = j = 0
    total = 0.0
    while i < len(a) and j < len(b):
        total += max(0, min(a[i][1], b[j][1]) - max(a[i][0], b[j][0]))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return total


def riff_check(path, details=False):
    """Check declared RIFF/chunk boundaries without decoding PCM payload."""
    with path.open("rb") as f:
        size = f.seek(0, 2)
        f.seek(0)
        def result(status):
            return (status, size) if details else status
        h = f.read(12)
        if len(h) != 12 or h[:4] != b"RIFF" or h[8:] != b"WAVE":
            return result("non_RIFF_WAVE_requires_review")
        boundary = struct.unpack("<I", h[4:8])[0] + 8
        if boundary > size:
            return result("truncated_RIFF")
        data_seen = False
        while f.tell() + 8 <= boundary:
            chunk, n = struct.unpack("<4sI", f.read(8))
            if f.tell() + n > min(size, boundary):
                return result("truncated_chunk")
            data_seen |= chunk == b"data"
            f.seek(n + n % 2, 1)
        return result("ok" if data_seen else "missing_data_chunk")


def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def buffered_map(fn, items):
    """Eight small I/O reads at once, at most 32 outstanding results in memory."""
    items = iter(items)
    with ThreadPoolExecutor(max_workers=8) as pool:
        while batch := list(islice(items, 32)):
            yield from pool.map(fn, batch)


def read_header(item):
    stem, ps, p = item
    try:
        try:
            with p.open("rb") as f:
                with wave.open(f) as w:
                    info = SimpleNamespace(samplerate=w.getframerate(), channels=w.getnchannels(),
                        frames=w.getnframes(), duration=w.getnframes()/w.getframerate(),
                        subtype={1:"PCM_U8",2:"PCM_16",3:"PCM_24",4:"PCM_32"}.get(w.getsampwidth(),"PCM_other"))
        except wave.Error:
            info = sf.info(p)
        status, size = riff_check(p, details=True)
        return stem, ps, p, info, status, size, None
    except Exception as e:
        return stem, ps, p, None, None, None, type(e).__name__


def read_json(item):
    stem, ps, p = item
    try:
        d = json.loads(p.read_text(encoding="utf-8-sig"))
        if not isinstance(d, dict) or not isinstance(d.get("utterances"), list):
            raise ValueError("unexpected schema")
        return stem, ps, p, d, None
    except Exception as e:
        return stem, ps, p, None, type(e).__name__


def read_quick(item):
    split, ref, p, size = item
    try:
        with p.open("rb") as f:
            head = f.read(4096)
            f.seek(max(0, size-4096))
            tail = f.read(4096)
        return split, ref, p, size, hashlib.sha256(head+tail).hexdigest(), None
    except OSError as e:
        return split, ref, p, size, None, type(e).__name__


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "eda_outputs")
    ap.add_argument("--samples-per-stratum", type=int, default=5,
                    help="Calls per split/gender/reporter-duration quartile; default up to 80 calls")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-acoustics", action="store_true")
    args = ap.parse_args()
    root, out = args.data_root.resolve(), args.out_dir.resolve()
    if out == root or root in out.parents:
        ap.error("Output must be outside original data root")
    if args.samples_per_stratum < 1:
        ap.error("samples-per-stratum must be positive")
    if not all((root / s).is_dir() for s in ("Training", "Validation")):
        ap.error("data-root must contain both Training and Validation directories")
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    issues, rows, segrows, audiorows, examples = [], [], [], [], []
    schemas = collections.defaultdict(collections.Counter)
    inventories, paths, labels = {}, {}, {}
    identities = collections.defaultdict(lambda: collections.defaultdict(list))
    audio_index = []

    def issue(split, ref, kind, detail=""):
        issues.append(dict(split=split, file_ref=ref, issue=kind, detail=detail))

    for split in ("Training", "Validation"):
        grouped = {".wav": collections.defaultdict(list), ".json": collections.defaultdict(list)}
        ignored = collections.Counter()
        for directory, dirs, files in os.walk(root / split):
            dirs.sort()
            for name in sorted(files):
                p = Path(directory) / name
                if p.suffix.lower() not in grouped:
                    continue
                if p.name.startswith("._") or "__MACOSX" in p.parts:
                    ignored[p.suffix.lower()] += 1
                    continue
                grouped[p.suffix.lower()][unicodedata.normalize("NFC", p.stem)].append(p)
        wav, js = grouped[".wav"], grouped[".json"]
        unique_pairs = {k for k in wav.keys() & js.keys() if len(wav[k]) == len(js[k]) == 1}
        inv = dict(wav_files=sum(map(len, wav.values())), json_files=sum(map(len, js.values())),
                   matched_unique_pairs=len(unique_pairs), json_only_stems=len(js.keys()-wav.keys()),
                   wav_only_stems=len(wav.keys()-js.keys()),
                   ambiguous_shared_stems=len(wav.keys() & js.keys())-len(unique_pairs),
                   duplicate_wav_stems=sum(len(v)>1 for v in wav.values()),
                   duplicate_json_stems=sum(len(v)>1 for v in js.values()),
                   ignored_macos_sidecars=dict(ignored),
                   directories=sorted({unicodedata.normalize("NFC", str(p.parent.relative_to(root)))
                                       for group in grouped.values() for ps in group.values() for p in ps}))
        inventories[split] = inv
        for ext, group in grouped.items():
            for stem, ps in group.items():
                if len(ps) > 1:
                    issue(split, token(stem), "duplicate_stem" + ext, str(len(ps)))
        for stem in js.keys()-wav.keys():
            issue(split, token(stem), "missing_wav")
        for stem in wav.keys()-js.keys():
            issue(split, token(stem), "missing_json")
        print(split, inv, flush=True)
        headers = {}
        items = ((stem, ps, p) for stem, ps in wav.items() for p in ps)
        for index, (stem, ps, p, info, status, size, error) in enumerate(buffered_map(read_header, items)):
            ref = token(stem)
            if error is None:
                r = dict(split=split, file_ref=ref, sample_rate=info.samplerate,
                         channels=info.channels, frames=info.frames, duration_s=info.duration,
                         subtype=info.subtype, size_bytes=size, riff_status=status)
                audiorows.append(r)
                if len(ps) == 1:
                    headers[stem] = r
                if status != "ok":
                    issue(split, ref, "wav_structure", status)
                audio_index.append((split, ref, p, size))
            else:
                issue(split, ref, "wav_header_failure", error)
            if (index+1) % 1000 == 0:
                print(split, "headers", index+1, flush=True)
        items = ((stem, ps, p) for stem, ps in js.items() for p in ps)
        for index, (stem, ps, p, d, error) in enumerate(buffered_map(read_json, items)):
            ref = token(stem)
            if error is not None:
                issue(split, ref, "json_parse_or_schema_failure", error)
                continue
            schema(d, schemas)
            if sum(e["split"] == split for e in examples) < 3:
                examples.append(dict(split=split, file_ref=ref, total_utterances=len(d["utterances"]),
                                     example=redact(d)))
            for key in ("_id", "recordId", "audioPath"):
                if isinstance(d.get(key), str) and d[key]:
                    identities[key][token(d[key])].append((split, ref))
            identities["filename_stem"][ref].append((split, ref))
            identities["filename_prefix"][token(stem.split("_")[0])].append((split, ref))
            gender = d.get("gender")
            if gender not in ("M", "F"):
                issue(split, ref, "invalid_gender")
                gender = "unknown"
            intervals = collections.defaultdict(list)
            speakers = collections.Counter()
            invalid = 0
            for u in d["utterances"]:
                if not isinstance(u, dict):
                    issue(split, ref, "invalid_utterance_object")
                    invalid += 1
                    continue
                sp = u.get("speaker")
                if type(sp) is not int or sp not in (0, 1):
                    issue(split, ref, "unexpected_speaker")
                speakers[str(sp)] += 1
                a, b = u.get("startAt"), u.get("endAt")
                if not all(type(x) in (int, float) and math.isfinite(x) for x in (a,b)) or a < 0 or b <= a:
                    invalid += 1
                    issue(split, ref, "invalid_segment_time")
                    continue
                duration_s = (b-a)/1000  # Subtract in ms first: exact 1s must not become <1s.
                a, b = a/1000, b/1000
                role = sp if type(sp) is int and sp in (0,1) else "unknown"
                intervals[role].append((a,b))
                oob = stem in headers and b > headers[stem]["duration_s"] + .001
                if oob:
                    issue(split, ref, "segment_exceeds_wav")
                segrows.append(dict(split=split, file_ref=ref, gender=gender if role == 1 else "not_applicable",
                                    speaker=role, start_s=a, end_s=b, duration_s=duration_s,
                                    matched=stem in unique_pairs, exceeds_wav=oob))
            rep = intervals[1]
            rep_sum = sum(b-a for a,b in rep)
            rep_union = sum(b-a for a,b in merged(rep))
            cross = overlap_seconds(rep, intervals[0])
            if rep_sum-rep_union > 1e-8:
                issue(split, ref, "reporter_self_overlap")
            h = headers.get(stem)
            top_end = d.get("endAt")
            delta = (top_end / 1000 - h["duration_s"] if h and type(top_end) in (int,float) else None)
            if delta is not None and abs(delta) > .02:
                issue(split, ref, "json_end_wav_mismatch_gt_20ms")
            rows.append(dict(split=split, file_ref=ref, gender=gender, matched=stem in unique_pairs,
                             n_segments=len(d["utterances"]), n_speaker_values=len(speakers),
                             n_reporter_segments=speakers.get("1",0), n_valid_reporter_segments=len(rep),
                             invalid_segments=invalid, reporter_sum_s=rep_sum, reporter_union_s=rep_union,
                             reporter_dispatcher_overlap_s=cross,
                             reporter_overlap_ratio=cross/rep_union if rep_union else 0,
                             json_end_minus_wav_s=delta,
                             seoul_address=isinstance(d.get("address"),str) and "서울" in d["address"],
                             id_matches_stem_prefix=d.get("_id")==stem.split("_")[0],
                             record_id_matches_stem=d.get("recordId")==stem,
                             audio_path_stem_matches=Path(str(d.get("audioPath",""))).stem==stem))
            if stem in unique_pairs and h and h["riff_status"] == "ok" and rep:
                paths[(split,ref)] = wav[stem][0]
                labels[(split,ref)] = rep
            if (index+1) % 5000 == 0:
                print(split, "JSON", index+1, flush=True)

    calls, segments, audio = pd.DataFrame(rows), pd.DataFrame(segrows), pd.DataFrame(audiorows)
    if calls.empty or segments.empty or audio.empty:
        dump(out / "inventory.json", inventories)
        dump(out / "issues.json", issues)
        raise SystemExit("Insufficient parsed data; inventory and issues saved, EDA not complete")
    calls.to_csv(out / "calls.csv", index=False)
    segments.to_csv(out / "segments.csv", index=False)
    audio.to_csv(out / "audio_headers.csv", index=False)
    dump(out / "schema.json", schemas)
    dump(out / "schema_examples_redacted.json", examples)
    dump(out / "inventory.json", inventories)

    leak = {}
    for key, ids in identities.items():
        cross = [refs for refs in ids.values() if len({s for s,r in refs}) > 1]
        leak[key] = dict(unique_values=len(ids), duplicate_values=sum(len(v)>1 for v in ids.values()),
                         cross_split_values=len(cross), cross_split_file_references=cross)
    # Exact-byte duplicates must have identical size and first/last bytes.
    # This staged filter is exhaustive for exact bytes, not for re-encoded/cropped speech.
    by_size = collections.defaultdict(list)
    for row in audio_index:
        by_size[row[3]].append(row)
    candidate_groups = [v for v in by_size.values() if len(v)>1]
    quick = collections.defaultdict(list)
    quick_failures = 0
    candidates = (item for group in candidate_groups for item in group)
    for n, (split,ref,p,size,signature,error) in enumerate(buffered_map(read_quick,candidates)):
        if error is None:
            quick[(size,signature)].append((split,ref,p))
        else:
            quick_failures += 1
            issue(split,ref,"duplicate_scan_read_failure",error)
        if (n+1)%5000==0:
            print("Duplicate prefilter files", n+1, flush=True)
    exact = collections.defaultdict(list)
    full_hashed = 0
    for group in quick.values():
        if len(group)<2:
            continue
        for split,ref,p in group:
            try:
                exact[digest_file(p)].append((split,ref))
                full_hashed += 1
            except OSError as e:
                quick_failures += 1
                issue(split,ref,"full_hash_read_failure",type(e).__name__)
    duplicates = [v for v in exact.values() if len(v)>1]
    leak["exact_wav_bytes"] = dict(header_readable_files=len(audio_index),
        size_prefilter_files=sum(map(len,candidate_groups)), full_hashed_files=full_hashed,
        read_failures=quick_failures, duplicate_groups=duplicates,
        cross_split_groups=sum(len({s for s,r in v})>1 for v in duplicates),
        limitation="Header failures excluded. Does not detect re-encoding, cropping, or recurring speakers.")
    dump(out / "leakage.json", leak)

    summary = {"inventory":inventories, "duration_unit":"seconds (JSON milliseconds / 1000)",
               "groups":{}, "audio":{}, "leakage":leak}
    for split in ("Training","Validation"):
        for scope in ("all_labels", "matched"):
            c = calls[calls.split.eq(split)]
            s = segments[segments.split.eq(split)]
            if scope == "matched":
                c, s = c[c.matched], s[s.matched]
            key = f"{split}/{scope}"
            genders = {}
            for gender in ("M","F","unknown"):
                cg = c[c.gender.eq(gender)]
                sg = s[s.speaker.eq(1) & s.gender.eq(gender)]
                genders[gender] = dict(calls=len(cg), call_ratio=len(cg)/len(c) if len(c) else None,
                    reporter_segments=len(sg), segment_duration_s=stats(sg.duration_s),
                    short_segments={str(t):int((sg.duration_s<t).sum()) for t in (.5,1,2)},
                    reporter_segments_per_call=stats(cg.n_reporter_segments),
                    reporter_sum_s=stats(cg.reporter_sum_s), reporter_union_s=stats(cg.reporter_union_s),
                    multiple_reporter_segments_ratio=float((cg.n_reporter_segments>1).mean()) if len(cg) else None,
                    overlap_s=stats(cg.reporter_dispatcher_overlap_s),
                    overlap_ratio=stats(cg.reporter_overlap_ratio))
            summary["groups"][key] = dict(genders=genders, all_segments=len(s),
                all_segment_duration_s=stats(s.duration_s),
                all_short_segments={str(t):int((s.duration_s<t).sum()) for t in (.5,1,2)},
                segments_per_call=stats(c.n_segments), speaker_values_per_call=stats(c.n_speaker_values),
                zero_reporter_calls=int(c.n_reporter_segments.eq(0).sum()),
                seoul_calls=int(c.seoul_address.sum()),
                id_matches_stem_prefix=int(c.id_matches_stem_prefix.sum()),
                record_id_matches_stem=int(c.record_id_matches_stem.sum()),
                audio_path_stem_matches=int(c.audio_path_stem_matches.sum()))
        a = audio[audio.split.eq(split)]
        summary["audio"][split] = dict(headers_read=len(a), duration_s=stats(a.duration_s),
            sample_rates={str(k):int(v) for k,v in a.sample_rate.value_counts().items()},
            channels={str(k):int(v) for k,v in a.channels.value_counts().items()},
            subtypes=a.subtype.value_counts().to_dict(),
            structure_status=a.riff_status.value_counts().to_dict(), total_bytes=int(a.size_bytes.sum()),
            json_end_minus_wav_s=stats(calls[calls.split.eq(split)].json_end_minus_wav_s))
    dump(out / "summary.json", summary)
    print("Metadata and duplicate scan complete", flush=True)

    acoustics, selected = [], []
    if not args.skip_acoustics:
        import librosa
        rng = random.Random(args.seed)
        eligible = calls[calls.apply(lambda r:(r.split,r.file_ref) in paths,axis=1)].copy()
        for (split,gender), group in eligible.groupby(["split","gender"]):
            if gender not in ("M","F"):
                continue
            group = group.sort_values("file_ref").copy()
            group["quartile"] = pd.qcut(group.reporter_union_s.rank(method="first"),4,labels=False)
            for q,g in group.groupby("quartile"):
                for idx in rng.sample(list(g.index), min(args.samples_per_stratum,len(g))):
                    r=group.loc[idx]
                    selected.append(dict(split=split,gender=gender,file_ref=r.file_ref,
                                         quartile=int(q),stratum_population=len(g)))
        for n,r in enumerate(selected):
            key=(r["split"],r["file_ref"])
            try:
                # Exactly one complete call in memory; limited representative feature windows.
                y,sr=sf.read(paths[key], dtype="float32",always_2d=True)
                mono=y.mean(axis=1)
                frame=max(1,int(sr*.02))
                flat=y[:len(y)//frame*frame]
                rms=np.sqrt(np.mean(flat.reshape(-1,frame,y.shape[1])**2,axis=(1,2)))
                base=dict(r, sample_rate=sr, decoded_frames=len(y),
                          peak=float(np.max(np.abs(y))), rms=float(np.sqrt(np.mean(y*y))),
                          silence_frame_ratio=float(np.mean(rms<.001)),
                          near_full_scale_sample_ratio=float(np.mean(np.abs(y)>=.999)),
                          exact_zero_sample_ratio=float(np.mean(y==0)))
                if len(y)!=int(audio[(audio.split==r['split']) & (audio.file_ref==r['file_ref'])].frames.iloc[0]):
                    issue(*key,"decoded_frame_count_mismatch")
                rep=merged(labels[key])
                chunks=[mono[max(0,round(a*sr)):min(len(mono),round(b*sr))] for a,b in rep]
                chunks=[x for x in chunks if len(x)]
                rep_samples=sum(len(x) for x in chunks)
                base["reporter_rms"]=float(np.sqrt(sum(float(np.dot(x,x)) for x in chunks)/rep_samples))
                base["reporter_near_full_scale_ratio"]=sum(int((np.abs(x)>=.999).sum()) for x in chunks)/rep_samples
                # Uniform segment selection, exclude <0.5s for pitch; random max-3s crop.
                valid=[(a,b) for a,b in labels[key] if b-a>=.5 and a<len(mono)/sr]
                chosen=rng.sample(valid,min(3,len(valid)))
                f0s, voiced, centroids, energies, mfccs=[],[],[],[],[]
                windows=[]
                for a,b in chosen:
                    b=min(b,len(mono)/sr)
                    a=a+rng.random()*max(0,b-a-3)
                    b=min(b,a+3)
                    x=mono[round(a*sr):round(b*sr)]
                    if len(x)<512:
                        continue
                    windows.append([a,b])
                    f0,vf,vp=librosa.pyin(x,sr=sr,fmin=65,fmax=min(650,sr/2-1),
                                         frame_length=512,hop_length=160,center=False)
                    f0s.extend(f0[np.isfinite(f0)].tolist())
                    voiced.extend(vf.tolist())
                    centroids.extend(librosa.feature.spectral_centroid(y=x,sr=sr,n_fft=512,
                                                                       hop_length=160,center=False)[0].tolist())
                    energies.extend(librosa.feature.rms(y=x,frame_length=512,hop_length=160,
                                                       center=False)[0].tolist())
                    mfccs.append(librosa.feature.mfcc(y=x,sr=sr,n_mfcc=13,n_fft=512,
                                                      hop_length=160,center=False,n_mels=40))
                base.update(feature_windows=windows, pitch_voiced_ratio=float(np.mean(voiced)) if voiced else None,
                            f0_median_hz=float(np.median(f0s)) if f0s else None,
                            f0_q25_hz=float(np.percentile(f0s,25)) if f0s else None,
                            f0_q75_hz=float(np.percentile(f0s,75)) if f0s else None,
                            centroid_mean_hz=float(np.mean(centroids)) if centroids else None,
                            feature_rms_mean=float(np.mean(energies)) if energies else None)
                if mfccs:
                    m=np.concatenate(mfccs,axis=1)
                    base.update({f"mfcc_{i+1}_mean":float(v) for i,v in enumerate(m.mean(axis=1))})
                    base.update({f"mfcc_{i+1}_std":float(v) for i,v in enumerate(m.std(axis=1))})
                acoustics.append(base)
            except Exception as e:
                issue(*key,"sample_audio_or_features_failure",type(e).__name__)
            if (n+1)%10==0:
                print("Sample acoustics",n+1,"/",len(selected),flush=True)
        dump(out / "sample_selection.json",selected)
        pd.DataFrame(acoustics).to_csv(out / "sample_acoustics.csv",index=False)
    summary["sample_acoustics"]={}
    if acoustics:
        ac=pd.DataFrame(acoustics)
        for (split,g),group in ac.groupby(["split","gender"]):
            summary["sample_acoustics"][f"{split}/{g}"]={k:stats(group[k]) for k in
                ("peak","rms","reporter_rms","silence_frame_ratio","near_full_scale_sample_ratio",
                 "pitch_voiced_ratio","f0_median_hz","centroid_mean_hz")}
    summary["issues"] = dict(collections.Counter(x["issue"] for x in issues))
    summary["issue_counts_by_split"] = {s:dict(collections.Counter(x["issue"] for x in issues if x["split"]==s))
                                        for s in ("Training","Validation")}
    dump(out / "summary.json",summary)
    pd.DataFrame(issues,columns=["split","file_ref","issue","detail"]).to_csv(out / "issues.csv",index=False)
    make_plots(out,calls,segments,acoustics)
    dump(out / "run_metadata.json",dict(data_root=str(root),arguments=vars(args)|{"data_root":str(root),"out_dir":str(out)},
        python=sys.version,platform=platform.platform(), elapsed_seconds=time.time()-start,
        versions={p:importlib.metadata.version(p) for p in ("numpy","pandas","soundfile","librosa","matplotlib")},
        method={"headers":"all WAV, wave (libsndfile fallback) + RIFF bounds; no full payload integrity claim",
                "duplicates":"size -> 4KiB head/tail -> full SHA256 for candidates; all header-readable WAV",
                "silence":"20ms non-overlapping frames RMS < 0.001 (-60dBFS); not a speech VAD",
                "clipping":"abs(sample)>=0.999 proxy; does not detect all clipped/rescaled signals",
                "pitch":"pYIN 65-650Hz, 512-frame, 160-hop; up to 3 random >=0.5s segments/call, max 3s each",
                "sample":"equal samples per split/gender/reporter-union-duration quartile; not population weighted",
                "speaker":"repository mapping 0 dispatcher / 1 reporter; role IDs are not person IDs"}))
    print("Completed:",out, "seconds:",round(time.time()-start),flush=True)


def make_plots(out,calls,segments,acoustics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors={"M":"#3677b5","F":"#c65078"}
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axs,("Training","Validation")):
        g=calls[calls.split==split]
        for i,gender in enumerate(("M","F")):
            a=len(g[g.gender==gender]); b=len(g[(g.gender==gender)&g.matched])
            ax.bar(i-.16,a,.32,color=colors[gender],alpha=.4)
            ax.bar(i+.16,b,.32,color=colors[gender])
            ax.text(i,a,str(a)+" / "+str(b),ha="center",va="bottom",fontsize=9)
        ax.set(xticks=[0,1],xticklabels=["M","F"],title=split,ylabel="Calls: all labels / matched")
        ax.margins(y=.2)
    fig.tight_layout();fig.savefig(out/"gender_distribution.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,split in zip(axs,("Training","Validation")):
        for gender in ("M","F"):
            x=segments[(segments.split==split)&(segments.speaker==1)&(segments.gender==gender)].duration_s
            ax.hist(x,bins=np.geomspace(.001,max(100,float(x.max())),70),density=True,
                    histtype="step",label=gender,color=colors[gender])
        ax.set(xscale="log",xlabel="Reporter segment seconds (log scale)",ylabel="Density",title=split)
        ax.legend()
    fig.tight_layout();fig.savefig(out/"segment_duration_distribution.png",dpi=150);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    labels=["Train M","Train F","Valid M","Valid F"]
    groups=[calls[(calls.split==s)&(calls.gender==g)] for s in ("Training","Validation") for g in ("M","F")]
    for ax,col,title in zip(axs,("reporter_union_s","n_reporter_segments"),("Reporter union seconds / call","Reporter segments / call")):
        ax.boxplot([g[col] for g in groups],tick_labels=labels,showfliers=False)
        ax.set(title=title,ylabel=title)
    fig.tight_layout();fig.savefig(out/"gender_duration_comparison.png",dpi=150);plt.close(fig)
    if acoustics:
        a=pd.DataFrame(acoustics)
        fig,axs=plt.subplots(2,2,figsize=(10,7))
        for ax,col,title in zip(axs.flat,("f0_median_hz","pitch_voiced_ratio","reporter_rms","centroid_mean_hz"),
                               ("Raw pYIN F0 (Hz)","pYIN voiced frame ratio","Reporter RMS","Spectral centroid (Hz)")):
            ax.boxplot([a[(a.split==s)&(a.gender==g)][col].dropna() for s in ("Training","Validation") for g in ("M","F")],
                       tick_labels=labels,showfliers=True)
            ax.set(title=title)
        fig.suptitle("Stratified sample; descriptive only, no model or accuracy estimate")
        fig.tight_layout();fig.savefig(out/"sample_acoustic_features.png",dpi=150);plt.close(fig)


if __name__=="__main__":
    main()
