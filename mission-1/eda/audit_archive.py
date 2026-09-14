"""Optional ZIP directory/JSON cross-check; never extracts WAVs or edits archives."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("archive",type=Path)
    ap.add_argument("--out",type=Path,default=Path(__file__).parent/"eda_outputs/archive_crosscheck.json")
    args=ap.parse_args()
    ids={k:defaultdict(set) for k in ("_id","recordId","audioPath")}
    result={}
    with zipfile.ZipFile(args.archive) as z:
        names=[n for n in z.namelist() if "__MACOSX" not in n and not Path(n).name.startswith("._")]
        for split in ("Training","Validation"):
            paths=[n for n in names if split in Path(n).parts]
            js=[n for n in paths if n.lower().endswith(".json")]
            wav={Path(n).stem for n in paths if n.lower().endswith(".wav")}
            genders,missing,dates,lengths=Counter(),Counter(),Counter(),Counter()
            for n in js:
                d=json.loads(z.read(n))
                genders[d.get("gender")]+=1
                if Path(n).stem not in wav:
                    missing[d.get("gender")]+=1
                    dates[Path(n).stem.split("_")[-1]]+=1
                lengths[len(str(d.get("recordId","")))]+=1
                for key in ids:
                    if d.get(key):
                        ids[key][hashlib.sha256(str(d[key]).encode()).hexdigest()].add(split)
            result[split]=dict(json_count=len(js),wav_count=len(wav),genders=dict(genders),
                missing_wav_gender=dict(missing),missing_wav_date_tokens=dict(dates),
                recordId_lengths=dict(lengths))
        result["cross_split_id_values"]={k:sum(len(v)>1 for v in values.values()) for k,values in ids.items()}
    result["source"]="ZIP central directory and every JSON; WAV payloads not decoded or CRC-verified"
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
