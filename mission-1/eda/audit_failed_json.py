"""Compare unparsed local JSON with a ZIP copy, without repairing original files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import unicodedata
import zipfile
import pandas as pd


def digest(b):
    return hashlib.sha256(b).hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True)
    ap.add_argument('--results',type=Path,default=Path(__file__).parent/'eda_outputs')
    args=ap.parse_args()
    calls=pd.read_csv(args.results/'calls.csv')
    known=set(zip(calls.split,calls.file_ref))
    results=[]
    with zipfile.ZipFile(args.archive) as z:
        lookup={(split,Path(n).name):n for n in z.namelist()
                if n.endswith('.json') and not Path(n).name.startswith('._') and '__MACOSX' not in n
                for split in ('Training','Validation') if split in Path(n).parts}
        for split in ('Training','Validation'):
            for directory,dirs,files in os.walk(args.data_root/split):
                for name in sorted(files):
                    if not name.endswith('.json') or name.startswith('._') or '__MACOSX' in Path(directory).parts:
                        continue
                    p=Path(directory)/name
                    ref=digest(unicodedata.normalize('NFC',p.stem).encode())
                    if (split,ref) in known:
                        continue
                    r={'split':split,'file_ref':ref}
                    try:
                        b=p.read_bytes()
                        r.update(local_bytes=len(b),local_sha256=digest(b))
                        try:
                            json.loads(b.decode('utf-8-sig'))
                            r['local_parse']='success'
                        except Exception as e:
                            r.update(local_parse=type(e).__name__,line=getattr(e,'lineno',None),column=getattr(e,'colno',None))
                        try:
                            json.loads(b)
                            r['local_autodetected_encoding_parse']='success'
                        except Exception as e:
                            r['local_autodetected_encoding_parse']=type(e).__name__
                        a=z.read(lookup[(split,name)])
                        d=json.loads(a)
                        r.update(archive_bytes=len(a),archive_sha256=digest(a),archive_parse='success',
                                 archive_gender=d.get('gender'),archive_segments=len(d['utterances']),bytes_equal=a==b,
                                 different_byte_positions=sum(x!=y for x,y in zip(a,b))+abs(len(a)-len(b)))
                    except Exception as e:
                        r['audit_failure']=type(e).__name__
                    results.append(r)
    (args.results/'json_failure_audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(json.dumps(results,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
