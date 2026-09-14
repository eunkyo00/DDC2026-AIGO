"""Synthetic audit checks; no real data or classification model is used."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import run_eda as eda


class EdaChecks(unittest.TestCase):
    def test_acoustic_known_tones(self):
        import librosa
        import numpy as np
        sr=8000
        for hz in (110,220):
            y=(.1*np.sin(2*np.pi*hz*np.arange(sr)/sr)).astype('float32')
            pitch,voiced,_=librosa.pyin(y,sr=sr,fmin=65,fmax=650,frame_length=512,hop_length=160,center=False)
            self.assertLess(abs(float(np.nanmedian(pitch))-hz),5)
            self.assertGreater(float(voiced.mean()),.9)
            mfcc=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13,n_fft=512,hop_length=160,center=False,n_mels=40)
            self.assertEqual(mfcc.shape[0],13)
            self.assertTrue(np.isfinite(mfcc).all())

    def test_overlap_and_redaction(self):
        self.assertEqual(eda.merged([(0,2),(1,3),(4,5)]), [[0,3],[4,5]])
        self.assertEqual(eda.overlap_seconds([(0,2),(1,3)],[(2,4)]), 1)
        self.assertEqual(eda.overlap_seconds([(0,1)],[(1,2)]), 0)
        self.assertNotIn("private", json.dumps(eda.redact({"text":"private","gender":"M"})))

    def test_audit_end_to_end(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)
            data=base/"data"
            for split in ("Training","Validation"):
                (data/split).mkdir(parents=True)
                for gender in ("M","F"):
                    stem=split+gender
                    p=data/split/(stem+".wav")
                    with wave.open(str(p),"wb") as f:
                        f.setparams((1,2,8000,0,"NONE","not compressed"))
                        f.writeframes(b"\x00\x00"*8000)
                    self.assertEqual(eda.riff_check(p),"ok")
                    d={"gender":gender,"startAt":0,"endAt":1000,"recordId":"shared-original",
                       "utterances":[{"speaker":1,"startAt":0,"endAt":700,"text":"private"},
                                     {"speaker":1,"startAt":500,"endAt":1000},
                                     {"speaker":0,"startAt":800,"endAt":1100},
                                     {"speaker":1,"startAt":50,"endAt":20}]}
                    (data/split/(stem+".json")).write_text(json.dumps(d))
            boundary=dict(d,utterances=d['utterances']+[
                {'speaker':1,'startAt':3010,'endAt':3510},
                {'speaker':1,'startAt':3010,'endAt':4010},
                {'speaker':1,'startAt':3010,'endAt':5010}])
            (data/"Training"/"missing.json").write_text(json.dumps(boundary))
            (data/"Training"/"._ignored.wav").write_bytes(b"sidecar")
            out=base/"result"
            subprocess.run([sys.executable,str(Path(eda.__file__)),"--data-root",str(data),
                            "--out-dir",str(out),"--skip-acoustics"],check=True,capture_output=True)
            s=json.loads((out/"summary.json").read_text())
            self.assertEqual(s["inventory"]["Training"]["json_only_stems"],1)
            self.assertEqual(s["inventory"]["Training"]["wav_files"],2)
            self.assertEqual(s["issues"]["invalid_segment_time"],5)
            self.assertEqual(s["issues"]["segment_exceeds_wav"],4)
            self.assertEqual(s["leakage"]["exact_wav_bytes"]["cross_split_groups"],1)
            self.assertEqual(s["leakage"]["recordId"]["cross_split_values"],1)
            self.assertEqual(s['groups']['Training/all_labels']['genders']['F']['short_segments']['1'],5)
            self.assertEqual(s['groups']['Training/all_labels']['genders']['F']['short_segments']['2'],6)
            self.assertAlmostEqual(s["groups"]["Training/matched"]["genders"]["M"]["reporter_union_s"]["mean"],1)
            self.assertAlmostEqual(s["groups"]["Training/matched"]["genders"]["M"]["reporter_sum_s"]["mean"],1.2)
            self.assertNotIn("private",(out/"schema_examples_redacted.json").read_text())
            subprocess.run([sys.executable,str(Path(eda.__file__).with_name('build_report.py')),
                            '--results',str(out),'--out',str(base/'report.md')],check=True,capture_output=True)
            self.assertTrue((base/'report.md').exists())
            p.write_bytes(p.read_bytes()[:-2])
            self.assertEqual(eda.riff_check(p),"truncated_RIFF")


if __name__=="__main__":
    unittest.main()
