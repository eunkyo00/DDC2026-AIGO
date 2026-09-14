"""Inspect three selected calls for the repeated ~329 Hz pYIN median; no speech export."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import zipfile

os.environ.setdefault('NUMBA_CACHE_DIR','/tmp/ddc-m1-numba')
os.environ.setdefault('MPLCONFIGDIR','/tmp/ddc-m1-mpl')
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import welch


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-root',type=Path,required=True)
    ap.add_argument('--archive',type=Path,required=True,help='Used only as a filename index; audio read from data-root')
    ap.add_argument('--results',type=Path,default=Path(__file__).parent/'eda_outputs')
    args=ap.parse_args()
    ac=pd.read_csv(args.results/'sample_acoustics.csv')
    chosen=[ac[(ac.gender==g)&(ac.f0_median_hz.between(329,330))].iloc[0] for g in ('M','F')]
    chosen.append(ac[ac.gender=='M'].sort_values('f0_median_hz').iloc[0])
    with zipfile.ZipFile(args.archive) as z:
        index={(split,hashlib.sha256(Path(n).stem.encode()).hexdigest()):Path(*Path(n).parts[Path(n).parts.index(split):])
               for n in z.namelist() if n.endswith('.wav') and not Path(n).name.startswith('._') and '__MACOSX' not in n
               for split in ('Training','Validation') if split in Path(n).parts}
    fig,axs=plt.subplots(3,2,figsize=(11,9))
    diagnostics=[]
    for row,r in enumerate(chosen):
        y,sr=sf.read(args.data_root/index[(r.split,r.file_ref)],dtype='float32')
        if y.ndim==2:y=y.mean(axis=1)
        windows=[]
        for start,end in ast.literal_eval(r.feature_windows):
            x=y[round(start*sr):round(end*sr)]
            f0,v,_=librosa.pyin(x,sr=sr,fmin=65,fmax=650,frame_length=512,hop_length=160,center=False)
            f0_wide,_,_=librosa.pyin(x,sr=sr,fmin=65,fmax=1000,frame_length=512,hop_length=160,center=False)
            freq,power=welch(x,fs=sr,nperseg=min(len(x),2048))
            band=(freq>=320)&(freq<=340)
            windows.append(dict(start_s=start,end_s=end,
                f0_320_340_ratio=float(np.mean((f0>=320)&(f0<=340))),
                f0_median_hz=float(np.nanmedian(f0)) if np.isfinite(f0).any() else None,
                f0_median_hz_fmax1000=float(np.nanmedian(f0_wide)) if np.isfinite(f0_wide).any() else None,
                power_320_340_ratio=float(power[band].sum()/power.sum()),
                power_650_670_ratio=float(power[(freq>=650)&(freq<=670)].sum()/power.sum()),
                peak_spectral_hz=float(freq[np.argmax(power)])))
        selected=max(windows,key=lambda w:w['f0_320_340_ratio'])
        x=y[round(selected['start_s']*sr):round(selected['end_s']*sr)]
        S=np.abs(librosa.stft(x,n_fft=1024,hop_length=80,center=False))**2
        db=librosa.power_to_db(S,ref=np.max)
        axs[row,0].imshow(db,origin='lower',aspect='auto',extent=[0,len(x)/sr,0,sr/2],vmin=-70,vmax=0,cmap='magma')
        axs[row,0].set(ylim=(0,1500),xlabel='Seconds within selected crop',ylabel='Hz',title=f'{r.split} {r.gender}: diagnostic crop')
        freq,power=welch(x,fs=sr,nperseg=min(len(x),2048))
        axs[row,1].plot(freq,10*np.log10(power+1e-20))
        axs[row,1].axvline(329.48,color='red',linestyle='--',label='Repeated pYIN bin')
        axs[row,1].set(xlim=(0,1500),xlabel='Hz',ylabel='Power spectral density (dB/Hz)',title='Welch spectrum')
        axs[row,1].legend()
        diagnostics.append(dict(split=r.split,gender=r.gender,file_ref=r.file_ref,
                                selection='first repeated-bin call per gender; lowest-F0 male control',windows=windows,
                                plotted_window=selected))
    fig.tight_layout();fig.savefig(args.results/'pitch_peak_diagnostics.png',dpi=150);plt.close(fig)
    result=dict(repeated_bin_calls=int(ac.f0_median_hz.between(329,330).sum()),sample_calls=len(ac),
                diagnostics=diagnostics,limitation='Three diagnostic calls, not population prevalence; source of tonal components unconfirmed')
    tone=(.1*np.sin(2*np.pi*660*np.arange(8000)/8000)).astype('float32')
    result['synthetic_660_hz_sensitivity']={}
    for upper in (650,1000):
        pitch,voiced,_=librosa.pyin(tone,sr=8000,fmin=65,fmax=upper,frame_length=512,hop_length=160,center=False)
        result['synthetic_660_hz_sensitivity'][str(upper)]=dict(f0_median_hz=float(np.nanmedian(pitch)),voiced_ratio=float(voiced.mean()))
    (args.results/'pitch_peak_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
