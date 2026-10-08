"""Paired sequence-bootstrap diagnostics, provenance and depth plots."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from handpose.evaluation.metrics import sequence_bootstrap

root=Path('outputs/ace-audit'); report=Path('reports/ace_audit')
names=['saved_decode_first81','fresh_640_81','fresh_480_81']
clips=sorted(p.stem for p in (root/'fresh_640_81').glob('clip-*.npz'))
if len(clips)!=3: raise RuntimeError('Expected all three held-out sequences')
pairs=[('fresh_480_81','fresh_640_81'),('fresh_640_81','saved_decode_first81')]
comparisons={}
for a,b in pairs:
    deltas=[]; ids=[]; per_sequence={}
    for clip in clips:
        x=np.load(root/a/(clip+'.npz')); y=np.load(root/b/(clip+'.npz'))
        np.testing.assert_allclose(x['ground_truth'],y['ground_truth'],atol=1e-9)
        gt=x['ground_truth']; pa=x['final_mano']; pb=y['final_mano']
        valid=np.isfinite(pa).all(-1)&np.isfinite(pb).all(-1)&np.isfinite(gt).all(-1)
        diff=(np.linalg.norm(pa-gt,axis=-1)-np.linalg.norm(pb-gt,axis=-1))[valid]*1000
        deltas.extend(diff); ids.extend([clip]*len(diff))
        per_sequence[clip]=dict(delta_mm=float(diff.mean()),paired_joints=len(diff))
    comparisons[a+' minus '+b]=dict(delta_mm=float(np.mean(deltas)),
        ci95_sequence_bootstrap_mm=sequence_bootstrap(deltas,ids,seed=42,samples=10000),
        per_sequence=per_sequence,paired_joints=len(deltas),sequences=3)
fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
for ax,clip in zip(axes,clips):
    for name,color in zip(names,['#888888','#2464b4','#d34c36']):
        p=np.load(root/name/(clip+'.npz'))
        ax.plot(np.arange(81)/30,p['final_mano'][:,0,2]*1000,label=name,color=color)
    ax.plot(np.arange(81)/30,p['ground_truth'][:,0,2]*1000,label='MANO reference',color='black',linewidth=2)
    ax.set_title(clip); ax.set_ylabel('Wrist depth (mm)'); ax.grid(alpha=.2)
axes[0].legend(fontsize=8); axes[-1].set_xlabel('Time in clip (seconds)')
fig.tight_layout(); fig.savefig(report/'wrist_depth.png',dpi=160); plt.close(fig)
files=[]
for directory in ['outputs/ace-audit-480-81','outputs/ace-audit-640-81','outputs/ace-audit']:
    for p in sorted(Path(directory).rglob('*')):
        if p.is_file(): files.append(dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(report/'paired_comparisons.json').write_text(json.dumps(comparisons,indent=2))
(report/'artifact_manifest.json').write_text(json.dumps(files,indent=2))
print(json.dumps(comparisons,indent=2))
