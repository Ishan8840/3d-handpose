"""Paired diagnostics and measured report for frozen ACE rigid variants."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from handpose.geometry.camera import load_stereo
from handpose.evaluation.metrics import sequence_bootstrap, evaluate

report=Path('reports/ace_rigid'); selection=json.load(open(report/'frozen_selection.json'))['selection']
root=Path('outputs/ace-rigid/held_out')/selection['se3']
data={p.parent.name:dict(np.load(p)) for p in sorted(root.glob('*/predictions.npz'))}
for clip,d in data.items():
    w=np.load(f'outputs/adam-temporal-held/{clip}/predictions.npz')
    np.testing.assert_array_equal(w['timestamps'],d['timestamps'])
    d['wilor_temporal']=np.where(w['validity'][...,None],w['joints_3d'],np.nan)
methods=['ace_final','anchored','se3','wilor','wilor_temporal']
gt=np.concatenate([d['ground_truth'] for d in data.values()])
overall={m:evaluate(np.concatenate([d[m] for d in data.values()]),gt) for m in methods}
(report/'held_out_comparison.json').write_text(json.dumps(overall,indent=2))
results={}
for method,base in [('anchored','ace_final'),('se3','ace_final'),('se3','anchored'),('anchored','wilor'),('se3','wilor'),('anchored','wilor_temporal')]:
    diffs=[]; ids=[]; sequences={}; tips=[]; wrists=[]
    for clip,d in data.items():
        g=d['ground_truth']; a=d[method]; b=d[base]
        mask=np.isfinite(a).all(-1)&np.isfinite(b).all(-1)&np.isfinite(g).all(-1)
        joint_delta=(np.linalg.norm(a-g,axis=-1)-np.linalg.norm(b-g,axis=-1))*1000
        delta=joint_delta[mask]
        tips.extend(joint_delta[:,[4,8,12,16,20]][mask[:,[4,8,12,16,20]]])
        wrists.extend(joint_delta[:,0][mask[:,0]])
        diffs.extend(delta);ids.extend([clip]*len(delta))
        sequences[clip]=dict(mean_delta_mm=float(delta.mean()),paired_joints=len(delta))
    results[method+' minus '+base]=dict(mean_delta_mm=float(np.mean(diffs)),
        ci95_sequence_bootstrap_mm=sequence_bootstrap(diffs,ids,seed=42,samples=10000),sequences=sequences,
        paired_fingertip_delta_mm=float(np.mean(tips)),paired_wrist_delta_mm=float(np.mean(wrists)))
C=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
diagnostics={};fig,axes=plt.subplots(3,1,figsize=(10,9),sharex=True)
for ax,(clip,d) in zip(axes,data.items()):
    camera=load_stereo(f'outputs/ace-input-{clip}/calibration.json')
    with open(f'outputs/ace-upright-held_out/{clip}/left/left.pkl','rb') as f:left=pickle.load(f)
    with open(f'outputs/ace-upright-held_out/{clip}/right/right.pkl','rb') as f:right=pickle.load(f)
    valid=np.isfinite(d['anchored']).all((1,2))&np.isfinite(d['se3']).all((1,2))
    reproj={}; errors={}; depths={}
    for method in ['ace_final','anchored','se3','wilor']:
        e=np.linalg.norm(d[method]-d['ground_truth'],axis=-1)*1000
        errors[method]=dict(observed_mpjpe_mm=float(np.nanmean(e)),
            catastrophic_joint_error_over100mm_fraction=float(np.nanmean(e[np.isfinite(e)]>100)))
        depths[method]=float(np.nanmean((d[method][:,0,2]-d['ground_truth'][:,0,2])*1000))
        ax.plot(np.arange(len(e))/30,d[method][:,0,2]*1000,label=method)
        if method not in ['anchored','se3']:continue
        residual=[]
        for i in np.flatnonzero(valid):
            support=d['stereo_evidence'][i]
            for cam,exp in zip(camera,[left,right]):
                uv=cam.project(d[method][i]@C.T)
                residual.extend(np.linalg.norm(uv-exp['joints_2d'][i,1]*640,axis=1)[support])
        reproj[method]=float(np.mean(residual))
    diagnostics[clip]=dict(reprojection_px_on_same_stereo_support=reproj,metrics=errors,wrist_signed_depth_mm=depths)
    ax.plot(np.arange(len(valid))/30,d['ground_truth'][:,0,2]*1000,color='black',linewidth=2,label='reference')
    ax.set_title(clip);ax.set_ylabel('Wrist depth (mm)');ax.grid(alpha=.2)
axes[0].legend(ncol=5,fontsize=8);axes[-1].set_xlabel('Seconds');fig.tight_layout()
fig.savefig(report/'wrist_depth.png',dpi=150);plt.close(fig)
(report/'paired_comparisons.json').write_text(json.dumps(results,indent=2))
(report/'failure_diagnostics.json').write_text(json.dumps(diagnostics,indent=2))
manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('outputs/ace-rigid').rglob('*') if p.is_file()}
(report/'artifact_hashes.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(results,indent=2));print(json.dumps(diagnostics,indent=2))
