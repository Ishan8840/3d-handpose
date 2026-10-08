"""Paired error analysis and figures from executed object ablations only."""
import json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from handpose.evaluation.metrics import sequence_bootstrap

R=Path('reports/object_aware');selection=json.load(open(R/'frozen_selection.json'))
weight=f"{selection['weight']:g}"
labels=['wilor_stereo','existing_mano_temporal','hand_only_w1','visibility_w1',selection['collision'],selection['contact'],'collision_oracle_w'+weight,'collision_oracle_all_w'+weight,'contact_oracle_all_w'+weight]
clips=[Path(e['path']).stem for e in json.load(open('configs/datasets/hot3d.json')) if e['split']=='held_out']
truth=[];pool={k:[] for k in labels};seq=[];object_error=[];accepted=[];contact={k:[] for k in labels}
obj_report=json.load(open(R/'held_out_object_results.json'))
byframe={(r['clip'],r['frame']):r for r in obj_report['frames']}
for name in clips:
 gt=np.load(Path('outputs/adam-temporal-held')/name/'ground_truth_mano21.npz')['joints_3d'];truth.append(gt);seq.extend([name]*len(gt))
 poses=np.load(f'outputs/object-pose/{name}.npz');accepted.extend(poses['accepted'])
 object_error.extend([byframe.get((name,i),{}).get('translation_centroid_mm',np.nan) for i in range(len(gt))])
 for label in labels:
  path=(Path('outputs/wilor-held/stereo')/name/'predictions.npz' if label=='wilor_stereo' else Path('outputs/adam-temporal-held')/name/'predictions.npz' if label=='existing_mano_temporal' else Path('outputs/object-hand/held_out')/label/name/'predictions.npz')
  d=np.load(path);pool[label].append(np.where(d['validity'][...,None],d['joints_3d'],np.nan))
  contact[label].extend((d['contact_weight']>0).sum(-1) if 'contact_weight' in d else np.zeros(len(gt)))
gt=np.concatenate(truth);xyz={k:np.concatenate(v) for k,v in pool.items()};errors={k:np.linalg.norm(v-gt,axis=-1)*1000 for k,v in xyz.items()}
mean=lambda v:np.divide(np.nansum(v,axis=1),np.isfinite(v).sum(1),out=np.full(len(v),np.nan),where=np.isfinite(v).sum(1)>0)
analysis={};base=errors['hand_only_w1'];fig,axes=plt.subplots(2,2,figsize=(12,8))
for label in labels:
 e=errors[label];valid=np.isfinite(e);values=np.sort(e[valid]);axes[0,0].plot(values,np.arange(1,len(values)+1)/len(values),label=label)
 axes[0,1].plot(np.nanmean(e,axis=0),label=label)
 if label=='wilor_stereo':continue
 delta=mean(e-base);depth=mean(np.abs(xyz['hand_only_w1'][...,2]-gt[...,2])*1000)
 oe=np.array(object_error);v=np.isfinite(delta)&np.isfinite(oe)&np.array(accepted)
 corr=float(spearmanr(oe[v],delta[v]).statistic) if v.sum()>3 and np.std(delta[v])>1e-9 else None
 common=np.isfinite(e)&np.isfinite(base)
 residual_corr=float(np.corrcoef(e[common],base[common])[0,1])
 analysis[label]=dict(error_pearson_vs_refit=residual_corr,paired_delta_mm=float(np.nanmean(e-base)),paired_sequence_ci95=sequence_bootstrap(delta,seq),object_error_vs_hand_delta_spearman=corr,object_error_correlation_frames=int(v.sum()),per_clip_delta_mm={c:float(np.nanmean(delta[np.array(seq)==c])) for c in clips},contact_active_frames=int((np.array(contact[label])>0).sum()))
 if label in [selection['collision'],'contact_oracle_all_w'+weight]:
  axes[1,0].scatter(depth,delta,s=8,alpha=.6,label=label)
  axes[1,1].scatter(oe[v],delta[v],s=15,alpha=.6,label=label)
axes[0,0].set(xlim=(0,100),xlabel='Absolute joint error (mm)',ylabel='Observed-joint CDF')
axes[0,1].set(xlabel='Canonical MANO joint index',ylabel='Absolute error (mm)')
axes[1,0].set(xlabel='Hand-only mean depth error (mm)',ylabel='Paired MPJPE change (mm)')
axes[1,1].set(xlabel='Accepted object centroid error (mm)',ylabel='Paired MPJPE change (mm)')
for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=6)
fig.tight_layout();fig.savefig(R/'error_analysis.png',dpi=160)
(R/'paired_analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False))
# Side-by-side camera overlays; selected examples explicitly diagnostic, not random.
import cv2
from handpose.models.base import EDGES
C=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
comp='contact_oracle_all_w'+weight;delta=mean(errors[comp]-base);finite=np.flatnonzero(np.isfinite(delta));ordered=finite[np.argsort(delta[finite])]
examples=[int(ordered[0]),int(ordered[len(ordered)//2]),int(ordered[-1])];rows=[]
for index in examples:
 ci=index//150;frame=index%150;name=clips[ci]
 cap=cv2.VideoCapture(f'outputs/ace-input-{name}/left.mp4');cap.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,im=cap.read();cap.release()
 if not ok:continue
 panes=[]
 for label in ['hand_only_w1',selection['contact'],comp]:
  canvas=im.copy()
  for points,color in [(gt[index],(60,210,60)),(xyz[label][index],(255,80,255))]:
   p=points@C.T;uv=p[:,:2]/p[:,2:]*320+319.5
   for u,v in EDGES:
    if np.isfinite(uv[[u,v]]).all() and (p[[u,v],2]>0).all():cv2.line(canvas,tuple(uv[u].astype(int)),tuple(uv[v].astype(int)),color,2)
  cv2.rectangle(canvas,(0,0),(640,48),(0,0,0),-1);cv2.putText(canvas,label,(8,19),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1);cv2.putText(canvas,f'{name} frame {frame}, green=reference',(8,40),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1)
  panes.append(cv2.resize(canvas,(400,400)))
 rows.append(np.hstack(panes))
if rows:cv2.imwrite(str(R/'hand_comparison.png'),np.vstack(rows))
files={}
for root in [Path('outputs/object-hand'),Path('outputs/object-pose'),Path('outputs/object-input')]:
 for p in sorted(root.rglob('*.npz')):files[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
(R/'artifact_hashes.json').write_text(json.dumps(files,indent=2))
print(json.dumps(analysis,indent=2))
runtimes={}
for split in ['development','held_out']:
    models={}
    for folder in sorted((Path('outputs/object-hand')/split).glob('*')):
        records=[json.load(open(p)) for p in sorted(folder.glob('clip-*/metadata.json'))]
        if records:models[folder.name]=dict(clips=len(records),frames=len(records)*150,hand_fit_seconds=sum(d['seconds'] for d in records),excludes='object-pose estimation, mesh download and SDF preparation')
    runtimes[split]=models
runtimes['object_pose']={p.stem:json.load(open(p)) for p in sorted(Path('outputs/object-pose').glob('clip-*.json'))}
runtimes['peak_gpu_memory']='not instrumented for these runs'
(R/'runtime.json').write_text(json.dumps(runtimes,indent=2))
