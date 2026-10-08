"""Compare correlated failures and accuracy including all missing joint slots."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from handpose.evaluation.overlap import compare_errors
p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--manifest',default='configs/datasets/hot3d.json');p.add_argument('--split',default='held_out');p.add_argument('--output',default='reports/expanded_analysis');a=p.parse_args()
runs=json.load(open(a.runs));entries=[e for e in json.load(open(a.manifest)) if e['split']==a.split];out=Path(a.output);out.mkdir(parents=True,exist_ok=True);preds={};diagnostics={}
for name,root in runs.items():
 ps=[];gs=[]
 for e in entries:
  d=Path(root)/Path(e['path']).stem;q=np.load(d/'predictions.npz');ps.append(np.where(q['validity'][...,None],q['joints_3d'],np.nan));gs.append(np.load(d/'ground_truth.npz')['joints_3d'])
 xyz=np.concatenate(ps);gt=np.concatenate(gs);preds[name]=xyz
 eligible=np.isfinite(gt).all(-1);valid=np.isfinite(xyz).all(-1)&eligible;error=(xyz-gt)*1000;distance=np.linalg.norm(error,axis=-1)
 energy=float(np.nansum(error**2));n=valid.sum(-1);mean=np.divide(np.nansum(error,axis=1),n[:,None],out=np.zeros((len(n),3)),where=n[:,None]>0)
 item={'eligible_joint_slots':int(eligible.sum()),'observed_slots':int(valid.sum()),'depth_squared_error_fraction':float(np.nansum(error[:,:,2]**2)/energy) if energy else None,'common_translation_squared_error_fraction':float((n[:,None]*mean**2).sum()/energy) if energy else None,'thresholds':{}}
 for t in (8,10,20,50,100):item['thresholds'][str(t)]={'accurate_slot_rate':float(((distance<t)&valid).sum()/eligible.sum()),'capped_missing_mm':float(np.where(valid,np.minimum(distance,t),t)[eligible].mean())}
 diagnostics[name]=item
names=list(runs);n=len(names);cor=np.eye(n);jac=np.eye(n)
for i,j in itertools.combinations(range(n),2):
 d=compare_errors(preds[names[i]],preds[names[j]],gt);cor[i,j]=cor[j,i]=d['error_spearman_common'] if d['error_spearman_common'] is not None else np.nan;jac[i,j]=jac[j,i]=d['failure_jaccard']
fig,axs=plt.subplots(1,2,figsize=(max(12,n*1.1),max(6,n*.48)))
for ax,values,title,low,high in zip(axs,[cor,jac],['Observed-error Spearman correlation','Failure overlap (missing or >20 mm): Jaccard'],[-1,0],[1,1]):
 im=ax.imshow(values,vmin=low,vmax=high,cmap='coolwarm');ax.set_xticks(range(n),names,rotation=75,ha='right',fontsize=7);ax.set_yticks(range(n),names,fontsize=7);ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.7)
fig.tight_layout();fig.savefig(out/'overlap.png',dpi=150);plt.close(fig)
fig,ax=plt.subplots(figsize=(9,6))
for name in names:
 vals=diagnostics[name]['thresholds'];ax.plot([8,10,20,50,100],[vals[str(t)]['accurate_slot_rate']*100 for t in (8,10,20,50,100)],marker='o',label=name)
ax.set(xlabel='Absolute error threshold (mm)',ylabel='Accurate annotated joint slots (%)',title='Accuracy and missing observations jointly counted');ax.legend(fontsize=7);ax.grid(alpha=.25);fig.tight_layout();fig.savefig(out/'accuracy_coverage.png',dpi=150);plt.close(fig)
(out/'diagnostics.json').write_text(json.dumps({'split':a.split,'translation_definition':'Per-frame mean joint error; descriptive energy decomposition, never an alignment of predictions','models':diagnostics},indent=2,allow_nan=False))
