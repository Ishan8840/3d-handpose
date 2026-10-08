import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from handpose.visualization.plots import diagnostic_plots
from handpose.evaluation.comparison import error_coverage
runs=json.load(open('configs/experiments/held_out_runs.json'));out=Path('reports/figures/held-out');out.mkdir(parents=True,exist_ok=True);curves={};strata={}
for model,root in runs.items():
    predictions=[];truths=[];confidence=[];visibility={}
    for folder in sorted(Path(root).iterdir()):
        if not (folder/'predictions.npz').exists():continue
        pred=np.load(folder/'predictions.npz');gt=np.load(folder/'ground_truth.npz');dest=out/model.replace(' ','_')/folder.name
        worst=diagnostic_plots(pred['joints_3d'],gt['joints_3d'],pred['timestamps'],dest);(dest/'worst_frames.json').write_text(json.dumps(worst,indent=2))
        predictions.append(pred['joints_3d']);truths.append(gt['joints_3d']);confidence.append(pred['confidence'])
        p=folder/'visibility_metrics.json'
        if p.exists():visibility[folder.name]=json.loads(p.read_text())
    curves[model]=error_coverage(np.concatenate(predictions),np.concatenate(truths),np.concatenate(confidence))
    (out/(model.replace(' ','_')+'_visibility.json')).write_text(json.dumps(visibility,indent=2))
    strata[model]={}
    for label in ('visibility_low','visibility_partial','visibility_high'):
        metrics=[v[label] for v in visibility.values()]
        observed=sum(m['observed_joints'] for m in metrics);eligible=sum(m['eligible_joints'] for m in metrics)
        strata[model][label]={'error_mm':sum(m['absolute_mpjpe_mm']*m['observed_joints'] for m in metrics if m['absolute_mpjpe_mm'] is not None)/observed if observed else None,'coverage':observed/eligible if eligible else None}
fig,ax=plt.subplots(figsize=(7,5))
for name,curve in curves.items():
    valid=[m for m in curve if m['absolute_mpjpe_mm'] is not None]
    ax.plot([100*m['joint_coverage'] for m in valid],[m['absolute_mpjpe_mm'] for m in valid],marker='o',label=name)
ax.set_xlabel('Joint coverage (%)');ax.set_ylabel('Observed absolute MPJPE (mm)');ax.legend();fig.tight_layout();fig.savefig(out/'error_coverage.png');plt.close(fig)
(out/'error_coverage.json').write_text(json.dumps(curves,indent=2,allow_nan=False))

fig,axes=plt.subplots(1,2,figsize=(11,4));labels=['visibility_low','visibility_partial','visibility_high'];x=np.arange(3)
for i,(model,values) in enumerate(strata.items()):
    for ax,key,scale in ((axes[0],'error_mm',1),(axes[1],'coverage',100)):
        ax.bar(x+(i-1)*.25,[values[k][key]*scale if values[k][key] is not None else np.nan for k in labels],width=.25,label=model)
for ax in axes:ax.set_xticks(x,['<0.3','0.3–0.8','≥0.8']);ax.set_xlabel('Modeled hand visibility: minimum across views');ax.legend(fontsize=8)
axes[0].set_ylabel('Observed absolute MPJPE (mm)');axes[1].set_ylabel('Joint coverage (%)');axes[1].set_ylim(0,100);fig.tight_layout();fig.savefig(out/'visibility.png');plt.close(fig)
(out/'visibility_summary.json').write_text(json.dumps(strata,indent=2,allow_nan=False))
