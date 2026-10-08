"""Development-only grid: select metric fusion and temporal settings."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from handpose.inference.fusion import fuse_metric
from handpose.temporal.smoothing import smooth
from handpose.evaluation.metrics import evaluate

p=argparse.ArgumentParser();p.add_argument('--runs',default='configs/experiments/expanded_development_runs.json');p.add_argument('--output',default='reports/fusion_development.json');a=p.parse_args()
runs=json.loads(Path(a.runs).read_text());entries=[e for e in json.load(open('configs/datasets/hot3d.json')) if e['split']=='development']
data={}
for name,root in runs.items():
    data[name]=[]
    for e in entries:
        path=Path(root)/Path(e['path']).stem
        f=dict(np.load(path/'predictions.npz'));f['gt']=np.load(path/'ground_truth.npz')['joints_3d']
        f['joints_3d']=np.where(f['validity'][...,None],f['joints_3d'],np.nan)
        data[name].append(f)
records=[]
# All stereo/direct-multiview sources; monocular variants are separately benchmarked.
names=[n for n in runs if 'mono' not in n.lower() and 'lift' not in n.lower()]
for first,second in itertools.permutations(names,2):
    for weight,gate,window in itertools.product((.25,.5,.75),(.02,.05,.1), (0.,.1)):
        ps=[];gs=[];per=[]
        for left,right in zip(data[first],data[second]):
            np.testing.assert_array_equal(left['timestamps'],right['timestamps'])
            np.testing.assert_allclose(left['gt'],right['gt'],equal_nan=True)
            x,source=fuse_metric(left['joints_3d'],right['joints_3d'],weight,gate)
            if window:
                confidence=np.where(np.isfinite(x).all(-1),1.,0.)
                x,_=smooth(x,confidence,left['timestamps'],window,.1)
            ps.append(x);gs.append(left['gt'])
            per.append(evaluate(x,left['gt'])['capped_error_with_missing_penalty_mm'])
        m=evaluate(np.concatenate(ps),np.concatenate(gs))
        records.append(dict(primary=first,secondary=second,weight_secondary=weight,agreement_m=gate,temporal_seconds=window,
                            per_sequence_capped_mm=per,metrics={k:m[k] for k in ('absolute_mpjpe_mm','fingertips_mm','joint_coverage','p90_mm','capped_error_with_missing_penalty_mm')}))
records.sort(key=lambda r:r['metrics']['capped_error_with_missing_penalty_mm'])
Path(a.output).write_text(json.dumps({'split':'development','selection_warning':'Many settings and only3 sequences; held-out verification required','candidates':records},indent=2,allow_nan=False))
for r in records[:10]:print(r)
