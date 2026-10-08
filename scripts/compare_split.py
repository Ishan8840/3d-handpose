"""Paired participant/sequence comparison on identical frames and GT conventions."""
import argparse,json
from pathlib import Path
import numpy as np
from handpose.evaluation.metrics import evaluate,sequence_bootstrap
from handpose.evaluation.comparison import paired_comparison
p=argparse.ArgumentParser();p.add_argument('--manifest',default='data/hot3d/manifest.json');p.add_argument('--split',required=True);p.add_argument('--runs',required=True,help='JSON mapping model labels to output roots');p.add_argument('--output',required=True);p.add_argument('--mano21',action='store_true');a=p.parse_args()
entries=[e for e in json.load(open(a.manifest)) if e['split']==a.split];runs=json.load(open(a.runs));reference={};predictions={};summary={};ids=[]
for e in entries:ids.extend([e['sequence']]*e['max_frames'])
for name,root in runs.items():
    ps=[];gs=[];persequence=[]
    for e in entries:
        folder=Path(root)/Path(e['path']).stem;pred=np.load(folder/'predictions.npz');gt=np.load(folder/('ground_truth_mano21.npz' if a.mano21 else 'ground_truth.npz'))
        if len(pred['timestamps'])!=e['max_frames'] or not np.array_equal(pred['timestamps'],gt['timestamps']) or not np.array_equal(pred['frame_ids'],gt['frame_ids']):raise ValueError('Mismatched frames')
        key=e['sequence'];r=(gt['joints_3d'],gt['timestamps'])
        if key in reference:
            if not np.allclose(reference[key][0],r[0],equal_nan=True) or not np.array_equal(reference[key][1],r[1]):raise ValueError('Ground truth differs between methods')
        else:reference[key]=r
        xyz=np.where(pred['validity'][...,None],pred['joints_3d'],np.nan);ps.append(xyz);gs.append(gt['joints_3d']);persequence.append(evaluate(xyz,gt['joints_3d'],timestamps_ns=pred['timestamps']))
    P=np.concatenate(ps);G=np.concatenate(gs);m=evaluate(P,G);m['tracking_discontinuities']=sum(v['tracking_discontinuities'] for v in persequence)
    m['per_sequence']=dict(zip([e['sequence'] for e in entries],persequence));eligible=np.isfinite(G).all(-1);err=np.linalg.norm(P-G,axis=-1)*1000
    loss=np.where(np.isfinite(err),np.minimum(err,100),100);perframe=np.divide(np.where(eligible,loss,0).sum(1),eligible.sum(1),out=np.full(len(P),np.nan),where=eligible.sum(1)>0)
    m['capped_score_sequence_bootstrap_95ci']=sequence_bootstrap(perframe,ids);summary[name]=m;predictions[name]=P
paired={}
for i,first in enumerate(runs):
    for second in list(runs)[i+1:]:paired[first+' minus '+second]=paired_comparison(predictions[first],predictions[second],G,ids)
out=Path(a.output);out.mkdir(parents=True,exist_ok=True);(out/'comparison.json').write_text(json.dumps({'split':a.split,'gt':'MANO-21' if a.mano21 else 'UmeTrack-19','frames':len(G),'sequences':len(entries),'models':summary,'paired':paired},indent=2,allow_nan=False))
for name,m in summary.items():print(name,{k:m[k] for k in ('absolute_mpjpe_mm','wrist_mm','joint_coverage','p90_mm','capped_error_with_missing_penalty_mm')})
print('Paired',paired)
