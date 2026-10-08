"""Evaluation-only access to anatomical hand and object reference annotations."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from handpose.evaluation.metrics import evaluate,sequence_bootstrap
p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['development','held_out']);a=p.parse_args()
report=Path('reports/object_aware');report.mkdir(parents=True,exist_ok=True)
root=Path('outputs/object-hand')/a.split
manifest=[e for e in json.load(open('configs/datasets/hot3d.json')) if e['split']==a.split]
pool={};truth=[];seqids=[];scores={};occlusion=[]
for e in manifest:
 name=Path(e['path']).stem
 gtpath=Path('outputs/ace-audit/dev-groundtruth' if a.split=='development' else 'outputs/adam-temporal-held')/name/'ground_truth_mano21.npz'
 gt=np.load(gtpath)['joints_3d'];truth.append(gt);
 ocpath=Path(f'outputs/object-input/{name}_occlusion.npz')
 occlusion.append(np.load(ocpath)['selected_object_ray_occluded'] if ocpath.exists() else np.zeros(gt.shape[:2],bool))
 seqids.extend([name]*len(gt));scores[name]={}
 source=Path('outputs/wilor-dev/stereo' if a.split=='development' else 'outputs/wilor-held/stereo')/name/'predictions.npz'
 files={'wilor_stereo':source}
 historical=Path('outputs/adam-temporal-held')/name/'predictions.npz'
 if a.split=='held_out' and historical.exists():files['existing_mano_temporal']=historical
 for folder in sorted(root.glob('*')):
  path=folder/name/'predictions.npz'
  if path.exists():files[folder.name]=path
 for label,path in files.items():
  d=np.load(path);xyz=np.where(d['validity'][...,None],d['joints_3d'],np.nan)
  if len(xyz)!=len(gt):raise ValueError('Frame mismatch')
  pool.setdefault(label,[]).append(xyz);scores[name][label]=evaluate(xyz,gt)
gt=np.concatenate(truth);pooled={};paired={};xyzs={}
for label,rows in pool.items():
 if len(rows)!=len(manifest):continue
 xyz=np.concatenate(rows);xyzs[label]=xyz;m=evaluate(xyz,gt)
 error=np.linalg.norm(xyz-gt,axis=-1)*1000;valid=np.isfinite(error)
 m['catastrophic_observed_joint_fraction_gt100mm']=float((error[valid]>100).mean()) if valid.any() else None
 m['catastrophic_frame_fraction_any_joint_gt100mm']=float((error>100).any(-1)[valid.any(-1)].mean()) if valid.any() else None
 tipmask=np.zeros_like(valid);tipmask[:,[4,8,12,16,20]]=True
 occ=np.concatenate(occlusion)&tipmask&valid
 m['selected_object_occluded_fingertip_error_mm']=float(error[occ].mean()) if occ.any() else None
 m['selected_object_occluded_fingertips_observed']=int(occ.sum())
 m['selected_object_occluded_fingertips_eligible']=int((np.concatenate(occlusion)&tipmask&np.isfinite(gt).all(-1)).sum())
 pooled[label]=m
controls=['wilor_stereo','hand_only_w1','visibility_w1','collision_w1','contact_w1']
for label,xyz in xyzs.items():
 for control in controls:
  if control not in xyzs or control==label:continue
  err=np.linalg.norm(xyz-gt,axis=-1)*1000;ref=np.linalg.norm(xyzs[control]-gt,axis=-1)*1000
  common=np.isfinite(err)&np.isfinite(ref);delta=np.where(common,err-ref,np.nan)
  frame=np.divide(np.nansum(delta,axis=-1),common.sum(-1),out=np.full(len(gt),np.nan),where=common.sum(-1)>0)
  paired[label+' minus '+control]=dict(joint_weighted_delta_mm=float(np.nanmean(delta)),mean_frame_delta_mm=float(np.nanmean(frame)),sequence_bootstrap_frame_delta_ci95=sequence_bootstrap(frame,seqids),shared_joints=int(common.sum()),improved_joints=int((delta<0).sum()),worsened_joints=int((delta>0).sum()))
result=dict(split=a.split,frames=len(gt),sequences=len(manifest),protocol='absolute native-left optical MANO21, m converted to mm, no alignment',pooled=pooled,sequences_detail=scores,paired=paired)
(report/f'{a.split}_hand_results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
for label,m in pooled.items():print(label,round(m['absolute_mpjpe_mm'],3),round(100*m['joint_coverage'],2),flush=True)
