"""Evaluation-only selected-object ray occlusion; never imported by inference."""
import argparse,json
from pathlib import Path
import numpy as np
from handpose.objects.geometry import load_metric_mesh
p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();meshes={}
for e in json.load(open('configs/datasets/hot3d.json')):
 if e['split']!=a.split:continue
 name=Path(e['path']).stem;d=np.load(f'outputs/object-input/{name}.npz');p=np.load(f'outputs/object-pose/{name}.npz');ref=np.load(f'outputs/object-input/{name}_reference.npz')['T_left_from_object']
 gt=np.load(Path('outputs/ace-audit/dev-groundtruth' if a.split=='development' else 'outputs/adam-temporal-held')/name/'ground_truth_mano21.npz')['joints_3d']
 occluded=np.zeros(gt.shape[:2],bool);eligible=np.zeros_like(occluded)
 for f,j in enumerate(p['selected']):
  if j<0:continue
  oid=int(d['object_ids'][j]);T=ref[f,j]
  if not np.isfinite(T).all():continue
  if oid not in meshes:meshes[oid]=load_metric_mesh(f'data/hot3d/object_models/obj_{oid:06d}.glb')
  good=np.isfinite(gt[f]).all(-1);ids=np.flatnonzero(good)
  if not len(ids):continue
  origin=-T[:3,3]@T[:3,:3];rays=gt[f,good]@T[:3,:3];depth=np.linalg.norm(rays,axis=1);rays/=depth[:,None]
  hits,which,_=meshes[oid].ray.intersects_location(np.repeat(origin[None],len(ids),0),rays,multiple_hits=True)
  for point,k in zip(hits,which):
   if np.linalg.norm(point-origin)<depth[k]-.002:occluded[f,ids[k]]=True
  eligible[f,ids]=True
 np.savez_compressed(f'outputs/object-input/{name}_occlusion.npz',selected_object_ray_occluded=occluded,eligible=eligible)
 print(name,'occluded tips',int(occluded[:,[4,8,12,16,20]].sum()),flush=True)
