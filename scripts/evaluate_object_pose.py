"""Evaluation-only object pose errors, symmetries, and reference ray occlusion."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from handpose.objects.geometry import load_metric_mesh
p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args()
info=json.load(open('data/hot3d/object_models/models_info.json'));meshes={};rows=[]
for entry in json.load(open('configs/datasets/hot3d.json')):
 if entry['split']!=a.split:continue
 name=Path(entry['path']).stem;d=np.load(f'outputs/object-input/{name}.npz');poses=np.load(f'outputs/object-pose/{name}.npz');ref=np.load(f'outputs/object-input/{name}_reference.npz')['T_left_from_object']
 for f,j in enumerate(poses['selected']):
  if j<0:continue
  oid=int(d['object_ids'][j]);T=poses['T_left_from_object'][f];G=ref[f,j]
  if not np.isfinite(G).all():continue
  if oid not in meshes:meshes[oid]=load_metric_mesh(f'data/hot3d/object_models/obj_{oid:06d}.glb')
  mesh=meshes[oid];v=mesh.vertices[::max(1,len(mesh.vertices)//1500)];center=mesh.centroid
  sym=[np.eye(4)]+[np.array(s).reshape(4,4) for s in info[str(oid)].get('symmetries_discrete',[])]
  # Discretization only for evaluation; report angular resolution explicitly.
  for continuous in info[str(oid)].get('symmetries_continuous',[]):
   axis=np.array(continuous['axis']);offset=np.array(continuous['offset']);base=sym.copy();sym=[]
   for angle in np.arange(0,2*np.pi,np.pi/60):
    S=np.eye(4);S[:3,:3]=Rotation.from_rotvec(axis*angle).as_matrix();S[:3,3]=offset-S[:3,:3]@offset
    sym.extend([S@b for b in base])
  angle=min(np.degrees(Rotation.from_matrix(T[:3,:3].T@(G@s)[:3,:3]).magnitude()) for s in sym)
  pred=v@T[:3,:3].T+T[:3,3];gt=v@G[:3,:3].T+G[:3,3]
  adds=cKDTree(gt).query(pred)[0].mean()*1000
  row=dict(clip=name,frame=f,object_id=oid,accepted=bool(poses['accepted'][f]),mask_score=float(poses['scores'][f]),spread_mm=float(poses['spread'][f]*1000),translation_origin_mm=float(np.linalg.norm(T[:3,3]-G[:3,3])*1000),translation_centroid_mm=float(np.linalg.norm(center@T[:3,:3].T+T[:3,3]-(center@G[:3,:3].T+G[:3,3]))*1000),symmetry_reduced_rotation_deg=float(angle),adds_mm=float(adds))
  rows.append(row)
keys=['translation_origin_mm','translation_centroid_mm','symmetry_reduced_rotation_deg','adds_mm']
summary={}
for name,subset in [('all_attempted',rows),('accepted',[r for r in rows if r['accepted']])]:
 summary[name]=dict(count=len(subset),**{k:float(np.mean([r[k] for r in subset])) if subset else None for k in keys})
Path('reports/object_aware').mkdir(parents=True,exist_ok=True)
Path(f'reports/object_aware/{a.split}_object_results.json').write_text(json.dumps(dict(summary=summary,frames=rows,continuous_symmetry_sampling_degrees=3,orientation_observable=False),indent=2))
print(json.dumps(summary),flush=True)
