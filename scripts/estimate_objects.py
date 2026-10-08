"""GT-free selected-object pose estimator for saved WiLoR/mask inputs."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from handpose.objects.geometry import load_metric_mesh
from handpose.objects.pose import estimate_pose

p=argparse.ArgumentParser();p.add_argument('--split',choices=['development','held_out'],required=True);p.add_argument('--stride',type=int,default=1);a=p.parse_args()
root=Path('outputs/object-input');out=Path('outputs/object-pose');out.mkdir(exist_ok=True)
meshes={};manifest=json.load(open('configs/datasets/hot3d.json'))
for e in manifest:
 if e['split']!=a.split:continue
 name=Path(e['path']).stem;d=np.load(root/f'{name}.npz');w=np.load(Path('outputs/wilor-dev/stereo' if a.split=='development' else 'outputs/wilor-held/stereo')/name/'predictions.npz')
 np.testing.assert_array_equal(d['timestamps'],w['timestamps'])
 N=len(w['timestamps']);pose=np.full((N,4,4),np.nan); selected=np.full(N,-1,int);scores=np.full(N,np.nan);spread=np.full(N,np.nan);accepted=np.zeros(N,bool)
 previous={};start=time.time()
 for f in range(0,N,a.stride):
  valid=w['validity'][f]&np.isfinite(w['joints_3d'][f]).all(-1)
  if valid.sum()<6:continue
  points=w['joints_3d'][f,valid]; candidates=[]
  for j,oid in enumerate(d['object_ids']):
   masks=d['masks'][f,j]; distances=[]
   if not (masks.sum((-1,-2))>20).all():continue
   for v,T in enumerate([np.eye(4),d['T_right_from_left'][f]]):
    q=points@T[:3,:3].T+T[:3,3];uv=q@d['K'].T;uv=uv[:,:2]/uv[:,2:];ij=np.rint(uv).astype(int)
    inside=(ij>=0).all(-1)&(ij<160).all(-1)
    if inside.sum()<3:distances.append(1e3);continue
    dist=distance_transform_edt(~masks[v].astype(bool));distances.append(float(np.sort(dist[ij[inside,1],ij[inside,0]])[:3].mean()))
   candidates.append((max(distances),j,int(oid)))
  if not candidates:continue
  distance,j,oid=min(candidates)
  if distance>12:continue
  if oid not in meshes:
   mesh=load_metric_mesh(f'data/hot3d/object_models/obj_{oid:06d}.glb');meshes[oid]=mesh.convex_hull.vertices
  fit=estimate_pose(meshes[oid],d['masks'][f,j],d['K'],d['T_right_from_left'][f],previous.get(oid))
  if fit is None:continue
  selected[f]=j;pose[f]=fit['T'];scores[f]=fit['score'];spread[f]=fit['translation_hypothesis_spread_m'];accepted[f]=fit['accepted']
  if fit['accepted']:previous[oid]=fit['T']
  if f%20==0:print(name,f,oid,round(scores[f],2),round(spread[f],3),flush=True)
 np.savez_compressed(out/f'{name}.npz',T_left_from_object=pose,selected=selected,scores=scores,spread=spread,accepted=accepted,timestamps=d['timestamps'])
 (out/f'{name}.json').write_text(json.dumps(dict(seconds=time.time()-start,selected=int((selected>=0).sum()),accepted=int(accepted.sum()),stride=a.stride,orientation_observable=False)))
 print(name,'completed',int(accepted.sum()),time.time()-start,flush=True)
