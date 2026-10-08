"""Run a controlled hand-object ablation; no reference hand labels are read."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from handpose.objects.geometry import load_metric_mesh,voxel_sdf
from handpose.objects.hand_fit import fit_hand
p=argparse.ArgumentParser();p.add_argument('--split',required=True,choices=['development','held_out']);p.add_argument('--mode',required=True,choices=['hand_only','visibility','collision','contact']);p.add_argument('--oracle',action='store_true');p.add_argument('--oracle-all',action='store_true');p.add_argument('--weight',type=float,default=1.);p.add_argument('--steps',type=int,default=160);a=p.parse_args()
if a.oracle_all:a.oracle=True
if a.oracle and a.mode not in ['collision','contact']:p.error('Oracle only changes geometric object constraints')
label=a.mode+('_oracle_all' if a.oracle_all else '_oracle' if a.oracle else '')+f'_w{a.weight:g}'
out=Path('outputs/object-hand')/a.split/label;out.mkdir(parents=True,exist_ok=True)
fields={};cache=Path('outputs/object-sdf');cache.mkdir(exist_ok=True)
for entry in json.load(open('configs/datasets/hot3d.json')):
 if entry['split']!=a.split:continue
 name=Path(entry['path']).stem;source=Path('outputs/wilor-dev/stereo' if a.split=='development' else 'outputs/wilor-held/stereo')/name
 obs=dict(np.load(source/'predictions.npz'));parameters=json.load(open(source/'mano_parameters.json'))
 data=np.load(f'outputs/object-input/{name}.npz')
 poses=dict(selected=np.full(len(obs['timestamps']),-1),timestamps=obs['timestamps']) if a.mode=='hand_only' else np.load(f'outputs/object-pose/{name}.npz')
 np.testing.assert_array_equal(data['timestamps'],obs['timestamps']);np.testing.assert_array_equal(poses['timestamps'],obs['timestamps'])
 reference=np.load(f'outputs/object-input/{name}_reference.npz') if a.oracle else None
 objects=[]
 for i,j in enumerate(poses['selected']):
  if j<0:objects.append(None);continue
  oid=int(data['object_ids'][j]);mask=data['masks'][i,j];boundary=[]
  for m in mask:
   boundary.append(np.abs(distance_transform_edt(m>0)-distance_transform_edt(m==0)))
  T=reference['T_left_from_object'][i,j] if a.oracle else poses['T_left_from_object'][i]
  accepted=(a.oracle_all or bool(poses['accepted'][i])) and np.isfinite(T).all()
  objects.append(dict(id=oid,T=T,masks=mask,boundary_distance=np.array(boundary),accepted=accepted))
  if accepted and oid not in fields:
   path=cache/f'{oid:06d}.npz'
   if not path.exists():
    mesh=load_metric_mesh(f'data/hot3d/object_models/obj_{oid:06d}.glb');grid,origin,pitch=voxel_sdf(mesh)
    np.savez_compressed(path,grid=grid,origin=origin,pitch=pitch,watertight=mesh.is_watertight)
   f=np.load(path);fields[oid]=(f['grid'],f['origin'],float(f['pitch']))
 K=data['K'].copy();K[:2,:2]*=4;K[:2,2]=(K[:2,2]+.5)*4-.5
 start=time.time();xyz,vertices,contact,eligible=fit_hand(obs,parameters,K,data['T_right_from_left'],objects,fields,a.mode,a.weight,a.steps)
 folder=out/name;folder.mkdir(exist_ok=True)
 np.savez_compressed(folder/'predictions.npz',joints_3d=xyz,validity=np.isfinite(xyz).all(-1),timestamps=obs['timestamps'],frame_ids=obs['frame_ids'],vertices=vertices,contact_weight=contact,model_fitted=eligible,original_validity=obs['validity'])
 (folder/'metadata.json').write_text(json.dumps(dict(mode=a.mode,oracle_object_pose=a.oracle,hand_ground_truth_used=False,weight=a.weight,steps=a.steps,seconds=time.time()-start,coordinate_frame='native left optical',units='meters',joint_convention='MANO21',object_acceptance='all mask-selected objects' if a.oracle_all else 'same estimated-pose acceptance for oracle and estimated branches',missing='raw stereo fallback if MANO cannot fit; no empty frame filling',contact='uncalibrated geometric soft weights, not ContactOpt learned prediction'),indent=2))
 print(label,name,time.time()-start,'fitted',int(eligible.sum()),'contacts',int((contact>0).sum()),flush=True)
