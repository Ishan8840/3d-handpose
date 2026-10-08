"""Apply a GT-free hand/object ablation to synchronized saved stereo observations.

Object input: masks, object_ids, K (160px pinhole), T_right_from_left, timestamps.
Pose input: selected object indices, T_left_from_object, accepted, timestamps.
These inputs are produced by prepare_object_inputs.py / estimate_objects.py for HOT3D.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt
from handpose.objects.geometry import load_metric_mesh,voxel_sdf
from handpose.objects.hand_fit import fit_hand
from handpose.models.base import JOINT_NAMES

def main():
 p=argparse.ArgumentParser();p.add_argument('--observations',required=True);p.add_argument('--mano-parameters',required=True);p.add_argument('--object-input',required=True);p.add_argument('--object-pose',required=True);p.add_argument('--mesh-dir',required=True);p.add_argument('--output',required=True);p.add_argument('--mode',choices=['hand_only','visibility','collision','contact'],default='contact');p.add_argument('--weight',type=float,default=1.);p.add_argument('--steps',type=int,default=160);a=p.parse_args()
 obs=dict(np.load(a.observations));parameters=json.load(open(a.mano_parameters));d=np.load(a.object_input);poses=np.load(a.object_pose)
 np.testing.assert_array_equal(d['timestamps'],obs['timestamps']);np.testing.assert_array_equal(poses['timestamps'],obs['timestamps'])
 if d['masks'].shape[-2:]!=(160,160):raise ValueError('Expected documented 160px object mask calibration')
 objects=[];fields={}
 for i,j in enumerate(poses['selected']):
  if j<0:objects.append(None);continue
  oid=int(d['object_ids'][j]);mask=d['masks'][i,j];T=poses['T_left_from_object'][i]
  if poses['accepted'][i] and not np.isfinite(T).all():raise ValueError('Accepted object must have a finite pose')
  boundary=np.array([np.abs(distance_transform_edt(m>0)-distance_transform_edt(m==0)) for m in mask])
  objects.append(dict(id=oid,T=T,masks=mask,boundary_distance=boundary,accepted=bool(poses['accepted'][i])))
  if oid not in fields and poses['accepted'][i]:fields[oid]=voxel_sdf(load_metric_mesh(Path(a.mesh_dir)/f'obj_{oid:06d}.glb'))
 K=d['K'].copy();K[:2,:2]*=4;K[:2,2]=(K[:2,2]+.5)*4-.5
 start=time.time();xyz,vertices,contact,eligible=fit_hand(obs,parameters,K,d['T_right_from_left'],objects,fields,a.mode,a.weight,a.steps)
 out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 arrays=dict(joints_3d=xyz,validity=np.isfinite(xyz).all(-1),timestamps=obs['timestamps'],frame_ids=obs['frame_ids'],vertices=vertices,contact_weight=contact,original_validity=obs['validity'],model_fitted=eligible)
 arrays['confidence']=obs['confidence']
 arrays['observation_type']=np.where(eligible[:,None]&arrays['validity'],5,obs.get('observation_type',0))
 wrist=np.full((len(xyz),4,4),np.nan);wrist[:,:3,3]=xyz[:,0];wrist[:,3]=[0,0,0,1]
 arrays['wrist_poses']=wrist
 np.savez_compressed(out/'predictions.npz',**arrays)
 for key in ['joints_3d','validity','timestamps','confidence','wrist_poses']:np.save(out/f'{key}.npy',arrays[key])
 (out/'metadata.json').write_text(json.dumps(dict(coordinate_frame='native left optical',units='meters',joint_convention='anatomical MANO21',joint_order=list(JOINT_NAMES),wrist_rotation='not exported; NaN',mode=a.mode,weight=a.weight,steps=a.steps,seconds=time.time()-start,reference_hand_labels_read=False,object_pose_source=a.object_pose,observation_classification='model fitted on eligible frames; original_validity preserves stereo support; otherwise raw stereo fallback',confidence='input confidence is detector evidence, not calibrated posterior uncertainty',limitations='mask-conditioned; approximate voxel collision; geometric contact prior, not official learned ContactOpt'),indent=2))

if __name__=='__main__':main()
