"""Score trusted saved ACE pickles against raw HOT3D MANO annotations.

GT is loaded only in this scoring process. Never pass these annotations to
inference. Uses camera-frame MANO translation with the shape-root correction.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
import tarfile
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def main():
    torch.set_num_threads(4)
    parser=argparse.ArgumentParser()
    parser.add_argument('--predictions',type=Path,default=Path('outputs/ace-upright-held_out'))
    parser.add_argument('--output',type=Path,default=Path('reports/ace_audit/paper_protocol.json'))
    parser.add_argument('--gt-root',type=Path,default=Path('outputs/adam-temporal-held'))
    a=parser.parse_args(); root=Path.cwd()
    sys.path[:0]=[str(root/'src'),str(root/'third_party/ace'),str(root/'third_party/hand_tracking_toolkit')]
    os.environ['ACE_EGO_HAND_MANO_DIR']=str(root/'checkpoints/mano_converted')
    from ace_ego_hand.mano_utils import mano_forward_batch_full
    import smplx
    from hand_tracking_toolkit.camera import from_json
    from hand_tracking_toolkit.hand_models.mano_hand_model import MANOHandModel
    from handpose.evaluation.ace_protocol import Hand, score
    spec=importlib.util.spec_from_file_location('ace_viz',root/'third_party/ace/scripts/viz_preds.py')
    viz=importlib.util.module_from_spec(spec); spec.loader.exec_module(viz)
    models={side:smplx.MANO(str(root/'checkpoints/mano_converted'),is_rhand=side,
                           use_pca=False,flat_hand_mean=False).eval() for side in (False,True)}
    # Reference must use the official toolkit, including its left shapedirs fix.
    # Predictions retain ACE's official decoder conventions unchanged.
    official_gt=MANOHandModel(str(root/'checkpoints/mano_converted'))
    gtmodels={False:official_gt.mano_layer_left,True:official_gt.mano_layer_right}
    canonical={}; canonical_flat={}
    for side,flag in [('left',False),('right',True)]:
        with torch.no_grad():
            j,v=mano_forward_batch_full(torch.eye(3)[None],torch.eye(3).repeat(1,15,1,1),torch.zeros(1,10),models[flag])
        canonical[side]=Hand(side,j[0].numpy(),v[0].numpy(),np.zeros(3),np.eye(3),np.zeros((21,2)))
        flat=smplx.MANO(str(root/'checkpoints/mano_converted'),is_rhand=flag,
                        use_pca=False,flat_hand_mean=True).eval()
        with torch.no_grad():
            fj,fv=mano_forward_batch_full(torch.eye(3)[None],torch.eye(3).repeat(1,15,1,1),torch.zeros(1,10),flat)
        canonical_flat[side]=Hand(side,fj[0].numpy(),fv[0].numpy(),np.zeros(3),np.eye(3),np.zeros((21,2)))
    order=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
    # Native Quest image to upright: x'=-y, y'=x, z'=z.
    C=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    allframes=[]; per_sequence={}; coordinate_checks=[]
    for directory in sorted(a.predictions.glob('clip-*')):
        with (directory/'left/left.pkl').open('rb') as f: pred=pickle.load(f)
        joints,verts=viz.mano_cam_joints(pred,models,'cpu',want_verts=True)
        intr=pred['intrinsics']; size=(int(intr['image_width']),int(intr['image_height']))
        K=np.array([[intr['fx'],0,intr['cx']],[0,intr['fy'],intr['cy']],[0,0,1.]])
        frames=[]; gt_reference=np.load(root/a.gt_root/directory.name/'ground_truth_mano21.npz')['joints_3d']
        with tarfile.open(root/'data/hot3d'/f'{directory.name}.tar') as tar:
            def read(name): return json.load(tar.extractfile(name))
            beta=torch.tensor(read('__hand_shapes.json__')['mano'],dtype=torch.float32)[None]
            keys=sorted(n[:-10] for n in tar.getnames() if n.endswith('.info.json'))[:len(joints)]
            for index,key in enumerate(keys):
                camera=from_json(read(key+'.cameras.json')['1201-1'])
                T=np.linalg.inv(camera.T_world_from_eye); R=C@T[:3,:3]; t=C@T[:3,3]
                labels=read(key+'.hands.json'); targets=[]; predictions=[]
                for slot,(side,flag) in enumerate([('left',False),('right',True)]):
                    if np.isfinite(joints[index,slot]).all():
                        predictions.append(Hand(side,joints[index,slot],verts[index,slot],pred['cam_trans'][index,slot],pred['go'][index,slot].reshape(3,3),pred['joints_2d'][index,slot]*size,float(pred['exists_2d'][index,slot])))
                    d=(labels.get(side) or {}).get('mano_pose')
                    if not d: continue
                    model=gtmodels[flag]; theta=torch.tensor(d['thetas'],dtype=torch.float32)[None]
                    hp=theta  # Official HOT3D model expands the 15 PCA coefficients.
                    xf=np.asarray(d['wrist_xform'])
                    with torch.no_grad():
                        o=model(betas=beta,hand_pose=hp,global_orient=torch.tensor(xf[:3],dtype=torch.float32)[None],transl=torch.tensor(xf[3:],dtype=torch.float32)[None])
                        neutral=model(betas=beta,hand_pose=torch.zeros_like(theta),global_orient=torch.zeros(1,3),transl=torch.zeros(1,3))
                    j=torch.cat([o.joints[:,:16],o.vertices[:,[744,320,443,554,671]]],1)[0,order].numpy()
                    v=o.vertices[0].numpy(); j0=neutral.joints[0,0].numpy()
                    # MANO global orientation rotates about its shaped wrist, not zero.
                    tau=R@xf[3:]+t+R@j0-j0
                    camera_rotation=R@Rotation.from_rotvec(xf[:3]).as_matrix()
                    with torch.no_grad():
                        check=model(betas=beta,hand_pose=hp,
                            global_orient=torch.tensor(Rotation.from_matrix(camera_rotation).as_rotvec(),dtype=torch.float32)[None],
                            transl=torch.tensor(tau,dtype=torch.float32)[None])
                    reexpressed=check.vertices[0].numpy()
                    error=float(np.max(np.linalg.norm(reexpressed-(v@R.T+t),axis=1)))
                    if error>2e-5: raise RuntimeError(f'MANO camera-frame translation check failed: {error}')
                    if side=='right':
                        reference_error=float(np.max(np.linalg.norm(j@R.T+t-gt_reference[index]@C.T,axis=1)))
                        if reference_error>2e-5: raise RuntimeError(f'Prior GT convention mismatch: {reference_error}')
                        coordinate_checks.append({'sequence':directory.name,'frame':key,'gt_match_max_m':reference_error,'mano_transform_max_m':error})
                    targets.append(Hand(side,j@R.T+t,v@R.T+t,tau,camera_rotation,np.zeros((21,2))))
                frames.append((predictions,targets,K,size))
        per_sequence[directory.name]=score(frames,canonical)
        allframes.extend(frames)
    result={'protocol':'ACE appendix A.1 reimplementation; author parity unverified',
            'sequences':per_sequence,'pooled':score(allframes,canonical),
            'sensitivity_bbox_1_2':score(allframes,canonical,bbox_scale=1.2),
            'sensitivity_same_side_first':score(allframes,canonical,same_side_first=True)}
    result['canonical_convention']='zero MANO pose parameters with ACE flat_hand_mean=False'
    result['sensitivity_flat_hand_canonical']=score(allframes,canonical_flat)
    result['coordinate_checks']={'ground_truth_decoder':'official hand_tracking_toolkit MANOHandModel, including left shapedirs correction',
        'prediction_decoder':'official ACE MANO conventions unchanged', 'right_hand_frames':len(coordinate_checks),
        'gt_match_max_m':max(c['gt_match_max_m'] for c in coordinate_checks),
        'mano_transform_max_m':max(c['mano_transform_max_m'] for c in coordinate_checks)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
