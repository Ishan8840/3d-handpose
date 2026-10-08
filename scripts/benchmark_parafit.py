"""UA-Fit solver-core ablation on predicted observations, not full UA-Fit model."""
import argparse,inspect,json,sys,time,shutil,pickle
from pathlib import Path
import cv2
import numpy as np
import torch
from scipy.sparse import csr_matrix
from handpose.data.hot3d import iter_clip
from handpose.data.show3d import iter_scene
from handpose.models.base import Prediction
from handpose.inference.cli import save_predictions
from handpose.evaluation.metrics import evaluate

p=argparse.ArgumentParser();p.add_argument('--source',default='outputs/wilor-dev/stereo');p.add_argument('--output',default='outputs/parafit-dev');p.add_argument('--split',default='development');p.add_argument('--manifest',default='configs/datasets/hot3d.json');a=p.parse_args()
sys.path.insert(0,str(Path('third_party/manotorch').absolute()))
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
for k,v in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if k not in np.__dict__:setattr(np,k,v)
from parafit import LMSolver,PinholeCameras,ReprojectionEnergy
from parafit.energies.anchor3d import Anchor3DEnergy
from parafit.energies.pose_prior import PosePriorEnergy
from parafit.models.mano import ManoModel
assets=Path('checkpoints/parafit/models');assets.mkdir(parents=True,exist_ok=True)
for side in ('LEFT','RIGHT'):
    target=assets/f'MANO_{side}.pkl'
    if target.is_symlink():target.unlink()
    if not target.exists():
        with (Path('checkpoints/mano_converted')/target.name).open('rb') as f:asset=pickle.load(f)
        asset['J_regressor']=csr_matrix(asset['J_regressor'])
        with target.open('wb') as f:pickle.dump(asset,f)
tensor=lambda x:torch.tensor(x,dtype=torch.float32,device='cuda')
rotation=np.array([[0,1,0],[-1,0,0],[0,0,1]],float)
for e in json.load(open(a.manifest)):
    if e['split']!=a.split:continue
    rotation=np.eye(3) if 'annotations' in e else np.array([[0,1,0],[-1,0,0],[0,0,1]],float)
    source=Path(a.source)/Path(e['path']).stem;obs=np.load(source/'predictions.npz')
    params=json.loads((source/'mano_parameters.json').read_text())
    eligible=(obs['validity'].sum(-1)>=6)&np.array([bool(p) for p in params])
    ids=np.flatnonzero(eligible)
    samples=list(iter_scene(e['path'],e['annotations'],e['max_frames'])) if 'annotations' in e else list(iter_clip(e['path'],e['max_frames']))
    for solver_name in ('analytic_lm','adam'):
        result=np.full_like(obs['joints_3d'],np.nan);start=time.perf_counter();losses=[]
        if len(ids):
            beta=np.median([params[i][0]['betas'] for i in ids],axis=0)
            for chunk in [ids[k:k+16] for k in range(0,len(ids),16)]:
                model=ManoModel('checkpoints/parafit',betas=tensor(beta)[None],center_idx=None,flat_hand_mean=True,joints='kinematic')
                model.mano_layer=model.mano_layer.cuda();model.J_regressor=model.mano_layer.th_J_regressor
                model.tip_corrective=True
                init=np.zeros((len(chunk),51),np.float32)
                for j,i in enumerate(chunk):
                    rotations=np.concatenate([np.asarray(params[i][0]['global_orient']),np.asarray(params[i][0]['hand_pose'])])
                    rotations[0]=rotation@rotations[0]
                    init[j,:48]=np.concatenate([cv2.Rodrigues(r)[0].ravel() for r in rotations])
                theta=tensor(init)
                with torch.no_grad():template=model.forward(theta).landmarks.cpu().numpy()
                for j,i in enumerate(chunk):
                    good=obs['validity'][i]
                    init[j,48:]=np.mean(obs['joints_3d'][i,good]-template[j,good],axis=0)
                theta=tensor(init)
                cams=PinholeCameras(tensor([[c.K for c in samples[i]['cameras']] for i in chunk]),tensor([[c.T_camera_from_left for c in samples[i]['cameras']] for i in chunk]))
                uv=np.stack([obs['joints_2d_left'][chunk],obs['joints_2d_right'][chunk]],1)
                uv_mask=np.isfinite(uv).all(-1)&obs['validity'][chunk,None,:]
                target=tensor(np.nan_to_num(obs['joints_3d'][chunk]));valid=tensor(obs['validity'][chunk])
                reference=theta.clone();prior_precision=tensor([1.]*48+[0.]*3)
                energies=[ReprojectionEnergy(cams,tensor(np.nan_to_num(uv)),precision=tensor(uv_mask)*.25),
                          Anchor3DEnergy(target,precision=valid,weight=10000.),
                          PosePriorEnergy(reference,weight=.1,precision=prior_precision)]
                if solver_name=='analytic_lm':
                    solved=LMSolver(max_iters=30,damping_mode='diag').solve(model,theta,energies)
                    theta=solved.params.detach()
                    losses.append(solved.diagnostics['final_cost'].detach().cpu().tolist())
                else:
                    theta=theta.detach().requires_grad_(True);optim=torch.optim.Adam([theta],lr=.005)
                    for step in range(200):
                        optim.zero_grad();state=model.forward(theta);projected,_=cams.project_and_jac(state.landmarks)
                        loss=(((projected-tensor(np.nan_to_num(uv)))**2).sum(-1)*tensor(uv_mask)*.25).sum()
                        loss+=((state.landmarks-target).square().sum(-1)*valid).sum()*10000
                        loss+=((theta-reference).square()*prior_precision).sum()*.1
                        loss.backward();optim.step()
                    theta=theta.detach()
                with torch.no_grad():result[chunk]=model.forward(theta).landmarks.cpu().numpy()
        out=Path(a.output)/solver_name/source.name;out.mkdir(parents=True,exist_ok=True)
        predictions=[Prediction(int(t),x,l,r,c,np.isfinite(x).all(-1),np.full((3,3),np.nan)) for t,x,l,r,c in zip(obs['timestamps'],result,obs['joints_2d_left'],obs['joints_2d_right'],obs['confidence'])]
        save_predictions(predictions,out,{'model':'UA-Fit parafit solver core + WiLoR observations','solver':solver_name,'observation_code':5,'learned_uncertainty':False,'shape':'median predicted WiLoR beta','joint_convention':'kinematic+smplx tips','seconds':time.perf_counter()-start,'iterations':30 if solver_name=='analytic_lm' else 200,'final_cost':losses})
        saved=dict(np.load(out/'predictions.npz'));saved['original_observation_type']=obs['observation_type'];np.savez_compressed(out/'predictions.npz',**saved)
        shutil.copy2(source/'ground_truth.npz',out/'ground_truth.npz')
        gt=np.load(out/'ground_truth.npz');m=evaluate(result,gt['joints_3d'],timestamps_ns=obs['timestamps'])
        (out/'metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False));print(source.name,solver_name,m['absolute_mpjpe_mm'],m['joint_coverage'],flush=True)
