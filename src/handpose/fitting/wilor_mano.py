"""WiLoR-initialized MANO fitting; no annotations enter this function."""
from pathlib import Path
import inspect,sys,pickle,time
import numpy as np
import cv2
from scipy.sparse import csr_matrix

def fit_mano_sequence(obs,params,cameras,rotation=None,solver_name='adam'):
    import torch
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
    rotation=np.eye(3) if rotation is None else np.asarray(rotation)
    eligible=(obs['validity'].sum(-1)>=6)&np.array([bool(p) for p in params])
    ids=np.flatnonzero(eligible)
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
            cams=PinholeCameras(tensor([[c.K for c in cameras[i]] for i in chunk]),tensor([[c.T_camera_from_left for c in cameras[i]] for i in chunk]))
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
    return result, {'seconds':time.perf_counter()-start,'final_cost':losses}
