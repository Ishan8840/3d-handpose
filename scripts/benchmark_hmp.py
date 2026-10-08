"""Dyn-HaMR released HMP latent-prior ablation; not a full Dyn-HaMR reproduction."""
import argparse,inspect,json,os,sys,shutil,time
from pathlib import Path
import numpy as np
import torch
import smplx
from handpose.geometry.camera import transform
from handpose.models.base import Prediction
from handpose.inference.cli import save_predictions
from handpose.evaluation.metrics import evaluate

parser=argparse.ArgumentParser();parser.add_argument('--source',default='outputs/wilor-dev/stereo');parser.add_argument('--output',default='outputs/hmp-dev');parser.add_argument('--split',default='development');cli=parser.parse_args()
base=Path.cwd();source=base/'third_party/dynhamr/dyn-hamr/HMP'
sys.path.insert(0,str(source));os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
for k,v in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if k not in np.__dict__:setattr(np,k,v)
from nemf.generative import Architecture,Arguments
from rotations import matrix_to_rotation_6d
from utils import estimate_angular_velocity,estimate_linear_velocity
args=Arguments(str(source),'hmp_config.yaml');args.smpl.smpl_body_model=str(base/'checkpoints/mano_converted')
args.dataset_dir=args.save_dir=str(base/'checkpoints/dynhamr/hmp_model')
prior=Architecture(args,1);prior.load(optimal=True);prior.eval()
for module in prior.models:
    for param in module.parameters():param.requires_grad_(False)
tensor=lambda a:torch.tensor(np.asarray(a),dtype=torch.float32,device='cuda')
mano=smplx.MANOLayer(str(base/'checkpoints/mano_converted'),is_rhand=True,use_pca=False).cuda()
order=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
roll=np.array([[0,1,0],[-1,0,0],[0,0,1]])
for e in json.load(open('configs/datasets/hot3d.json')):
    if e['split']!=cli.split:continue
    folder=Path(cli.source)/Path(e['path']).stem
    obs=np.load(folder/'predictions.npz');params=json.loads((folder/'mano_parameters.json').read_text())
    poses=np.full_like(obs['joints_3d'],np.nan);tracking=np.load(folder/'ground_truth.npz')['T_world_from_left']
    good=np.flatnonzero([bool(p) for p in params]);start=time.perf_counter();loss_history=[]
    if len(good)>=20:
        closest=np.array([good[np.argmin(abs(good-i))] for i in range(len(poses))])
        beta=np.median([params[i][0]['betas'] for i in good],axis=0)
        local=np.array([params[i][0]['hand_pose'] for i in closest])
        root=np.array([tracking[i,:3,:3]@roll@np.array(params[j][0]['global_orient'])[0] for i,j in enumerate(closest)])
        targets=np.array([transform(x,T) for x,T in zip(obs['joints_3d'],tracking)])
        for begin in range(0,len(poses),128):
            n=min(128,len(poses)-begin);ids=np.minimum(np.arange(begin,begin+128),len(poses)-1)
            mask=obs['validity'][ids].copy();mask[n:]=False
            if mask.sum()<21:continue
            rot=torch.cat([torch.eye(3,device='cuda').expand(128,1,3,3),tensor(local[ids])],1)
            with torch.no_grad():
                pos,xf=prior.fk(rot)
                prior.set_input({'pos':pos[None], 'velocity':estimate_linear_velocity(pos[None],dt=1/30),
                                 'global_xform':matrix_to_rotation_6d(xf[None,:,:,:3,:3]),
                                 'angular':estimate_angular_velocity(rot[None],dt=1/30),
                                 'root_orient':matrix_to_rotation_6d(tensor(root[ids]))[None]})
                _,mean_l,_=prior.encode_local();_,mean_g,_=prior.encode_global()
            z=mean_l.detach().clone().requires_grad_();g=mean_g.detach()
            reference=z.detach().clone();target=tensor(np.nan_to_num(targets[ids]));weight=tensor(mask)
            translations=[]
            with torch.no_grad():
                initial=mano(global_orient=tensor(root[ids])[:,None],hand_pose=tensor(local[ids]),betas=tensor(beta)[None].expand(128,-1))
                j=torch.cat([initial.joints[:,:16],initial.vertices[:,[744,320,443,554,671]]],1)[:,order]
                initial_j=j.cpu().numpy()
            for k in range(128):
                observed=mask[k]
                if observed.any():translations.append(np.mean(targets[ids[k],observed]-initial_j[k,observed],0))
                else:translations.append(np.full(3,np.nan))
            translations=np.asarray(translations);valid_t=np.isfinite(translations).all(-1)
            for axis in range(3):translations[:,axis]=np.interp(np.arange(128),np.flatnonzero(valid_t),translations[valid_t,axis])
            trans=tensor(translations).requires_grad_();opt=torch.optim.Adam([z,trans],lr=.01)
            for step in range(100):
                opt.zero_grad();decoded=prior.decode(z,g,128)
                rotations=prior.fk.global_to_local(decoded['rotmat'].reshape(-1,16,3,3))
                out=mano(global_orient=tensor(root[ids])[:,None],hand_pose=rotations[:,1:],betas=tensor(beta)[None].expand(128,-1),transl=trans)
                joints=torch.cat([out.joints[:,:16],out.vertices[:,[744,320,443,554,671]]],1)[:,order]
                loss=((joints-target).square().sum(-1)*weight).sum()/weight.sum().clamp_min(1)
                loss+=1e-5*(z-reference).square().mean()+.01*(trans[1:]-trans[:-1]).square().mean()
                loss.backward();opt.step();loss_history.append(float(loss.detach()))
            with torch.no_grad():
                decoded=prior.decode(z,g,128)
                rotations=prior.fk.global_to_local(decoded['rotmat'].reshape(-1,16,3,3))
                out=mano(global_orient=tensor(root[ids])[:,None],hand_pose=rotations[:,1:],betas=tensor(beta)[None].expand(128,-1),transl=trans)
                result=torch.cat([out.joints[:,:16],out.vertices[:,[744,320,443,554,671]]],1)[:,order].cpu().numpy()
            for k in range(n):poses[begin+k]=transform(result[k],np.linalg.inv(tracking[begin+k]))
    out=Path(cli.output)/folder.name;out.mkdir(parents=True,exist_ok=True)
    preds=[Prediction(int(t),x,l,r,np.where(np.isfinite(x).all(-1),.5,0),np.isfinite(x).all(-1),np.full((3,3),np.nan)) for t,x,l,r in zip(obs['timestamps'],poses,obs['joints_2d_left'],obs['joints_2d_right'])]
    save_predictions(preds,out,{'model':'Dyn-HaMR HMP latent fitting ablation','observation_code':3,'full_dynhamr_reproduction':False,'minimum_detected_frames':20,'fit_steps':100,'seconds':time.perf_counter()-start,'loss':loss_history})
    saved=dict(np.load(out/'predictions.npz'));saved['original_observation_type']=obs['observation_type'];np.savez_compressed(out/'predictions.npz',**saved)
    shutil.copy2(folder/'ground_truth.npz',out/'ground_truth.npz')
    gt=np.load(out/'ground_truth.npz')['joints_3d'];m=evaluate(poses,gt,timestamps_ns=obs['timestamps'])
    (out/'metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False));print(folder.name,m['absolute_mpjpe_mm'],m['joint_coverage'],flush=True)
