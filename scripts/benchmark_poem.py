"""POEM-v2 two-view inference with predicted crops, following upstream preprocessing."""
import argparse, json, os, inspect, sys, time
from pathlib import Path
import numpy as np
import cv2
import torch
from handpose.data.hot3d import iter_clip
from handpose.inference.cli import save_predictions
from handpose.models.base import Prediction
from handpose.evaluation.metrics import evaluate
from handpose.geometry.rectification import rotate_clockwise_camera

p=argparse.ArgumentParser();p.add_argument('--clip',required=True);p.add_argument('--observations',required=True);p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=30)
a=p.parse_args();base=Path.cwd();clip=Path(a.clip).resolve();obs=np.load(a.observations);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if name not in np.__dict__:setattr(np,name,value)
root=base/'third_party/poem';sys.path[:0]=[str(root),str(base/'third_party/manotorch')];os.chdir(root)
import lib.models
from lib.utils.config import get_config
from lib.utils.builder import build_model
from lib.utils.transform import _affine_transform, _affine_transform_post_rot
cfg=get_config('config/release/train_large.yaml');cfg.defrost();cfg.MODEL.BACKBONE.PRETRAINED='';cfg.MODEL.PRETRAINED=str(base/'checkpoints/poem/large.pth.tar');cfg.freeze()
model=build_model(cfg.MODEL,data_preset=cfg.DATA_PRESET,train=cfg.TRAIN).cuda().eval()
# Upstream lazily loads assets/bps.npy during forward; retain its repository cwd.
np.random.seed(42);torch.manual_seed(42)
preds=[];truths=[];meshes=[];latencies=[]
for i,s in enumerate(iter_clip(str(clip),a.frames)):
    uv=[obs[k][i] for k in ('joints_2d_left','joints_2d_right')]
    pred=Prediction(s['timestamp_ns'],np.full((21,3),np.nan),*uv,np.zeros(21),np.zeros(21,bool),np.full((3,3),np.nan))
    images=[];intr=[];extr=[];vertices=np.full((778,3),np.nan);start=time.perf_counter()
    for image,camera,points in zip((s['left'],s['right']),s['cameras'],uv):
        good=np.isfinite(points).all(-1)
        if good.sum()<6:break
        h,w=image.shape[:2];cam,R=rotate_clockwise_camera(camera,w,h);image=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
        pts=np.stack([h-1-points[good,1],points[good,0]],-1);low=pts.min(0);high=pts.max(0);center=(low+high)/2;scale=max(200,float((high-low).max())*2)
        A=_affine_transform(center=center,scale=scale,out_res=(256,256),rot=0)
        crop=cv2.warpAffine(image,A[:2],(256,256));images.append(torch.from_numpy(crop[:,:,::-1].copy()).permute(2,0,1).float()/255-.5)
        post=_affine_transform_post_rot(center=center,scale=scale,optical_center=cam.K[:2,2],out_res=(256,256),rot=0)
        intr.append(post@cam.K);extr.append(np.linalg.inv(cam.T_camera_from_left))
    if len(images)==2:
        batch={'image':torch.stack(images).cuda(),'cam_view_num':np.array([2]),'target_cam_intr':torch.tensor(np.array(intr),dtype=torch.float32,device='cuda')[None],'target_cam_extr':torch.tensor(np.array(extr),dtype=torch.float32,device='cuda')[None],'master_id':torch.tensor([0],device='cuda'),'master_serial':['left'],'cam_serial':[['left','right']]}
        with torch.inference_mode():result=model(batch,0,'inference',epoch_idx=0)
        xyz=result['pred_joints_3d'][0].cpu().numpy()@R
        vertices=result['pred_verts_3d'][0].cpu().numpy()@R
        valid=np.isfinite(xyz).all(-1)&(xyz[:,2]>0);xyz[~valid]=np.nan
        pred.joints_3d_camera=xyz;pred.joint_validity=valid;pred.joint_confidence=valid.astype(float)
    preds.append(pred);truths.append(s['ground_truth']);meshes.append(vertices);latencies.append(time.perf_counter()-start)
    if i%10==0:print('frame',i,'valid',int(pred.joint_validity.sum()),flush=True)
save_predictions(preds,out,{'model':'POEM-v2-large','crop_source':str(a.observations),'predicted_crops':True,'two_views':True,'seconds_per_frame':float(np.mean(latencies))})
np.save(out/'mesh_vertices.npy',meshes)
gt=np.array(truths);ts=np.array([x.timestamp_ns for x in preds]);np.savez_compressed(out/'ground_truth.npz',joints_3d=gt,validity=np.isfinite(gt).all(-1),timestamps=ts,frame_ids=np.arange(len(gt)))
metrics=evaluate(np.array([x.joints_3d_camera for x in preds]),gt,np.array([x.joint_validity for x in preds]),timestamps_ns=ts);metrics['seconds_per_frame']=float(np.mean(latencies));(out/'metrics.json').write_text(json.dumps(metrics,indent=2,allow_nan=False));print(json.dumps(metrics))
