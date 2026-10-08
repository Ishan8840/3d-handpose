"""EgoForce hand-only input ablation using predicted WiLoR detector crops.

Runs official HALO, preprocessing, MANO/arm models and ray-space solver.
Forearm detection is absent and explicitly represented by the missing-arm input.
"""
import os,sys,inspect
from pathlib import Path
import numpy as np
from .base import Observation


class EgoForce:
    def __init__(self,root='.',forearm=False):
        import torch
        self.torch=torch;self.base=Path(root).absolute()
        source=self.base/'third_party/egoforce';sys.path.insert(0,str(source))
        # Reuse the already compiled PyTorch3D wheel, without replacing extra-env deps.
        sys.path.append(str(self.base/'.venv-poem/lib/python3.12/site-packages'))
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
        for k,v in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
            if k not in np.__dict__:setattr(np,k,v)
        from settings import config
        from models import HALO,LimbModel
        from core import get_limb,compute_camera_space_mesh
        from demo.demo_hand_arm_loader import DemoHandArmLoader
        from camera_models import PinholeCameraModel
        from ultralytics import YOLO
        config.MANO_PATH=str(self.base/'checkpoints/mano_converted');self.config=config
        self.model=HALO(config)
        self.model.load_state_dict(torch.load(config.POSE_3D.CHECKPOINT_PATH,map_location='cpu',weights_only=False),strict=True)
        self.model=self.model.cuda().eval();self.limb=LimbModel(config,device='cuda',use_pose_pca=False)
        self.detector=YOLO(str(self.base/'third_party/wilor/pretrained_models/detector.pt')).to('cuda')
        self.arm_detector=None
        if forearm:
            sys.path[:0]=[str(source/'thirdparty/mmdetection'),str(source/'thirdparty/datapipes')]
            from mmdet.apis import DetInferencer
            self.arm_detector=DetInferencer(str(source/'demo/rtmdet_tiny_8xb32-300e_combined_cutmix.py'),weights=config.DETECTION.HAND_ARM_PATH,device='cuda')
        self.loader_cls=DemoHandArmLoader;self.camera_cls=PinholeCameraModel
        self.get_limb=get_limb;self.solve=compute_camera_space_mesh
        self.checkpoint=config.POSE_3D.CHECKPOINT_PATH;self.load_status={'strict':True}

    def predict(self,image,camera):
        torch=self.torch
        boxes=self.detector(image,conf=.3,verbose=False)[0].boxes.data.cpu().numpy()
        boxes=boxes[boxes[:,5]==1];result=[]
        K=camera.K
        cam=self.camera_cls((K[0,0],K[1,1]),K[:2,2],image.shape[1],image.shape[0])
        loader=self.loader_cls(self.config,cam,undistort_inp=True,hand_type='right')
        arms=[]
        if self.arm_detector is not None and len(boxes):
            detections=self.arm_detector(image[:,:,::-1].copy(),return_vis=False)['predictions'][0]
            arms=[np.asarray(b) for b,label,score in zip(detections['bboxes'],detections['labels'],detections['scores']) if label==1 and score>=.3]
        for box in boxes:
            center=(box[:2]+box[2:4])/2;size=max(box[2]-box[0],box[3]-box[1])*1.5
            bbox=np.r_[center-size/2,center+size/2]
            bbox[[0,2]]=bbox[[0,2]].clip(0,image.shape[1]-1)
            bbox[[1,3]]=bbox[[1,3]].clip(0,image.shape[0]-1)
            if min(bbox[2:]-bbox[:2])<3:continue
            inputs={'hand':{'bbox':bbox,'keypoint':np.zeros((21,2))}}
            if arms:
                arm=min(arms,key=lambda b:np.linalg.norm((b[:2]+b[2:])/2-center))
                inputs['arm']={'bbox':arm,'keypoint':np.zeros((2,2))}
            data,meta=loader.transform(image[:,:,::-1].copy(),inputs)
            data={k:v.cuda()[None,None].contiguous() for k,v in data.items() if torch.is_tensor(v)}
            meta={k:v.cuda()[None] for k,v in meta.items()}
            with torch.inference_mode():
                pred=self.model(data['hand_crop'],data['hand_sparse_kpe'],data['arm_crop'],data['arm_sparse_kpe'])
                limb=self.get_limb(self.config,self.limb,pred['global_orient'],pred['betas'],pred['hand_pose'],torch.zeros((1,1,3),device='cuda'),torch.ones((1,1),device='cuda'),pred['arm_shape'],pred['arm_R'])
                limb.hand.crop_j2d=pred['hand_kpts_2d'].squeeze(1)
                limb.arm.crop_j2d=pred['arm_kpts_2d'].squeeze(1)
                limb.hand.confidence=pred['hand_kpt_w'].squeeze(1)
                limb.arm.confidence=pred['arm_kpt_w'].squeeze(1)
                solved=self.solve(self.config,meta,limb)
            uv=pred['hand_kpts_2d'][0,0].cpu().numpy()/np.asarray(self.config.POSE_3D.IMAGE_SIZE)*meta['hand_crop_size'][0].cpu().numpy()+meta['hand_bbox'][0,:2].cpu().numpy()
            conf=pred['hand_kpt_w'].squeeze().cpu().numpy()
            xyz=solved.hand.joints[0].cpu().numpy()
            result.append({'observation':Observation(uv,conf),'joints_camera':xyz,'mano':{}})
        return result
