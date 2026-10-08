"""Official WiLoR/HaMeR reconstruction with WiLoR's released hand detector."""
import importlib
import inspect
import os
from pathlib import Path
import sys
import numpy as np
from .base import Observation


class MeshRegressor:
    def __init__(self, name='wilor', root=None, checkpoint=None):
        import torch
        self.torch=torch
        base=Path(root or '.').absolute()
        self.base=base; self.name=name
        # Compatibility for the trusted legacy public checkpoints and Chumpy.
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        if not hasattr(inspect,'getargspec'): inspect.getargspec=inspect.getfullargspec
        for key,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
            if key not in np.__dict__: setattr(np,key,value)
        source=base/'third_party'/name
        sys.path.insert(0,str(source))
        previous=Path.cwd();os.chdir(source)
        try:
            models=importlib.import_module(name+'.models')
            get_config=importlib.import_module(name+'.configs').get_config
            if name=='wilor':
                cfg=get_config(str(source/'pretrained_models/model_config.yaml'),update_cachedir=True)
                weights=source/'pretrained_models/wilor_final.ckpt'
                klass=models.WiLoR
            else:
                weights=base/'checkpoints/hamer/hamer_ckpts/checkpoints/hamer.ckpt'
                cfg=get_config(str(weights.parent.parent/'model_config.yaml'),update_cachedir=True)
                klass=models.HAMER
            if checkpoint is not None:
                weights=Path(checkpoint).absolute()
            self.checkpoint=str(weights)
            cfg.defrost();cfg.MODEL.BBOX_SHAPE=[192,256]
            cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None)
            cfg.MANO.MODEL_PATH=str(base/'checkpoints/mano_converted')
            cfg.MANO.DATA_DIR=str(base/'third_party/wilor/mano_data')
            cfg.MANO.MEAN_PARAMS=str(base/'third_party/wilor/mano_data/mano_mean_params.npz')
            cfg.freeze();self.cfg=cfg
            self.model=klass(cfg,init_renderer=False)
            checkpoint=torch.load(weights,map_location='cpu',weights_only=False)
            status=self.model.load_state_dict(checkpoint['state_dict'],strict=False)
            self.load_status={'missing_keys':status.missing_keys,'unexpected_keys':status.unexpected_keys}
            # Never silently accept missing learned layers.
            if status.missing_keys: raise RuntimeError(self.load_status)
            self.model=self.model.cuda().eval()
            self.dataset=importlib.import_module(name+'.datasets.vitdet_dataset').ViTDetDataset
        finally:
            os.chdir(previous)
        from ultralytics import YOLO
        self.detector=YOLO(str(base/'third_party/wilor/pretrained_models/detector.pt')).to('cuda')

    def predict(self,image):
        torch=self.torch
        detection=self.detector(image,conf=.3,verbose=False)[0]
        boxes=detection.boxes.data.detach().cpu().numpy()
        # Official WiLoR detector: class1 right, class0 left.
        boxes=boxes[boxes[:,5]==1]
        if not len(boxes):return []
        dataset=self.dataset(self.cfg,image,boxes[:,:4],np.ones(len(boxes)),rescale_factor=2.0)
        result=[]
        for i in range(len(dataset)):
            batch={k:torch.as_tensor(v,device='cuda')[None] for k,v in dataset[i].items()}
            with torch.inference_mode():out=self.model(batch)
            uv=out['pred_keypoints_2d'][0].cpu().numpy()*float(batch['box_size'][0])+batch['box_center'][0].cpu().numpy()
            xyz=out['pred_keypoints_3d'][0].cpu().numpy()
            result.append(dict(observation=Observation(uv,np.full(21,float(boxes[i,4]))),
                               joints_relative=xyz,
                               mano={k:v[0].cpu().numpy() for k,v in out['pred_mano_params'].items()}))
        return result
