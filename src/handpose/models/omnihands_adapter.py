"""Official OmniHands multiview network with predicted two-view hand crops."""
import os,sys,inspect,types
from pathlib import Path
import numpy as np
from .base import Observation


class OmniHands:
    def __init__(self,root='.'):
        import torch,smplx
        self.torch=torch;base=Path(root).absolute();source=base/'third_party/omnihands'
        sys.path.insert(0,str(source));os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
        for k,v in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
            if k not in np.__dict__:setattr(np,k,v)
        # Skip unrelated training-dataset imports, retaining the actual inference modules.
        for name in ('hands_multiview.datasets','hands_4d.datasets'):
            pkg=types.ModuleType(name);pkg.__path__=[str(source/Path(*name.split('.')))]
            sys.modules[name]=pkg
        original_create=smplx.create
        def create(path,*args,**kwargs):
            if str(path)=='/workspace/hamer_twohand/_DATA/data/':
                side='RIGHT' if kwargs.get('is_rhand',True) else 'LEFT'
                path=str(base/'checkpoints/mano_converted'/f'MANO_{side}.pkl')
            return original_create(path,*args,**kwargs)
        smplx.create=create
        from hands_multiview.models import Hands_Multi
        from hands_multiview.configs import get_config
        from hands_4d.datasets.vitdet_dataset import ViTDetInterDataset_Batch
        cfg=get_config(str(source/'checkpoints/config_multi.yaml'),update_cachedir=True)
        cfg.defrost();cfg.MANO.MODEL_PATH=str(base/'checkpoints/mano_converted')
        cfg.MANO.MEAN_PARAMS=str(source/'mano_files/mano_mean_params.npz')
        cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS',None);cfg.freeze()
        previous=Path.cwd();os.chdir(source)
        try:
            self.model=Hands_Multi(cfg,init_renderer=False)
        finally:
            os.chdir(previous);smplx.create=original_create
        self.checkpoint=str(source/'checkpoints/Demo_Multiview.pth')
        state=torch.load(self.checkpoint,map_location='cpu',weights_only=False)
        status=self.model.load_state_dict(state,strict=False)
        self.load_status={'missing_keys':status.missing_keys,'unexpected_keys':status.unexpected_keys}
        if status.missing_keys:raise RuntimeError(self.load_status)
        self.model=self.model.cuda().eval();self.cfg=cfg;self.dataset=ViTDetInterDataset_Batch
        from ultralytics import YOLO
        self.detector=YOLO(str(base/'third_party/wilor/pretrained_models/detector.pt')).to('cuda')

    def predict_views(self,images):
        torch=self.torch;boxes={'right':{},'left':{}};scores=[]
        for i,image in enumerate(images):
            det=self.detector(image,conf=.3,verbose=False)[0].boxes.data.cpu().numpy()
            right_score=0
            for side,cls in [('right',1),('left',0)]:
                selected=det[det[:,5]==cls]
                if len(selected):
                    row=selected[np.argmax(selected[:,4])];box=row[:4]
                    if side=='right':right_score=float(row[4])
                else:
                    box=np.array([0,0,image.shape[1],image.shape[0]],float)
                boxes[side][str(i)]=box
            scores.append(right_score)
        if min(scores)==0:return [[],[]]
        data=self.dataset(self.cfg,images,boxes,rescale_factor=2.,stack_all=True)[0]
        batch={k:v.cuda()[None] for k,v in data.items() if torch.is_tensor(v)}
        self.model.seq_len=2;self.model.aggregate_head.seq_len=2
        with torch.inference_mode():out=self.model(batch)
        uv=out['joints2d_world_right'].cpu().numpy()
        xyz=out['joints3d_world_right'].cpu().numpy()
        return [[{'observation':Observation(uv[i],np.full(21,scores[i])),
                  'joints_relative':xyz[i],'mano':{}}] for i in range(2)]
