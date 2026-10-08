import os,sys
from pathlib import Path
import numpy as np

class FoundationStereoAdapter:
    def __init__(self,root='third_party/foundationstereo',checkpoint='checkpoints/foundationstereo/23-51-11/model_best_bp2.pth'):
        import torch
        from omegaconf import OmegaConf
        os.environ['XFORMERS_DISABLED']='1'
        sys.path.insert(0,str(Path(root).resolve()))
        from core.foundation_stereo import FoundationStereo
        from core.utils.utils import InputPadder
        cfg=OmegaConf.load(str(Path(checkpoint).parent/'cfg.yaml'))
        cfg.vit_size=cfg.get('vit_size','vitl');cfg.valid_iters=32
        self.model=FoundationStereo(cfg)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model.load_state_dict(state['model'],strict=True);self.model.cuda().eval();self.padder=InputPadder
    def predict(self,left,right):
        import torch
        images=[torch.as_tensor(np.ascontiguousarray(x[:,:,::-1])).cuda().float().permute(2,0,1)[None] for x in (left,right)]
        padder=self.padder(images[0].shape,divis_by=32,force_square=False);images=padder.pad(*images)
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            disparity=self.model(*images,iters=32,test_mode=True)
        return padder.unpad(disparity.float()).cpu().numpy().squeeze()
