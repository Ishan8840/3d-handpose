"""Compatibility bootstrap for upstream Python 3.8-era dependencies."""
import inspect,sys,os
from pathlib import Path
import numpy as np
# Chumpy 0.70 API aliases removed in modern Python/NumPy, no numerical changes.
if not hasattr(inspect,'getargspec'): inspect.getargspec=inspect.getfullargspec
for name,value in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if name not in np.__dict__:setattr(np,name,value)
root=Path('third_party/poem').resolve();sys.path[:0]=[str(root),str(Path('third_party/manotorch').resolve())];os.chdir(root)
import torch
import lib.models
from lib.utils.config import get_config
from lib.utils.builder import build_model
cfg=get_config('config/release/train_large.yaml');cfg.defrost();cfg.MODEL.BACKBONE.PRETRAINED='';cfg.MODEL.PRETRAINED='../../checkpoints/poem/large.pth.tar';cfg.freeze()
model=build_model(cfg.MODEL,data_preset=cfg.DATA_PRESET,train=cfg.TRAIN).cuda().eval()
print('POEM model loaded successfully',flush=True)
