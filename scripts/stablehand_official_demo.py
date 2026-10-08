"""Run official cached-data diagnostic; supplied subject shape is GT, not fair inference."""
import os,sys,runpy,inspect
from pathlib import Path
import numpy as np

base=Path.cwd();source=base/'third_party/stablehand'
if not hasattr(inspect,'getargspec'):inspect.getargspec=inspect.getfullargspec
for k,v in [('bool',bool),('int',int),('float',float),('complex',complex),('object',object),('unicode',str),('str',str)]:
    if k not in np.__dict__:setattr(np,k,v)
sys.path.insert(0,str(source));os.chdir(source)
sys.argv=['infer_clips','--ckpt','save/dit_hot3d/model.pt','--qn_ckpt','save/qn/model.pt',
          '--qn_sigma_override','26.18328668456495,3.9939303136949373',
          '--depth_signal_dir','data/hot3d/depth_signal','--clips','clip-002736',
          '--seed','42','--n_steps','20','--no_rrd','--mano_dir',str(base/'checkpoints/mano_converted'),
          '--out_dir',str(base/'outputs/stablehand-official-demo')]
runpy.run_module('sample.infer_clips',run_name='__main__')
