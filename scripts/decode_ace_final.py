"""Decode final right-hand MANO joints from trusted ACE pickle; no GT required.

Run in the ACE Python environment. Output uses the export's processed camera
frame, with no rotation, scale alignment or root-relative conversion.
"""
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
import smplx


def main():
    p=argparse.ArgumentParser()
    for name in ['prediction','ace-root','mano-dir','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args(); sys.path.insert(0,str(Path(a.ace_root).resolve()))
    from ace_ego_hand.mano_utils import mano_forward_batch_full
    torch.set_num_threads(4)
    with open(a.prediction,'rb') as f:data=pickle.load(f)
    model=smplx.MANO(a.mano_dir,is_rhand=True,use_pca=False,flat_hand_mean=False).eval()
    arrays=[torch.from_numpy(np.nan_to_num(data[key][:,1]).astype(np.float32)) for key in ['go','hp','betas']]
    with torch.no_grad():joints,_=mano_forward_batch_full(*arrays,model)
    xyz=joints.numpy()+data['cam_trans'][:,1,None,:]
    xyz[~data['claim'][:,1]]=np.nan
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,joints_3d=xyz,validity=np.isfinite(xyz).all(-1))


if __name__=='__main__':main()
