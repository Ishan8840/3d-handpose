"""Read verified official ACE inference exports (pickle from trusted local runs)."""
import pickle
import numpy as np
from .base import Observation

class ACEExport:
    def __init__(self,path,image_size,upright=False):
        self.upright=upright
        with open(path,'rb') as f:self.data=pickle.load(f)
        self.width,self.height=image_size
        if self.data['joints_cam_direct'].shape[2:]!=(21,3):raise ValueError('Unexpected ACE joint shape')
    def observations(self,index):
        if not self.data['claim'][index,1]:return []
        # Official projector produces sigmoid-normalized [0,1] x/y coordinates.
        points=self.data['joints_2d'][index,1]*[self.width,self.height]
        if self.upright: points=np.c_[points[:,1],self.width-1-points[:,0]]
        return [Observation(points,np.full(21,self.data['exists_2d'][index,1]))]
    def direct(self,index):
        """Auxiliary direct-3D diagnostic, NOT the official final MANO output.

        For final monocular output use the official MANO decoder as demonstrated
        in scripts/audit_ace_decode.py. This method remains unchanged to preserve
        historical experiments and their provenance.
        """
        x=self.data['joints_cam_direct'][index,1].copy()
        if self.upright: x=x@np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        return x
