"""Two-view UmeTrack adaptation with predicted stereo crops and generic shape.

No GT skeleton, crops or annotations are passed to this adapter. Original model
and skinning are used. Native wrist and palm semantics are not 21-joint anatomy.
"""
import json
import sys
from pathlib import Path
import cv2
import numpy as np
from .base import Prediction
from handpose.data.hot3d import TO_21

class UmeTrackAdapter:
    def __init__(self,root='third_party/umetrack'):
        import torch
        self.torch=torch; self.root=Path(root)
        sys.path[:0]=[str(self.root.resolve()),str(Path('third_party/pytorch3d').resolve())]
        from lib.models.model_loader import load_pretrained_model
        from lib.tracker.tracker import HandTracker,HandTrackerOpts
        from lib.common.hand import HandModel
        self.tracker=HandTracker(load_pretrained_model(str(self.root/'pretrained_models/pretrained_weights.torch')),HandTrackerOpts())
        d=json.loads((self.root/'dataset/generic_hand_model.json').read_text())
        self.generic=HandModel(**{k:torch.tensor(v) if isinstance(v,list) else v for k,v in d.items()})

    def predict_stereo(self,left_image,right_image,left_camera,right_camera,initial):
        from lib.common.camera import PinholePlaneCameraModel
        from lib.common.crop import gen_crop_parameters_from_points
        from lib.common.hand import scaled_hand_model
        from lib.tracker.tracker import InputFrame,ViewData
        from lib.tracker.perspective_crop import landmarks_from_hand_pose
        from handpose.inference.pipeline import reconstruct
        points=initial.joints_3d_camera[initial.joint_validity]*1000
        if len(points)<6:
            self.tracker.reset_history()
            return reconstruct(None,None,[],[],initial.timestamp_ns)
        center=np.median(points,axis=0); points=points[np.linalg.norm(points-center,axis=1)<180]
        if len(points)<6:return reconstruct(None,None,[],[],initial.timestamp_ns)
        views=[];crops={}
        for i,(image,camera) in enumerate(zip((left_image,right_image),(left_camera,right_camera))):
            T=np.linalg.inv(camera.T_camera_from_left);T[:3,3]*=1000
            cam=PinholePlaneCameraModel(image.shape[1],image.shape[0],(camera.K[0,0],camera.K[1,1]),(camera.K[0,2],camera.K[1,2]),[],T)
            views.append(ViewData(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),cam,0))
            crops[i]=gen_crop_parameters_from_points(cam,points,(96,96),mirror_img_x=True,focal_multiplier=.8)
        # No memory across changing left-camera reference frames in this first spatial ablation.
        self.tracker.reset_history()
        tracked=self.tracker.track_frame_and_calibrate_scale(InputFrame(views),{1:crops})
        pose=tracked.hand_poses[1]; scale=float(tracked.predicted_scales[1])
        landmarks=landmarks_from_hand_pose(scaled_hand_model(self.generic,scale),pose,1)*.001
        xyz=np.full((21,3),np.nan)
        for dst,src in TO_21.items():xyz[dst]=landmarks[src]
        xyz[0]=np.nan  # Native UmeTrack wrist is not anatomical MANO wrist.
        validity=np.isfinite(xyz).all(1)
        return Prediction(initial.timestamp_ns,xyz,initial.joints_2d_left,initial.joints_2d_right,validity.astype(float),validity,np.full((3,3),np.nan)).validate()
