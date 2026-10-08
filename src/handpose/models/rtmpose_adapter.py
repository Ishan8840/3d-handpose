"""RTMDet + RTMPose hand5. MediaPipe supplies side identity only.

RTMPose hand5 has no handedness head. Associate its independently detected
hands to MediaPipe's right-hand wrist; unmatched identities remain missing.
This is explicitly a detector/identity-assisted RTMPose baseline.
"""
import numpy as np
from .base import Observation
from .mediapipe_adapter import MediaPipeAdapter

class RTMPoseAdapter:
    def __init__(self):
        from rtmlib import Hand
        self.model=Hand(backend='onnxruntime',device='cpu')
        self.identity=MediaPipeAdapter()

    def predict(self,image):
        identities=self.identity.predict(image)
        if not identities: return []
        boxes=self.model.det_model(image)
        if len(boxes)==0: return []
        keypoints,scores=self.model.pose_model(image,bboxes=boxes)
        if len(keypoints)==0: return []
        output=[]
        for identity in identities:
            distance=np.linalg.norm(keypoints[:,0]-identity.pixels[0],axis=-1)
            i=int(np.argmin(distance))
            hand_extent=np.linalg.norm(np.ptp(identity.pixels,axis=0))
            if distance[i] > .5*hand_extent: continue
            confidence=np.clip(scores[i],0,1)
            confidence[confidence<.3]=0
            output.append(Observation(keypoints[i],confidence))
        return output

    def close(self): self.identity.close()

class EnsembleAdapter:
    def __init__(self):
        self.rtm=RTMPoseAdapter()

    def predict(self,image):
        mp=self.rtm.identity.predict(image); rtm=self.rtm.predict(image)
        if not mp: return rtm
        if not rtm: return mp
        a,b=mp[0],rtm[0]
        # Heuristic evidence weights; cross-model scores are not calibrated.
        w=a.confidence; v=b.confidence
        agree=np.linalg.norm(a.pixels-b.pixels,axis=1)<20
        pixels=a.pixels.copy(); conf=w.copy()
        pixels[agree]=(a.pixels[agree]*w[agree,None]+b.pixels[agree]*v[agree,None])/(w[agree,None]+v[agree,None])
        conf[agree]=(w[agree]+v[agree])/2
        use=(~agree)&(v>w); pixels[use]=b.pixels[use]; conf[use]=v[use]
        return [Observation(pixels,conf)]

    def close(self): self.rtm.close()

class RTMPoseMPCropAdapter(RTMPoseAdapter):
    """Ablation: predicted MediaPipe bounding boxes, RTMPose joint regression."""
    def predict(self,image):
        observations=[]
        for identity in self.identity.predict(image):
            lo=identity.pixels.min(0); hi=identity.pixels.max(0)
            margin=.15*(hi-lo)
            boxes=np.r_[lo-margin,hi+margin][None].astype(np.float32)
            k,c=self.model.pose_model(image,bboxes=boxes)
            confidence=np.clip(c[0],0,1); confidence[confidence<.3]=0
            observations.append(Observation(k[0],confidence))
        return observations
