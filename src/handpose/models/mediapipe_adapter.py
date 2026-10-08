import cv2
import numpy as np
from .base import Observation

class MediaPipeAdapter:
    """Full image detector. MediaPipe handedness assumes selfie/mirrored input.

    Feed horizontally flipped unmirrored camera images, then map pixels back.
    Confidence is HAND-level handedness score, not calibrated joint uncertainty.
    """
    def __init__(self):
        import mediapipe as mp
        self.model = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2, model_complexity=1, min_detection_confidence=.3)

    def predict(self, image):
        h, w = image.shape[:2]
        result = self.model.process(cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB))
        observations = []
        for points, side in zip(result.multi_hand_landmarks or [], result.multi_handedness or []):
            label = side.classification[0]
            if label.label == 'Right':
                uv = np.array([[(1-p.x)*w-1, p.y*h] for p in points.landmark])
                observations.append(Observation(uv, np.full(21, label.score)))
        return observations

    def close(self):
        self.model.close()

class RotatedMediaPipeAdapter(MediaPipeAdapter):
    """Four orientation hypotheses, inverse mapped before stereo matching."""
    def predict(self,image):
        h,w=image.shape[:2]; observations=[]
        for turns in range(4):
            rotated=np.ascontiguousarray(np.rot90(image,turns))
            for observation in super().predict(rotated):
                x,y=observation.pixels.T.copy()
                if turns==1: observation.pixels=np.c_[w-1-y,x]
                elif turns==2: observation.pixels=np.c_[w-1-x,h-1-y]
                elif turns==3: observation.pixels=np.c_[y,h-1-x]
                observations.append(observation)
        return observations

class UprightMediaPipeAdapter(MediaPipeAdapter):
    """HOT3D Quest native sensor orientation: clockwise to upright, then inverse map."""
    def predict(self,image):
        h,w=image.shape[:2]
        observations=super().predict(cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE))
        for obs in observations:
            x,y=obs.pixels.T.copy();obs.pixels=np.c_[y,h-1-x]
        return observations
