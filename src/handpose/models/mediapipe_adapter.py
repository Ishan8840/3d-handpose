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
