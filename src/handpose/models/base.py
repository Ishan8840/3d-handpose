from dataclasses import dataclass
import numpy as np

JOINT_NAMES = ['wrist'] + [f'{finger}_{joint}' for finger in ('thumb','index','middle','ring','pinky') for joint in ('mcp','pip','dip','tip')]
JOINT_NAMES[1:5] = ['thumb_cmc', 'thumb_mcp', 'thumb_ip', 'thumb_tip']
FINGERTIPS = [4, 8, 12, 16, 20]
EDGES = [(0, start) for start in (1,5,9,13,17)] + [(i,i+1) for start in (1,5,9,13,17) for i in range(start,start+3)]

@dataclass
class Observation:
    pixels: np.ndarray
    confidence: np.ndarray
    hand_side: str = 'right'

@dataclass
class Prediction:
    timestamp_ns: int
    joints_3d_camera: np.ndarray
    joints_2d_left: np.ndarray
    joints_2d_right: np.ndarray
    joint_confidence: np.ndarray
    joint_validity: np.ndarray
    wrist_rotation: np.ndarray
    hand_side: str = 'right'
    coordinate_frame: str = 'left_camera_optical'

    def validate(self):
        for name, shape in [('joints_3d_camera',(21,3)), ('joints_2d_left',(21,2)), ('joints_2d_right',(21,2)), ('joint_confidence',(21,)), ('joint_validity',(21,)), ('wrist_rotation',(3,3))]:
            if np.shape(getattr(self,name)) != shape:
                raise ValueError(f'{name} must have shape {shape}')
        if not np.isfinite(self.joints_3d_camera[self.joint_validity]).all():
            raise ValueError('Valid joints must be finite')
        if np.isfinite(self.joints_3d_camera[~self.joint_validity]).any():
            raise ValueError('Invalid 3D joints must be NaN')
        return self


def reorder(joints, source_names, target_names=JOINT_NAMES):
    if len(set(source_names)) != len(source_names):
        raise ValueError('Duplicate source joint names')
    return np.asarray(joints)[..., [source_names.index(n) for n in target_names], :]
