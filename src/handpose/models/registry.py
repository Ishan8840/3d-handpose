"""Only executable adapters belong to the runtime registry."""
from .mediapipe_adapter import MediaPipeAdapter, RotatedMediaPipeAdapter
from .rtmpose_adapter import RTMPoseAdapter, EnsembleAdapter, RTMPoseMPCropAdapter
REGISTRY = {'mediapipe': MediaPipeAdapter, 'mediapipe_rotated': RotatedMediaPipeAdapter, 'rtmpose': RTMPoseAdapter, 'ensemble': EnsembleAdapter, 'rtmpose_mp_crop': RTMPoseMPCropAdapter}

def create(name):
    if name not in REGISTRY:
        raise ValueError(f'{name} is not integrated. Available: {list(REGISTRY)}')
    return REGISTRY[name]()
