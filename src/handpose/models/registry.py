"""Only executable adapters belong to the runtime registry."""
from .mediapipe_adapter import MediaPipeAdapter
REGISTRY = {'mediapipe': MediaPipeAdapter}

def create(name):
    if name not in REGISTRY:
        raise ValueError(f'{name} is not integrated. Available: {list(REGISTRY)}')
    return REGISTRY[name]()
