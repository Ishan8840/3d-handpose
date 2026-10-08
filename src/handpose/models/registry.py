"""Only executable adapters belong to the runtime registry."""
from .mediapipe_adapter import MediaPipeAdapter, RotatedMediaPipeAdapter, UprightMediaPipeAdapter
from .rtmpose_adapter import RTMPoseAdapter, EnsembleAdapter, RTMPoseMPCropAdapter
REGISTRY = {'mediapipe': MediaPipeAdapter, 'mediapipe_upright': UprightMediaPipeAdapter, 'mediapipe_rotated': RotatedMediaPipeAdapter, 'rtmpose': RTMPoseAdapter, 'ensemble': EnsembleAdapter, 'rtmpose_mp_crop': RTMPoseMPCropAdapter}

def create(name):
    if name not in REGISTRY:
        raise ValueError(f'{name} is not integrated. Available: {list(REGISTRY)}')
    return REGISTRY[name]()

# Offline and multiview runners use separate environments/interfaces but share
# this discoverable catalog. Registration does not imply successful execution.
MODEL_SPECS = {
    **{name: {'interface':'image_2d','environment':'baseline','executable':True} for name in REGISTRY},
    'ace': {'interface':'stereo_video','environment':'.venv-ace','runner':'handpose.inference.ace_cli','executable':True},
    'poem': {'interface':'predicted_crop_multiview','environment':'.venv-poem','runner':'scripts/benchmark_poem.py','executable':True},
    'umetrack': {'interface':'stereo_initialized','environment':'.venv','runner':'scripts/benchmark_hot3d.py --model umetrack','executable':True},
    'foundationstereo': {'interface':'rectified_stereo_depth','environment':'.venv','runner':'scripts/benchmark_dense.py','executable':True},
    'uafit': {'interface':'analytical_mano','executable':False,'reason':'Released uncertainty checkpoint not located in inspected official checkpoint instructions'},
}

MODEL_SPECS.update({name: {"interface":"predicted_crop_mesh_stereo", "environment":".venv-extra", "runner":"scripts/benchmark_mesh_regressor.py --model "+name, "executable":True} for name in ("wilor","hamer","egoforce","omni")})
MODEL_SPECS.update({"parafit":{"interface":"analytical_solver_component", "environment":".venv-extra", "runner":"scripts/benchmark_parafit.py", "executable":True}, "hmp":{"interface":"learned_temporal_prior_component", "environment":".venv-extra", "runner":"scripts/benchmark_hmp.py", "executable":True}, "stablehand":{"interface":"official_cached_feature_demo_GT_shape", "environment":".venv-extra", "runner":"scripts/stablehand_official_demo.py", "executable":True}})
