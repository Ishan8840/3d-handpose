"""Evaluation-only reference export for development clips lacking MANO21 archives."""
import json
import sys
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(4)
root=Path.cwd()
sys.path[:0]=[str(root/'src'),str(root/'third_party/hand_tracking_toolkit')]
from handpose.data.hot3d import mano_ground_truth

for entry in json.loads((root/'data/hot3d/manifest.json').read_text()):
    if entry['split']!='development': continue
    out=root/'outputs/ace-audit/dev-groundtruth'/Path(entry['path']).stem
    out.mkdir(parents=True,exist_ok=True)
    gt=mano_ground_truth(entry['path'],entry['max_frames'])
    np.savez_compressed(out/'ground_truth_mano21.npz',joints_3d=gt)
