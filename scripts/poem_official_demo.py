"""Headless official three-view demo diagnostic; not a fair stereo benchmark."""
import json
import pickle
import runpy
import time
from pathlib import Path

import cv2
import numpy as np
import torch

base = Path.cwd()
data = base / 'data/poem-official/example_data'
out = base / 'outputs/poem-official-demo'
out.mkdir(parents=True, exist_ok=True)
state = runpy.run_path(str(base / 'scripts/poem_probe.py'))
from tool.infer_hand import format_batch, extract_pred

cv2.imshow = lambda *args: None
cv2.waitKey = lambda *args: -1
seq = 'pour__2025_0325_1117_56'
names = ['camera_1', 'camera_2', 'camera_3']
calib = data / 'calib/calib__2025_0319_1534_41'
def load_map(folder):
    result = {}
    for name in names:
        with (calib / folder / (name + '.pkl')).open('rb') as f:
            result[name] = np.array(pickle.load(f), dtype=np.float32)
    return result
intr, extr = load_map('cam_intr'), load_map('cam_extr')
side = json.loads((data / 'hand_labels.json').read_text())[seq]
captures = [cv2.VideoCapture(str(data / 'data_v2' / seq / (n + '.mkv'))) for n in names]
assert all(c.isOpened() for c in captures)
joints, meshes, frames, master_joints = [], [], [], []
start = time.perf_counter()
for i in range(300):
    images = []
    for capture in captures:
        ok, image = capture.read()
        assert ok
        images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    boxes = []
    for name in names:
        box = np.load(data / 'human_mask_hand' / seq / name / 'bbox' / f'{i:05d}.npy').astype(np.float32)
        if box.ndim > 1:
            box = box[0]
        boxes.append(box)
    if any(len(box) == 0 for box in boxes):
        continue
    batch = format_batch(images, boxes, side == 'left', names, intr, extr,
                         (1280, 720), state['cfg'].DATA_PRESET.IMAGE_SIZE, 'cuda')
    with torch.inference_mode():
        pred = state['model'](batch, 0, 'inference', epoch_idx=0)
    payload = extract_pred(pred, batch, side == 'left', extr)
    joints.append(payload['joints']); meshes.append(payload['verts']); frames.append(i)
    master_joints.append(pred['pred_joints_3d'][0].cpu().numpy())
    print('frame', i, flush=True)
    if len(frames) == 30:
        break
for capture in captures:
    capture.release()
assert len(frames) == 30
assert np.asarray(joints).shape[1:] == (21, 3)
assert np.asarray(meshes).shape[1:] == (778, 3)
assert np.isfinite(joints).all() and np.isfinite(meshes).all()
np.savez_compressed(out / 'predictions.npz', joints_world=joints, vertices_world=meshes,
                    joints_master=master_joints, frame_ids=frames)
metadata = dict(sequence=seq, frames=len(frames), views=3, provided_demo_crops=True,
                checkpoint='POEM-v2 large', hand_side=side, metric_accuracy_evaluated=False,
                upstream_functions=['format_batch', 'extract_pred'],
                io_changes='OpenCV sequential RGB decoding and disabled GUI display',
                seconds=time.perf_counter()-start,
                positive_master_depth_fraction=float((np.asarray(master_joints)[..., 2] > 0).mean()))
(out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
print(json.dumps(metadata), flush=True)
