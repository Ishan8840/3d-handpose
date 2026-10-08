"""Official known-skeleton/GT-crop smoke; NOT a fair stereo benchmark."""
import json,sys
from pathlib import Path
import numpy as np
sys.path[:0]=[str(Path('third_party/umetrack').resolve()),str(Path('third_party/pytorch3d').resolve())]
from lib.models.model_loader import load_pretrained_model
from lib.tracker.tracker import HandTracker,HandTrackerOpts
from lib.tracker.video_pose_data import SyncedImagePoseStream
from lib.tracker.perspective_crop import landmarks_from_hand_pose
model=load_pretrained_model('third_party/umetrack/pretrained_models/pretrained_weights.torch');model.eval();tracker=HandTracker(model,HandTrackerOpts())
stream=SyncedImagePoseStream('data/umetrack-official/recording_00.mp4');errors=[];used=[];preds=[];gts=[]
for i,(frame,gt) in enumerate(stream):
    if i>=30:break
    skeleton=stream._hand_pose_labels.hand_model
    crops=tracker.gen_crop_cameras([v.camera for v in frame.views],stream._hand_pose_labels.camera_angles,skeleton,gt,min_num_crops=1)
    result=tracker.track_frame(frame,skeleton,crops)
    for side,pose in result.hand_poses.items():
        prediction=landmarks_from_hand_pose(skeleton,pose,side);truth=landmarks_from_hand_pose(skeleton,gt[side],side)
        errors.extend(np.linalg.norm(prediction-truth,axis=-1));preds.append(prediction);gts.append(truth);used.append(len(crops[side]))
out=Path('outputs/umetrack-official-smoke');out.mkdir(parents=True,exist_ok=True)
np.savez_compressed(out/'native_predictions.npz',prediction=preds,ground_truth=gts)
(out/'result.json').write_text(json.dumps({'frames':30,'available_cameras':len(stream._hand_pose_labels.cameras),'used_crop_counts':sorted(set(used)),'mean_error_native_units':float(np.mean(errors)),'ground_truth_crops':True,'ground_truth_subject_skeleton':True,'eligibility':'reproduction diagnostic only; excluded from two-view leaderboard'},indent=2));print((out/'result.json').read_text())
