"""Separate anatomical MANO-21 evaluation, never merged with UmeTrack-19 scores."""
import argparse,json
from pathlib import Path
import numpy as np
from handpose.data.hot3d import mano_ground_truth
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--roots',nargs='+',required=True);a=p.parse_args()
for e in json.load(open('data/hot3d/manifest.json')):
    if e['split']!=a.split:continue
    name=Path(e['path']).stem;gt=mano_ground_truth(e['path'],e['max_frames'])
    for root in a.roots:
        folder=Path(root)/name;path=folder/'predictions.npz'
        if not path.exists():continue
        pred=np.load(path)
        if len(pred['timestamps'])!=len(gt):raise ValueError('Frame mismatch')
        m=evaluate(pred['joints_3d'],gt,pred['validity'],timestamps_ns=pred['timestamps']);m['gt_convention']='MANO-21';m['sequence']=e['sequence']
        (folder/'mano21_metrics.json').write_text(json.dumps(m,indent=2,allow_nan=False))
        np.savez_compressed(folder/'ground_truth_mano21.npz',joints_3d=gt,validity=np.isfinite(gt).all(-1),timestamps=pred['timestamps'],frame_ids=pred['frame_ids'])
        print(root,name,m['absolute_mpjpe_mm'],m['wrist_mm'],m['joint_coverage'],flush=True)
