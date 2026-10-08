"""Post-freeze development diagnostic; no changes to held-out selected settings."""
import json
from pathlib import Path
import numpy as np
from handpose.data.hot3d import iter_clip
from handpose.fitting.mano import fit_sequence
from handpose.evaluation.metrics import evaluate
source=np.load('outputs/b0-dev/clip-000175/predictions.npz');dense=np.load('outputs/dense-dev/predictions.npz')
np.testing.assert_allclose(source['joints_3d'],dense['stereo'],equal_nan=True,atol=1e-7)
samples=list(iter_clip('data/hot3d/clip-000175.tar',len(source['timestamps'])))
Ks=[np.stack([s['cameras'][v].K for s in samples]) for v in (0,1)];T=np.stack([s['cameras'][1].T_camera_from_left for s in samples]);points=dense['dense'].copy()
reliable=source['validity']&np.isfinite(points).all(-1)&(np.linalg.norm(points-source['joints_3d'],axis=-1)<.03);points[~reliable]=np.nan
out=Path('outputs/dense-mano-dev');out.mkdir(parents=True,exist_ok=True);results={}
for label,depth in [('stereo_mano',None),('stereo_depth_mano',points)]:
    xyz,params=fit_sequence(source['joints_3d'],source['validity'],source['joints_2d_left'],source['joints_2d_right'],*Ks,T,'checkpoints/mano_converted',dense_points=depth,depth_weight=.1)
    # GT is excluded from fitting and read only to score the completed output.
    gt=np.load('outputs/b0-dev/clip-000175/ground_truth.npz')['joints_3d'];results[label]=evaluate(xyz,gt,timestamps_ns=source['timestamps'])
    np.savez_compressed(out/f'{label}.npz',joints_3d=xyz,validity=np.isfinite(xyz).all(-1),timestamps=source['timestamps'],frame_ids=source['frame_ids'],observation_type=np.where(np.isfinite(xyz).all(-1),5,0))
    (out/f'{label}_parameters.json').write_text(json.dumps(params));print(label,results[label]['absolute_mpjpe_mm'],results[label]['joint_coverage'],flush=True)
results['protocol']={'split':'development','sequence':'P0003_01a10066','post_freeze_diagnostic':True,'depth_weight':.1,'surface_points':int(reliable.sum()),'depth_mask':'within3cm of independently triangulated landmark; not semantic visibility ground truth'}
(out/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
