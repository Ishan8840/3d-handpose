"""Rescore saved ACE outputs; no hand reference enters MANO decoding."""
import argparse
import importlib.util
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--predictions', default='outputs/ace-upright-held_out')
    p.add_argument('--report-name', default='saved_decode')
    p.add_argument('--frames', type=int)
    p.add_argument('--gt-root', default='outputs/adam-temporal-held')
    a = p.parse_args(); root = a.root.resolve()
    sys.path.insert(0, str(root/'src'))
    sys.path.insert(0, str(root/'third_party/ace'))
    os.environ['ACE_EGO_HAND_MANO_DIR'] = str(root/'checkpoints/mano_converted')
    import smplx
    from handpose.evaluation.metrics import evaluate
    spec = importlib.util.spec_from_file_location('ace_viz', root/'third_party/ace/scripts/viz_preds.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    models = {side:smplx.MANO(str(root/'checkpoints/mano_converted'),is_rhand=side,
                             use_pca=False,flat_hand_mean=False).eval() for side in (False,True)}
    result = {'protocol': 'Saved-export rescore, right hand, anatomical MANO21; not paper detection protocol', 'sequences': {}}
    pooled = {k: [] for k in ['final_mano', 'auxiliary_direct', 'gt']}
    out = root/'outputs/ace-audit'/a.report_name; out.mkdir(parents=True, exist_ok=True)
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    for directory in sorted((root/a.predictions).glob('clip-*')):
        with (directory/'left/left.pkl').open('rb') as f: pred = pickle.load(f)
        final, _ = module.mano_cam_joints(pred, models, 'cpu')
        final = final[:,1] @ rotation
        direct = pred['joints_cam_direct'][:,1] @ rotation
        if a.frames:
            final=final[:a.frames]; direct=direct[:a.frames]
        gtfile = root/a.gt_root/directory.name/'ground_truth_mano21.npz'
        gt = np.load(gtfile)['joints_3d'][:len(final)]
        result['sequences'][directory.name] = {k: evaluate(v, gt) for k,v in [('final_mano',final),('auxiliary_direct',direct)]}
        for k,v in [('final_mano',final),('auxiliary_direct',direct),('gt',gt)]: pooled[k].append(v)
        np.savez_compressed(out/(directory.name+'.npz'), final_mano=final, auxiliary_direct=direct, ground_truth=gt)
    gt = np.concatenate(pooled['gt'])
    result['pooled'] = {k:evaluate(np.concatenate(pooled[k]),gt) for k in ['final_mano','auxiliary_direct']}
    result['decomposition']={}
    for k in ['final_mano','auxiliary_direct']:
        xyz=np.concatenate(pooled[k]); valid=np.isfinite(xyz).all((1,2)) & np.isfinite(gt).all((1,2))
        d=(xyz[valid]-gt[valid])*1000; root_error=d[:,:1]; articulation=d-root_error
        result['decomposition'][k]={
            'observed_frames':int(valid.sum()),'total_frames':len(gt),
            'wrist_signed_depth_mm':float(root_error[:,:,2].mean()),
            'wrist_depth_mae_mm':float(np.abs(root_error[:,:,2]).mean()),
            'absolute_mse_mm2':float((d*d).sum(-1).mean()),
            'root_mse_mm2':float((root_error*root_error).sum(-1).mean()),
            'articulation_mse_mm2':float((articulation*articulation).sum(-1).mean()),
            'cross_term_mm2':float(2*(root_error*articulation).sum(-1).mean()),
            'identity':'absolute MSE = root MSE + articulation MSE + cross term; MPJPE is not additive'}
    report = root/'reports/ace_audit'; report.mkdir(parents=True,exist_ok=True)
    (report/(a.report_name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result['pooled'],indent=2))


if __name__ == '__main__': main()
