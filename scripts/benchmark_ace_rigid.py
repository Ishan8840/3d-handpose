"""Controlled saved-export ACE rigid stereo experiments; references score only."""
import json
import pickle
import time
from dataclasses import asdict
from pathlib import Path
import numpy as np
from handpose.geometry.camera import load_stereo
from handpose.fitting.stereo_rigid import fit_stereo_rigid, RigidConfig
from handpose.evaluation.metrics import evaluate, sequence_bootstrap

ROOT=Path('outputs/ace-rigid'); REPORT=Path('reports/ace_rigid')
C=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
configs={'default':RigidConfig(),'palm_only':RigidConfig(fit_palm_only=True),
         'loose_consensus':RigidConfig(anchor_outlier_m=.04)}


def experiment(split, config_name):
    config=configs[config_name]; decoded='development_decode' if split=='development' else 'saved_decode'
    archives=Path('outputs/ace-audit')/decoded
    scores={}; pool={name:[] for name in ['ace_final','anchored','se3','wilor']}; targets=[]
    for file in sorted(archives.glob('clip-*.npz')):
        name=file.stem; archive=np.load(file); pose=archive['final_mano']
        # Only final_mano reaches the reconstruction function, never GT.
        cams=load_stereo(f'outputs/ace-input-{name}/calibration.json')
        timestamps=np.load(f'outputs/ace-input-{name}/timestamps.npz')
        if len(timestamps['left'])!=len(pose) or np.any(np.abs(timestamps['left']-timestamps['right'])>1_000_000):
            raise ValueError('ACE stereo export timestamps do not match decoded poses')
        exports=[]
        for side in ['left','right']:
            with open(f'outputs/ace-upright-{split}/{name}/{side}/{side}.pkl','rb') as f: exports.append(pickle.load(f))
        output=ROOT/split/config_name/name; output.mkdir(parents=True,exist_ok=True)
        anchored=[]; se3=[]; rotations=[]; evidence=[]; inliers=[]; statuses=[]; start=time.perf_counter()
        for i,xyz in enumerate(pose):
            pixels=[np.asarray(e['joints_2d'][i,1],float)*640 for e in exports]
            confidence=np.stack([np.full(21,e['exists_2d'][i,1] if e['claim'][i,1] else 0.) for e in exports],axis=1)
            for view,uv in enumerate(pixels):
                confidence[~(np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<640).all(1)),view]=0
            r=fit_stereo_rigid(xyz@C.T,*pixels,confidence,cams,config)
            anchored.append(r['anchored']@C); se3.append(r['se3']@C)
            rotations.append(C.T@r['rotation']@C)
            evidence.append(r['stereo_valid']); inliers.append(r['anchor_inliers']); statuses.append(r['status'])
        elapsed=time.perf_counter()-start
        gt=archive['ground_truth']; targets.append(gt)
        wp=Path('outputs/wilor-dev/stereo' if split=='development' else 'outputs/wilor-held/stereo')/name/'predictions.npz'
        w=np.load(wp); wilor=np.where(w['validity'][...,None],w['joints_3d'],np.nan)
        np.testing.assert_array_equal(w['timestamps'],timestamps['left'])
        arrays=dict(ace_final=pose,anchored=np.asarray(anchored),se3=np.asarray(se3),wilor=wilor)
        for key,value in arrays.items(): pool[key].append(value)
        np.savez_compressed(output/'predictions.npz',**arrays,ground_truth=gt,
            stereo_evidence=np.array(evidence),anchor_inliers=np.array(inliers),rotation_delta=rotations,
            statuses=np.array(statuses),timestamps=w['timestamps'],frame_ids=w['frame_ids'],
            anchored_validity=np.isfinite(anchored).all(-1),se3_validity=np.isfinite(se3).all(-1))
        scores[name]={key:evaluate(value,gt) for key,value in arrays.items()}
        (output/'metadata.json').write_text(json.dumps(dict(config=asdict(config),seconds=elapsed,frames=len(gt),
            coordinate_frame='native left camera optical',units='meters',hand_side='right',
            joint_convention='anatomical MANO21',ground_truth_used_in_fitting=False,
            articulation='fixed ACE MANO; rigid rotation and translation only',
            observation_classification='all output joints model-inferred; stereo_evidence and anchor_inliers are separate support masks',
            confidence='ACE hand existence, not calibrated per-joint probability'),indent=2))
    gt=np.concatenate(targets)
    result=dict(config=asdict(config),sequences=scores,pooled={k:evaluate(np.concatenate(v),gt) for k,v in pool.items()})
    (REPORT/f'{split}_{config_name}.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    return result


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    dev={k:experiment('development',k) for k in configs}
    selection={variant:min(configs,key=lambda k:dev[k]['pooled'][variant]['capped_error_with_missing_penalty_mm']) for variant in ['anchored','se3']}
    freeze=dict(selection=selection,criterion='Development capped100mm error with missing-joint penalty; no held-out tuning',configs={k:asdict(v) for k,v in configs.items()})
    (REPORT/'frozen_selection.json').write_text(json.dumps(freeze,indent=2))
    for k in sorted(set(selection.values())):
        result=experiment('held_out',k)
        print(k,json.dumps(result['pooled']),flush=True)


if __name__=='__main__': main()
