"""Analyze identical-frame prediction archives without changing model selection."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from handpose.evaluation.overlap import compare_errors

p=argparse.ArgumentParser()
p.add_argument('--runs', required=True)
p.add_argument('--manifest', default='configs/datasets/hot3d.json')
p.add_argument('--split', required=True)
p.add_argument('--output', required=True)
a=p.parse_args()
runs=json.loads(Path(a.runs).read_text())
entries=[e for e in json.loads(Path(a.manifest).read_text()) if e['split']==a.split]
predictions={}; ground_truth=None; sequence_ids=[]
for name, root in runs.items():
    poses=[]; truths=[]
    for entry in entries:
        folder=Path(root)/Path(entry['path']).stem
        pred=np.load(folder/'predictions.npz'); gt=np.load(folder/'ground_truth.npz')
        assert np.array_equal(pred['timestamps'], gt['timestamps'])
        assert np.array_equal(pred['frame_ids'], gt['frame_ids'])
        assert len(pred['timestamps']) == entry['max_frames']
        poses.append(np.where(pred['validity'][...,None], pred['joints_3d'], np.nan))
        truths.append(gt['joints_3d'])
    predictions[name]=np.concatenate(poses)
    truth=np.concatenate(truths)
    if ground_truth is None:
        ground_truth=truth
    else:
        np.testing.assert_allclose(ground_truth, truth, equal_nan=True)
for entry in entries:
    sequence_ids.extend([entry['sequence']]*entry['max_frames'])
sequence_ids=np.asarray(sequence_ids)
result={'split':a.split,'frames':len(ground_truth),'pairs':{},
        'caution':'Oracle uses ground truth; correlations are descriptive, joint samples are not independent.'}
for first,second in itertools.combinations(runs,2):
    item=compare_errors(predictions[first],predictions[second],ground_truth)
    item['by_sequence']={s:compare_errors(predictions[first][sequence_ids==s],predictions[second][sequence_ids==s],ground_truth[sequence_ids==s]) for s in np.unique(sequence_ids)}
    result['pairs'][first+' vs '+second]=item
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(result,indent=2,allow_nan=False))
for pair,item in result['pairs'].items():
    print(pair,{k:item[k] for k in ('error_spearman_common','failure_jaccard','both_missing_rate','oracle_joint_selection_capped_mm')})
