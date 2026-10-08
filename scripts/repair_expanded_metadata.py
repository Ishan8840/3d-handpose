"""Repair provenance descriptions; never modify poses or evaluation errors."""
import json
from pathlib import Path
import numpy as np
for manifest_file in ('expanded_development_runs.json','expanded_held_runs.json','expanded_show3d_runs.json'):
 runs=json.load(open(Path('configs/experiments')/manifest_file))
 for name,root in runs.items():
  for p in Path(root).glob('*/metadata.json'):
   m=json.loads(p.read_text())
   if name.startswith('EgoForce'):
    m.update(landmarks='native learned 2D head',translation='official ray-space solver',input_views=2 if 'stereo' in name else 1)
    m.pop('monocular_translation',None)
   elif name.startswith('OmniHands'):
    m.update(input_views=2,translation='calibrated least squares metric translation' if 'lift' in name else 'calibrated stereo triangulation')
    m.pop('monocular_translation',None)
    if 'lift' in name:
     m['observation_code']=4;q=dict(np.load(p.parent/'predictions.npz'));q['observation_type']=np.where(q['validity'],4,0).astype(np.uint8);np.savez_compressed(p.parent/'predictions.npz',**q)
   p.write_text(json.dumps(m,indent=2)+'\n')
for report,runs_file in [('expanded_held_comparison','expanded_held_runs.json'),('expanded_held_mano21','expanded_held_runs.json'),('expanded_development_comparison','expanded_development_runs.json'),('expanded_show3d_comparison','expanded_show3d_runs.json')]:
 path=Path('reports')/report/'comparison.json';doc=json.loads(path.read_text());runs=json.load(open(Path('configs/experiments')/runs_file))
 for name,m in doc['models'].items():
  counts=dict(direct_stereo_initial=0,multiview_model=0,temporal_fill_only=0,anatomical_fill=0);total=0
  for folder in Path(runs[name]).iterdir():
   if not (folder/'predictions.npz').exists():continue
   pred=np.load(folder/'predictions.npz');gt=np.load(folder/('ground_truth_mano21.npz' if report.endswith('mano21') else 'ground_truth.npz'));eligible=np.isfinite(gt['joints_3d']).all(-1);total+=int(eligible.sum());valid=pred['validity']&eligible
   original=pred['original_observation_type'] if 'original_observation_type' in pred else pred['observation_type']
   pre=pred['pre_temporal_observation_type'] if 'pre_temporal_observation_type' in pred else np.where(pred['observation_type']==5,5,original)
   counts['direct_stereo_initial']+=int(((original==1)&valid).sum());counts['multiview_model']+=int(((original==4)&valid).sum());counts['temporal_fill_only']+=int(((pre==0)&valid).sum());counts['anatomical_fill']+=int(((original==0)&(pre==5)&valid).sum())
  m['provenance_coverage']={k:v/total if total else None for k,v in counts.items()}
 path.write_text(json.dumps(doc,indent=2,allow_nan=False)+'\n')
