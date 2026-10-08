"""Versioned metadata correction for early pilot artifacts; pose values unchanged."""
import json
from pathlib import Path
import numpy as np
for path in Path('outputs').rglob('metadata.json'):
    metadata=json.loads(path.read_text());model=metadata.get('model')
    if model not in ('umetrack','POEM-v2-large'):continue
    predictions=path.parent/'predictions.npz'
    if not predictions.exists():continue
    with np.load(predictions) as data:arrays=dict(data)
    arrays['observation_type']=np.where(arrays['validity'],4,0).astype(np.uint8)
    np.savez_compressed(predictions,**arrays)
    metadata['observation_types']={'0':'missing','1':'stereo_observed','2':'single_view_inferred','3':'temporal','4':'multiview_model_inferred','5':'anatomical_fit'}
    metadata['observation_code']=4;metadata['confidence_semantics']='binary valid model output, not calibrated uncertainty';metadata['provenance_revision']='2: distinguish multiview model inference from observed stereo triangulation; no coordinate changes'
    path.write_text(json.dumps(metadata,indent=2))
path=Path('outputs/b1-ablations-dev/mano.npz')
if path.exists():
    with np.load(path) as data:arrays=dict(data)
    arrays['observation_type']=np.where(arrays['validity'],5,0).astype(np.uint8);np.savez_compressed(path,**arrays)
