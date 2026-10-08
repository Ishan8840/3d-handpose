import argparse,json
from pathlib import Path
import numpy as np
from handpose.data.hot3d import mano_ground_truth
from handpose.evaluation.metrics import evaluate
p=argparse.ArgumentParser();p.add_argument('--clip',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args()
pred=np.load(a.predictions);gt=mano_ground_truth(a.clip,len(pred['joints_3d']))
r=evaluate(pred['joints_3d'],gt,pred['validity'],timestamps_ns=pred['timestamps']);r['gt_convention']='MANO-21, separate from UmeTrack-19 results';Path(a.output).write_text(json.dumps(r,indent=2,allow_nan=False));print(json.dumps(r))
