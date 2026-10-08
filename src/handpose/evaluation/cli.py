import argparse
import json
import numpy as np
from .metrics import evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--predictions',required=True); p.add_argument('--ground-truth',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); pred=np.load(a.predictions); gt=np.load(a.ground_truth)
    for key in ('timestamps','frame_ids'):
        if key not in pred or key not in gt or not np.array_equal(pred[key],gt[key]):
            raise ValueError(f'Matching {key} required; never evaluate different frames')
    result=evaluate(pred['joints_3d'],gt['joints_3d'],pred['validity'],gt['validity'],pred['timestamps'])
    with open(a.output,'w') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result,indent=2))
