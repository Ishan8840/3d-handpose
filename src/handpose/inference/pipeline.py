import numpy as np
from handpose.models.base import Prediction
from handpose.triangulation.weighted import triangulate

def reconstruct(left, right, left_observations, right_observations, timestamp_ns, **kwargs):
    best=None; best_score=-1
    for a in left_observations:
        for b in right_observations:
            xyz,valid=triangulate(left,right,a.pixels,b.pixels,np.stack([a.confidence,b.confidence],axis=1),**kwargs)
            score=np.sum(np.minimum(a.confidence,b.confidence)*valid)
            if score>best_score:
                best_score=score
                best=Prediction(timestamp_ns,xyz,a.pixels,b.pixels,np.where(valid,np.minimum(a.confidence,b.confidence),0),valid,np.full((3,3),np.nan))
    if best is None:
        best=Prediction(timestamp_ns,np.full((21,3),np.nan),np.full((21,2),np.nan),np.full((21,2),np.nan),np.zeros(21),np.zeros(21,bool),np.full((3,3),np.nan))
    return best.validate()
