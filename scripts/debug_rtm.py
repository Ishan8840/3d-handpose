from handpose.data.hot3d import iter_clip
from handpose.models.rtmpose_adapter import RTMPoseAdapter
import numpy as np
m=RTMPoseAdapter()
for i,s in enumerate(iter_clip('data/hot3d/clip-000000.tar',10)):
    for side in ('left','right'):
        im=s[side]; mp=m.identity.predict(im); boxes=m.model.det_model(im); k,c=m.model.pose_model(im,bboxes=boxes)
        print(i,side,'mp',len(mp),'boxes',boxes,'score',c.min() if c.size else None,c.max() if c.size else None,'wrists',k[:,0].tolist() if len(k) else [],'mpw',[x.pixels[0].tolist() for x in mp],flush=True)
m.close()
