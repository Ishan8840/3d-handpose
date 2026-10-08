"""Independent official-camera ray round-trip, rotation and timestamp checks."""
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

root=Path.cwd(); sys.path[:0]=[str(root/'src'),str(root/'third_party/hand_tracking_toolkit')]
from hand_tracking_toolkit.camera import from_json
from handpose.data.hot3d import pinhole_view

records=[]
for entry in json.loads((root/'data/hot3d/manifest.json').read_text()):
    if entry['split']=='smoke': continue
    with tarfile.open(entry['path']) as tar:
        def read(n): return json.load(tar.extractfile(n))
        keys=sorted(n[:-10] for n in tar.getnames() if n.endswith('.info.json'))[:150]
        source=from_json(read(keys[0]+'.cameras.json')['1201-1'])
        timestamps=np.array([read(k+'.info.json')['image_timestamps_ns']['1201-1'] for k in keys],np.int64)
        for size in (480,640):
            target=pinhole_view(source,size,size/2)
            x,y=np.meshgrid(np.linspace(0,size-1,7),np.linspace(0,size-1,7))
            uv=np.c_[x.ravel(),y.ravel()]; rays=target.window_to_eye(uv)
            source_uv=source.eye_to_window(rays)
            # OVR624 exposes forward projection only. Invert that official
            # projection numerically, starting from the optical axis.
            inverse=[]
            for pixel in source_uv:
                fit=least_squares(lambda xy:source.eye_to_window(np.array([[xy[0],xy[1],1.]]))[0]-pixel,
                                  np.zeros(2),xtol=1e-12,ftol=1e-12,gtol=1e-12)
                if not fit.success: raise RuntimeError('Numerical lens inverse failed')
                inverse.append([*fit.x,1.])
            restored=target.eye_to_window(np.array(inverse))
            inverse_error=float(np.max(np.linalg.norm(restored-uv,axis=1)))
            C=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
            rotated=target.eye_to_window(rays@C.T)
            rotation_error=float(np.max(np.abs(rotated-np.c_[size-1-uv[:,1],uv[:,0]])))
            if inverse_error>.01 or rotation_error>1e-8: raise RuntimeError('Camera convention check failed')
            records.append(dict(sequence=entry['sequence'],split=entry['split'],
                source_model=type(source).__name__,source_size=[source.width,source.height],
                inverse_method='numerical inversion of official forward projection; 49 rays',
                target_size=size,target_K=target.uv_to_window_matrix().tolist(),
                fisheye_pinhole_roundtrip_max_px=inverse_error,clockwise_rotation_max_px=rotation_error,
                median_frame_period_ns=float(np.median(np.diff(timestamps))),
                timestamps_strictly_increasing=bool(np.all(np.diff(timestamps)>0))))
out=root/'reports/ace_audit/camera_checks.json'; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(records,indent=2)); print(json.dumps(records,indent=2))
