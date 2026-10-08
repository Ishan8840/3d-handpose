"""Render held-out overlays from archived input video and saved predictions/GT."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from handpose.geometry.camera import Camera
from handpose.models.base import EDGES
parser=argparse.ArgumentParser();parser.add_argument('--root',default='outputs/ace-hybrid-held-out');parser.add_argument('--output',default='reports/figures/failures');args=parser.parse_args()
camera=Camera(np.array([[320.,0,319.5],[0,320,319.5],[0,0,1]]));out=Path(args.output);out.mkdir(parents=True,exist_ok=True);records=[]
for folder in sorted(Path(args.root).iterdir()):
    if not folder.is_dir():continue
    pred=np.load(folder/'predictions.npz');gt=np.load(folder/'ground_truth.npz');error=np.linalg.norm(pred['joints_3d']-gt['joints_3d'],axis=-1)*1000
    count=np.isfinite(error).sum(1);score=np.divide(np.nansum(error,axis=1),count,out=np.full(len(error),-1.),where=count>0)
    choices={'largest_observed_error':int(np.argmax(score))}
    speed=np.linalg.norm(np.diff(gt['joints_3d'][:,8],axis=0),axis=-1)
    if np.isfinite(speed).any():choices['fast_index_tip_motion']=int(np.nanargmax(speed)+1)
    visibility=Path('outputs/mp-held-out')/folder.name/'visibility.json'
    if visibility.exists():
        values=np.array([min(v.values()) if isinstance(v,dict) and v else np.nan for v in json.load(open(visibility))])
        if np.isfinite(values).any():choices['lowest_modeled_visibility']=int(np.nanargmin(values))
    cap=cv2.VideoCapture(str(Path('outputs')/('ace-input-'+folder.name)/'left.mp4'))
    if not cap.isOpened():raise ValueError('Missing archived input video')
    for category,index in choices.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
        if not ok:raise ValueError('Frame decode failed')
        image=cv2.rotate(image,cv2.ROTATE_90_COUNTERCLOCKWISE)
        for points,color in ((camera.project(gt['joints_3d'][index]),(0,0,255)),(camera.project(pred['joints_3d'][index]),(0,255,0))):
            for j,k in EDGES:
                if np.isfinite(points[[j,k]]).all():cv2.line(image,tuple(np.rint(points[j]).astype(int)),tuple(np.rint(points[k]).astype(int)),color,2)
        cv2.putText(image,'GT red / prediction green',(8,22),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1)
        filename=f'{folder.name}_{category}_{index}.jpg';cv2.imwrite(str(out/filename),image)
        records.append({'sequence':folder.name,'category':category,'frame':index,'observed_mpjpe_mm':float(score[index]) if score[index]>=0 else None,'image':filename})
    cap.release()
(out/'index.json').write_text(json.dumps(records,indent=2,allow_nan=False))
