"""Offline ACE stereo video pipeline. No annotations or GT-dependent crops."""
import argparse,json,subprocess,time
from pathlib import Path
import cv2
import numpy as np
from handpose.geometry.camera import load_stereo
from handpose.geometry.rectification import rotate_clockwise_camera
from handpose.models.ace_adapter import ACEExport
from handpose.models.base import EDGES
from handpose.inference.pipeline import reconstruct
from handpose.inference.cli import save_predictions
from handpose.temporal.smoothing import smooth


def main(argv=None):
    p=argparse.ArgumentParser()
    for key in ('left','right','calibration','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--model',default='ace');p.add_argument('--timestamps');p.add_argument('--rotate-cw',action='store_true');p.add_argument('--temporal-seconds',type=float,default=0)
    p.add_argument('--anatomy-weight',type=float,default=0)
    p.add_argument('--ace-root',default='third_party/ace');p.add_argument('--ace-python',default='.venv-ace/bin/python');p.add_argument('--ace-left-export');p.add_argument('--ace-right-export')
    a=p.parse_args(argv);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);cameras=load_stereo(a.calibration);exports=[];sizes=[];fps_values=[];counts=[];start=time.perf_counter()
    if any(np.any(c.distortion) for c in cameras):raise ValueError('ACE requires pinhole video; undistort video/calibration first')
    if bool(a.ace_left_export)!=bool(a.ace_right_export):raise ValueError('Supply both ACE exports or neither')
    for side,path,camera,existing in zip(('left','right'),(a.left,a.right),cameras,(a.ace_left_export,a.ace_right_export)):
        cap=cv2.VideoCapture(path)
        if not cap.isOpened():raise ValueError('Cannot open '+path)
        w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT));fps=cap.get(cv2.CAP_PROP_FPS);sizes.append((w,h));fps_values.append(fps)
        side_dir=out/'ace_work'/side;side_dir.mkdir(parents=True,exist_ok=True);video=Path(path).resolve();in_cam=camera;in_size=(w,h);count=0
        writer=None
        if a.rotate_cw:
            in_cam,_=rotate_clockwise_camera(camera,w,h);in_size=(h,w)
            if not existing:
                video=(side_dir/'input.mp4').resolve();writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,in_size)
                if not writer.isOpened():raise ValueError('Video encoder unavailable')
        while True:
            ok,im=cap.read()
            if not ok:break
            count+=1
            if writer:writer.write(cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE))
        cap.release()
        if writer:writer.release()
        counts.append(count)
        if existing:exports.append(ACEExport(existing,(h,w) if a.rotate_cw else (w,h),a.rotate_cw));continue
        K=in_cam.K
        camera_file=side_dir/'camera.json';camera_file.write_text(json.dumps({'image_width':in_size[0],'image_height':in_size[1],'frames':[{'intrinsics':{'fx':K[0,0],'fy':K[1,1],'cx':K[0,2],'cy':K[1,2]}}]}))
        result=(side_dir/'result').resolve()
        with (side_dir/'inference.log').open('w') as log:
            subprocess.run([str(Path(a.ace_python).resolve()),'infer_video.py','--video',str(video),'--camera',str(camera_file.resolve()),'--opt','options/ace_ego_hand_k.yml','--ckpt','checkpoints/ace_ego_hand_k.pt','--out',str(result),'--encode_w','640'],cwd=Path(a.ace_root).resolve(),stdout=log,stderr=subprocess.STDOUT,check=True)
        exports.append(ACEExport(result/(video.stem+'.pkl'),in_size,a.rotate_cw))
    if counts[0]!=counts[1] or counts[0]==0 or min(fps_values)<=0 or abs(fps_values[0]-fps_values[1])>1e-4:raise ValueError('Mismatched stereo streams')
    ts=np.arange(counts[0],dtype=float)/fps_values[0]*1e9
    if a.timestamps:
        data=np.load(a.timestamps)
        if any(len(data[k])!=counts[0] for k in ('left','right')) or np.any(np.abs(data['left']-data['right'])>1_000_000):raise ValueError('Invalid stereo timestamps')
        ts=data['left']
    if np.any(np.diff(ts)<=0):raise ValueError('Nonmonotonic timestamps')
    if any(len(e.data['claim'])!=counts[0] for e in exports):raise ValueError('ACE export frame count differs from input')
    predictions=[reconstruct(*cameras,exports[0].observations(i),exports[1].observations(i),int(ts[i])) for i in range(counts[0])]
    if a.anatomy_weight:
        from handpose.fitting.stereo import refine_sequence
        xyz=refine_sequence(np.array([q.joints_3d_camera for q in predictions]),np.array([q.joints_2d_left for q in predictions]),np.array([q.joints_2d_right for q in predictions]),np.array([q.joint_confidence for q in predictions]),[cameras]*len(predictions),a.anatomy_weight)
        for q,x in zip(predictions,xyz):q.joints_3d_camera=x
    original=np.array([q.joint_validity for q in predictions]);provenance=np.where(original,1,0).astype(np.uint8)
    if a.temporal_seconds:
        xyz,provenance=smooth(np.array([q.joints_3d_camera for q in predictions]),np.array([q.joint_confidence for q in predictions]),ts,a.temporal_seconds,max_gap_seconds=.1)
        for q,x in zip(predictions,xyz):q.joints_3d_camera=x;q.joint_validity=np.isfinite(x).all(-1)
    save_predictions(predictions,out,{'model':'ace_stereo','anatomy_weight':a.anatomy_weight,'temporal_seconds':a.temporal_seconds,'elapsed_seconds':time.perf_counter()-start,'unit_convention_source':json.loads(Path(a.calibration).read_text()).get('unit_convention_source','provided calibration declares meters'),'input_rotation_clockwise':a.rotate_cw})
    path=out/'predictions.npz'
    with np.load(path) as f:arrays=dict(f)
    arrays.update(observation_type=provenance,original_observation_type=np.where(original,1,0).astype(np.uint8));np.savez_compressed(path,**arrays)
    cap=cv2.VideoCapture(a.left);writer=cv2.VideoWriter(str(out/'visualization.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps_values[0],sizes[0])
    if not writer.isOpened():raise ValueError('Video encoder unavailable')
    for q in predictions:
        ok,im=cap.read()
        if not ok:raise ValueError('Video decode changed during visualization')
        uv=cameras[0].project(q.joints_3d_camera)
        for j,k in EDGES:
            if q.joint_validity[[j,k]].all():cv2.line(im,tuple(np.rint(uv[j]).astype(int)),tuple(np.rint(uv[k]).astype(int)),(0,255,0),2)
        writer.write(im)
    cap.release();writer.release()
