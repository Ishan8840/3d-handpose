"""Weight-free real video decoding/encoding + metric export integration test."""
import json,sys
import pytest
import cv2
import numpy as np
from handpose.geometry.camera import Camera
from handpose.models.base import Observation
from handpose.inference import cli

@pytest.mark.parametrize('backend',['mediapipe','wilor'])
def test_video_cli_exports_metric_pose_and_missing_frame(tmp_path,monkeypatch,backend):
    K=np.array([[120.,0,80],[0,120,60],[0,0,1]])
    T=np.eye(4);T[0,3]=-.1;cameras=[Camera(K),Camera(K,T)]
    xyz=np.random.default_rng(3).normal(0,.02,(21,3));xyz[:,2]+=.6
    for side,value in [('left',40),('right',100)]:
        writer=cv2.VideoWriter(str(tmp_path/f'{side}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),30,(160,120));assert writer.isOpened()
        for i in range(3):writer.write(np.full((120,160,3),value if i!=1 else 0,np.uint8))
        writer.release()
    calibration={'units':'meters','transform_convention':'T_camera_from_left'}
    for side,camera in zip(('left','right'),cameras):calibration[side]={'K':camera.K.tolist(),'T_camera_from_left':camera.T_camera_from_left.tolist(),'distortion':[],'model':'opencv'}
    path=tmp_path/'calibration.json';path.write_text(json.dumps(calibration));out=tmp_path/'output'
    class Detector:
        def predict(self,image):
            if image.mean()<10:return []
            return [Observation(cameras[int(image.mean()>60)].project(xyz),np.ones(21))]
        def close(self):pass
    monkeypatch.setattr(cli,'create',lambda name:Detector())
    if backend=='wilor':
        from handpose.inference import mesh_cli
        class MeshDetector(Detector):
            checkpoint='synthetic-test';load_status={}
            def predict(self,image):
                return [{'observation':o,'mano':{}} for o in super().predict(image)]
        monkeypatch.setattr(mesh_cli,'MeshRegressor',lambda *args,**kwargs:MeshDetector())
    monkeypatch.setattr(sys,'argv',['handpose-infer','--model',backend,'--left',str(tmp_path/'left.mp4'),'--right',str(tmp_path/'right.mp4'),'--calibration',str(path),'--output',str(out)])
    cli.main();pred=np.load(out/'predictions.npz')
    np.testing.assert_allclose(pred['joints_3d'][0],xyz,atol=1e-8)
    assert np.isnan(pred['joints_3d'][1]).all() and not pred['validity'][1].any()
    assert pred['timestamps'].tolist()==[0,33333333,66666667]
    for filename in ('joints_3d.npy','timestamps.npy','confidence.npy','validity.npy','wrist_poses.npy','metadata.json','visualization.mp4'):assert (out/filename).is_file()
    assert json.loads((out/'metadata.json').read_text())['coordinate_frame']=='left_camera_optical'
