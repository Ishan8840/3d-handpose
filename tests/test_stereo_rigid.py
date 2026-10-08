import numpy as np
from scipy.spatial.transform import Rotation
from handpose.geometry.camera import Camera
from handpose.fitting.stereo_rigid import fit_stereo_rigid, RigidConfig


def scene():
    K=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    T=np.eye(4); T[0,3]=-.08
    cameras=[Camera(K),Camera(K,T)]
    points=np.random.default_rng(20).normal(0,.025,(21,3)); points[0]=0
    points[:,2]+=.5
    return cameras,points


def test_anchor_corrects_translation_without_changing_articulation():
    cams,gt=scene(); uv=[c.project(gt) for c in cams]
    result=fit_stereo_rigid(gt+[.05,-.02,.12],*uv,np.ones((21,2)),cams)
    np.testing.assert_allclose(result['anchored'],gt,atol=1e-6)
    np.testing.assert_allclose(result['se3'],gt,atol=1e-6)


def test_se3_corrects_rotation_and_preserves_shape():
    cams,gt=scene(); uv=[c.project(gt) for c in cams]
    R=Rotation.from_euler('y',12,degrees=True).as_matrix()
    ace=(gt-gt[:1])@R.T+gt[:1]+[0,0,.1]
    result=fit_stereo_rigid(ace,*uv,np.ones((21,2)),cams,RigidConfig(anchor_sigma_m=.1))
    assert result['status']=='ok'
    assert np.linalg.norm(result['se3']-gt,axis=1).mean()<.001
    distances=lambda x:np.linalg.norm(x[:,None]-x[None],axis=-1)
    np.testing.assert_allclose(distances(result['se3']),distances(ace),atol=1e-12)


def test_outlier_and_missing_stereo_are_not_zero_error_predictions():
    cams,gt=scene(); uv=[c.project(gt) for c in cams]
    uv[1][5]+=[0,70]
    result=fit_stereo_rigid(gt+[0,0,.1],*uv,np.ones((21,2)),cams)
    assert not result['stereo_valid'][5]
    np.testing.assert_allclose(result['anchored'],gt,atol=1e-6)
    result=fit_stereo_rigid(gt,*uv,np.zeros((21,2)),cams)
    assert np.isnan(result['anchored']).all() and np.isnan(result['se3']).all()


def test_gt_free_cli_both_modes(tmp_path):
    import json,pickle,subprocess,sys
    from pathlib import Path
    cams,gt=scene(); poses=np.stack([gt+[0,0,.1]]*2)
    np.savez(tmp_path/'poses.npz',joints_3d=poses)
    np.savez(tmp_path/'times.npz',left=[0,33333333],right=[0,33333333])
    cal=dict(units='meters',transform_convention='T_camera_from_left')
    for view,cam in zip(['left','right'],cams):
        cal[view]=dict(K=cam.K.tolist(),T_camera_from_left=cam.T_camera_from_left.tolist())
        uv=cam.project(gt)/[640,480]
        e=dict(claim=np.ones((2,2),bool),exists_2d=np.ones((2,2)),
            joints_2d=np.tile(uv,(2,2,1,1)),intrinsics=dict(fx=500.,fy=500.,cx=320.,cy=240.,image_width=640,image_height=480))
        if view=='right':e['claim'][1,1]=False
        (tmp_path/(view+'.pkl')).write_bytes(pickle.dumps(e))
    (tmp_path/'cal.json').write_text(json.dumps(cal))
    script=Path(__file__).resolve().parents[1]/'scripts/run_ace_rigid.py'
    for mode in ['anchored','se3']:
        out=tmp_path/mode
        subprocess.run([sys.executable,str(script),'--left-predictions',str(tmp_path/'left.pkl'),
            '--right-predictions',str(tmp_path/'right.pkl'),'--final-poses',str(tmp_path/'poses.npz'),
            '--timestamps',str(tmp_path/'times.npz'),'--calibration',str(tmp_path/'cal.json'),
            '--output',str(out),'--mode',mode],check=True)
        p=np.load(out/'predictions.npz')
        np.testing.assert_allclose(p['joints_3d'][0],gt,atol=1e-6)
        assert np.isnan(p['joints_3d'][1]).all() and not p['validity'][1].any()
