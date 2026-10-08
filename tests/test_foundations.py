import numpy as np
import pytest
from handpose.geometry.camera import Camera, transform, fundamental, epipolar_distance, crop_transform, disparity_depth
from handpose.triangulation.weighted import triangulate
from handpose.evaluation.metrics import evaluate, sequence_bootstrap
from handpose.inference.pipeline import reconstruct
from handpose.inference.cli import save_predictions
from handpose.models.base import Observation

@pytest.fixture
def stereo():
    K=np.array([[500.,0,320],[0,500,240],[0,0,1]])
    T=np.eye(4); T[0,3]=-.1
    return Camera(K),Camera(K,T)

def points():
    rng=np.random.default_rng(4)
    return rng.uniform([-.1,-.1,.4],[.1,.1,.9],(21,3))

def test_projection_triangulation_metric(stereo):
    l,r=stereo; x=points(); a,b=l.project(x),r.project(x)
    y,v=triangulate(l,r,a,b)
    assert v.all(); np.testing.assert_allclose(y,x,atol=1e-10)
    assert np.all(a[:,0]>b[:,0])
    np.testing.assert_allclose(disparity_depth(a[:,0]-b[:,0],500,.1),x[:,2])
    np.testing.assert_allclose(l.backproject(a,x[:,2]),x)
    np.testing.assert_allclose(r.backproject(b,x[:,2]),x)
    np.testing.assert_allclose(epipolar_distance(l,r,a,b),0,atol=1e-12)

def test_distortion(stereo):
    l,r=stereo; l.distortion=r.distortion=np.array([.08,-.03,.001,.002,.001])
    x=points(); y,v=triangulate(l,r,l.project(x),r.project(x))
    assert v.all(); np.testing.assert_allclose(y,x,atol=1e-7)

def test_axis_transform_and_crop(stereo):
    _,r=stereo; x=np.array([[.1,.2,.8]])
    np.testing.assert_allclose(transform(x,r.T_camera_from_left),[[0,.2,.8]])
    H=crop_transform(10,20,100,200,200,100)
    np.testing.assert_allclose(H@np.array([10,20,1]),[.5,-.25,1])

def test_invalid_geometry(stereo):
    l,r=stereo; x=points(); a,b=l.project(x),r.project(x)
    b[0,1]+=20; b[1]=np.nan; b[2]=a[2]
    y,v=triangulate(l,r,a,b)
    assert not v[:3].any(); assert np.isnan(y[:3]).all()
    y,v=triangulate(l,r,b,a)
    assert not v.any()

def test_metrics_alignment_missing():
    gt=np.stack([points()]*3); pred=gt+[.01,0,0]
    m=evaluate(pred,gt)
    assert m['absolute_mpjpe_mm']==pytest.approx(10)
    assert m['wrist_mm']==pytest.approx(10)
    assert m['wrist_relative_mpjpe_mm']==pytest.approx(0,abs=1e-10)
    assert m['pa_mpjpe_mm']==pytest.approx(0,abs=1e-10)
    pred[0]=np.nan; m=evaluate(pred,gt)
    assert m['joint_coverage']==pytest.approx(2/3)
    assert m['capped_error_with_missing_penalty_mm']==pytest.approx(40)
    pred[:]=np.nan; m=evaluate(pred,gt)
    assert m['absolute_mpjpe_mm'] is None
    assert m['capped_error_with_missing_penalty_mm']==100

def test_pa_scale_not_absolute():
    gt=points()[None]; pred=2*gt
    m=evaluate(pred,gt)
    assert m['absolute_mpjpe_mm']>400
    assert m['pa_mpjpe_mm']<1e-9

def test_end_to_end_weightless(stereo,tmp_path):
    l,r=stereo; x=points()
    pred=reconstruct(l,r,[Observation(l.project(x),np.ones(21))],[Observation(r.project(x),np.ones(21))],42)
    save_predictions([pred],tmp_path,{})
    result=np.load(tmp_path/'predictions.npz')
    np.testing.assert_allclose(result['joints_3d'][0],x)
    assert result['timestamps'][0]==42
    assert result['validity'].all()
    assert (tmp_path/'metadata.json').exists()

def test_bootstrap_clusters():
    assert sequence_bootstrap([1],[0]) is None
    np.testing.assert_allclose(sequence_bootstrap([2,2,2],[0,1,2]),[2,2])

def test_rotated_camera_projection(stereo):
    from handpose.geometry.rectification import rotate_clockwise_camera
    for cam in stereo:
        rotated,R=rotate_clockwise_camera(cam,640,480)
        xyz=points();uv=cam.project(xyz)
        np.testing.assert_allclose(rotated.project(xyz@R.T),np.c_[479-uv[:,1],uv[:,0]],atol=1e-10)

def test_rectified_horizontal_disparity(stereo):
    import cv2
    from handpose.geometry.rectification import rectify
    l,r=stereo; Rl,Rr,Pl,Pr,Q,maps=rectify(l,r,(640,480))
    xyz=points();a=l.project(xyz);b=r.project(xyz)
    d=a[:,0]-b[:,0];h=np.c_[a,d,np.ones(len(d))]@Q.T
    np.testing.assert_allclose(h[:,:3]/h[:,3:],xyz,atol=1e-10)

def test_rotation_and_nonparallel_cameras():
    from scipy.spatial.transform import Rotation
    K=np.array([[650.,0,320],[0,620,240],[0,0,1]])
    T=np.eye(4);T[:3,:3]=Rotation.from_euler('y',7,degrees=True).as_matrix();T[:3,3]=-T[:3,:3]@np.array([.08,.01,0])
    l,r=Camera(K),Camera(K,T);x=points();p,v=triangulate(l,r,l.project(x),r.project(x))
    assert v.all();np.testing.assert_allclose(p,x,atol=1e-10)

def test_reject_nan_camera_translation():
    T=np.eye(4);T[0,3]=np.nan
    with pytest.raises(ValueError,match='finite'):
        Camera(np.eye(3),T)
