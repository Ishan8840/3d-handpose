import numpy as np
from handpose.fitting.stereo import refine_frame
from handpose.geometry.camera import Camera

def test_robust_refinement_rejects_misaligned_prior():
    K=np.array([[500,0,320],[0,500,240],[0,0,1.]])
    T=np.eye(4);T[0,3]=-.1;left=Camera(K);right=Camera(K,T)
    gt=np.random.default_rng(42).normal(0,.03,(21,3));gt[:,2]+=.6
    initial=gt+np.array([.001,-.002,.003]);uvl=left.project(gt);uvr=right.project(gt)
    a=refine_frame(initial,uvl,uvr,np.ones(21),left,right)
    b=refine_frame(initial,uvl,uvr,np.ones(21),left,right,prior=gt+1,prior_weight=10)
    assert np.linalg.norm(a-gt)<np.linalg.norm(initial-gt)
    np.testing.assert_allclose(a,b)
