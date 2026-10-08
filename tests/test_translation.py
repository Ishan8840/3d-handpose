import numpy as np
from handpose.geometry.camera import Camera
from handpose.geometry.translation import translation_from_rays


def test_metric_translation_from_distorted_pixels():
    rng=np.random.default_rng(9)
    joints=rng.normal(size=(21,3))*.035
    t=np.array([.07,-.11,.54])
    cam=Camera(np.array([[600,0,300],[0,590,230],[0,0,1]]),distortion=np.array([.01,-.02,0,0,0]))
    recovered=translation_from_rays(joints,cam.project(joints+t),cam)
    np.testing.assert_allclose(recovered,t,atol=1e-8)
