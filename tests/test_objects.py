import numpy as np
import pytest
from handpose.objects.geometry import decode_mask
from handpose.objects.pose import estimate_pose


def test_hot3d_rle_is_one_based_row_major():
    m=decode_mask(dict(height=2,width=4,rle=[1,2,7,2]))
    np.testing.assert_array_equal(m,[[1,1,0,0],[0,0,1,1]])
    with pytest.raises(ValueError):decode_mask(dict(height=1,width=2,rle=[0,2]))


def test_stereo_object_pose_metric_translation():
    import cv2
    # Asymmetric cuboid, exact convex masks, known nonzero stereo baseline.
    from itertools import product
    v=np.array(list(product([-.04,.04],[-.025,.025],[-.03,.03])))
    K=np.array([[160.,0,79.5],[0,160,79.5],[0,0,1]])
    T=np.eye(4);T[0,3]=-.08;truth=np.array([.025,.015,.6])
    masks=[]
    for t in [np.zeros(3),T[:3,3]]:
        xyz=v+truth+t;uv=xyz@K.T;uv=uv[:,:2]/uv[:,2:]
        m=np.zeros((160,160),np.uint8);cv2.fillConvexPoly(m,cv2.convexHull(uv.astype(np.float32)).astype(int),1);masks.append(m)
    fit=estimate_pose(v,np.array(masks),K,T,starts=4)
    assert fit is not None and fit['accepted']
    assert np.linalg.norm(fit['T'][:3,3]-truth)<.02
    assert not fit['orientation_observable']


def test_empty_masks_are_missing_objects():
    assert estimate_pose(np.zeros((8,3)),np.zeros((2,160,160)),np.eye(3),np.eye(4)) is None


def test_mask_bounds_and_disjoint_runs():
    with pytest.raises(ValueError):decode_mask(dict(height=2,width=3,rle=[6,2]))
    with pytest.raises(ValueError):decode_mask(dict(height=2,width=3,rle=[1]))
    assert not decode_mask(dict(height=2,width=3,rle=[])).any()
