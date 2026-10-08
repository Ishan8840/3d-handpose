import numpy as np
import pytest
from types import SimpleNamespace as S
from handpose.data.egostandard import canonical_hand,synchronized_indices
from handpose.models.base import JOINT_NAMES

def test_explicit_world_to_camera_units():
    T=np.eye(4);T[:3,3]=[1,0,0]
    points=[S(pos=S(x=1100,y=200,z=500)) for _ in range(21)]
    np.testing.assert_allclose(canonical_hand(points,JOINT_NAMES,.001,T),np.tile([.1,.2,.5],(21,1)))
    with pytest.raises(ValueError):canonical_hand(points,['unknown']*21,.001,T)

def test_unique_timestamp_matching_keeps_missing_frames():
    np.testing.assert_array_equal(synchronized_indices([0,10,20,30],[1,21,31],2),[0,-1,1,2])
    with pytest.raises(ValueError):synchronized_indices([1,1],[1])
