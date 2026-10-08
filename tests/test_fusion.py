import numpy as np
from handpose.inference.fusion import fuse_metric


def test_agreement_disagreement_and_missing_provenance():
    a=np.array([[0,0,1],[0,0,1],[np.nan]*3,[np.nan]*3])
    b=np.array([[.02,0,1],[.5,0,1],[0,0,2],[np.nan]*3])
    x,s=fuse_metric(a,b,.25,.05)
    np.testing.assert_allclose(x[:3],[[.005,0,1],[0,0,1],[0,0,2]])
    assert np.isnan(x[3]).all()
    assert s.tolist()==[3,1,2,0]
