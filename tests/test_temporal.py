import numpy as np
from handpose.temporal.smoothing import smooth

def test_linear_motion_and_gaps():
    t=np.arange(20)*10000000
    x=np.ones((20,21,3))*np.arange(20)[:,None,None]*.001
    x[9]=np.nan
    y,s=smooth(x,np.ones((20,21)),t,window_seconds=.06,max_gap_seconds=.03)
    np.testing.assert_allclose(y[:,0,0],np.arange(20)*.001,atol=1e-12)
    assert (s[9]==3).all()
    y,s=smooth(x,np.ones((20,21)),t,max_gap_seconds=0)
    assert np.isnan(y[9]).all()
