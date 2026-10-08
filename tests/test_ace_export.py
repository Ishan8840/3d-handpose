import pickle
import numpy as np
from handpose.models.ace_adapter import ACEExport

def test_rectangular_rotation_inverse_and_claim(tmp_path):
    # Original WxH=640x480, rotated input WxH=480x640.
    data={'joints_cam_direct':np.zeros((1,2,21,3)),'joints_2d':np.tile([.25,.5],(1,2,21,1)),'claim':np.array([[False,True]]),'exists_2d':np.ones((1,2))}
    path=tmp_path/'trusted.pkl';path.write_bytes(pickle.dumps(data))
    exported=ACEExport(path,(480,640),True)
    np.testing.assert_allclose(exported.observations(0)[0].pixels,np.tile([320,359],(21,1)))
    exported.data['claim'][0,1]=False
    assert exported.observations(0)==[]

def test_venv_interpreter_path_preserves_environment():
    import subprocess,sys
    from pathlib import Path
    # resolve() follows the venv's Python symlink and silently selects system packages.
    from handpose.inference.ace_cli import interpreter_path
    executable=interpreter_path(sys.executable)
    prefix=subprocess.check_output([executable,'-c','import sys; print(sys.prefix)'],text=True).strip()
    assert prefix==sys.prefix
