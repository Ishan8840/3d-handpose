import numpy as np


def translation_from_rays(joints, pixels, camera):
    """Fit translation only; joints must already have metric scale and camera axes."""
    xy=camera.normalized(pixels)
    matrix=np.zeros((len(joints),2,3));matrix[:,0,0]=1;matrix[:,1,1]=1
    matrix[:,:,2]=-xy
    target=xy*joints[:,2:]-joints[:,:2]
    return np.linalg.lstsq(matrix.reshape(-1,3),target.reshape(-1),rcond=None)[0]
