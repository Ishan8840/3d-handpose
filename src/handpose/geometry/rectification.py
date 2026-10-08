import cv2
import numpy as np

def rectify(left,right,size):
    """Return rectified cameras/maps; transform metric results back via R_left.T."""
    if left.model!='opencv' or right.model!='opencv':raise ValueError('First undistort native fisheye using its official camera model')
    T=right.T_camera_from_left@np.linalg.inv(left.T_camera_from_left)
    Rl,Rr,Pl,Pr,Q,_,_=cv2.stereoRectify(left.K,left.distortion,right.K,right.distortion,size,T[:3,:3],T[:3,3],flags=cv2.CALIB_ZERO_DISPARITY,alpha=0)
    maps=[cv2.initUndistortRectifyMap(cam.K,cam.distortion,R,P[:,:3],size,cv2.CV_32FC1) for cam,R,P in ((left,Rl,Pl),(right,Rr,Pr))]
    return Rl,Rr,Pl,Pr,Q,maps

def rotate_clockwise_camera(camera,width,height):
    """Re-express optical axes and pixel grid under a 90-degree CW image rotation."""
    from .camera import Camera
    if np.any(camera.distortion):raise ValueError('Undistort before rotating the camera')
    R=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    A=np.array([[0.,-1,height-1],[1,0,0],[0,0,1]])
    H=np.eye(4);H[:3,:3]=R
    return Camera(A@camera.K@R.T,H@camera.T_camera_from_left@H.T),R
