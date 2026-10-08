"""OpenCV optical axes: +x right, +y down, +z forward; translations in meters."""
from dataclasses import dataclass, field
import json
import cv2
import numpy as np


def transform(points, T):
    return np.asarray(points) @ T[:3, :3].T + T[:3, 3]


def skew(t):
    x, y, z = t
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])


@dataclass
class Camera:
    K: np.ndarray
    T_camera_from_left: np.ndarray = field(default_factory=lambda: np.eye(4))
    distortion: np.ndarray = field(default_factory=lambda: np.zeros(5))
    model: str = "opencv"

    def __post_init__(self):
        self.K = np.asarray(self.K, float)
        self.T_camera_from_left = np.asarray(self.T_camera_from_left, float)
        self.distortion = np.asarray(self.distortion, float)
        if self.model not in ("opencv", "fisheye"):
            raise ValueError("Unsupported camera model: use official HOT3D camera unprojection")
        if self.K.shape != (3, 3) or self.T_camera_from_left.shape != (4, 4):
            raise ValueError("Expected K (3,3), T (4,4)")
        if not np.isfinite(self.T_camera_from_left).all() or not np.isfinite(self.distortion).all():
            raise ValueError("Calibration must contain finite transforms and distortion")
        R = self.T_camera_from_left[:3, :3]
        if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(R), 1):
            raise ValueError("Extrinsic rotation must be proper orthonormal")
        if not np.allclose(self.T_camera_from_left[3], [0, 0, 0, 1]):
            raise ValueError("Invalid homogeneous transform")
        if not np.isfinite(self.K).all() or min(self.K[0, 0], self.K[1, 1]) <= 0:
            raise ValueError("Invalid intrinsics")

    @property
    def P(self):
        return self.K @ self.T_camera_from_left[:3]

    def project(self, points):
        p = transform(points, self.T_camera_from_left)
        if self.model == "fisheye":
            uv, _ = cv2.fisheye.projectPoints(p.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K, self.distortion)
        else:
            uv, _ = cv2.projectPoints(p, np.zeros(3), np.zeros(3), self.K, self.distortion)
        uv = uv.reshape(-1, 2)
        uv[p[:, 2] <= 0] = np.nan
        return uv

    def normalized(self, pixels):
        p = np.asarray(pixels, float).reshape(-1, 1, 2)
        fn = cv2.fisheye.undistortPoints if self.model == "fisheye" else cv2.undistortPoints
        return fn(p, self.K, self.distortion).reshape(-1, 2)

    def backproject(self, pixels, depth):
        rays = np.c_[self.normalized(pixels), np.ones(len(pixels))]
        return transform(rays * np.asarray(depth)[..., None], np.linalg.inv(self.T_camera_from_left))


def essential(left, right):
    T = right.T_camera_from_left @ np.linalg.inv(left.T_camera_from_left)
    return skew(T[:3, 3]) @ T[:3, :3]


def fundamental(left, right):
    return np.linalg.inv(right.K).T @ essential(left, right) @ np.linalg.inv(left.K)


def epipolar_distance(left, right, a, b):
    """Symmetric point-to-line distance, UNDISTORTED pixels."""
    F = fundamental(left, right)
    a, b = np.c_[a, np.ones(len(a))], np.c_[b, np.ones(len(b))]
    lb, la = a @ F.T, b @ F
    residual = np.abs(np.sum(b * lb, axis=1))
    return residual * .5 * (1 / np.maximum(np.linalg.norm(lb[:, :2], axis=1), 1e-12) + 1 / np.maximum(np.linalg.norm(la[:, :2], axis=1), 1e-12))


def crop_transform(x, y, width, height, out_width, out_height):
    """Pixel-center affine map for OpenCV resize after crop."""
    sx, sy = out_width / width, out_height / height
    return np.array([[sx, 0, sx * (.5 - x) - .5], [0, sy, sy * (.5 - y) - .5], [0, 0, 1.]])


def disparity_depth(disparity, focal_px, baseline_m):
    d = np.asarray(disparity, float)
    return np.divide(focal_px * baseline_m, d, out=np.full_like(d, np.nan), where=d > 0)


def load_stereo(path):
    d = json.load(open(path))
    if d.get("units") != "meters" or d.get("transform_convention") != "T_camera_from_left":
        raise ValueError("Calibration must declare units=meters and transform_convention=T_camera_from_left")
    cameras = [Camera(**d[k]) for k in ("left", "right")]
    if not np.allclose(cameras[0].T_camera_from_left, np.eye(4)):
        raise ValueError("Left camera transform must be identity")
    if np.linalg.norm(cameras[1].T_camera_from_left[:3, 3]) < 1e-6:
        raise ValueError("Stereo baseline is zero")
    return cameras
