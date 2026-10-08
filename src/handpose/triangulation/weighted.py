import numpy as np
from handpose.geometry.camera import epipolar_distance, transform


def triangulate(left, right, uv_left, uv_right, confidence=None, max_epipolar_px=5., min_ray_angle_deg=.5, max_reprojection_px=8.):
    a, b = np.asarray(uv_left, float), np.asarray(uv_right, float)
    n = len(a)
    c = np.ones((n, 2)) if confidence is None else np.asarray(confidence, float)
    if c.shape != (n, 2):
        raise ValueError("confidence must be (N,2)")
    X = np.full((n, 3), np.nan)
    valid = np.isfinite(a).all(1) & np.isfinite(b).all(1) & np.isfinite(c).all(1) & (c > 0).all(1)
    if not valid.any():
        return X, valid
    ids = np.flatnonzero(valid)
    an, bn = left.normalized(a[ids]), right.normalized(b[ids])
    au = (np.c_[an, np.ones(len(ids))] @ left.K.T)[:, :2]
    bu = (np.c_[bn, np.ones(len(ids))] @ right.K.T)[:, :2]
    epi = epipolar_distance(left, right, au, bu)
    P, Q = left.T_camera_from_left[:3], right.T_camera_from_left[:3]
    for k, j in enumerate(ids):
        ra = P[:, :3].T @ np.r_[an[k], 1.]
        rb = Q[:, :3].T @ np.r_[bn[k], 1.]
        angle = np.degrees(np.arccos(np.clip(np.dot(ra, rb) / np.linalg.norm(ra) / np.linalg.norm(rb), -1, 1)))
        if epi[k] > max_epipolar_px or angle < min_ray_angle_deg:
            valid[j] = False
            continue
        A = np.array([an[k, 0]*P[2]-P[0], an[k, 1]*P[2]-P[1], bn[k, 0]*Q[2]-Q[0], bn[k, 1]*Q[2]-Q[1]])
        A *= np.repeat(np.sqrt(c[j]), 2)[:, None]
        _, _, vh = np.linalg.svd(A)
        if abs(vh[-1, 3]) < 1e-12:
            valid[j] = False
            continue
        p = vh[-1, :3] / vh[-1, 3]
        ok = all(transform(p[None], cam.T_camera_from_left)[0, 2] > 0 for cam in (left, right))
        if ok:
            ok = max(np.linalg.norm(left.project(p[None])[0]-a[j]), np.linalg.norm(right.project(p[None])[0]-b[j])) <= max_reprojection_px
        valid[j] = ok
        if ok:
            X[j] = p
    return X, valid
