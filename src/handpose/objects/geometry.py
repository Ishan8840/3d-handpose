"""Mesh and HOT3D mask utilities with explicit metric conventions."""
import numpy as np
from scipy.ndimage import distance_transform_edt


def decode_mask(data):
    runs=np.asarray(data['rle'],dtype=np.int64)
    if len(runs)%2: raise ValueError('Malformed start/length RLE')
    size=data['height']*data['width']; mask=np.zeros(size,bool)
    for start,length in runs.reshape(-1,2):
        if start<1 or length<0 or start-1+length>size: raise ValueError('Invalid RLE bounds')
        mask[start-1:start-1+length]=True
    return mask.reshape(data['height'],data['width'])


def load_metric_mesh(path):
    import trimesh
    scene=trimesh.load(path,force='scene',process=True,skip_materials=True)
    mesh=scene.to_geometry()  # Applies GLB node transforms; official models are meters.
    if not .005<float(np.max(mesh.extents))<2: raise ValueError('Unexpected metric mesh dimensions')
    return mesh


def voxel_sdf(mesh,pitch=.003):
    """Filled 3 mm voxel signed distance: positive outside, negative inside.

    Discretized solid approximation, not exact triangle-level penetration distance.
    """
    voxel=mesh.voxelized(pitch).fill(); inside=np.pad(voxel.matrix,3)
    origin=voxel.transform[:3,3]-3*pitch
    distance=(distance_transform_edt(~inside)-distance_transform_edt(inside))*pitch
    return distance.astype(np.float32),origin.astype(np.float32),pitch
