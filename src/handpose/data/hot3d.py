"""Public HOT3D clip reader. GT is isolated from detector inputs.

Use official toolkit for FISHEYE624 and UmeTrack skinning. Canonical GT lacks
thumb CMC. Wrist is excluded by default because toolkit warns its UmeTrack
location differs from MANO's anatomical wrist. No fabricated 21-joint GT.
"""
import json
import tarfile
import cv2
import numpy as np
from handpose.geometry.camera import Camera, transform

# Toolkit indices: tips 0..4, wrist 5, thumb MCP/IP 6/7, other fingers 8..19.
TO_21 = {0:5, 2:6, 3:7, 4:0, 5:8, 6:9, 7:10, 8:1, 9:11, 10:12, 11:13, 12:2, 13:14, 14:15, 15:16, 16:3, 17:17, 18:18, 19:19, 20:4}


def pinhole_view(source, size=640, focal=320.):
    from hand_tracking_toolkit.camera import PinholePlaneCameraModel
    return PinholePlaneCameraModel(size,size,(focal,focal),((size-1)/2,)*2,[],source.T_world_from_eye)


def remap(source,destination):
    w,h=destination.width,destination.height
    x,y=np.meshgrid(np.arange(w),np.arange(h)); uv=np.c_[x.ravel(),y.ravel()]
    rays=destination.window_to_eye(uv)
    # Identical optical pose; rotation-only undistortion requires no scene depth.
    source_uv=source.eye_to_window(rays).astype(np.float32)
    return source_uv[:,0].reshape(h,w),source_uv[:,1].reshape(h,w)


def iter_clip(path,max_frames=None,size=640,focal=320.):
    import torch
    from hand_tracking_toolkit.camera import from_json
    from hand_tracking_toolkit.hand_models.umetrack_hand_model import from_json as model_from_json, UmeTrackHandPose, forward_kinematics
    from hand_tracking_toolkit.math_utils import quat_trans_to_matrix
    with tarfile.open(path) as tar:
        def read(name): return json.load(tar.extractfile(name))
        names=set(tar.getnames())
        if '__hand_shapes.json__' not in names: raise ValueError('Missing public training hand shapes')
        shape=model_from_json(read('__hand_shapes.json__')['umetrack'])
        frames=sorted(n.removesuffix('.info.json') for n in names if n.endswith('.info.json'))
        maps=None
        for key in frames[:max_frames]:
            info=read(key+'.info.json'); raw=read(key+'.cameras.json')
            streams=('1201-1','1201-2')
            ts=[info['image_timestamps_ns'][s] for s in streams]
            if abs(ts[0]-ts[1])>1000000: raise ValueError('Stereo timestamps differ by >1 ms')
            sources=[from_json(raw[s]) for s in streams]
            dest=[pinhole_view(s,size,focal) for s in sources]
            if maps is None: maps=[remap(s,d) for s,d in zip(sources,dest)]
            images=[]
            for stream,m in zip(streams,maps):
                im=cv2.imdecode(np.frombuffer(tar.extractfile(f'{key}.image_{stream}.jpg').read(),np.uint8),cv2.IMREAD_COLOR)
                images.append(cv2.remap(im,*m,cv2.INTER_LINEAR))
            T_left_from_world=np.linalg.inv(dest[0].T_world_from_eye)
            K=dest[0].uv_to_window_matrix()
            cams=[Camera(K),Camera(dest[1].uv_to_window_matrix(),np.linalg.inv(dest[1].T_world_from_eye)@dest[0].T_world_from_eye)]
            hands=read(key+'.hands.json')
            gt=np.full((21,3),np.nan); visibility=None
            if hands.get('right') and hands['right'].get('umetrack_pose'):
                d=hands['right']['umetrack_pose']; t=d['T_world_from_wrist']
                T=quat_trans_to_matrix(*t['quaternion_wxyz'],*t['translation_xyz'])
                pose=UmeTrackHandPose(torch.tensor(1),torch.tensor(d['joint_angles'],dtype=torch.float32),torch.tensor(T,dtype=torch.float32))
                landmarks=forward_kinematics(pose,shape,requires_mesh=False)[0].numpy()
                for dst,src in TO_21.items(): gt[dst]=landmarks[src]
                gt=transform(gt,T_left_from_world)
                gt[0]=np.nan  # Anatomical wrist convention is not comparable.
                visibility=hands['right'].get('visibilities_modeled')
            yield dict(frame_id=key,timestamp_ns=int(ts[0]),left=images[0],right=images[1],cameras=cams,ground_truth=gt,visibility=visibility,info=info,T_world_from_left=dest[0].T_world_from_eye)


def mano_ground_truth(path,max_frames,model_path='checkpoints/mano_converted'):
    """21 MANO anatomical joints in left pinhole optical frame (separate GT track)."""
    import torch,smplx
    from hand_tracking_toolkit.camera import from_json
    model=smplx.MANO(model_path,is_rhand=True,use_pca=True,num_pca_comps=15,flat_hand_mean=False)
    order=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
    with tarfile.open(path) as tar:
        read=lambda n:json.load(tar.extractfile(n))
        shape=read('__hand_shapes.json__')['mano']
        keys=sorted(n.removesuffix('.info.json') for n in tar.getnames() if n.endswith('.info.json'))[:max_frames]
        all_gt=[]
        for key in keys:
            hand=read(key+'.hands.json').get('right',{});gt=np.full((21,3),np.nan)
            if hand and hand.get('mano_pose'):
                d=hand['mano_pose'];xf=torch.tensor(d['wrist_xform'],dtype=torch.float32)[None]
                with torch.no_grad():o=model(betas=torch.tensor(shape,dtype=torch.float32)[None],hand_pose=torch.tensor(d['thetas'],dtype=torch.float32)[None],global_orient=xf[:,:3],transl=xf[:,3:])
                joints=torch.cat([o.joints[:,:16],o.vertices[:,[744,320,443,554,671]]],dim=1)[0,order].numpy()
                cam=from_json(read(key+'.cameras.json')['1201-1']);gt=transform(joints,np.linalg.inv(cam.T_world_from_eye))
            all_gt.append(gt)
    return np.stack(all_gt)
