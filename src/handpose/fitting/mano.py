"""Offline differentiable MANO fitting. Uses predictions only, never GT.

Shared shape across the sequence; independent poses; robust 3D/reprojection
losses. This is an experimental ablation, not a claim of improved accuracy.
"""
import numpy as np


def fit_sequence(xyz,valid,uv_left,uv_right,K_left,K_right,T_right_from_left,model_path,steps=200,lr=.02,device='cuda',dense_points=None,depth_weight=.1):
    import torch
    import smplx
    xyz=np.asarray(xyz); valid=np.asarray(valid)&np.isfinite(xyz).all(-1)
    eligible=valid.sum(1)>=6
    result=np.full_like(xyz,np.nan)
    if not eligible.any(): return result,{'eligible_frames':eligible.tolist()}
    tensor=lambda x:torch.tensor(x,dtype=torch.float32,device=device)
    indices=np.flatnonzero(eligible); n=len(indices)
    model=smplx.MANO(model_path,is_rhand=True,use_pca=False,flat_hand_mean=False,batch_size=n).to(device)
    pose=torch.zeros((n,45),device=device,requires_grad=True)
    orient=torch.zeros((n,3),device=device,requires_grad=True)
    beta=torch.zeros((1,10),device=device,requires_grad=True)
    target=tensor(np.nan_to_num(xyz[eligible])); mask=tensor(valid[eligible])
    translation=target.sum(1)/mask.sum(1)[:,None]
    translation=translation.detach().requires_grad_()
    uv=[tensor(np.nan_to_num(x[eligible])) for x in (uv_left,uv_right)]
    uv_mask=[tensor(np.isfinite(x[eligible]).all(-1)) for x in (uv_left,uv_right)]
    Ks=[tensor(np.broadcast_to(k,(len(xyz),3,3))[eligible]) for k in (K_left,K_right)]
    Ts=tensor(np.broadcast_to(T_right_from_left,(len(xyz),4,4))[eligible])
    depth_target=depth_mask=None
    if dense_points is not None:
        depth=np.asarray(dense_points)[eligible]
        depth_mask=tensor(np.isfinite(depth).all(-1))
        depth_target=tensor(np.nan_to_num(depth))
    optimizer=torch.optim.Adam([pose,orient,beta,translation],lr=lr)
    # MANO regressor ordering + surface tips -> MediaPipe anatomical ordering.
    order=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
    def forward():
        output=model(hand_pose=pose,global_orient=orient,betas=beta.expand(n,-1),transl=translation)
        joints=torch.cat([output.joints[:,:16],output.vertices[:,[744,320,443,554,671]]],dim=1)[:,order]
        return joints,output
    def robust(error,delta):return delta**2*(torch.sqrt(1+(error/delta)**2)-1)
    history=[]
    for step in range(steps):
        optimizer.zero_grad(); joints,mesh=forward()
        loss3=(robust(joints-target,.01).sum(-1)*mask).sum()/mask.sum().clamp_min(1)
        reproj=0
        for view in range(2):
            camera=joints if view==0 else joints@Ts[:,:3,:3].transpose(1,2)+Ts[:,:3,3][:,None]
            h=camera@Ks[view].transpose(1,2); pixels=h[...,:2]/h[...,2:].clamp_min(.05)
            w=uv_mask[view]*mask
            reproj+=(robust((pixels-uv[view])/500,.01).sum(-1)*w).sum()/w.sum().clamp_min(1)
        loss_depth=0
        if depth_target is not None:
            distance=torch.cdist(mesh.vertices,depth_target).min(dim=1).values
            loss_depth=(robust(distance,.005)*depth_mask).sum()/depth_mask.sum().clamp_min(1)
        loss=loss3+reproj*.1+depth_weight*loss_depth+1e-5*pose.square().mean()+1e-5*beta.square().mean()
        loss.backward(); optimizer.step(); history.append(float(loss.detach()))
    joints,output=forward(); result[eligible]=joints.detach().cpu().numpy()
    return result,dict(eligible_frames=eligible.tolist(),loss=history,betas=beta.detach().cpu().numpy().tolist(),pose=pose.detach().cpu().numpy().tolist(),orientation=orient.detach().cpu().numpy().tolist(),translation=translation.detach().cpu().numpy().tolist())
