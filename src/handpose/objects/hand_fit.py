"""MANO fitting to predicted stereo observations with optional object constraints.

Reference hand labels are not accepted. Object transforms are external inputs;
callers must label reference-object-pose diagnostics explicitly.
"""
import numpy as np
import cv2

ORDER=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]
TIPS=[744,320,443,554,671]
C=np.array([[0,-1,0],[1,0,0],[0,0,1.]])


def fit_hand(obs,parameters,K,extrinsics,objects,fields,mode='hand_only',weight=1.,steps=160):
    import torch
    import torch.nn.functional as F
    import smplx
    torch.set_num_threads(4)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    tensor=lambda a:torch.as_tensor(a,dtype=torch.float32,device=device)
    N=len(obs['timestamps']);result=np.where(obs['validity'][...,None],obs['joints_3d'],np.nan).copy()
    eligible=np.array([bool(p) for p in parameters])&(obs['validity'].sum(-1)>=6)
    indices=np.flatnonzero(eligible)
    used=np.zeros((N,5),np.float32);vertices_out=np.full((N,778,3),np.nan,np.float32)
    if not len(indices):return result,vertices_out,used,eligible
    beta=np.median([parameters[i][0]['betas'] for i in indices],axis=0).reshape(1,10)
    model=smplx.MANO('checkpoints/mano_converted',is_rhand=True,use_pca=False,flat_hand_mean=True).to(device)
    field_cache={}
    for oid,(grid,origin,pitch) in fields.items():field_cache[oid]=(tensor(grid)[None,None],tensor(origin),pitch,tensor(np.array(grid.shape)-1))
    for offset in range(0,len(indices),24):
        ids=indices[offset:offset+24];B=len(ids)
        initial=[]
        for i in ids:
            p=parameters[i][0];rot=np.concatenate([np.asarray(p['global_orient']),np.asarray(p['hand_pose'])]);rot[0]=C.T@rot[0]
            initial.append(np.concatenate([cv2.Rodrigues(r)[0].ravel() for r in rot]))
        reference=tensor(np.array(initial));angles=reference.clone().requires_grad_(True)
        def forward():
            out=model(global_orient=angles[:,:3],hand_pose=angles[:,3:],betas=tensor(beta).expand(B,-1))
            joints=torch.cat([out.joints[:,:16],out.vertices[:,TIPS]],1)[:,ORDER]
            return joints,out.vertices
        with torch.no_grad():template,_=forward()
        valid=tensor(obs['validity'][ids]);target=tensor(np.nan_to_num(obs['joints_3d'][ids]))
        translation=(((target-template)*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]).detach().requires_grad_(True)
        uv=tensor(np.stack([obs['joints_2d_left'][ids],obs['joints_2d_right'][ids]],1))
        # Stored WiLoR pixels and coordinates are native 640 pinhole.
        uv_valid=torch.isfinite(uv).all(-1)&valid[:,None,:].bool();uv=torch.nan_to_num(uv)
        cameras=tensor(np.stack([np.repeat(np.eye(4)[None],B,0),extrinsics[ids]],1));intrinsic=tensor(K)
        visibility=np.ones((B,2,21),np.float32)
        object_groups={}
        for b,i in enumerate(ids):
            obj=objects[i]
            if obj is None:continue
            if mode!='hand_only':
                for v in range(2):
                    pix=(obs['joints_2d_left'] if v==0 else obs['joints_2d_right'])[i]
                    pix=np.nan_to_num((pix+.5)/4-.5,nan=-1).round().astype(int)
                    good=(pix>=0).all(-1)&(pix<160).all(-1)
                    occluded=np.zeros(21,bool);occluded[good]=obj['masks'][v,pix[good,1],pix[good,0]]>0
                    visibility[b,v,occluded]=.25
            if obj['accepted']:object_groups.setdefault(obj['id'],[]).append((b,obj))
        def signed_distance(vertices):
            values=[]
            for oid,items in object_groups.items():
                batches=[b for b,o in items];Ts=tensor(np.array([o['T'] for b,o in items]));R=Ts[:,:3,:3];t=Ts[:,:3,3]
                local=torch.bmm(vertices[batches]-t[:,None],R)
                grid,origin,pitch,shape=field_cache[oid]
                coords=(local-origin)/(pitch*shape)*2-1
                query=coords[:,:,[2,1,0]][:,None,None]
                sdf=F.grid_sample(grid.expand(len(items),-1,-1,-1,-1),query,align_corners=True,padding_mode='border').reshape(len(items),-1)
                # Outside padded grid, exact distance is unknown but cannot penetrate.
                outside=torch.relu(torch.abs(coords)-1).amax(-1)>0
                sdf=torch.where(outside,torch.clamp(sdf,min=.012),sdf)
                values.append((batches,sdf,items))
            return values
        contact_weights={}
        if mode=='contact':
            with torch.no_grad():
                _,mesh=forward();mesh=mesh+translation[:,None]
                for batches,sdf,items in signed_distance(mesh):
                    dist=sdf[:,TIPS].cpu().numpy()
                    for row,(b,obj) in enumerate(items):
                        evidence=[]
                        for tip in [4,8,12,16,20]:
                            # Evidence in both visible masks (within 3 px of their boundary).
                            pixel_support=[]
                            for v in range(2):
                                p=(obs['joints_2d_left'] if v==0 else obs['joints_2d_right'])[ids[b],tip]
                                p=np.nan_to_num((p+.5)/4-.5,nan=-100).round().astype(int)
                                pixel_support.append(bool((p>=0).all() and (p<160).all() and obj['boundary_distance'][v,p[1],p[0]]<3))
                            evidence.append(all(pixel_support))
                        cw=np.exp(-.5*(dist[row]/.008)**2)*np.array(evidence)*(np.abs(dist[row])<.015)
                        contact_weights[b]=tensor(cw);used[ids[b]]=cw
        optim=torch.optim.Adam([angles,translation],lr=.002)
        for step in range(steps):
            optim.zero_grad();joints,mesh=forward();joints=joints+translation[:,None];mesh=mesh+translation[:,None]
            xyz=torch.einsum('bvij,bkj->bvki',cameras[:,:,:3,:3],joints)+cameras[:,:,:3,3][:,:,None]
            proj=xyz@intrinsic.T;proj=proj[...,:2]/proj[...,2:].clamp(min=.05)
            reproj=F.smooth_l1_loss(proj,uv,reduction='none',beta=4).sum(-1)
            loss=(reproj*uv_valid*tensor(visibility)).sum()/uv_valid.sum().clamp(min=1)
            loss+=10000*((joints-target).square().sum(-1)*valid).sum()/valid.sum()
            loss+=.1*(angles-reference).square().mean()
            if mode in ['collision','contact']:
                for batches,sdf,items in signed_distance(mesh):
                    # Soft 2 mm penetration allowance; meshes voxelized at 3 mm.
                    loss+=weight*100000*torch.relu(-sdf-.002).square().sum()/(B*778)
                    if mode=='contact':
                        for row,b in enumerate(batches):
                            if b in contact_weights:loss+=weight*20000*((sdf[row,TIPS]-.001).square()*contact_weights[b]).sum()/max(B,1)
            if not torch.isfinite(loss):raise ValueError('Nonfinite hand-object objective')
            loss.backward();optim.step()
        with torch.no_grad():
            joints,mesh=forward();result[ids]=(joints+translation[:,None]).cpu().numpy();vertices_out[ids]=(mesh+translation[:,None]).cpu().numpy()
    return result,vertices_out,used,eligible
