"""Diagnostic mask/mesh alignment; references are displayed only for evaluation."""
from pathlib import Path
import numpy as np,cv2
from handpose.objects.geometry import load_metric_mesh
out=Path('reports/object_aware');out.mkdir(parents=True,exist_ok=True)
images=[]
for name in ['clip-000175','clip-000648','clip-001035','clip-001168']:
 d=np.load(f'outputs/object-input/{name}.npz');p=np.load(f'outputs/object-pose/{name}.npz');ref=np.load(f'outputs/object-input/{name}_reference.npz')['T_left_from_object']
 eligible=np.flatnonzero(p['selected']>=0)
 if not len(eligible):continue
 f=int(eligible[len(eligible)//2]);j=p['selected'][f];mesh=load_metric_mesh(f'data/hot3d/object_models/obj_{int(d["object_ids"][j]):06d}.glb');v=mesh.convex_hull.vertices
 panes=[]
 for label,T in [('reference',ref[f,j]),('estimated',p['T_left_from_object'][f])]:
  im=np.repeat((d['masks'][f,j,0]*140)[:,:,None],3,axis=2)
  xyz=v@T[:3,:3].T+T[:3,3];uv=xyz@d['K'].T;uv=uv[:,:2]/uv[:,2:]
  cv2.polylines(im,[cv2.convexHull(uv.astype(np.float32)).astype(int)],True,(0,255,0),1)
  im=cv2.resize(im,(320,320),interpolation=cv2.INTER_NEAREST);cv2.putText(im,f'{name} f{f} {label}',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,255,255),1);panes.append(im)
 images.append(np.hstack(panes))
cv2.imwrite(str(out/'object_mask_alignment.png'),np.vstack(images))
