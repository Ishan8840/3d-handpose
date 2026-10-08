"""Export mask-conditioned inputs and separate reference object poses for HOT3D.
Reference hand annotations never enter the input archive. Amodal masks are excluded.
"""
import json, tarfile, hashlib, sys
from pathlib import Path
import numpy as np
import cv2
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path('third_party/hand_tracking_toolkit').resolve()))
from hand_tracking_toolkit.camera import from_json
from handpose.data.hot3d import pinhole_view, remap
from handpose.objects.geometry import decode_mask
from huggingface_hub import HfApi, hf_hub_download

out=Path('outputs/object-input'); out.mkdir(parents=True,exist_ok=True)
all_ids=set(); inventory=[]
for entry in json.load(open('configs/datasets/hot3d.json')):
    if entry['split']=='smoke': continue
    path=Path(entry['path']); name=path.stem
    with tarfile.open(path) as tar:
        read=lambda p:json.load(tar.extractfile(p))
        keys=sorted(n[:-10] for n in tar.getnames() if n.endswith('.info.json'))
        objects=[read(k+'.objects.json') for k in keys]
        ids=sorted({int(i) for d in objects for i in d}); all_ids.update(ids)
        masks=np.zeros((len(keys),len(ids),2,160,160),np.uint8)
        reference=np.full((len(keys),len(ids),4,4),np.nan)
        relative=[]; timestamps=[]; maps=None
        for f,key in enumerate(keys):
            raw=read(key+'.cameras.json'); sources=[from_json(raw[s]) for s in ['1201-1','1201-2']]
            dest=[pinhole_view(s,160,80.) for s in sources]
            if maps is None: maps=[remap(s,d) for s,d in zip(sources,dest)]
            relative.append(np.linalg.inv(sources[1].T_world_from_eye)@sources[0].T_world_from_eye)
            info=read(key+'.info.json');timestamps.append(info['image_timestamps_ns']['1201-1'])
            for j,oid in enumerate(ids):
                instances=objects[f].get(str(oid),[])
                if not instances: continue
                if len(instances)!=1: raise ValueError('Multiple same-ID objects require instance matching')
                obj=instances[0]
                for v,stream in enumerate(['1201-1','1201-2']):
                    encoded=obj.get('masks_modal',{}).get(stream)
                    if encoded: masks[f,j,v]=cv2.remap(decode_mask(encoded).astype(np.uint8),*maps[v],cv2.INTER_NEAREST)
                p=obj['T_world_from_object']; T=np.eye(4)
                q=p['quaternion_wxyz']; T[:3,:3]=Rotation.from_quat([*q[1:],q[0]]).as_matrix();T[:3,3]=p['translation_xyz']
                reference[f,j]=np.linalg.inv(sources[0].T_world_from_eye)@T
        np.savez_compressed(out/f'{name}.npz',masks=masks,object_ids=ids,T_right_from_left=relative,
                            K=dest[0].uv_to_window_matrix(),timestamps=timestamps,frame_ids=keys)
        np.savez_compressed(out/f'{name}_reference.npz',T_left_from_object=reference)
        row=dict(clip=name,split=entry['split'],participant=entry['participant'],frames=len(keys),objects=ids,
                 paired_mask_frames={str(i):int((masks[:,j].sum((-1,-2))>20).all(-1).sum()) for j,i in enumerate(ids)})
        inventory.append(row);print(row,flush=True)
provenance_path=out/'provenance.json'
revision=json.load(open(provenance_path))['dataset_revision'] if provenance_path.exists() else HfApi().dataset_info('bop-benchmark/hot3d').sha
modeldir=Path('data/hot3d/object_models');modeldir.mkdir(parents=True,exist_ok=True)
files={}
for fname in ['models_info.json']+[f'obj_{i:06d}.glb' for i in sorted(all_ids)]:
    src=Path(hf_hub_download('bop-benchmark/hot3d','object_models/'+fname,repo_type='dataset',revision=revision))
    dst=modeldir/fname;dst.write_bytes(src.read_bytes());files[fname]=hashlib.sha256(dst.read_bytes()).hexdigest()
(out/'provenance.json').write_text(json.dumps(dict(dataset_revision=revision,meshes=files,clips=inventory,
    mask_source='HOT3D supplied SAM2 masks_modal; prompting provenance not established; mask-conditioned evaluation',
    coordinate_frame='native left optical, meters',reference_pose_file_suffix='_reference.npz'),indent=2))
