"""Controlled 480px/81-frame ACE K-given run using images/calibration only.

No hands.json or hand-shape annotation is opened by this inference script.
"""
import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path
import cv2
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--size',type=int,default=480)
    p.add_argument('--frames',type=int,default=81); a=p.parse_args()
    root=Path.cwd(); sys.path[:0]=[str(root/'src'),str(root/'third_party/hand_tracking_toolkit')]
    from hand_tracking_toolkit.camera import from_json
    from handpose.data.hot3d import pinhole_view, remap
    checkpoint=root/'third_party/ace/checkpoints/ace_ego_hand_k.pt'
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if sha!='33f328624ca532942b12d10c59b648bb9fc0801bcdfe59f35ba92645e51fc3ad':
        raise RuntimeError('Checkpoint differs from previous pinned official K-given checkpoint')
    destination=root/f'outputs/ace-audit-{a.size}-{a.frames}'
    manifest=json.loads((root/'data/hot3d/manifest.json').read_text())
    for entry in manifest:
        if entry['split']!='held_out': continue
        name=Path(entry['path']).stem; out=destination/name; out.mkdir(parents=True,exist_ok=True)
        video=out/'input.mp4'; writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),30,(a.size,a.size))
        if not writer.isOpened(): raise RuntimeError('Video writer failed')
        with tarfile.open(entry['path']) as tar:
            keys=sorted(n[:-10] for n in tar.getnames() if n.endswith('.info.json'))[:a.frames]
            for key in keys:
                camera=from_json(json.load(tar.extractfile(key+'.cameras.json'))['1201-1'])
                target=pinhole_view(camera,a.size,a.size/2)
                maps=remap(camera,target)
                rgb=cv2.imdecode(np.frombuffer(tar.extractfile(key+'.image_1201-1.jpg').read(),np.uint8),cv2.IMREAD_COLOR)
                writer.write(cv2.rotate(cv2.remap(rgb,*maps,cv2.INTER_LINEAR),cv2.ROTATE_90_CLOCKWISE))
        writer.release()
        camera_file=out/'camera.json'
        camera_file.write_text(json.dumps({'image_width':a.size,'image_height':a.size,'frames':[{'intrinsics':{'fx':a.size/2,'fy':a.size/2,'cx':(a.size-1)/2,'cy':(a.size-1)/2}}]}))
        command=[sys.executable,'infer_video.py','--video',str(video),'--camera',str(camera_file),'--opt','options/ace_ego_hand_k.yml','--ckpt',str(checkpoint),'--out',str(out/'left'),'--encode_w',str(a.size),'--mode','tiled']
        start=time.perf_counter()
        with (out/'inference.log').open('w') as log:
            subprocess.run(command,cwd=root/'third_party/ace',stdout=log,stderr=subprocess.STDOUT,check=True)
        # Preserve official dump unchanged, provide scorer's stable filename.
        import shutil
        shutil.copyfile(out/'left/input.pkl',out/'left/left.pkl')
        (out/'run.json').write_text(json.dumps(dict(command=command,checkpoint_sha256=sha,frames=keys,size=a.size,seconds=time.perf_counter()-start,ground_truth_used=False),indent=2))
        print('Completed',name,flush=True)


if __name__=='__main__': main()
