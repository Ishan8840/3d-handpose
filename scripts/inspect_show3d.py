import json
from pathlib import Path
m=json.load(open('data/show3d/manifest.json'))[0]
for f in [Path(m['path'])/'camera_calibration/headset0.json',Path(m['path'])/'metadata/frame_info.json',Path(m['annotations'])]:
    x=json.load(open(f));print(f,type(x),str(x)[:2300])
