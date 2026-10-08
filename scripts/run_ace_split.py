"""Execute both calibrated views with official ACE inference; retain each log."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--split',default='development');a=p.parse_args();base=Path.cwd()
for entry in json.load(open('data/hot3d/manifest.json')):
    if entry['split']!=a.split:continue
    name=Path(entry['path']).stem;video=base/'outputs'/f'ace-input-{name}';output=base/'outputs'/f'ace-upright-{a.split}'/name
    subprocess.run([str(base/'.venv/bin/python'),'scripts/export_stereo_example.py','--clip',entry['path'],'--frames',str(entry['max_frames']),'--upright','--output',str(video)],check=True)
    output.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    for side in ('left','right'):
        with (output/f'{side}.log').open('w') as log:
            subprocess.run([str(base/'.venv-ace/bin/python'),'infer_video.py','--video',str(video/f'{side}.mp4'),'--camera',str(video/'cam.json'),'--opt','options/ace_ego_hand_k.yml','--ckpt','checkpoints/ace_ego_hand_k.pt','--out',str(output/side),'--encode_w','640'],cwd=base/'third_party/ace',stdout=log,stderr=subprocess.STDOUT,check=True)
    with (output/'evaluation.log').open('w') as log:
        subprocess.run([str(base/'.venv/bin/python'),'scripts/evaluate_ace.py','--left',str(output/'left/left.pkl'),'--right',str(output/'right/right.pkl'),'--clip',entry['path'],'--frames',str(entry['max_frames']),'--upright','--output',str(output)],stdout=log,stderr=subprocess.STDOUT,check=True)
    (output/'runtime.json').write_text(json.dumps({'total_seconds_including_model_load':time.perf_counter()-start,'frames':entry['max_frames']}))
    print('completed',name,flush=True)
