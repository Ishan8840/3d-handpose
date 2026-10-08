import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--split',default='development');p.add_argument('--observations',required=True);p.add_argument('--output',required=True);a=p.parse_args()
for e in json.load(open('data/hot3d/manifest.json')):
    if e['split']!=a.split:continue
    name=Path(e['path']).stem;out=Path(a.output)/name;out.mkdir(parents=True,exist_ok=True)
    with (out/'run.log').open('w') as log:
        subprocess.run(['.venv-poem/bin/python','scripts/benchmark_poem.py','--clip',e['path'],'--observations',str(Path(a.observations)/name/'predictions.npz'),'--output',str(out),'--frames',str(e['max_frames'])],check=True,stdout=log,stderr=subprocess.STDOUT)
    print('Completed',name,flush=True)
