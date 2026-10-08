"""Download a small deterministic public-training subset, never challenge test GT."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

BASE='https://huggingface.co/datasets/bop-benchmark/hot3d/resolve/main/'

def download(url,path):
    if not path.exists():
        temp=path.with_suffix(path.suffix+'.partial')
        urllib.request.urlretrieve(url,temp)
        temp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',default='data/hot3d'); p.add_argument('--clips-per-participant',type=int,default=1); p.add_argument('--expanded',action='store_true')
    a=p.parse_args(); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    for name in ('clip_definitions.json','clip_splits.json'): download(BASE+name,out/name)
    definitions=json.loads((out/'clip_definitions.json').read_text()); splits=json.loads((out/'clip_splits.json').read_text())
    train=set(str(int(x)) for x in splits['train']['Quest3'])
    participants={}
    for key,d in definitions.items():
        if key in train and d['device']=='Quest3':
            participants.setdefault(d['sequence_id'].split('_')[0],[]).append(key)
    selected=sorted(participants)[:7 if a.expanded else 3]
    if len(selected)<3: raise RuntimeError('Need three separate training participants')
    manifest=[]
    for split,participant in zip(('smoke','development','held_out','development','held_out','development','held_out'),selected):
        for key in participants[participant][:a.clips_per_participant]:
            rel=f'train_quest3/clip-{int(key):06d}.tar'; path=out/Path(rel).name
            sha=download(BASE+rel,path)
            manifest.append(dict(split=split,participant=participant,sequence=definitions[key]['sequence_id'],path=str(path),sha256=sha,source_url=BASE+rel,max_frames=30 if split=='smoke' else 150))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))
if __name__=='__main__': main()
