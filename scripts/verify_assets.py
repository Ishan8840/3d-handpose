"""Verify local checkpoint bytes against the executed asset lock."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--lock',default='configs/checkpoints.lock.json');p.add_argument('--rtm-cache',default=str(Path.home()/'.cache/rtmlib/hub/checkpoints'));p.add_argument('--contains',default='');a=p.parse_args();failed=[];checked=0
for item in json.load(open(a.lock)):
    name=item['file']
    if a.contains not in name:continue
    path=Path(a.rtm_cache)/Path(name).name if name.startswith('rtmlib-cache/') else Path(name)
    if not path.exists():failed.append({'file':str(path),'status':'missing'});continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    if h.hexdigest()!=item['sha256']:failed.append({'file':str(path),'status':'hash_mismatch'})
    else:checked+=1
print(json.dumps({'verified':checked,'failures':failed},indent=2))
if failed:raise SystemExit(1)
if not checked:raise SystemExit('No matching assets')
