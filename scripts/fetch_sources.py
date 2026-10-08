"""Obtain locked source revisions without overwriting existing checkouts."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('names',nargs='+');a=p.parse_args();lock=json.load(open('configs/sources.lock.json'))
for name in a.names:
    entry=lock[name];path=Path('third_party')/name
    if path.exists():
        current=subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()
        if current!=entry['commit']:raise ValueError(f'{path}: existing commit differs from lock; preserve or move the checkout before fetching')
        print(name,'already at',current);continue
    path.parent.mkdir(exist_ok=True)
    subprocess.run(['git','clone','--no-checkout',entry['url'],str(path)],check=True)
    subprocess.run(['git','-C',str(path),'checkout','--detach',entry['commit']],check=True)
    print(name,entry['commit'])
