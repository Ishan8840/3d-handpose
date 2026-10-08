"""Fetch exactly the existing public HOT3D clips, with recorded hash checks."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request


def fetch(entry):
    p=Path(entry['path']); p.parent.mkdir(parents=True,exist_ok=True)
    if not p.exists():
        temporary=p.with_suffix('.partial')
        urllib.request.urlretrieve(entry['source_url'],temporary)
        temporary.replace(p)
    if hashlib.sha256(p.read_bytes()).hexdigest()!=entry['sha256']:
        raise RuntimeError(f'Hash mismatch: {p}')
    print('Verified',p,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument('--split',default='held_out')
    args=parser.parse_args()
    entries=json.loads(Path('data/hot3d/manifest.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,[e for e in entries if e['split']==args.split]))
