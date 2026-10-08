"""Record runtime versions, source revisions and cryptographic asset identities."""
import hashlib,json,subprocess
from pathlib import Path
root=Path.cwd();out=root/'configs/environments';out.mkdir(parents=True,exist_ok=True)
for env in ('.venv','.venv-ace','.venv-poem'):
    python=root/env/'bin/python'
    if not python.exists():continue
    freeze=subprocess.check_output([str(python),'-m','pip','freeze'],text=True)
    lines=[]
    for line in freeze.splitlines():
        if line.startswith('#') or line.startswith('-e ') or ' @ file:' in line:continue
        lines.append(line)
    (out/(env.strip('.')+'.txt')).write_text('\n'.join(lines)+'\n')
sources={}
for path in (root/'third_party').iterdir():
    if not (path/'.git').exists():continue
    sources[path.name]={'commit':subprocess.check_output(['git','-c','safe.directory='+str(path),'-C',str(path),'rev-parse','HEAD'],text=True).strip(),'url':subprocess.check_output(['git','-c','safe.directory='+str(path),'-C',str(path),'remote','get-url','origin'],text=True).strip(),'dirty':bool(subprocess.check_output(['git','-c','safe.directory='+str(path),'-C',str(path),'status','--porcelain'],text=True).strip())}
(root/'configs/executed_sources.json').write_text(json.dumps(sources,indent=2))
records=[]
paths=list((root/'checkpoints').rglob('*'))+list((root/'third_party/ace/checkpoints').glob('*'))+list((root/'third_party/umetrack').rglob('*.torch'))+list(Path('/root/.cache/rtmlib/hub/checkpoints').glob('*.onnx'))
for p in paths:
    if not p.is_file() or p.suffix not in ('.pth','.pt','.tar','.onnx','.torch','.pkl'):continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    records.append({'file':str(p.relative_to(root)) if p.is_relative_to(root) else 'rtmlib-cache/'+p.name,'bytes':p.stat().st_size,'sha256':h.hexdigest()})
(root/'configs/checkpoints.lock.json').write_text(json.dumps(records,indent=2));print('Captured',len(sources),'sources and',len(records),'weights')
