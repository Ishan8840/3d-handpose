"""Create portable provenance manifest of local research assets (no credential reads)."""
import hashlib,json,subprocess
from pathlib import Path

def main():
    out=Path('reports');out.mkdir(exist_ok=True)
    records=[]
    for root in ('checkpoints','outputs'):
        for path in sorted(Path(root).rglob('*')):
            if path.is_file() and path.suffix in ('.json','.npz','.pth','.pt','.onnx','.pkl'):
                h=hashlib.sha256()
                with path.open('rb') as f:
                    for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
                records.append({'path':str(path),'bytes':path.stat().st_size,'sha256':h.hexdigest()})
    (out/'artifacts.json').write_text(json.dumps(records,indent=2))
if __name__=='__main__':main()
