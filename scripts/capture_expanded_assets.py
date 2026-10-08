"""Hash executed research assets without collecting credentials or private keys."""
from pathlib import Path
import hashlib,json
patterns=['third_party/wilor/pretrained_models/*','checkpoints/anyhand/*.ckpt','checkpoints/hamer/hamer_ckpts/checkpoints/*.ckpt','third_party/egoforce/_DATA/*.pth','third_party/egoforce/_DATA/*.torchscript','third_party/omnihands/checkpoints/*.pth','third_party/stablehand/save/*/*.pt','checkpoints/dynhamr/hmp_model/**/*.pth','checkpoints/dynhamr/hmp_model/**/*.pt']
items=[]
for pattern in patterns:
 for path in Path('.').glob(pattern):
  if not path.is_file():continue
  h=hashlib.sha256()
  with path.open('rb') as f:
   while chunk:=f.read(8*1024*1024):h.update(chunk)
  items.append({'path':str(path),'size':path.stat().st_size,'sha256':h.hexdigest()})
Path('configs/models/expanded_checkpoint_hashes.json').write_text(json.dumps(items,indent=2)+'\n')
print('Hashed',len(items),'assets')
