"""Track small measured artifacts in Git while large outputs remain external."""
import hashlib,json
from pathlib import Path
records=[]
for path in sorted(Path('outputs').rglob('*.json')):
    if path.name not in ('metrics.json','results.json','mano21_metrics.json','runtime.json','ace-cold-profile-fixed.json','asset-verification.json'):continue
    invalid=any(part in ('ace-smoke','dense-smoke') for part in path.parts)
    records.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'valid_configuration':not invalid,'exclusion_reason':'Incorrect image orientation or vertical dense-stereo baseline; diagnostic only' if invalid else None,'measurements':json.loads(path.read_text())})
Path('reports/experiment_metrics.json').write_text(json.dumps(records,indent=2,allow_nan=False))
print('Recorded',len(records),'measured artifacts')
