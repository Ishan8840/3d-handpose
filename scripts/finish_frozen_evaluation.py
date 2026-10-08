"""Wait for launched frozen predictions, then run fair comparisons once complete."""
import json,subprocess,time
from pathlib import Path
runs=json.load(open('configs/experiments/held_out_runs.json'));entries=[e for e in json.load(open('data/hot3d/manifest.json')) if e['split']=='held_out']
expected=[Path(root)/Path(e['path']).stem/'metrics.json' for root in runs.values() for e in entries]
deadline=time.monotonic()+3600
while not all(p.exists() for p in expected):
    if time.monotonic()>deadline:raise TimeoutError('Frozen prediction jobs did not finish; inspect their logs')
    time.sleep(10)
python='.venv/bin/python'
def run(script,*args):subprocess.run([python,'scripts/'+script,*args],check=True)
run('repair_provenance.py')
run('score_mano21_split.py','--split','held_out','--roots',*runs.values())
run('compare_split.py','--split','held_out','--runs','configs/experiments/held_out_runs.json','--output','reports/held_out_comparison')
run('compare_split.py','--split','held_out','--runs','configs/experiments/held_out_runs.json','--output','reports/held_out_mano21','--mano21')
for root in runs.values():
    for e in entries:run('evaluate_visibility.py','--predictions',str(Path(root)/Path(e['path']).stem),'--clip',e['path'])
run('score_mano21_split.py','--split','development','--roots','outputs/ace-hybrid-dev')
run('compare_split.py','--split','development','--runs','configs/experiments/development_runs.json','--output','reports/development_comparison')
run('capture_reproducibility.py')
print('Frozen held-out evaluation complete',flush=True)
