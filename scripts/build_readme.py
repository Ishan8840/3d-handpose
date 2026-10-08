"""Publish the measured report in README, preserving installation instructions.

Uses committed reports only: no model weights or dataset downloads required.
"""
import json
import os
from pathlib import Path
import re

readme=Path('README.md')
old=readme.read_text()
marker='<!-- BEGIN INSTALLATION -->'
installation=old.split(marker,1)[1].split('<!-- END INSTALLATION -->',1)[0].strip() if marker in old else old[old.index('Python 3.12:'):].strip()
installation=installation.replace('`configs/checkpoints.lock.json` records 13 downloaded checkpoint identities.', '`configs/checkpoints.lock.json` and `configs/models/expanded_checkpoint_hashes.json` record downloaded checkpoint identities.')
installation=installation.replace('Frozen settings are in `configs/experiments/frozen_hot3d.yaml`.', 'Frozen settings are in `configs/experiments/frozen_hot3d.yaml`, `expanded_frozen.json`, and `fitted_temporal_frozen.json`.')
report=Path('reports/final_research_report.md').read_text().split('\n',1)[1].strip()
# Report links/images are relative to reports/; README lives at repository root.
def relocate(match):
 target=match.group(1)
 if '://' in target or target.startswith('#'):return match.group(0)
 return ']('+os.path.normpath(str(Path('reports')/target))+')'
report=re.sub(r'\]\(([^)]+)\)',relocate,report)

def table(models):
 lines=['| Method | Absolute MPJPE mm | Wrist mm | Fingertips mm | Joint coverage % | P90 mm | Capped100 mm |','|---|---:|---:|---:|---:|---:|---:|']
 for name,m in models.items():
  values=[m['absolute_mpjpe_mm'],m['wrist_mm'],m['fingertips_mm'],m['joint_coverage']*100,m['p90_mm'],m['capped_error_with_missing_penalty_mm']]
  lines.append('| '+name+' | '+' | '.join('—' if v is None else f'{v:.2f}' for v in values)+' |')
 return '\n'.join(lines)
# Publish every anatomical held-out variant, not only the report's shortlist.
heading='## Anatomical MANO-21 held-out comparison'
start=report.index('| Method',report.index(heading));end=report.index('\n\nRelative to ACE',start)
report=report[:start]+table(json.load(open('reports/expanded_held_mano21/comparison.json'))['models'])+report[end:]

ace=json.load(open('reports/ace_standalone_held.json'))
section='''## ACE-Ego-Hand alone versus stereo

These three configurations use the **same 450 held-out HOT3D frames and common
19-joint reference**. They are separate from the anatomical MANO-21 table above.

| ACE configuration | Absolute MPJPE mm | PA-MPJPE mm, aligned | Joint coverage % |
|---|---:|---:|---:|
'''
for name,m in ace['models'].items():section+=f"| {name} | {m['absolute_mpjpe_mm']:.2f} | {m['pa_mpjpe_mm']:.2f} | {m['joint_coverage']*100:.2f} |\n"
section+='''
Standalone ACE predicts a much better aligned hand configuration than absolute
metric placement: **107.48 mm absolute error versus 8.69 mm after Procrustes
alignment**. Alignment changes translation, rotation and scale, so this gap must
not be attributed entirely to depth. ACE's 2D observations remain useful: calibrated
stereo reduces absolute error to **30.03 mm**, then spatial/temporal refinement
to **28.28 mm**. This conclusion applies to the tested checkpoint and protocol.
[Machine-readable ACE results and source hashes](reports/ace_standalone_held.json).

'''
report=report.replace('## All executed expanded held-out variants: common19',section+'## All executed expanded held-out variants: common19')

pilots='''## Smaller baseline and optimization pilots

These runs use different subsets and must **not** be ranked against the 450-frame
leaderboards as though they evaluated the same observations. Smoke clips have
30 frames; the development pilots below use clip-000175 (150 frames). Dense smoke
runs marked invalid in the experiment ledger are excluded.

| Recorded artifact / variant | Absolute MPJPE mm | Joint coverage % | Capped100 mm |
|---|---:|---:|---:|
'''
ledger=json.load(open('reports/experiment_metrics.json'))
for record in ledger:
 path=record['path'];m=record['measurements']
 selected=any(s in path for s in ('/b0-smoke/','/b0-dev/','/b1-smoke/','/b1-mpcrop-dev/','/b2-dev/','/b1-ablations-dev/','/dense-dev/','/dense-mano-dev/'))
 if not selected or not record['valid_configuration'] or not isinstance(m,dict):continue
 entries={path:m} if 'absolute_mpjpe_mm' in m else {path+' / '+k:v for k,v in m.items() if isinstance(v,dict) and 'absolute_mpjpe_mm' in v}
 for label,v in entries.items():
  vals=[v['absolute_mpjpe_mm'],100*v['joint_coverage'],v['capped_error_with_missing_penalty_mm']]
  pilots+='| `'+label+'` | '+' | '.join('—' if x is None else f'{x:.2f}' for x in vals)+' |\n'
pilots+='''
All recorded first-round measurements, protocol exclusions and artifact hashes
are in the [experiment ledger](reports/experiment_metrics.json). The
[development fusion grid](reports/fusion_development.json),
[fitted temporal sweep](reports/fitted_temporal_development.json),
[POEM official demo verification](reports/poem_official_demo.json), and
[first-round report](reports/round_one_research_report.md) retain the additional
ablations and diagnostics. A demo with provided crops or GT shape is not a fair
predicted-input accuracy result.

'''
report=report.replace('## Execution matrix and remaining blockers',pilots+'## Execution matrix and remaining blockers')
intro='''# Stereo hand-pose research: measured results and inference

Calibrated stereo hand reconstruction in metric camera coordinates, with executable
inference, participant-separated benchmarks, uncertainty intervals and error-overlap
analysis. **Recommended tested pipeline: WiLoR stereo → MANO Adam fitting → 50 ms
local temporal refinement.** The sub-10 mm / >95% coverage targets remain unmet.

This README contains the full measured comparison and execution limits. Tables
come from committed experiment JSON, not published paper numbers.

- [Recommended inference command](#inference-and-reproducibility)
- [Installation and input/output contract](#installation-and-inputoutput-contract)
- [Anatomical held-out results](#anatomical-mano-21-held-out-comparison)
- [ACE alone versus stereo](#ace-ego-hand-alone-versus-stereo)
- [SHOW3D results](#show3d-cross-dataset-check)
- [Error overlap](#correlated-errors-and-complementary-information)
- [Executed methods and blockers](#execution-matrix-and-remaining-blockers)
- [Full research report](reports/final_research_report.md)

'''
# Clean inherited report typography without altering code, links or heading anchors.
parts=re.split(r'(```[\s\S]*?```|`[^`]+`|\]\([^)]+\)|^#+[^\n]*$)',report,flags=re.MULTILINE)
for i in range(0,len(parts),2):
 parts[i]=re.sub(r'(?<=[a-z])(?=\d)',' ',parts[i])
 parts[i]=re.sub(r'(?<=\d)(mm|ms|MiB|GiB|frames|steps|iterations|seconds|joints|s)\b',r' \1',parts[i])
 parts[i]=re.sub(r',(?=\S)',', ',parts[i])
report=''.join(parts)
readme.write_text(intro+report+'\n\n## Installation and input/output contract\n\n'+marker+'\n\n'+installation+'\n\n<!-- END INSTALLATION -->\n')
print('Updated README from committed measurement reports')
