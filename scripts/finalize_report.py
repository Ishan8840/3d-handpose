"""Render completed frozen comparisons, retaining explicit scope limitations."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('reports');common=json.loads((root/'held_out_comparison/comparison.json').read_text());anatomical=json.loads((root/'held_out_mano21/comparison.json').read_text());dev=json.loads((root/'development_comparison/comparison.json').read_text())

def table(data):
    lines=['| Model | Abs MPJPE mm | Wrist mm | Tips mm | Joint coverage % | P90 mm | Capped score mm |','|---|---:|---:|---:|---:|---:|---:|']
    for name,m in data['models'].items():
        keys=['absolute_mpjpe_mm','wrist_mm','fingertips_mm','joint_coverage','p90_mm','capped_error_with_missing_penalty_mm'];values=[]
        for key in keys:
            value=m[key];values.append('—' if value is None else f'{value*(100 if key=="joint_coverage" else 1):.2f}')
        lines.append('| '+' | '.join([name]+values)+' |')
    return '\n'.join(lines)

def interval(ci):return 'unavailable' if ci is None else f'[{ci[0]:.2f}, {ci[1]:.2f}]'

paired='\n'.join(f"- {name}: {m['mean_paired_delta_mm']:.2f} mm; paired sequence-bootstrap 95% CI {interval(m['sequence_bootstrap_95ci'])}." for name,m in common['paired'].items())
winner=min(common['models'],key=lambda name:common['models'][name]['capped_error_with_missing_penalty_mm']);candidates=[n for n,m in anatomical['models'].items() if m['wrist_mm'] is not None];anatomical_winner=min(candidates,key=lambda n:anatomical['models'][n]['capped_error_with_missing_penalty_mm'])
profile_path=Path('outputs/ace-cold-profile-fixed.json')
profile=json.loads(profile_path.read_text()) if profile_path.exists() else None
profile_text=(f"A fresh 30-frame ACE hybrid run completed in {profile['elapsed_seconds']:.1f} seconds, including both model loads, refinement and output encoding. The sampled device-wide GPU-memory peak was {profile['peak_sampled_device_memory_MiB'][0]:,} MiB. This is a cold-start integration profile, not steady-state throughput or an exact allocator peak." if profile else 'Fresh inference profiling is pending.')

text=f'''# Stereo hand-pose research report

## Outcome and scope

A working stereo inference project and a frozen three-pipeline HOT3D comparison
were executed on the supplied A100 (40 GB) VM. This is a bounded research round,
not proof of the most accurate possible pipeline. Several requested extensions
remain incomplete, as listed below. No published metric is used as an experimental
result, and no hidden challenge test annotations were accessed.

The lowest common-19 coverage-aware held-out score belongs to **{winner}**.
Among pipelines that output all 21 anatomical joint slots including wrist, the
lowest MANO-21 coverage-aware score belongs to **{anatomical_winner}**. Differences
must be read with the paired uncertainty intervals and coverage, not just observed
MPJPE. MANO-reference evaluation is distinct from native UmeTrack annotation.

## Frozen held-out results

All three methods use the same 450 frames from three participants: P0010, P0017,
P0021. Parameters were committed in `3ee7b9d` before held-out execution. The
primary score is mean min(error,100 mm), with missing annotated joints assigned
100 mm. Absolute errors use no root, scale, rigid, or Procrustes alignment.

### Native common-19 landmarks

{table(common)}

### Separate MANO-21 anatomical reference

{table(anatomical)}

![Held-out coverage-aware error and coverage](figures/held_out_comparison.png)

UmeTrack cannot provide an interchangeable anatomical wrist or thumb CMC; those
slots remain missing and are penalized in MANO-21 coverage-aware evaluation.
Eligible annotations can lie outside either image; these difficult frames are
retained. Visibility-stratified results, where available, are separate diagnostics.
No model is allowed to improve its observed error by silently deleting evaluated
frames. Joint coverage is not the same as full-frame tracking coverage; both are
available in the comparison JSON files.

### Paired uncertainty

{paired}

Negative differences favor the first model. Intervals resample sequences, not
individual correlated joints. Only three independent clusters are available, so
these intervals are coarse and cannot support broad state-of-the-art claims.
An interval containing zero indicates statistical indistinguishability under
this procedure. Pretrained contamination remains a separate limitation.

## Development selection and ablations

Development participants P0003, P0013 and P0018 are disjoint from held-out and
smoke participant P0002. No hyperparameter was tuned using held-out errors.

{table(dev)}

MediaPipe, independent RTMDet/RTMPose, confidence fusion, predicted-MediaPipe-crop
RTMPose, ACE monocular/stereo, POEM large two-view and UmeTrack predicted-crop
inference were executed. Some simpler baseline comparisons cover fewer development
sequences; consult the experiment log and individual artifacts for exact scopes.
They are not silently pooled into the frozen450-frame comparison.

On the two development sequences with ACE observations, robust stereo plus a
sequence-consistent bone-length term and 0.1 s local-linear temporal refinement
changed 21.307→20.169mm and 36.869→35.411mm. The third sequence remained missing.
The strongest tested independent multiview prior was UmeTrack under the available
predicted-crop protocol; adding its gated prior did not beat the stronger anatomy
setting. No pose alignment was applied to make priors agree.

MANO fitting of RTMPose observations increased coverage but worsened observed
error 15.000→22.360mm on the first development sequence. FoundationStereo large
with correct horizontal rectification changed 27.103→26.634mm under conservative
surface-depth fusion at unchanged coverage; depth-only samples were worse at
34.298mm. Neither component was included in the frozen pipelines. These are
single-sequence ablations, not conclusive universal comparisons.

A post-freeze development diagnostic fitted MANO to the MediaPipe observations
with and without gated dense surface points. Absolute error changed
23.865→23.771mm at identical85.33% joint coverage over150 frames. The depth term
uses nearest mesh-surface distance for points within3cm of triangulated joints;
it does not assume that skin depth equals an anatomical joint center. This
single-clip0.094mm difference does not establish a reliable gain. It did not
change the frozen configurations or held-out selection.

## Architecture and failure analysis

The ACE hybrid runs released calibrated monocular inference on each view,
converts normalized landmarks to original pixels, and performs confidence-weighted
metric triangulation with epipolar, ray-angle, reprojection and cheirality checks.
Robust optimization balances reprojection, stereo position, and per-sequence
median bone lengths. Offline local-linear smoothing uses a 0.1 s window and fills
only bracketed gaps of at most0.1 s. No ground-truth crops, masks, poses, or shapes
enter this pipeline. The default lightweight CLI remains MediaPipe; ACE is
explicitly selected with `--model ace`.

Native HOT3D Quest images require a clockwise rotation for ACE. Camera and pixel
transforms are inverted before output, retaining original-left-camera metric
coordinates (+x right,+y down,+z forward). A regression against the benchmark
adapter agreed to within0.000602 mm with identical validity masks.

Wrong-hand identity can dominate error: an inspected development overlay shows
MediaPipe selecting the visible left hand while the right hand is mostly outside
the image. UmeTrack and POEM cannot reliably repair the incorrect initialization.
Unfiltered ACE triangulation increases coverage but produces large depth outliers
under partial visibility; geometric filtering trades coverage for reliability.
Small stereo disparity makes depth sensitive to landmark errors. Model surfaces
and anatomical joint centers also differ, explaining why raw dense surface depth
is not automatically a better joint constraint. Learned priors can look plausible
while misplacing the entire hand; ACE's105.048mm monocular smoke error alongside
7.775mm PA error demonstrates this distinction.

![Inspected held-out failure: image-boundary hand](figures/failures/clip-000648_largest_observed_error_6.jpg)

The example above has 89.71 mm observed common-19 error. Ground truth is red and
prediction green. The hand is near the image boundary with incomplete evidence.
Further worst-error, fast-motion and low-visibility overlays are in figures/failures.

## Data, checkpoints and reproducibility

HOT3D public training clips are obtained from the official BOP distribution.
Official FISHEYE624 projection and UmeTrack skinning are used. SHOW3D public hand-v3
scenes were downloaded at revision 070d5586edf369dfa17a374b4e9f80d5611577d5;
provided 2D annotations were checked against projected 3D points. Its baseline
results are separate from HOT3D; a full three-model SHOW3D comparison remains undone.
Both of these evaluated stereo streams are monochrome.

EgoStandard's authorized bucket sample contains 547 synchronized RGB stereo frames
at 1920×1456,30 Hz. Video, intrinsics/extrinsics, timestamps, native hand transforms,
bad-frame labels and operator IDs were decoded. MediaPipe inference produced all
547 output rows;20.632% of joint slots had valid stereo estimates. This is not a
GT coverage or accuracy measurement. The pose joint mapping is undocumented in
the embedded schemas, linked technical documentation denied access, and the
translation scale is explicitly marked unverified. No anatomical accuracy score
is claimed for EgoStandard. Source bytes are hashed because the bucket is mutable.

The supplied MANO archive is stored outside Git. Source revisions are recorded in
`configs/sources.lock.json`; executed package snapshots and checkpoint hashes are
in `configs/environments` and `configs/checkpoints.lock.json`. ACE's released
checkpoint and backbone revisions are fixed in its downloader. Some environments
use Python 3.12/PyTorch 2.11 rather than original upstream versions; executed
compatibility fixes are retained. FoundationStereo's torch-hub DINO dependency
was fetched from a moving reference, so that pilot still has a reproducibility
gap. Research-model environment setup is partly manual; the clean baseline
installation and weight-free tests are supported.

ACE lists HOT3D in its training data. This local participant split does not rule
out pretrained benchmark contamination. Other pretrained dataset overlaps have
not been exhaustively audited. Annotation/model joint-definition differences and
potential GT error limit millimeter-level claims.

## Runtime, official reproduction and limitations

Runtime measurements are in each run's metadata or `runtime.json`. ACE totals
include model loading for both views; detector timings generally exclude decoding
and initialization. Jobs overlapped during research, so these are not a controlled
speed comparison and are not placed in a misleading single leaderboard column.
A standardized isolated runtime and peak-memory comparison remains incomplete.

{profile_text}

UmeTrack's official known-skeleton example was also executed on 30 frames. It has
four available cameras and selects two crops, using GT crop guidance and the GT
subject skeleton. Its native-unit error 9.662 is a reproduction diagnostic only,
excluded from our two-camera predicted-crop leaderboard.

UA-Fit's inspected checkpoint instructions say its uncertainty checkpoint is
forthcoming; its analytical solver was not reproduced. POEM's official
`format_batch` and `extract_pred` functions also executed on30 usable frames of
its supplied three-camera demonstration with the released large checkpoint.
All21 joints and778 vertices were finite, with positive master-camera joint
depth. OpenCV sequential RGB decoding and disabled GUI display allowed headless
execution. Provided demo crops and three views make this a reproduction
diagnostic, excluded from the fair stereo leaderboard; no demo accuracy is claimed.
Additional candidates and learned temporal priors are documented in
[additional_candidates.md](additional_candidates.md); none has a claimed score.
No new model was trained or paid infrastructure provisioned.

The remaining work includes authoritative EgoStandard conventions, broader
manipulation activities and participants, exhaustive pretrained-overlap checks,
more independent temporal/occlusion evaluation, robust hand identity, released
UA-Fit weights, further candidates, fully automated heavy-model installs, and
controlled runtime/VRAM measurements. The aspirational accuracy and coverage
targets must be assessed from the table above; they are not assumed achieved.

## Inference and artifacts

```bash
python scripts/run_inference.py --model ace \\
  --left data/example/left.mp4 --right data/example/right.mp4 \\
  --calibration data/example/calibration.json \\
  --timestamps data/example/timestamps.npz --rotate-cw \\
  --anatomy-weight 1 --temporal-seconds 0.1 --output outputs/example-ace
```

Use `--rotate-cw` only for the corresponding native orientation. Nonzero-distortion
input must be undistorted first for ACE. Ground truth is not needed. Required
NPY/NPZ, timestamps, confidence, validity, wrist-pose, metadata and overlay-video
outputs are generated. Unavailable wrist orientation is NaN, never fabricated.
Provenance distinguishes stereo observations, multiview model inference, anatomical
fitting, and temporal reconstruction; original observation masks are preserved.

![Visibility-stratified error and coverage](figures/held-out/visibility.png)

![Error–coverage curves](figures/held-out/error_coverage.png)

Modeled hand visibility is an annotation-derived proxy, not a dedicated per-joint
occlusion label. Blank bars mean no observed eligible joints, not zero error.

Measured JSON comparisons and per-sequence metrics accompany this report.
[experiment_metrics.json](experiment_metrics.json) preserves the completed pilot
measurements and marks invalid configurations explicitly. Dataset split manifests
and hashes are tracked in configs/datasets. Pilot
plots and worst-frame indices are under `reports/figures`; the full experimental
record is in PROGRESS.md and the ignored outputs directory. Large assets remain
outside Git.
'''
(root/'final_research_report.md').write_text(text)
names=list(common['models']);fig,axes=plt.subplots(1,2,figsize=(11,4))
axes[0].bar(names,[common['models'][n]['capped_error_with_missing_penalty_mm'] for n in names]);axes[0].set_ylabel('Coverage-aware capped error (mm)')
axes[1].bar(names,[100*common['models'][n]['joint_coverage'] for n in names]);axes[1].set_ylabel('Joint coverage (%)');axes[1].set_ylim(0,100)
for ax in axes:ax.tick_params(axis='x',rotation=20)
fig.tight_layout();(root/'figures').mkdir(exist_ok=True);fig.savefig(root/'figures/held_out_comparison.png');plt.close(fig)
print('Report written. Common19 leader:',winner,'; full21 leader:',anatomical_winner)
