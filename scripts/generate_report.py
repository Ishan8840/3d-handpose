"""Build an auditable progress report only from completed on-disk measurements."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from handpose.visualization.plots import diagnostic_plots
p=argparse.ArgumentParser();p.add_argument('--outputs',default='outputs');p.add_argument('--report',default='reports/final_research_report.md');a=p.parse_args();root=Path(a.outputs);report=Path(a.report);report.parent.mkdir(parents=True,exist_ok=True)
rows=[];artifacts=[]
for path in sorted(root.rglob('metrics.json')):
    if 'dense-smoke' in str(path) or '/ace-smoke/' in str(path):continue
    metrics=json.loads(path.read_text())
    values=[(path.parent.relative_to(root).as_posix(),metrics)] if 'absolute_mpjpe_mm' in metrics else [(path.parent.relative_to(root).as_posix()+'/'+k,v) for k,v in metrics.items() if isinstance(v,dict) and 'absolute_mpjpe_mm' in v]
    for name,m in values:
        def fmt(key,scale=1):
            v=m.get(key);return '—' if v is None else f'{v*scale:.2f}'
        rows.append('| '+ ' | '.join([name,fmt('absolute_mpjpe_mm'),fmt('wrist_mm'),fmt('fingertips_mm'),fmt('joint_coverage',100),fmt('p90_mm'),fmt('capped_error_with_missing_penalty_mm'),fmt('seconds_per_frame')])+' |')
    artifacts.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
text='''# Stereo hand-pose research report — interim, incomplete

This report contains actual executed measurements. It is not a final model
selection or a claim that the aspirational accuracy targets were achieved.
Held-out selection, wider ablations, and independent reproduction remain pending.

## Protocol and interpretation

Coordinates are metric original-left-camera optical (+x right, +y down, +z forward).
Absolute MPJPE uses no alignment. PA scores are stored separately and never used
as absolute position accuracy. Coverage below is joint coverage over annotated
eligible joints; unavailable observations are NaN and incur 100 mm in the capped
score. Observed errors and coverage must be read together. Rows from different
clips, annotation conventions, or splits are not directly comparable.

HOT3D and SHOW3D native UmeTrack scores compare 19 common joints, excluding wrist
and thumb CMC. A dash for wrist is missing reference information, not zero error.
Separate MANO-derived 21-joint evaluations are labeled explicitly in their files.
Public training data is partitioned locally by participant; these are not official
challenge test results. ACE's reported training includes HOT3D, so benchmark
contamination cannot be ruled out. Other pretrained data overlap remains under audit.

The expanded manifest uses three development and three held-out participants per
dataset, plus a separate smoke participant. Current sampled clips overrepresent
pick-up interactions and do not establish performance on every requested activity.

## Completed measurements

Runtime excludes decoding/model loading unless the associated artifact says
otherwise. Runs have not yet been standardized for GPU concurrency, so timing
is diagnostic, not a fair speed leaderboard. A1 means monocular; A2 means
stereo without epipolar/reprojection gates; A3 retains geometric gates.

| Experiment / sequence | Abs MPJPE mm | Wrist mm | Tips mm | Coverage % | P90 mm | Capped score mm | s/frame |
|---|---:|---:|---:|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

## Findings and failures

- MediaPipe can miss hands and confuse hand side, especially during partial visibility.
  An inspected expanded-development failure selects the visible left hand while
  the annotated right hand is largely outside the image, producing >600 mm errors.
  Such frames remain in scores. Four-orientation search improves first-clip
  coverage but does not solve identity errors on diverse sequences.
- RTMDet missed all eligible smoke hands in the independent detector baseline.
  Predicted MediaPipe crops enable RTMPose to run; this is a distinct assisted
  detector ablation and still loses considerable coverage.
- ACE requires correct native-image orientation. Upright smoke stereo gives
  28.019 mm at 99.65% coverage, while monocular absolute error is 105.048 mm
  despite PA error 7.775 mm. A visually plausible aligned hand is insufficient.
- POEM large executes on two views using predicted crops and its released weights.
  Its smoke coverage is constrained by the crop source. The implementation follows
  official preprocessing, but an independent official-demo reproduction remains pending.
- UmeTrack executed on two calibrated views with predicted initial crops. The
  unknown-user-scale head is used; memory resets because the moving camera frame
  is not a stationary world frame. This is a two-view adaptation, not the official
  four-view benchmark. Its wrist convention is not anatomical MANO wrist.
- MANO fitting on the first RTMPose development clip increases coverage but worsens
  observed error (15.000 to 22.360 mm). It is not selected as an improvement.
- FoundationStereo surface-depth samples alone worsen joint accuracy. Conservative
  fusion on the first development clip changes 27.103 to 26.634 mm at unchanged
  coverage. This small single-sequence gain does not establish generalization.
- The initial dense experiment used a vertical baseline and is invalid for the
  horizontal disparity network. The corrected experiment rotates camera axes,
  rectifies horizontally, and transforms 3D points back to the original left frame.
- Short local-linear temporal refinement helps one RTMPose development clip;
  broader paired evaluation is pending. Camera-frame acceleration includes camera
  motion and is not a pure world-space articulation smoothness metric.

## Dataset additions and blockers

SHOW3D public hand-v3 annotations and stereo camera calibration are decoded;
GT projection was checked against the published 2D landmarks. Held-out official
test annotations are not used. HOT3D/SHOW3D camera views used here are monochrome.

LightwheelAI/EgoStandard access was verified by downloading a 547-frame MCAP.
It includes stereo RGB1920x1456 at30Hz, intrinsics/extrinsics, pose streams,
bad-frame annotations, and operator identifiers. Embedded schemas do not name
hand joints. The linked detailed EgoDemo documentation denied access. Native
poses and timestamps are preserved; no anatomical accuracy score is claimed.
The inference integration test explicitly records its assumed translation scale.
Bucket files are mutable, so downloaded bytes are identified by SHA-256.

UA-Fit's inspected checkpoint directory says the checkpoint is coming soon.
The analytical solver has not been reproduced; no UA-Fit result is claimed.
Additional candidate methods and learned temporal priors have not been benchmarked.
No new model has been trained and no paid infrastructure was provisioned.

## Software and remaining work

The package provides tested projection, distortion, coordinate changes, weighted
triangulation, metrics, stereo video inference, explicit missing outputs, and
separate model environments. Source commits are in configs/sources.lock.json;
full environment/checkpoint locking is still being consolidated.

There is no frozen final architecture yet. Complete the expanded development
comparisons, verify output joint conventions, tune hybrid/temporal objectives only
on development data, freeze three configurations, then run held-out comparisons
with sequence-bootstrap paired uncertainty. Annotation source error and pretrained
contamination limit any final claim, even after these experiments finish.
'''
report.write_text(text);(report.parent/'measured_artifacts.json').write_text(json.dumps(artifacts,indent=2));print(report)
for folder in ('b0-rotated-dev/clip-000175','poem-smoke','ace-upright-smoke'):
    path=root/folder
    if not (path/'predictions.npz').exists():continue
    pred=np.load(path/'predictions.npz');gt=np.load(path/'ground_truth.npz')
    dest=report.parent/'figures'/folder.replace('/','_')
    worst=diagnostic_plots(pred['joints_3d'],gt['joints_3d'],pred['timestamps'],dest)
    (dest/'worst_frames.json').write_text(json.dumps(worst,indent=2))
