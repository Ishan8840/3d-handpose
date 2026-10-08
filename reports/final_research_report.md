# Stereo hand-pose research report — interim, incomplete

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
| ace-upright-development/clip-000175/A1_monocular | 137.77 | — | 139.27 | 100.00 | 158.62 | 100.00 | — |
| ace-upright-development/clip-000175/A3_stereo_epipolar | 21.31 | — | 28.58 | 93.40 | 38.85 | 26.50 | — |
| ace-upright-development/clip-000175/A2_stereo_ungated | 22.42 | — | 31.03 | 100.00 | 41.14 | 22.42 | — |
| ace-upright-development/clip-000838/A1_monocular | 114.94 | — | 116.25 | 68.00 | 134.03 | 97.78 | — |
| ace-upright-development/clip-000838/A3_stereo_epipolar | 36.87 | — | 41.21 | 57.09 | 50.09 | 60.23 | — |
| ace-upright-development/clip-000838/A2_stereo_ungated | 259.99 | — | 164.28 | 68.00 | 513.21 | 58.55 | — |
| ace-upright-smoke/A1_monocular | 105.05 | — | 106.13 | 100.00 | 127.51 | 95.17 | — |
| ace-upright-smoke/A3_stereo_epipolar | 28.02 | — | 37.14 | 99.65 | 56.85 | 28.11 | — |
| ace-upright-smoke/A2_stereo_ungated | 28.20 | — | 37.71 | 100.00 | 58.08 | 28.04 | — |
| b0-dev/clip-000175 | 27.10 | — | 37.94 | 75.96 | 51.03 | 42.45 | 0.11 |
| b0-rotated-dev/clip-000175 | 26.60 | — | 37.94 | 94.11 | 48.98 | 28.61 | 0.58 |
| b0-rotated-expanded-dev/clip-000175 | 26.60 | — | 37.94 | 94.11 | 48.98 | 28.61 | 0.54 |
| b0-rotated-expanded-dev/clip-000838 | 29.17 | — | 30.89 | 45.82 | 34.47 | 62.84 | 0.50 |
| b0-rotated-expanded-dev/clip-001119 | 629.65 | — | 597.25 | 25.47 | 716.49 | 100.00 | 0.43 |
| b0-smoke/clip-000000 | 21.87 | — | 31.11 | 28.07 | 41.84 | 78.07 | 0.13 |
| b0-upright-expanded-dev/clip-000175 | 25.38 | — | 33.66 | 77.37 | 48.87 | 41.23 | 0.12 |
| b0-upright-expanded-dev/clip-000838 | 18.52 | — | 19.98 | 44.25 | 35.69 | 63.94 | 0.11 |
| b0-upright-expanded-dev/clip-001119 | 637.91 | — | 585.35 | 9.44 | 736.99 | 100.00 | 0.09 |
| b1-mpcrop-dev/clip-000175 | 15.00 | — | 15.05 | 49.65 | 28.96 | 57.78 | 0.60 |
| b1-smoke/clip-000000 | — | — | — | 0.00 | — | 100.00 | 0.69 |
| b2-dev/clip-000175 | 27.53 | — | 39.76 | 76.25 | 51.99 | 42.14 | 1.10 |
| poem-smoke | 24.57 | — | 30.49 | 30.00 | 49.52 | 77.37 | 0.07 |
| show3d-b0-rotated-dev/aria_pick-up-put-down_3b63 | 20.60 | — | 26.95 | 38.09 | 41.89 | 69.74 | 0.76 |
| show3d-b0-smoke/aria_pick-up-put-down_2170 | — | — | — | 0.00 | — | 100.00 | 0.07 |
| ume-upright-expanded-dev/clip-000175 | 23.68 | — | 28.49 | 88.67 | 40.71 | 32.33 | 0.14 |
| ume-upright-expanded-dev/clip-000838 | 19.54 | — | 22.08 | 45.33 | 35.55 | 63.52 | 0.12 |
| ume-upright-expanded-dev/clip-001119 | 702.74 | — | 698.90 | 14.00 | 894.56 | 100.00 | 0.11 |
| umetrack-dev/clip-000175 | 22.45 | — | 26.87 | 85.33 | 36.26 | 33.82 | 0.13 |
| umetrack-smoke/clip-000000 | 24.98 | — | 29.48 | 30.00 | 44.29 | 77.50 | 0.20 |

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
