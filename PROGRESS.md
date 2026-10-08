# Progress

## Current phase: Phase 7–8 research round delivered; remaining scope below

Entries below are chronological; early pending items are superseded by later
measured milestones. See the delivered comparison and remaining-work sections.

## Initial bootstrap record

- Phase 0: Python 3.12 isolated environment installed; package imports; 8 tests pass.
- Hardware: supplied VM is an A100-SXM4 40 GB container, ~50 GB available disk;
  preinstalled PyTorch 2.11.0+cu128 reports CUDA available. Local machine has no nvidia-smi.
- Geometry: distortion, projection/backprojection, transforms, E/F, epipolar gating,
  weighted DLT, cheirality, ray-angle and reprojection checks implemented.
- Metrics: absolute/root-relative/PA errors, wrist/tips, coverage, capped missing
  penalty, percentiles, axes, depth bins, acceleration error, cluster bootstrap.
- End-to-end MediaPipe video CLI implemented; real-data execution pending.
- Official HOT3D/toolkit repositories inspected and cloned. Public training clips
  identified; actual clip download and joint mapping verification in progress.
- Full HOT3D VRS download requires signup. Public challenge test GT is unavailable
  and will not be accessed. Clip stereo cameras are monochrome, not stereo RGB.
- Failures resolved: local Python 3.14 incompatible with baseline wheels; installed
  isolated 3.12. SSH global config permissions required `-F /dev/null`.
- Next: obtain public training clips, participant-separated partitions, execute B0.
- No measured real-world accuracy and no final architecture selection yet.

## Real-data milestone

- Downloaded and hashed HOT3D Quest3 clips 000000 / 000175 / 000648 from public
  train distribution, participants P0002 / P0003 / P0010 (disjoint).
- B0 smoke: 30 frames, 21.870 mm observed absolute MPJPE, 31.113 mm tips,
  28.07% joint coverage, capped missing-penalty score 78.069 mm.
- B0 dev: 150 frames, 27.103 mm observed absolute MPJPE, 37.939 mm tips,
  75.96% joint coverage, capped score 42.449 mm.
- B1 independent RTMDet/RTMPose smoke: zero accepted joints. Detector often misses
  the right hand. Fixed upstream full-image fallback on empty detection lists.
- B1 predicted-MediaPipe-crop ablation dev: 15.000 mm MPJPE, 15.051 mm tips,
  49.65% coverage, capped score 57.781 mm. Not superior on primary metric.
- SHOW3D added per user instruction: three public train scenes downloaded from
  revision 070d5586edf369dfa17a374b4e9f80d5611577d5, distinct participants.
  v3 GT reprojection to provided 2D points audited. Initial 30-frame B0 coverage 0%.
- Both datasets use native UmeTrack GT; evaluate 19 comparable landmarks, not a
  claimed 21-joint anatomical wrist benchmark. Wrist and thumb CMC unavailable in
  this initial comparison; MANO archive supplied by user is now available.
- ACE checkpoint + backbone downloaded (~25 GB), preparation ongoing in separate
  environment. Upstream VideoX latest imports extra audio dependencies.
- UA-Fit checkpoint README says release coming soon; no executable weights yet.
- UmeTrack release and official sample data obtained; predicted-crop adaptation
  under development to avoid GT-assisted official crop protocol.
- 9 local weight-free tests pass. Spatial/temporal ablations running next.

## Spatial model milestone and dataset expansion

- 12 weight-free tests pass. Expanded HOT3D and SHOW3D manifests each contain
  seven distinct participants, with smoke/development/held-out assignment.
  Held-out predictions have not been used for model selection.
- Actual UmeTrack two-view predicted-crop inference: first HOT3D dev sequence
  22.449 mm observed MPJPE, 85.33% coverage, capped score 33.823 mm.
- Four-rotation MediaPipe hypothesis search on that dev sequence: 26.596 mm,
  94.11% coverage, capped score 28.610 mm. Expanded development run underway.
- Corrected native HOT3D 90-degree camera orientation for ACE: 30-frame smoke
  stereo 28.019 mm, 99.65% coverage, capped score 28.109 mm. Monocular metric
  position error remains 105.048 mm despite PA error 7.775 mm. No winner selected.
- Earlier native-orientation ACE experiments and vertical-baseline dense stereo
  are failed configurations, not eligible leaderboard candidates.
- FoundationStereo large checkpoint ran with horizontal rectification on 150 dev
  frames: conservative depth fusion 26.634 mm vs baseline 27.103 mm, unchanged
  75.96% coverage. Depth-only surface observations 34.298 mm. One sequence is
  insufficient evidence that the small fusion improvement generalizes.
- RTMPose predicted-crop MANO fitting worsened observed error (15.000 ->22.360 mm)
  while increasing coverage (49.65 ->70.67%). Temporal 0.1-second local-linear
  refinement gave 13.716 mm/54.39% coverage; requires wider development testing.
- Supplied MANO archive enables separate 21-joint HOT3D reference evaluation:
  rotated MediaPipe dev 25.692 mm absolute, wrist14.949 mm, tips37.700 mm,
  coverage94.35%. Keep distinct from common19 UmeTrack annotation scores.
- POEM large checkpoint loaded successfully; custom predicted-crop two-camera
  inference under validation. No POEM accuracy claim yet.
- User-added LightwheelAI/EgoStandard bucket accessible on VM. Downloaded one
  547-frame MCAP: actual stereo RGB1920x1456 at30Hz, intrinsics/extrinsics,
  world hand transforms, bad-frame annotations and operator identity present.
  Protobuf schema inspector and H264 extraction executed. Hand convention and
  annotation provenance still require verification before any accuracy score.
- Primary models' environments differ from some upstream versions; exact
  dependency/weight locks and portable setup still need consolidation.

## EgoStandard integration and expanded failures

- User confirms approved EgoStandard access but has no additional convention docs.
  Decoded 547 paired RGB frames, exported video/calibration/timestamps and native
  poses. Ran the actual inference CLI:547 outputs,20.632% valid joint slots,
  missing joints verified NaN. This is an inference integration result, not GT
  tracking coverage. Translation scale1 is explicitly marked unverified.
- Added strict explicit-convention conversion and timestamp matching tests;
  15 total tests pass including robust optimizer rejection of misaligned priors.
- POEM large smoke completed:24.568 mm observed/30% coverage/capped77.370 mm.
  Batch-view count and upstream relative BPS asset path incompatibilities fixed.
- Expanded MP development exposed wrong-hand selection on P0018:~630 mm at
  25.47% coverage. Known upright orientation also fails (~638 mm at9.44%).
  Inspected overlay confirms visible left hand selected while right is cropped.
- UmeTrack cannot repair erroneous side initialization (P0018~703 mm at14%).
- Upright ACE first dev: A3 21.307 mm/93.40% coverage/capped26.498 mm;
  A2 ungated22.419 mm/100% coverage/capped22.416 mm. Broader ACE run ongoing.
- Robust reprojection/stereo/anatomy/prior ablation implemented; prior requires
  same timestamps and metric frame, rejects discrepancies >5 cm. Dev run ongoing.
- Interim measured report and diagnostic plots generated; final selection remains
  pending. Environment snapshots and13 checkpoint hashes captured from VM.

## Frozen held-out evaluation in progress

- Commit3ee7b9d freezes three pipelines before held-out prediction execution.
  Participants:P0010/P0017/P0021,150 frames each, both common19 and MANO21 tracks.
- ACE hybrid development aggregate:25.971mm observed,50.42% joint coverage,
  capped61.439mm. MP rotations:120.185mm/55.13%/63.815mm;
  UmeTrack:86.648mm/49.33%/65.285mm; POEM:163.703mm/60.89%/68.580mm.
  Large MP/POEM errors include genuine wrong-hand failures. All paired95% cluster
  intervals reach zero with only three sequences; no statistical superiority claim.
- ACE hybrid uses anatomy weight1,0.1-second temporal window,0.1-second max gap.
  It improves the two observable development clips to20.169 and35.411mm.
- Required ACE CLI output files generated from stereo videos without GT;
  trusted-export reuse regression agrees with benchmark within6.03e-7m and
  identical validity. Added rectangular rotation regression:16 tests pass.
- Official UmeTrack smoke successfully executed30 native example frames:
  4 available cameras,2 selected GT-guided crops,known GT skeleton;
  error9.662 native units. Excluded from fair two-view results.
  Initial Git clone contained LFS video pointers; fetched and SHA-verified actual
  73MB example video before running. Missing PyAV dependency installed av16.0.1.
- Additional model release audit saved; these are unexecuted candidates, not
  fabricated scores or claimed hard blockers.


## Delivered frozen comparison and inference validation

- All three frozen pipelines completed 450 identical held-out HOT3D frames from
  three disjoint participants. Common-19 measured results:
  ACE hybrid: 28.278 mm absolute / 57.65% joint coverage / capped 58.281 mm.
  MediaPipe rotations: 47.797 mm absolute / 23.26% joint coverage / capped 84.515 mm.
  UmeTrack two-view: 34.654 mm absolute / 22.22% joint coverage / capped 84.859 mm.
- Separate MANO21 reference: ACE27.176mm absolute, wrist30.895mm,
  tips25.821mm, coverage58.01%, P9052.090mm, capped57.414mm.
  ACE has the best frozen score. Paired held-out cluster intervals exclude zero
  versus both comparators; only three clusters and possible HOT3D pretraining
  contamination limit generalization. Aspirational targets were not achieved.
- Complete fresh ACE hybrid inference from 30 video frames succeeded after fixing
  a venv-Python symlink resolution bug. Both model passes, optimization, smoothing,
  and all8 required output artifacts verified. Runtime175.917s including loading;
  sampled device-wide GPU peak14,606MiB. No GT used in inference.
- Local CPU MediaPipe CLI processed the actual 30-frame sample and preserved NaNs
  for missing joints. Package imports and21 weight-free tests pass, including
  real-video I/O, environment isolation and participant-split regression tests.
- 13 checkpoint hashes verified with zero mismatches. Dataset manifests/revisions
  are tracked under configs/datasets; executed snapshots under configs/environments.
- Final report, measured JSON comparisons, per-joint/depth/distribution/3D/trajectory
  plots, error–coverage curves, visibility diagnostics and9 failure overlays saved.
- SHOW3D three development scenes were evaluated with rotated MediaPipe; later
  scenes have very low/zero coverage. EgoStandard547-frame stereo inference
  integration completed; GT joint convention and unit documentation remain absent.

### Explicit remaining work (not claimed complete)

- Authoritative EgoStandard joint/coordinate conventions and quantitative evaluation.
- Full three-model SHOW3D comparison; more participants and manipulation activities.
- UA-Fit analytical solver/checkpoint.
- Additional candidate models and learned temporal priors (see release audit).
- Fully automated clean heavy-model installs; pin FoundationStereo's transient
  torch-hub dependency if revisiting that excluded pilot.
- Controlled comparative steady-state runtime/peak-memory measurements.
- Exhaustive pretrained benchmark-contamination and annotation-uncertainty audits.

## Post-freeze development and reproduction diagnostics

- Dense surface constraints plus MANO executed on150 development frames:
  stereo MANO23.865mm versus stereo+depth MANO23.771mm, both85.33% joint coverage.
  This small single-sequence difference did not change any frozen settings.
- Official POEM demo preprocessing/output conversion executed with the large
  released checkpoint on30 usable three-view frames with provided demo crops.
  Finite21-joint/778-vertex outputs and positive master-camera depth verified.
  This is separate from the predicted-crop two-camera accuracy comparison.
  The2.68GB official archive was downloaded and SHA256 verified; only one
  sequence's RGB videos, crops and calibration were extracted.
- User confirmed no additional EgoStandard convention documentation is available.
  Preserve native annotations and keep anatomical accuracy evaluation blocked;
  the supplied dataset card does not resolve joint ordering or transform units.
- Final local regression run:21 tests passed.
