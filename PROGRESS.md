# Progress

## Current phase: 1–2, first real-data baseline

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
