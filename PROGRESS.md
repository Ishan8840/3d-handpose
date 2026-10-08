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
