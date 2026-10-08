> **Expanded benchmark update:** the recommended accuracy/coverage compromise is
> WiLoR stereo + MANO Adam fitting + 50ms local temporal refinement. See
> [the measured report](reports/final_research_report.md) and
> [model setup](docs/model_setup.md). The original ACE comparison is retained as
> a prior round; the sub-10mm and >95% targets are not achieved.

# Stereo hand pose research

Working metric stereo inference and an executed, frozen three-pipeline HOT3D comparison.
ACE hybrid achieved the best held-out coverage-aware score: **27.18 mm absolute
MPJPE, 30.90 mm wrist error, 58.01% joint coverage** against the separate MANO-21
reference (450 frames, three participants). The requested accuracy/coverage
targets were not reached. See [the research report](reports/final_research_report.md)
for confidence intervals, pretrained-contamination concerns and unfinished work.

Python 3.12:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e '.[test,baseline]'
pytest -q
python scripts/setup_environment.py
python scripts/run_inference.py --left left.mp4 --right right.mp4 --calibration calibration.json --output outputs/example
```

Videos must already be synchronized, equal frame rate and frame count. Optional
`--timestamps timestamps.npz` supplies `left` and `right` nanosecond arrays;
default timestamps are nominal FPS from zero and do not verify physical sync.

Calibration JSON declares `units: "meters"`,
`transform_convention: "T_camera_from_left"`, and `left`/`right` objects with
`K` (3x3), `T_camera_from_left` (4x4), `distortion` (OpenCV coefficients),
`model` (`opencv` or `fisheye`). Left transform is identity. For a right camera
located 10 cm to the right with identical orientation, right T[0,3] = -0.1.
HOT3D FISHEYE624 requires the official toolkit adapter, not OpenCV fisheye.

Outputs use left optical coordinates (+x right, +y down, +z forward), meters.
Missing joints and unavailable wrist rotation are NaN; validity is explicit.
Order is wrist then thumb/index/middle/ring/pinky proximal-to-tip, 4 per finger.
Thumb anatomy is CMC/MCP/IP/tip. Confidence currently comes from MediaPipe's
handedness classifier for the MediaPipe baseline; other models document their
score semantics in metadata. None is calibrated per-joint uncertainty.

Evaluate matching NPZ predictions and GT with `handpose-evaluate --predictions
pred.npz --ground-truth gt.npz --output metrics.json`. Both require matching
`frame_ids`, `timestamps`, `joints_3d`, `validity`. Absolute and aligned metrics
are separate. Missing joints count against coverage and the capped 100 mm loss.

Data, checkpoints, third-party clones and generated outputs are excluded from Git.
See PROGRESS.md and reports for actual execution status and limitations.

Actual dataset adapters:

```bash
# On a PyTorch-enabled research environment, install the pinned official toolkit
# from configs/sources.lock.json, plus webdataset==0.2.111.
python scripts/prepare_hot3d.py
python scripts/benchmark_hot3d.py --split smoke --output outputs/b0-smoke
# SHOW3D: huggingface_hub==0.36.0 pyarrow==19.0.1
python scripts/prepare_show3d.py
python scripts/benchmark_hot3d.py --manifest data/show3d/manifest.json --split development --model mediapipe_rotated --output outputs/show3d-dev
```

The public datasets' native UmeTrack annotations have a different wrist definition
and no thumb CMC. The default comparison therefore measures **19 common joints**.
`evaluate_mano_gt.py` provides a separate 21-joint MANO annotation track once
legitimately obtained MANO files have been supplied. Do not merge these leaderboards.

HOT3D native images need a clockwise rotation for upright video models. The
`export_stereo_example.py --upright` helper rotates optical coordinates as well;
ACE export evaluation inversely maps predictions back to the original camera.
SHOW3D is already upright. Its per-frame world transforms refer to a moving rig,
not static physical world. The reader checks pose validity and GT reprojection.

EgoStandard access and schema audit:

```bash
hf buckets cp 'hf://buckets/LightwheelAI/EgoStandard/EgoStandard/mcap/wipe the keyboard/2b5d4457-b40c-4dbd-973c-4fdd82f9e06f.mcap' data/egostandard/sample.mcap
pip install mcap==1.3.0 lz4==4.4.5 zstandard==0.25.0
python scripts/inspect_egostandard.py data/egostandard/sample.mcap --output outputs/egostandard-inspection
```

The bucket is mutable; the inspector records SHA-256. This sample has paired RGB
streams, camera calibrations, 21-transform hand poses and bad-frame annotations.
Its joint names and pose units still require authoritative confirmation. It is
not yet an accuracy benchmark. `handpose.data.egostandard` requires explicit
joint names and units, validates world-to-camera conversion, and preserves
unmatched timestamps as missing. Linked EgoDemo technical documentation requires
separate access not available in the VM's current session.

Offline ACE stereo inference (separate model environment and released assets):

```bash
python scripts/run_inference.py --model ace \
  --left data/example/left.mp4 --right data/example/right.mp4 \
  --calibration data/example/calibration.json \
  --timestamps data/example/timestamps.npz --rotate-cw \
  --anatomy-weight 1 --temporal-seconds 0.1 --output outputs/ace-example
```

Use `--rotate-cw` for native HOT3D Quest images. Outputs remain in the original
left camera frame. `--temporal-seconds 0.1` enables offline local-linear refinement
with gaps capped at 0.1 seconds. Combined with `--anatomy-weight 1`, these are
the frozen ACE hybrid settings evaluated on held-out participants.
`--ace-left-export` and `--ace-right-export` reuse trusted local official ACE
pickle exports for regression checks; they are not required for inference.
Nonzero-distortion video must first be undistorted with matching calibration.

`configs/environments/` records the actually executed VM package versions;
`configs/checkpoints.lock.json` records 13 downloaded checkpoint identities.
These environment snapshots include inherited system packages and are audit
records, not yet a one-command research-model installer. The baseline installation
above is the supported clean installation path. Large-model setup remains partly
manual and is described in the research report.

Heavy-model setup details: [docs/model_setup.md](docs/model_setup.md).
Verify downloaded model bytes with `python scripts/verify_assets.py`; missing
optional model weights are reported explicitly. Use `--contains ace` to check a
subset. The complete executed model catalog is `handpose.models.registry.MODEL_SPECS`.

Dataset split identities, revisions and available hashes are tracked under
`configs/datasets/`. Frozen settings are in `configs/experiments/frozen_hot3d.yaml`.
