# Stereo hand pose research

Research in progress. Working metric stereo baseline; no accuracy winner claimed.

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
handedness classifier; it is not per-joint calibrated uncertainty.

Evaluate matching NPZ predictions and GT with `handpose-evaluate --predictions
pred.npz --ground-truth gt.npz --output metrics.json`. Both require matching
`frame_ids`, `timestamps`, `joints_3d`, `validity`. Absolute and aligned metrics
are separate. Missing joints count against coverage and the capped 100 mm loss.

Data, checkpoints, third-party clones and generated outputs are excluded from Git.
See PROGRESS.md and reports for actual execution status and limitations.
