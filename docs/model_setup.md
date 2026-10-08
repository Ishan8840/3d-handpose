# Executed model environments

The baseline clean installation is in README.md. These commands describe the
research environments executed on the supplied A100/CUDA12.8 VM. Exact installed
versions are in configs/environments; source revisions are in configs/sources.lock.json.
Original upstream environments differ. A clean installation of every heavy model
on an arbitrary machine has not been validated.

## Source checkouts

Run `python scripts/fetch_sources.py hot3d hand_tracking_toolkit rtmlib ace videox poem manotorch pytorch3d umetrack foundationstereo` to clone the recorded revisions.
Existing checkouts must already match their locks. Do not substitute the latest branch. ACE expects
VideoX-Fun at `third_party/ace/third_party`; in the executed layout this is a symlink
to `../videox`. The MANO archive supplied by the user is not part of Git.

## ACE

Use Python3.12 and a separate `.venv-ace`. The VM inherited PyTorch2.11.0+cu128 and
torchvision0.26.0+cu128 using `python -m venv --system-site-packages .venv-ace`.
For a clean GPU environment install those versions from the official cu128 wheel
index, then the recorded packages. The environment snapshot includes inherited
packages beyond inference requirements; audit-only packages need not be imported.

```bash
.venv-ace/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r configs/environments/venv-ace.txt
.venv-ace/bin/python third_party/ace/scripts/convert_mano_pkls.py \
  --src checkpoints/mano --dst checkpoints/mano_converted
.venv-ace/bin/python scripts/download_ace.py
```

From `third_party/ace`, run `../../.venv-ace/bin/python scripts/precompute_caption.py`.
This prepares the fixed caption embedding. The downloader pins both the calibrated
ACE checkpoint and Wan backbone revisions and places converted MANO files in the
expected path. The backbone occupies about25GB on disk. The inference CLI selects
this interpreter using `--ace-python`; it can point to another compatible environment.

## POEM-v2

A separate `.venv-poem` inherited the same PyTorch. Its snapshot is
configs/environments/venv-poem.txt. Build the pinned PyTorch3D source with the
matching CUDA toolkit. The pinned manotorch source is imported directly; old
chumpy requires the compatibility aliases recorded in benchmark_poem.py.

The released Google Drive checkpoint folder linked by the upstream POEM README
provided `large.pth.tar` (place at checkpoints/poem/large.pth.tar) and the
medium-MANO checkpoint. The plain medium checkpoint download was denied by the
public link during this run. Original MANO models are placed under
third_party/poem/assets/mano_v1_2/models. Keep the upstream working directory
because inference lazily accesses assets/bps.npy. The generated BPS sampling uses
seed42 and should be retained for byte-for-byte repeats.

```bash
PYTHONPATH=src:third_party/hand_tracking_toolkit .venv-poem/bin/python scripts/poem_probe.py
PYTHONPATH=src:third_party/hand_tracking_toolkit .venv-poem/bin/python scripts/benchmark_poem.py \
  --clip data/hot3d/clip-000000.tar \
  --observations outputs/b0-smoke/clip-000000/predictions.npz \
  --output outputs/poem-smoke --frames 30
```

The observation file supplies predicted crops only. Ground truth is used after
inference for evaluation.

The official three-view demo was also executed with the large checkpoint using
`scripts/poem_official_demo.py`. Download the archive from
`kelvin34501/POEM-v2_example_data` (Hugging Face model repository), verify the hash
in `configs/datasets/poem_official_demo.json`, and extract under
`data/poem-official/`. The script requires calibration, hand_labels.json, and the
RGB videos and crops for `pour__2025_0325_1117_56`. Additional demo imports require
`ffmpeg-python==0.2.0` and `open3d==0.19.0`; preserve `numpy==1.26.4` and
`scipy==1.14.1`. Run `.venv-poem/bin/python scripts/poem_official_demo.py`.
Thirty usable frames passed finite-joint/mesh checks. Supplied demo crops and
three views exclude this diagnostic from the fair two-camera leaderboard.

## UmeTrack and FoundationStereo

UmeTrack's pinned repository includes pretrained_models/pretrained_weights.torch.
The adapter imports its official model, cropping, and skinning code in a separate
process from POEM because both repositories use a top-level `lib` package.
Use the `.venv` snapshot with PyTorch available. `av==16.0.1` is additionally needed
for the official raw-data example. Git LFS pointers are not videos; the executed
recording_00.mp4 SHA256 is
125c20443a262354bb276c22a576bf36791a21659ec877d9b0f2e8aff563229f.

FoundationStereo's strongest downloaded checkpoint is
checkpoints/foundationstereo/23-51-11/model_best_bp2.pth, with its adjacent cfg.yaml.
The adapter uses the pinned source,32 iterations and float16 inference. Correct
horizontal rectification is mandatory. Its torch-hub DINO code was fetched from
main, which remains an explicit reproducibility gap for this pilot experiment.
The dense component is not included in the frozen final candidates.

## UA-Fit

The inspected official checkpoint instructions still indicate a forthcoming
checkpoint. No substitute weights, fabricated adapter output, or analytical-solver
result is provided.

## Expanded research environment and checkpoints

The second round used `.venv-extra` (Python3.12, PyTorch2.11.0+cu128 on A100).
`configs/environments/venv-extra.txt` records installed versions;
`configs/models/expanded_sources.json` and `expanded_releases.json` pin sources
and Hugging Face revisions. `expanded_checkpoint_hashes.json` records downloaded
weight sizes/SHA256. Clean installation on a different GPU remains unverified.

```bash
python scripts/fetch_sources.py wilor hamer egoforce omnihands anyhand stablehand dynhamr egohandicl parafit
python3.12 -m venv --system-site-packages .venv-extra
.venv-extra/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 -r configs/environments/venv-extra.txt
.venv-extra/bin/python -m pip install --no-deps -e third_party/parafit
```

MMCV2.1.0 was compiled for the A100 with `MAX_JOBS=4`,
`TORCH_CUDA_ARCH_LIST=8.0`, `--no-build-isolation`; this is required only for the
optional EgoForce forearm detector. EgoForce reuses the compiled PyTorch3D from
`.venv-poem`; this explicit dependency is in its adapter. Its vendored MMDetection
and datapipes source paths are loaded by the adapter, not stock MMDetection.
Legacy trusted checkpoints require the recorded NumPy/inspect compatibility
aliases and `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; the adapters apply these locally.

Download only pinned upstream assets and place them as follows:

| Source | Local destination |
|---|---|
| WiLoR HF Space `pretrained_models/wilor_final.ckpt`, detector.pt, model_config.yaml | `third_party/wilor/pretrained_models/` |
| WiLoR mean parameters from the same release | `third_party/wilor/mano_data/mano_mean_params.npz` |
| Official HaMeR demo archive, `hamer_ckpts` subtree | `checkpoints/hamer/hamer_ckpts/` |
| AnyHand `anyhand_wilor.ckpt`, `anyhand_hamer.ckpt` | `checkpoints/anyhand/` |
| EgoForce model_weights.pth and epoch_460.pth | `third_party/egoforce/_DATA/` |
| OmniHands Demo_Multiview.pth | `third_party/omnihands/checkpoints/` |
| StableHand dit_hot3d/model.pt and qn/model.pt plus args.json | `third_party/stablehand/save/` |
| Dyn-HaMR HMP encoder/NeMF weights and normalization files | `checkpoints/dynhamr/hmp_model/` |

The user-supplied MANO assets are required; no model files are committed.
`checkpoints/mano_converted` comes from the ACE converter above. The ParaFit
adapter makes a separate sparse-regressor copy under `checkpoints/parafit/models`.
StableHand's original demo also needs these under its `data_loaders/mano_models`.
It uses released cached features and GT subject shape: it is **not** a fair
predicted-input stereo baseline. Its data preprocessing remains unreleased in the
inspected commit. The HMP experiment is a prior component, not full Dyn-HaMR.

Reproduce measured variants with `scripts/benchmark_mesh_regressor.py`,
`benchmark_parafit.py`, `benchmark_hmp.py`, and `refine_fitted.py`; their defaults
and frozen experiment JSON files record all settings. AnyHand is a checkpoint
override of the corresponding WiLoR/HaMeR adapter. EgoForce `--forearm` uses its
released forearm detector but retains WiLoR hand detection; this is an adaptation,
not the exact official tracking/TensorRT demo.

For the recommended GT-free pipeline, run from the repository root:

```bash
.venv-extra/bin/python scripts/run_inference.py \
  --model wilor --mano-fit adam --temporal-seconds .05 \
  --left data/example/left.mp4 --right data/example/right.mp4 \
  --calibration data/example/calibration.json --output outputs/example-wilor
```

Add `--rotate-cw` only for sideways native HOT3D images. Add `--timestamps` for
an NPZ of synchronized `left`/`right` nanosecond timestamps. Undistort fisheye video
and provide its corresponding pinhole calibration first. Outputs always use the
original left optical camera frame in meters. `original_observation_type` retains
the initial stereo mask; fitted/temporal points are classified separately. Wrist
rotation is currently unavailable (NaN); MANO parameters saved by this CLI are
initializers, not fitted output parameters. HMP requires per-frame camera/world
tracking and is not the recommended generic stereo-video CLI.
