# Executed model environments

The baseline clean installation is in README.md. These commands describe the
research environments executed on the supplied A100/CUDA12.8 VM. Exact installed
versions are in configs/environments; source revisions are in configs/sources.lock.json.
Original upstream environments differ. A clean installation of every heavy model
on an arbitrary machine has not been validated.

## Source checkouts

Clone the upstream repositories into the names used by sources.lock.json and
checkout their recorded commits. Do not substitute the latest branch. ACE expects
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
inference for evaluation. Full official-demo reproduction remains outstanding.

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
