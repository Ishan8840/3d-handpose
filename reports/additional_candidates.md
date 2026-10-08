# Additional candidate release audit

These are release checks, not benchmark results. None of the candidates below
has been included in the frozen held-out comparison. Published metrics are not
substituted for measured stereo metric errors.

- [EgoForce](https://github.com/dfki-av/EgoForce) provides inference/evaluation
  entry points and a model-weight download script. Its installation includes
  MMCV, TensorRT, AnyCalib, PyTorch3D and camera-specific support. This is a
  promising independent metric-position baseline; it has not been installed here.
- [WiLoR](https://github.com/rolpotamias/WiLoR) documents detector and reconstruction
  checkpoint downloads and full-image demo inference. The supplied MANO files
  address its hand-model dependency. No execution result is claimed.
- [HaMeR](https://github.com/geopavlakos/hamer) provides the official reconstruction
  implementation. Its 3D output/crop-camera translation must be verified against
  calibrated camera coordinates before comparing absolute error.
- [OmniHands](https://github.com/LinDixuan/OmniHands) and
  [EgoHandICL](https://github.com/Nicous20/EgoHandICL) repositories were inspected.
  A complete usable checkpoint/inference combination has not been verified by
  execution. Placeholder documentation is not evidence of a working release.
- [AnyHand](https://arxiv.org/abs/2603.25726) describes a synthetic RGB/RGB-D dataset;
  a specific fine-tuned checkpoint has not been selected or executed here.
- [Dyn-HaMR checkpoint repository](https://huggingface.co/Zhengdi/Dyn-HaMR/tree/main)
  was identified. Its temporal pipeline has not been reproduced. StableHand was
  not integrated.

These remain uncompleted research candidates, not proven hardware or access
blockers. The report's claims are limited to the methods actually executed.
