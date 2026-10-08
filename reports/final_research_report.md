# Expanded stereo hand-pose research report

## Recommendation

Use **WiLoR observations in both calibrated views → confidence-gated stereo
triangulation → MANO fitting with Adam → 50ms local-linear temporal refinement**
for the best tested accuracy/coverage compromise. This recommendation is an
engineering choice across several metrics, not the winner of every individual
metric or proof of global state of the art.

On450 held-out HOT3D frames from three participants, the anatomical MANO-21
reference gives **24.27mm absolute MPJPE,
29.94mm wrist error, 26.98mm fingertip error,
67.33% joint coverage and 35.64mm P90**.
Median error is14.60mm. Complete21-joint frame coverage is also
67.33%. **The requested sub10mm and95% coverage
targets were not achieved.** Wrist orientation remains unavailable in the CLI.

Raw WiLoR stereo has lower observed MPJPE on fewer joints; HaMeR stereo has
better wrist/fingertip errors than the recommended fitted pipeline. HMP has100%
coverage and wins the original100mm-capped missing-penalty score, but makes much
less accurate poses. There is no single dominating model. The full tables and
threshold curves expose this tradeoff instead of silently changing the metric.

## Protocol and limits

HOT3D Quest stereo is monochrome, not RGB. SHOW3D and EgoStandard provide
RGB checks; grayscale-to-RGB replication does not make HOT3D color video.
All absolute metrics use meters in the original left camera optical frame,
converted to millimeters for reporting, with **no root, scale, rigid or
Procrustes alignment**. Missing predictions remain NaN. Capped100 means
mean(min(error,100mm)), with every missing annotated joint assigned100mm.
Coverage refers to annotated joint slots unless explicitly called frame coverage.

HOT3D development:450frames/P0003,P0013,P0018. Held-out:450frames/P0010,P0017,P0021.
Smoke:P0002/30frames. These are public training-partition clips, split by
participant for this project; no hidden test labels were used. Prior round
held-out results were already known. Expanded model/fitting configurations were
committed in`e14f95e` before their held-out runs; the later development-selected
50ms refinement was committed in`2b26be5` before applying it to held-out outputs.
No held-out values were used to tune numerical settings, but this is iterative
research on a reused small test set, not a fresh blind evaluation.

Native UmeTrack annotations only permit19 common joints. MANO-21 is a **separate
anatomical reference**, never mixed with native19 scores. UmeTrack lacks an
interchangeable anatomical wrist/thumb CMC and those remain missing. Pretrained
contamination is possible: ACE explicitly uses HOT3D and StableHand has HOT3D
weights. Complete checkpoint training-participant provenance is not established.

## Anatomical MANO-21 held-out comparison

| Method | Abs MPJPE mm | Wrist mm | Tips mm | Coverage % | P90 mm | Capped100 mm |
|---|---:|---:|---:|---:|---:|---:|
| WiLoR MANO Adam temporal | 24.27 | 29.94 | 26.98 | 67.33 | 35.64 | 45.69 |
| WiLoR MANO Adam | 24.90 | 30.03 | 28.16 | 67.33 | 37.57 | 46.22 |
| WiLoR MANO analytical LM | 24.98 | 27.44 | 28.82 | 67.33 | 40.17 | 46.61 |
| WiLoR stereo | 22.89 | 21.38 | 30.96 | 58.12 | 40.93 | 53.75 |
| HaMeR stereo | 23.05 | 18.74 | 24.90 | 57.23 | 44.87 | 55.37 |
| ACE hybrid | 27.18 | 30.90 | 25.82 | 58.01 | 52.09 | 57.41 |
| WiLoR HMP prior | 55.36 | 64.70 | 58.17 | 100.00 | 144.70 | 41.28 |

Relative to ACE, the recommendation improves the capped score by
11.72mm; paired sequence-bootstrap95% CI
[6.68,15.04]mm
for ACE minus recommended. Only three clusters are available: intervals are
coarse and cannot establish broad generalization. Full paired comparisons,
including differences statistically indistinguishable from zero, are in
[the anatomical comparison JSON](expanded_held_mano21/comparison.json).

The recommendation's PA-MPJPE is8.79mm, reported separately;
it is **not** its absolute positional accuracy.

## All executed expanded held-out variants: common19

| Method | Abs MPJPE mm | Wrist mm | Tips mm | Coverage % | P90 mm | Capped100 mm |
|---|---:|---:|---:|---:|---:|---:|
| ACE hybrid | 28.28 | — | 25.97 | 57.65 | 53.10 | 58.28 |
| MediaPipe rotations | 47.80 | — | 38.04 | 23.26 | 88.80 | 84.52 |
| UmeTrack two-view | 34.65 | — | 41.11 | 22.22 | 67.20 | 84.86 |
| WiLoR stereo | 23.91 | — | 30.84 | 58.21 | 42.90 | 54.13 |
| HaMeR stereo | 24.23 | — | 24.97 | 57.18 | 46.65 | 56.01 |
| WiLoR MANO analytical LM | 25.69 | — | 28.78 | 67.33 | 41.73 | 47.14 |
| WiLoR MANO Adam | 25.31 | — | 28.18 | 67.33 | 38.41 | 46.54 |
| WiLoR HMP prior | 55.42 | — | 58.33 | 100.00 | 146.93 | 41.05 |
| AnyHand WiLoR stereo | 24.89 | — | 34.22 | 58.05 | 45.22 | 55.01 |
| AnyHand WiLoR mono | 57.50 | — | 61.96 | 71.56 | 102.82 | 68.07 |
| AnyHand HaMeR stereo | 26.53 | — | 38.96 | 54.88 | 44.21 | 57.77 |
| AnyHand HaMeR mono | 47.11 | — | 51.57 | 71.56 | 84.16 | 61.04 |
| EgoForce hand-only stereo | 55.95 | — | 59.16 | 40.41 | 86.23 | 74.31 |
| EgoForce hand-only mono | 82.93 | — | 87.62 | 65.84 | 236.40 | 69.67 |
| OmniHands two-view stereo | 29.92 | — | 36.12 | 53.11 | 53.98 | 62.28 |
| OmniHands two-view lift | 103.86 | — | 108.19 | 69.78 | 190.26 | 80.84 |
| WiLoR mono | 39.93 | — | 44.17 | 71.56 | 69.77 | 56.96 |
| HaMeR mono | 44.04 | — | 50.07 | 71.56 | 71.36 | 59.05 |
| WiLoR MANO Adam temporal | 24.68 | — | 26.93 | 67.33 | 36.28 | 46.05 |

POEM was measured on development but did not advance. The original RTMPose,
MediaPipe/RTMPose fusion and dense-stereo pilots cover smaller subsets and are
retained in[the first-round report](round_one_research_report.md), not pooled
into this450-frame leaderboard. Monocular variants use calibrated translation
fitting from the predicted metric mesh; they do not use true scale or true crops.
OmniHands “lift” uses two image views and calibrated translation; it is not
monocular and its released network does not consume the stereo calibration.

## Development results and what improved

| Method | Abs MPJPE mm | Wrist mm | Tips mm | Coverage % | P90 mm | Capped100 mm |
|---|---:|---:|---:|---:|---:|---:|
| ACE hybrid | 25.97 | — | 31.17 | 50.42 | 41.53 | 61.44 |
| MediaPipe rotations | 120.18 | — | 103.00 | 55.13 | 610.73 | 63.82 |
| UmeTrack two-view | 86.65 | — | 89.94 | 49.33 | 70.75 | 65.29 |
| POEM two-view | 163.70 | — | 160.62 | 60.89 | 654.12 | 68.58 |
| WiLoR stereo | 30.84 | — | 34.47 | 51.72 | 38.42 | 58.28 |
| WiLoR mono | 96.18 | — | 100.83 | 57.33 | 141.70 | 75.62 |
| AnyHand WiLoR stereo | 37.98 | — | 42.70 | 51.50 | 44.19 | 60.16 |
| AnyHand WiLoR mono | 108.40 | — | 114.45 | 57.33 | 173.61 | 83.46 |
| HaMeR stereo | 34.06 | — | 33.85 | 50.98 | 40.76 | 59.27 |
| HaMeR mono | 100.39 | — | 106.25 | 57.33 | 151.65 | 79.75 |
| AnyHand HaMeR stereo | 36.86 | — | 35.74 | 51.13 | 41.06 | 59.39 |
| AnyHand HaMeR mono | 94.95 | — | 100.78 | 57.33 | 133.13 | 78.59 |
| EgoForce hand-only stereo | 36.20 | — | 44.93 | 47.53 | 52.56 | 64.17 |
| EgoForce hand-only mono | 61.56 | — | 65.81 | 55.54 | 58.39 | 62.04 |
| OmniHands two-view stereo | 39.80 | — | 41.53 | 49.39 | 53.52 | 64.49 |
| OmniHands two-view lift | 111.45 | — | 117.64 | 56.22 | 148.41 | 88.88 |
| WiLoR MANO analytical LM | 33.46 | — | 35.48 | 54.89 | 37.34 | 55.01 |
| WiLoR MANO Adam | 32.25 | — | 33.01 | 54.89 | 27.01 | 54.36 |
| WiLoR HMP prior | 35.34 | — | 35.55 | 66.67 | 107.17 | 52.45 |
| EgoForce forearm stereo | 33.84 | — | 40.40 | 46.32 | 49.31 | 64.74 |
| EgoForce forearm mono | 50.80 | — | 53.66 | 54.06 | 49.96 | 60.95 |

The analytical ParaFit solver is the released UA-Fit optimizer core, initialized
with predicted WiLoR pose/shape and observations. Adam uses the same measured2D,
stereo3D and pose-prior objective. Shared subject shape is the median of predicted
betas, never GT shape. Adam runs200steps/lr0.005, reprojection precision0.25,
stereo weight10000 and pose weight0.1; analytical LM runs30iterations. Complete
fitted joints may be inferred from anatomy and must not be called directly
stereo-observed. Analytical LM and Adam are close; inspect paired intervals
rather than treating their small difference as decisive. On anatomical held-out
data, LM minus Adam is0.39mm capped score (95% CI[-0.33,1.16]); Adam minus
50ms refinement is0.52mm (CI[-0.70,1.75]). Both intervals include zero: these
small improvements are statistically indistinguishable under this three-sequence
bootstrap, despite the point-estimate ranking.

A development-only ensemble sweep tried ordered source pairs, weights0.25/0.5/0.75,
agreement gates2/5/10cm, and optional100ms smoothing. Best capped score54.91mm
was worse than MANO Adam54.36mm and HMP52.45mm; its observed MPJPE86.40mm included
large errors. It was rejected before held-out execution. No GT oracle is deployed.

The learned HMP prior is an actual latent-optimization experiment using released
Dyn-HaMR weights, with known camera trajectories and predicted WiLoR pose/shape.
It is **not full Dyn-HaMR**. It fills long missing spans and achieves100% held-out
coverage, but approximately55mm observed error and145mm P90. Appearance of smooth
motion is insufficient evidence of correct metric position.

For the fitted pipeline,50ms local refinement improved development error
32.25→31.74mm without increasing coverage.100ms and200ms spread failures and
worsened observed error. On held-out common19,50ms improved25.31→24.68mm at
unchanged67.33% coverage. Per-sequence acceleration error fell from
20.54/40.72/16.37 to7.02/18.27/6.13 m/s² against the anatomical reference.
These are camera-frame metrics on valid triplets, not world-motion stability. Both smoothed/inferred and original observation masks
are preserved. The time window is a ±50ms neighborhood; gaps must be bracketed
and at most100ms, and at least3 observations are needed.

FoundationStereo was executed in round one. The calibrated dense-depth pilot
showed a small single-sequence gain; dense+MANO changed23.865→23.771mm at equal
coverage. This is insufficient evidence to add it to the selected pipeline.
No new model training was performed.

## SHOW3D cross-dataset check

The same900development frames from three distinct SHOW3D scenes were evaluated,
using predicted crops and each frame's official camera transforms. Configurations
were transferred from HOT3D. This is an external development check, not the
unexecuted SHOW3D held-out partition; its annotation convention is separate.

| Method | Abs MPJPE mm | Wrist mm | Tips mm | Coverage % | P90 mm | Capped100 mm |
|---|---:|---:|---:|---:|---:|---:|
| MediaPipe rotations | 20.49 | — | 26.88 | 13.50 | 41.57 | 89.26 |
| ACE hybrid | 30.13 | — | 33.39 | 60.94 | 58.77 | 57.40 |
| WiLoR stereo | 20.70 | — | 22.65 | 60.19 | 36.95 | 51.28 |
| HaMeR stereo | 20.51 | — | 23.22 | 58.10 | 37.30 | 52.90 |
| WiLoR MANO analytical LM | 22.03 | — | 23.61 | 79.22 | 39.43 | 37.30 |
| WiLoR MANO Adam | 21.19 | — | 22.95 | 79.22 | 38.83 | 36.65 |
| WiLoR MANO Adam temporal | 19.53 | — | 20.47 | 82.78 | 35.57 | 32.79 |

## Correlated errors and complementary information

![Error correlation and shared failures](expanded_analysis/overlap.png)

A failure means missing or>20mm. Spearman correlation uses **only jointly observed
joints**; Jaccard measures overlap of failure sets including misses. Joint samples
are correlated, so these descriptive correlations are not independent-sample
significance tests.

| Pair | Error Spearman | Failure Jaccard | Both missing % | Error-vector cosine |
|---|---:|---:|---:|---:|
| ACE hybrid vs WiLoR stereo | 0.21 | 0.64 | 32.87 | 0.28 |
| ACE hybrid vs WiLoR MANO Adam | 0.24 | 0.63 | 29.75 | 0.31 |
| WiLoR stereo vs HaMeR stereo | 0.52 | 0.74 | 36.89 | 0.61 |
| WiLoR stereo vs AnyHand WiLoR stereo | 0.55 | 0.76 | 37.23 | 0.60 |
| HaMeR stereo vs AnyHand HaMeR stereo | 0.46 | 0.75 | 37.93 | 0.56 |
| WiLoR stereo vs WiLoR MANO Adam | 0.57 | 0.73 | 32.30 | 0.70 |

ACE and WiLoR provide partly different localization errors, yet often fail on the
same hard observations. WiLoR, HaMeR and their AnyHand variants share the WiLoR
hand detector/crop protocol here; correlated misses therefore cannot be attributed
solely to their pose networks. Their related mesh regressors also predict2D by
projecting a reconstructed hand, so their landmark errors are not independent
heatmap evidence. AnyHand fine-tuning did not improve the overall held-out metric
versus the corresponding original stereo models.

On development, the WiLoR/AnyHand-WiLoR monocular error correlation was0.976,
consistent with strongly shared depth/scale weaknesses. Stereo reduces that
ambiguity using the known metric baseline. The detector scores are not calibrated
per-joint uncertainty, which limits simple confidence-weighted fusion.

For held-out ACE/WiLoR, a GT-selecting per-joint oracle reaches45.71mm capped
score versus54.13mm for raw WiLoR. That is diagnostic headroom, **not an executable
pipeline**; the tested deployable fusion did not realize it. Detailed per-sequence
and per-joint overlap statistics are in[the overlap JSON](error_overlap_held.json).

![Accuracy counting every annotated slot](expanded_analysis/accuracy_coverage.png)

- WiLoR MANO Adam temporal: 43.79% of all annotated common19 slots within20mm, counting misses as failures.
- WiLoR stereo: 36.76% of all annotated common19 slots within20mm, counting misses as failures.
- ACE hybrid: 24.76% of all annotated common19 slots within20mm, counting misses as failures.
- WiLoR HMP prior: 37.42% of all annotated common19 slots within20mm, counting misses as failures.

The per-frame mean error decomposition finds that much of MANO-fitting residual
energy is shared translation, rather than independent finger distortion. For
unsmoothed MANO Adam,94.5% of squared error is in this common component and65.2%
in the depth axis. These are unaligned diagnostic decompositions, heavily affected
by outliers, not causal estimates or improvements applied to predictions.

Inspected failures include a hand cut by the image border, overlapping hands,
and a catastrophic stereo-depth/fit failure onclip001035 frame42 exceeding1m.
A plausible projected skeleton does not certify depth. The recommendation still
has serious tails and coverage gaps; no post-hoc GT-based frame deletion was used.
[Worst-case overlays](expanded_analysis/failures/index.json) retain these cases.

## Execution matrix and remaining blockers

| Requested method/data | What actually executed | Limit |
|---|---|---|
| MediaPipe, RTMPose, detector fusion | Stereo baseline pilots and full MediaPipe comparisons | RTMPose/fusion pilots are smaller subsets |
| ACE | Official calibrated inference per view; mono/stereo/hybrid, HOT3D and SHOW3D | HOT3D pretraining contamination possible |
| POEM-v2 | Official demo plus predicted-crop two-camera development benchmark | Demo's provided crops are not fair benchmark evidence |
| UA-Fit | Actual released ParaFit analytical optimizer; matched Adam objective | Learned uncertainty checkpoint not located; not full UA-Fit |
| UmeTrack | Official smoke plus adapted two-view predicted-crop benchmark | Not a four-camera result; wrist convention differs |
| FoundationStereo | Rectified dense-depth and surface+MANO pilots | No convincing cross-sequence gain established |
| WiLoR, HaMeR | Released weights, predicted crops, mono/stereo, HOT3D and SHOW3D | Projected mesh landmarks, shared detector |
| AnyHand | Both released WiLoR and HaMeR fine-tuned checkpoints | Same detector/crops; no training performed |
| EgoForce | HALO hand-only stereo/mono; additional released-forearm-detector development ablation | Adapted hand detector; not official temporal/TensorRT tracker |
| OmniHands | Released multiview checkpoint with two predicted-crop views | Calibration used after network; missing opposite hand uses full-image crop |
| StableHand | Official cached-feature clip002736 demo,20steps/seed42 | GT shape conditioning; unreleased preprocessing; excluded from fair leaderboard |
| Dyn-HaMR | Released HMP prior, actual latent optimization | Full camera/hand optimization pipeline not reproduced |
| EgoHandICL | Repository and pinned HF release inspected | Referenced handicl.egohandicl_new module and usable released checkpoint not located; no inference claimed |
| HOT3D | 30smoke+450development+450held-out frames | Only3 held-out participants, reused test set |
| facebook/show3d-dataset | 900-frame multi-method development comparison | SHOW3D held-out partition not evaluated |
| LightwheelAI/EgoStandard | Authorized stereo sample downloaded;547-frame inference | Anatomical joint ordering/pose units remain unverified; no quantitative GT accuracy |

StableHand's measured demo world MPJPE was44.17mm (right47.13mm); PA2.63mm is
separate. This protocol conditions on GT shape and cached features, so it is not
ranked with predicted-input stereo methods. The supplied EgoStandard dataset card
does not resolve joint semantics. No access restrictions were bypassed.

## Inference and reproducibility

```bash
.venv-extra/bin/python scripts/run_inference.py \
  --model wilor --mano-fit adam --temporal-seconds .05 \
  --left data/example/left.mp4 --right data/example/right.mp4 \
  --calibration data/example/calibration.json --output outputs/example-wilor
```

Add`--rotate-cw` for native sideways HOT3D only; optionally provide synchronized
nanosecond timestamps. Input must be calibrated pinhole stereo (undistort first).
Output coordinates retain the original left optical frame,+x right,+y down,+z
forward, meters, canonical21-joint order. All required NPY/NPZ/metadata/video files
are exported, without GT. Wrist rotations are NaN. Saved MANO parameters are
predicted initializers, not fitted output parameters. Original stereo, anatomical
fit and temporal provenance are distinct; confidence is not a calibrated joint
probability. The bare CLI default remains the lightweight MediaPipe baseline;
use the explicit command above for the recommendation.

Pinned source commits, checkpoint hashes, release revisions, environment versions
and all experiment configurations are under`configs/`. See[setup details](../docs/model_setup.md).
26 weight-free regression tests pass, including actual video decode/encode and
metric export through both lightweight and mesh CLI routes. A real30-frame GPU
inference integration also passed. Clean heavy-model installation on a different
machine is not certified; recorded compatibility changes and environment snapshots
make the tested environment auditable.

Concurrent benchmark timings include different startup/caching loads and are not
used to rank speed. Dedicated CLI profiles are reported separately when available;
device-wide sampled GPU memory is not exact allocator peak. Accuracy was prioritized.

## Further research justified by these results

Prioritize reliable right-hand identity under overlap/cropping, calibrated
landmark uncertainty, and depth-outlier rejection learned/tuned only on fresh
development data. Test a wrist/translation initializer independently from finger
articulation rather than averaging whole related meshes. Obtain more unseen
participants and verify pretrained-data overlap before generalizing this ranking.
A broader dense-depth or learned temporal study is justified only with accurate
visible-surface association and explicit missing-frame scoring. EgoStandard needs
authoritative joint/coordinate documentation before anatomical benchmarking.

Dedicated final30-frame CLI: exit0, wall time25.85s, peak sampled device memory[3368]MiB (cold process, includes startup).

EgoStandard unsmoothed WiLoR+MANO integration: 547frames, 81.90% valid output joint slots. No GT accuracy is inferred from this coverage.

### Recorded expanded-run timing (not controlled speed ranking)

| Method | Mean seconds/frame, both reconstruction variants |
|---|---:|
| WiLoR | 0.113 |
| HaMeR | 0.124 |
| EgoForce hand-only | 0.230 |
| OmniHands | 0.249 |

These per-frame timers exclude model startup and input decoding, include mono/lift plus stereo export computation, and ran with concurrent jobs. MANO fitting and temporal refinement add work. Use the dedicated end-to-end CLI profile for the deployable pipeline; no controlled all-model steady-state speed claim is made.


## Object-aware follow-up

The controlled 900-frame development/evaluation experiment is documented in
[the object-aware report](object_aware/report.md). Estimated object collision/contact
and reference-object-pose diagnostics did not deliver a meaningful absolute MPJPE
improvement over their matched hand-only refit. Existing WiLoR baselines remain
unchanged. See the report for the small occluded-fingertip effects, pose ambiguity,
paired intervals, correlation analysis and runnable inference commands.
