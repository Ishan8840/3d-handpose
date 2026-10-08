# Object-aware experiment protocol

This experiment preserves the existing WiLoR stereo outputs. It tests whether known
object geometry improves metric MANO21 hand estimates, not just mesh appearance.
Results and configuration selection are generated from actual saved executions.

## Inputs and reference isolation

The six existing HOT3D Quest3 clips provide 900 synchronized monochrome stereo frames:
450 development frames from P0003/P0013/P0018 and 450 regression evaluation frames
from P0010/P0017/P0021. These are the previously used held-out participants, **not a
fresh confirmation cohort**. All frames remain in the denominator. SHOW3D common19
is not mixed into this MANO21 experiment.

`prepare_object_inputs.py` uses official camera projections to resample supplied
SAM2 `masks_modal` into 160×160 pinhole views. The mask source's prompting provenance
is not established; this is explicitly **mask-conditioned** reconstruction.
`masks_amodal` are generated using reference object poses and are excluded.
Mesh identity is known. Exact public GLB meshes retain their scene transforms and
meter scale. Object-pose references are exported to separate `_reference.npz` files.

`estimate_objects.py` reads no reference poses or hand labels. It selects the object
nearest the predicted right hand in both masks, then fits the convex silhouette of
the metric mesh in both cameras. Eight fixed-seed initial rotations and the previous
accepted pose initialize robust least squares. Fit cost and near-optimal hypothesis
translation spread gate acceptance. These are uncertainty diagnostics, not calibrated
confidence. Symmetry-reduced rotation errors use released BOP model symmetries
(continuous symmetries sampled at 3°); no unique observable orientation is claimed.

## Hand fitting and controls

- Raw WiLoR stereo remains unchanged.
- A: MANO hand-only fit to predicted stereo landmarks and both camera views.
- B: A with reduced landmark reprojection weights inside supplied visible object masks.
- C: B plus predicted object pose and soft mesh nonpenetration.
- E: C plus gated soft fingertip-surface contact.
- D: C with reference object pose, keeping estimated-pose acceptance fixed.
- Additional oracle: reference poses for every mask-selected object, with collision
  alone or collision plus contact. This exposes the cost of rejecting uncertain poses;
  it changes constraint availability, so it is separate from D.

Subject shape is the median predicted WiLoR shape. Fitting uses no reference hand
shape or pose. Missing poses are not filled across empty frames. Eligible MANO fits
can infer missing joints; original stereo validity is retained. A/B/C/E use the same
hand-fitting eligibility, so their error comparisons do not benefit from dropping frames.

The full object mesh is voxelized at 3 mm for a differentiable signed distance field.
This approximates exact surface collision. A 2 mm soft penetration allowance avoids
claiming sub-voxel physical precision. Contact weights require predicted fingertip
surface proximity and support near the visible mask boundary in **both views**.
They are uncalibrated geometric weights, not contact annotations or learned probabilities.
No universal fingertip attraction is imposed. No reference contact labels are available,
so contact-point accuracy is not reported as a measured outcome.

[ContactOpt](https://github.com/facebookresearch/ContactOpt) predicts desirable surface
contact and optimizes hand pose using that prediction. This experiment is inspired by
that formulation but **does not reproduce its pretrained contact predictor** or its
published benchmark. Its published gains cannot be transferred to HOT3D by assumption.

## Known estimator limitations

Convex silhouette fitting loses concavities, and visible silhouettes can be truncated
by image borders or occluding hands. Matching those incomplete silhouettes can move
or rotate a mesh incorrectly. Similar silhouettes do not establish accurate metric
pose. Translation spread of object origins is also sensitive to rotational ambiguity
when the mesh origin is off-center; centroid translation error is reported separately.
Object depth, hand depth and contact evidence can share the same observation errors.

The reference-pose oracle is a diagnostic of these specific losses and initializations.
It is **not a theoretical upper bound** on improvements achievable with object knowledge.
The current experiments do not include dense stereo depth, learned contact inference,
new network training, or temporal hand-object optimization.

## Reproduction

Use an existing CUDA PyTorch environment, then install `smplx==0.1.28` and
`pip install -e '.[objects]'`. The execution
VM used PyTorch 2.11.0+cu128, smplx 0.1.28, trimesh 4.6.8 and rtree 1.4.1. Keep the
already installed research-model environment rather than replacing its PyTorch stack.
Public clips and authorized converted MANO assets are prerequisites.

```bash
PYTHONPATH=src python scripts/prepare_object_inputs.py
PYTHONPATH=src python scripts/estimate_objects.py --split development
PYTHONPATH=src python scripts/estimate_objects.py --split held_out
PYTHONPATH=src python scripts/run_object_hand.py --split development --mode hand_only
PYTHONPATH=src python scripts/run_object_hand.py --split development --mode visibility
PYTHONPATH=src python scripts/run_object_hand.py --split development --mode collision --weight 1
PYTHONPATH=src python scripts/run_object_hand.py --split development --mode contact --weight 1
# Also run collision/contact with --weight .25; select only from development scores.
PYTHONPATH=src python scripts/object_occlusion_labels.py --split development
PYTHONPATH=src python scripts/evaluate_object_hand.py --split development
# Frozen weight is .25. Evaluation runs (same 450 frames):
PYTHONPATH=src python scripts/run_object_hand.py --split held_out --mode hand_only
PYTHONPATH=src python scripts/run_object_hand.py --split held_out --mode visibility
PYTHONPATH=src python scripts/run_object_hand.py --split held_out --mode collision --weight .25
PYTHONPATH=src python scripts/run_object_hand.py --split held_out --mode contact --weight .25
# Oracle diagnostics:
PYTHONPATH=src python scripts/run_object_hand.py --split held_out --mode collision --oracle --weight .25
PYTHONPATH=src python scripts/run_object_hand.py --split held_out --mode collision --oracle-all --weight .25
PYTHONPATH=src python scripts/run_object_hand.py --split held_out --mode contact --oracle-all --weight .25
PYTHONPATH=src python scripts/object_occlusion_labels.py --split held_out
PYTHONPATH=src python scripts/evaluate_object_pose.py --split held_out
PYTHONPATH=src python scripts/evaluate_object_hand.py --split held_out
PYTHONPATH=src python scripts/analyze_object_results.py
PYTHONPATH=src python scripts/report_object_experiment.py
```

Standalone refinement consumes saved predictions from the existing stereo pipeline,
calibrated masks, known mesh identities, and estimated object poses; it needs no hand GT:

```bash
PYTHONPATH=src python scripts/run_object_inference.py \
  --observations outputs/wilor-held/stereo/clip-000648/predictions.npz \
  --mano-parameters outputs/wilor-held/stereo/clip-000648/mano_parameters.json \
  --object-input outputs/object-input/clip-000648.npz \
  --object-pose outputs/object-pose/clip-000648.npz \
  --mesh-dir data/hot3d/object_models \
  --mode contact --weight .25 --output outputs/object-example
```

Inputs and outputs use native left optical axes (+x right, +y down, +z forward), meters,
and anatomical MANO21 joint ordering. Masks use the documented 160px pinhole calibration;
stored hand landmarks use the corresponding 640px views. WiLoR's upright MANO orientation
is explicitly rotated back to native optical coordinates before fitting.
