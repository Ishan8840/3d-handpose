"""Render the object experiment's measured tables, no external paper numbers."""
import json
from pathlib import Path
R=Path('reports/object_aware');r=json.load(open(R/'held_out_hand_results.json'));dev=json.load(open(R/'development_hand_results.json'));s=json.load(open(R/'frozen_selection.json'));a=json.load(open(R/'paired_analysis.json'));obj=json.load(open(R/'held_out_object_results.json'));w=f"{s['weight']:g}"
labels=[('WiLoR stereo (preserved)','wilor_stereo'),('Existing MANO + 50 ms temporal','existing_mano_temporal'),('A: controlled MANO refit','hand_only_w1'),('B: mask visibility','visibility_w1'),('C: estimated object + collision',s['collision']),('E: estimated object + contact',s['contact']),('D: reference pose, same acceptance','collision_oracle_w'+w),('Oracle: collision, all selected','collision_oracle_all_w'+w),('Oracle: contact, all selected','contact_oracle_all_w'+w)]
headers='| Pipeline | Abs MPJPE | Wrist | Fingertips | Coverage | P90 | P95 | Depth MAE | >100 mm joints |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'
rows=[]
for title,k in labels:
 m=r['pooled'][k];rows.append(f"| {title} | {m['absolute_mpjpe_mm']:.3f} | {m['wrist_mm']:.3f} | {m['fingertips_mm']:.3f} | {m['joint_coverage']*100:.2f}% | {m['p90_mm']:.3f} | {m['p95_mm']:.3f} | {m['depth_mae_mm']:.3f} | {m['catastrophic_observed_joint_fraction_gt100mm']*100:.2f}% |")
table=headers+'\n'.join(rows)
paired=[]
for title,k in labels:
 if k not in a:continue
 d=a[k];paired.append(f"| {title} | {d['paired_delta_mm']:+.3f} | {d['paired_sequence_ci95'][0]:+.3f} to {d['paired_sequence_ci95'][1]:+.3f} |")
occ=[]
for title,k in labels:
 m=r['pooled'][k];value=m['selected_object_occluded_fingertip_error_mm'];occ.append(f"| {title} | {value:.3f} | {m['selected_object_occluded_fingertips_observed']}/{m['selected_object_occluded_fingertips_eligible']} |")
objecttable='\n'.join(f"| {key} | {v['count']} | {v['translation_origin_mm']:.2f} | {v['translation_centroid_mm']:.2f} | {v['symmetry_reduced_rotation_deg']:.2f} | {v['adds_mm']:.2f} |" for key,v in obj['summary'].items())
text=f'''# Object-aware stereo hand reconstruction: measured experiment

**Recommendation: retain the existing WiLoR pipelines.** This implementation did not
demonstrate a practically meaningful object-aware improvement. Estimated-object
contact changed the controlled refit by {a[s['contact']]['paired_delta_mm']:+.4f} mm;
even reference-pose contact on all selected objects changed it by
{a['contact_oracle_all_w'+w]['paired_delta_mm']:+.4f} mm. These results do not justify
replacing raw WiLoR stereo or the existing MANO + temporal pipeline.

All hand errors below are **absolute native-left-camera MANO21 errors in millimeters**,
without alignment. Evaluation covers the same 450 frames from three separate
participants as the previous baseline; development used another 450 frames from
three participants. This is a controlled regression cohort, not a newly collected
held-out confirmation set. Each clip contains 150 consecutive frames (five seconds).

{table}

A is a new controlled SMPL-X MANO refit used to isolate the added losses. It is not an
exact rerun of the historical parafit Adam + 50 ms temporal implementation; that
pipeline and the raw 22.8916 mm baseline are both preserved and shown separately.
B/C/E/D share A's hand eligibility and initialization. Extra fitted joints are inferred,
not newly stereo-observed. Observed-joint errors and coverage must be read together.

## Paired changes from the controlled hand-only refit

Negative changes improve accuracy. Intervals bootstrap entire sequences, not individual
frames. Only three evaluation sequences are available, so intervals do not establish
broad generalization. The point estimate weights shared joints; intervals use per-frame
mean deltas clustered by sequence (weighting differs if partial frames are present).

| Pipeline | Paired MPJPE change (mm) | Sequence bootstrap 95% interval (mm) |
|---|---:|---:|
{chr(10).join(paired)}

The complete [paired comparisons](held_out_hand_results.json) include comparisons
against raw WiLoR, visibility-only and the refit control. Each result also contains
capped-100 mm error with missing predictions penalized, per-joint error, depth bins,
catastrophic-frame frequency, and exact observed-joint counts.

## Object pose quality before hand refinement

| Population | Frames | Origin translation mm | Centroid translation mm | Symmetry-reduced angle deg | ADD-S mm |
|---|---:|---:|---:|---:|---:|
{objecttable}

Selected objects are chosen using predicted right-hand proximity in both supplied
masks. Rejected or absent object estimates do not remove hand frames from evaluation.
The conservative pose gate accepts only {obj['summary']['accepted']['count']} of 450
frames. Its hypothesis spread is not a calibrated probability and did not reliably
select the most accurate estimates. Origin translation is sensitive to rotation for
objects with off-center origins; centroid translation is reported independently.
ADD-S uses nearest sampled surface points and is a diagnostic, not a substitute for
symmetry-aware pose accuracy. Continuous rotational symmetries use 3° sampling.

![Supplied masks with reference and estimated mesh projections](object_mask_alignment.png)

Visible masks do not reveal the complete silhouette. The convex-silhouette estimator
can explain a cropped or occluded observation with incorrect object depth/orientation.
The reference projection check confirms metric camera/mesh placement but also makes
this incomplete-observation failure visible. Geometric ambiguity remains even when
image fit residuals are small.

## Occluded fingertips

This uses reference mesh ray intersections for the **selected object's occlusion in
the left camera**, evaluated only after inference. It does not label all hand/object
occlusion. No reference hand poses enter fitting. The denominators below expose
missing predictions in this small difficult subset.

| Pipeline | Occluded fingertip error (mm) | Observed / reference-eligible fingertips |
|---|---:|---:|
{chr(10).join(occ)}

Contact-point error is unavailable: this experiment has no verified ground-truth
contact annotations. Heuristic contact weights must not be described as correct
contact labels. Collision uses a filled 3 mm voxel representation of the original mesh,
with a 2 mm soft penetration allowance, so tiny physical distances are approximate.

## Selection, correlations, and interpretation

The shared collision/contact weight is **{s['weight']}**, chosen from development
weights 0.25 and 1 using mean capped error with missing-joint penalties. Shared weight
keeps the contact-vs-collision ablation controlled. The frozen configuration was saved
before any held-out hand fitting. Other settings were fixed for this experiment.

[Paired analysis](paired_analysis.json) reports per-clip changes, improved/worsened
joints, and descriptive Spearman correlations between estimated object-centroid error
and changes in hand error. Correlations on the small accepted-frame subset are
exploratory; temporally adjacent frames are not independent evidence.

A known object excludes physically impossible hand locations but generally does not
uniquely localize the hand. Nonpenetration supplies no attraction when an erroneous
hand remains outside the object. The conservative contact gate also cannot repair a
large depth outlier when the initial fingertip is far from the surface. Wrong object
pose can introduce a new common translation bias across many joints. Perfect object
pose alone does not supply finger contact identity or resolve invisible articulation.

The oracle measures these particular constraints with reference object localization.
It is **not the maximum possible improvement** achievable with accurate object pose.
The all-selected oracle also increases constraint availability, so compare it separately
from the same-acceptance oracle. This is ContactOpt-inspired geometric optimization,
not execution of ContactOpt's learned contact prediction model.

![Error distributions, joints, depth and object-error relationship](error_analysis.png)

![Diagnostic best, median and worst oracle changes; reference in green](hand_comparison.png)

Overlay examples are deliberately selected from the best, median and worst paired
oracle changes, not random representative samples. Magenta is the reconstruction and
green the reference. The [runtime record](runtime.json) reports hand fitting and object pose separately; pose estimation
and mesh-field preparation are separate costs from the recorded hand fitting time.

## Reproduction and remaining work

See [protocol and commands](README.md), [frozen selection](frozen_selection.json),
[development results](development_hand_results.json), [mesh/data provenance](provenance.json),
and [artifact hashes](artifact_hashes.json). Model predictions, surfaces, masks and poses
remain outside Git; aggregate measurements, code, configurations and figures are committed.

No new model survey, training, dense stereo model or paid provisioning was performed.
The next stronger test requires object pose from evidence robust to occlusion and image
cropping, calibrated uncertainty, better contact evidence, and fresh participant-separated
clips. Further work should first show paired improvement on development clips before
any replacement of the preserved WiLoR pipeline. Nothing here demonstrates sub-10 mm
absolute accuracy or >95% coverage.
'''
(R/'report.md').write_text(text)
(R/'results_table.md').write_text(table+'\n')
print(table)
