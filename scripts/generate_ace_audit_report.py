"""Generate the ACE audit tables from executed experiment JSON, never estimates."""
import json
from pathlib import Path

report=Path('reports/ace_audit')
load=lambda name:json.loads((report/(name+'.json')).read_text())
rows=[('Archived held-out, 640 / 150','saved_decode',450),
      ('Archived development, 640 / 150','development_decode',450),
      ('Archived held-out first 81','saved_decode_first81',243),
      ('Fresh held-out, 640 / 81','fresh_640_81',243),
      ('Fresh held-out, 480 / 81','fresh_480_81',243)]
text=['# ACE-Ego-Hand reproduction audit — 2026-10-08',
      '', 'The historical 107.48 mm result was an auxiliary-head measurement, not '
      'the complete official MANO output. Correcting that distinction does not '
      'resolve the large camera-frame placement error. Fresh official K-given '
      'inference reproduces the archived behavior. Keep the WiLoR stereo baseline.',
      '', '## Executed experiments', '',
      'Every row below uses anatomical MANO-21 and the right hand in metric left-camera '
      'coordinates. No alignment is applied to absolute MPJPE. Coverage includes '
      'all labeled frames, including off-screen hands; it is not paper detection recall.',
      '', '| Run (pixels / input frames) | Scored frames | Absolute MPJPE mm | Wrist mm | Wrist-relative mm | PA mm | Coverage % |',
      '|---|---:|---:|---:|---:|---:|---:|']
for label,name,frames in rows:
    m=load(name)['pooled']['final_mano']
    text.append(f"| {label} | {frames} | {m['absolute_mpjpe_mm']:.2f} | {m['wrist_mm']:.2f} | {m['wrist_relative_mpjpe_mm']:.2f} | {m['pa_mpjpe_mm']:.2f} | {100*m['joint_coverage']:.2f} |")
text+=['', 'Six archived sequences contain 900 unique frames: three development '
       '(000175, 000838, 001119) and three held-out (000648, 001035, 001168), '
       '150 each. The six fresh executions reuse the first 81 frames of the three '
       'held-out sequences: 243 unique frames, not six new episodes. The historical '
       '107.48 mm common19 result remains a separate evaluation track.',
       '', '## Paper-style metrics', '',
       'MPJPE-p is wrist-relative; PA-p uses similarity alignment. CT-p measures '
       'MANO translation, not wrist position. Detection uses presence, on-screen '
       'gating and projected mesh overlap. Misses receive a canonical-hand cost; '
       '2D misses receive the image diagonal. Published evaluation uses 437 '
       '81-frame segments and 480×480 inputs. '
       '[Paper, Table 1 and Appendix A](https://arxiv.org/html/2608.20308v3).', '',
       '| Run | MPJPE-p mm | PA-p mm | EPE2D-p px | GO-p deg | CT-p m | Recall | F1 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|',
       '| Published HOT3D K-given (different split) | 12.888 | 6.436 | 6.418 | 7.924 | 0.025 | 0.998 | 0.996 |']
for label,name in [('Archived held-out 640','paper_protocol'),('Archived development 640','development_paper'),
                   ('Fresh held-out 640 / 81','fresh_640_81_paper'),('Fresh held-out 480 / 81','fresh_480_81_paper')]:
    m=load(name)['pooled']
    text.append('| '+label+' | '+' | '.join(f'{m[k]:.4f}' for k in ['MPJPE-p_mm','PA-p_mm','EPE2D-p_px','GO-p_deg','CT-p_m','recall','F1'])+' |')
text+=['', 'These scores evaluate both hands. The 640-pixel EPE columns are not '
       'directly comparable to the paper’s 480-pixel values. JSON also contains '
       'per-side counts, frame accuracy, matched-only metrics and sensitivity results.',
       '', '**Exact evaluator parity is not certified.** Official upstream HEAD '
       '`59da83b` still lists evaluation and data preparation as unreleased. '
       'The implementation follows Appendix A.1, but the text does not fully '
       'resolve dilation semantics, side-selection order, or the canonical '
       'hand’s mean-pose convention. These choices are exposed and tested.',
       '', '| Run | Default MPJPE-p | Flat-hand placeholder MPJPE-p | Default PA-p | Flat-hand placeholder PA-p |',
       '|---|---:|---:|---:|---:|']
for name in ['paper_protocol','development_paper','fresh_480_81_paper','fresh_640_81_paper']:
    x=load(name); a=x['pooled']; b=x['sensitivity_flat_hand_canonical']
    text.append(f"| {name} | {a['MPJPE-p_mm']:.3f} | {b['MPJPE-p_mm']:.3f} | {a['PA-p_mm']:.3f} | {b['PA-p_mm']:.3f} |")
text+=['', 'Default canonical construction uses zero pose parameters with '
       '`flat_hand_mean=False`, matching ACE’s MANO convention. The flat-hand '
       'alternative is reported rather than hidden. Dilation and matching-order '
       'sensitivity are stored in each scoring JSON.',
       '', '## Controlled differences and error decomposition', '']
for name,pair in load('paired_comparisons').items():
    lo,hi=pair['ci95_sequence_bootstrap_mm']
    text.append(f"- {name}: {pair['delta_mm']:+.3f} mm on shared observed joints; sequence-bootstrap 95% interval [{lo:+.3f}, {hi:+.3f}] mm; {pair['paired_joints']} paired joints.")
text+=['', 'Intervals resample three sequences, not independent frames. They '
       'describe this small local set and do not establish population-wide performance.', '',
       '| Output | Wrist signed depth mm | Root MSE mm² | Articulation MSE mm² | Cross term mm² | Absolute MSE mm² |',
       '|---|---:|---:|---:|---:|---:|']
for name in ['saved_decode','fresh_640_81','fresh_480_81']:
    d=load(name)['decomposition']['final_mano']
    assert abs(d['absolute_mse_mm2']-d['root_mse_mm2']-d['articulation_mse_mm2']-d['cross_term_mm2'])<1e-6
    text.append('| '+name+' | '+' | '.join(f'{d[k]:.3f}' for k in ['wrist_signed_depth_mm','root_mse_mm2','articulation_mse_mm2','cross_term_mm2','absolute_mse_mm2'])+' |')
text+=['', 'For each joint, error equals wrist error plus wrist-relative error. '
       'Squared error decomposes into root, articulation and cross terms; '
       'MPJPE itself is not additive. The consistent positive wrist-depth bias '
       'survives final MANO decoding and both fresh resolutions.', '',
       '![Measured wrist depth trajectories](wrist_depth.png)', '',
       '## Verified conventions and provenance', '',
       '- Official K-given checkpoint SHA256: '
       '`33f328624ca532942b12d10c59b648bb9fc0801bcdfe59f35ba92645e51fc3ad`. '
       'Fresh runs verify this hash before inference. All 605 trainable backbone '
       'tensors loaded; inference logs report no missing/unexpected keys.',
       '- K-given uses supplied intrinsics in `mixed_pnp`; `self_ray_decode` '
       'defaults false. Exported `pred_intrinsics` is a diagnostic and does not '
       'turn this into a K-free run. `nv_sigma=0`, LoRA rank/alpha 64, tap 15.',
       '- Quest3 OVR624 source cameras are 1280×1024. Official forward lens '
       'projection maps pinhole rays into the source images. Square pinhole '
       'views use focal=size/2 and principal point=(size−1)/2. This is our '
       'documented 90-degree horizontal-FOV choice, not a verified copy of '
       'the authors’ unpublished rectification.',
       '- Clockwise upright rotation uses x′=−y, y′=x, z′=z. Its pixel transform '
       'commutes with projection. Input orientation was visually inspected. '
       'Frame timestamps are increasing with median period 33,333,333 ns.',
       '- MANO-21 uses the anatomical wrist, 16 native joints and five mesh '
       'fingertips (744, 320, 443, 554, 671), in canonical finger order. '
       'The common19 UmeTrack reference is never substituted.',
       '- Ground truth uses the official HOT3D MANO loader, including its '
       'left-hand shapedirs correction. ACE prediction decoding remains '
       'unchanged; its left model does not apply that correction. Right-hand '
       'absolute results are unaffected by this left-only distinction.',
       '- For camera rotation C and translation t, MANO translation becomes '
       'Cτ+t+Cj₀−j₀, where j₀ is the shaped MANO root. Reexpressed meshes agree '
       'with directly transformed meshes below 0.00015 mm. Camera numerical '
       'round trips agree below 1e-9 pixels; these test implementation consistency, '
       'not physical recalibration of the rig.',
       '- Fresh inference opens video images and camera calibration only. '
       '`hands.json` and subject shape are read by separate evaluation processes. '
       'No reference wrist, pose, box, or shape is supplied to inference.',
       '- Raw outputs, input videos, run commands, hashes and logs are backed up '
       'under `outputs/ace-audit*`; [artifact manifest](artifact_manifest.json), '
       '[source provenance](provenance.json), [environment](environment.txt) '
       'and [camera checks](camera_checks.json) are retained.', '',
       '## What the 107.48 mm result means', '',
       '1. **Evaluation/output mismatch: confirmed.** The old result selected '
       '`joints_cam_direct`, bypassing MANO and translation decoding. It also '
       'used common19, not paper MANO21, and did not use the paper’s detection '
       'penalties. The historical result is retained but relabeled.',
       '2. **Large local depth/localization error: confirmed.** The final official '
       'output remains inaccurate in absolute coordinates. Fresh 640-pixel '
       'inference closely reproduces the archived run; the 480-pixel control '
       'worsens error on shared observed joints in all three sequences.',
       '3. **Simple axis, translation-center, metric-unit, or checkpoint-load '
       'bug: not supported by the checks.** This does not prove every aspect '
       'matches the authors’ training pipeline.',
       '4. **Domain shift versus intrinsic model weakness: unresolved.** '
       'These are grayscale Quest views with our chosen pinhole rectification. '
       'The authors’ recording IDs, exact rectification, evaluator and prepared '
       'segments are unavailable. Resolution sensitivity demonstrates input '
       'sensitivity, but does not identify the cause of the published/local gap.',
       '', 'Our participants are separated between local development and held-out '
       'sets, but every clip comes from public HOT3D training archives. ACE trained '
       'on HOT3D; actual recording overlap cannot be excluded. “Held-out” here '
       'means held out from our tuning, not proven unseen by ACE.', '',
       '## Baseline decision and reproducibility', '',
       'WiLoR is unchanged. Recomputed archive scores still match 22.8916 mm '
       'absolute raw-stereo MANO21 and 24.2666 mm for the higher-coverage '
       'MANO/temporal variant. No evidence from this audit justifies replacing '
       'them with ACE monocular output. Low PA error alone is not a positional '
       'accuracy result. The next useful reproduction step requires the authors’ '
       'prepared HOT3D inputs, split manifest and evaluator; object-aware work '
       'can then proceed with the preserved baseline.', '',
       'Use the ACE environment in `environment.txt`, the pinned source commits '
       'and authorized converted MANO assets. Point ACE’s `third_party` symlink '
       'to the pinned VideoX checkout. Commands from the repository root:', '',
       '```bash',
       'python scripts/prepare_ace_audit.py',
       'python scripts/prepare_ace_audit.py --split development',
       'python scripts/download_ace.py',
       '(cd third_party/ace && python scripts/precompute_caption.py)',
       'python scripts/run_ace_audit.py --size 480 --frames 81',
       'python scripts/run_ace_audit.py --size 640 --frames 81',
       'python scripts/audit_ace_decode.py --predictions outputs/ace-audit-480-81 --report-name fresh_480_81',
       'python scripts/score_ace_paper.py --predictions outputs/ace-audit-480-81 --output reports/ace_audit/fresh_480_81_paper.json',
       'python scripts/audit_ace_camera.py',
       'python scripts/analyze_ace_audit.py',
       'python scripts/generate_ace_audit_report.py',
       'python -m pytest -q',
       '```', '',
       'The archive-decode and analysis steps require the preserved prediction '
       'and reference archives listed in the artifact manifest. They are not '
       'downloaded model weights or bundled Git data. Machine-readable per-sequence '
       'results are adjacent to this report. No hidden challenge ground truth was used.', '']
(report/'report.md').write_text('\n'.join(text))
