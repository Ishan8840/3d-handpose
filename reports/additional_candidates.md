# Expanded candidate execution audit

This supersedes the first-round release-only audit. Measured results and protocol
limits are in [the final report](final_research_report.md).

| Candidate | Executed status |
|---|---|
| WiLoR | Released detector/mesh checkpoint, calibrated mono/stereo, HOT3D development+held-out and SHOW3D; GT-free CLI and EgoStandard inference |
| HaMeR | Released mesh checkpoint with predicted WiLoR detector crops; same HOT3D and SHOW3D comparisons |
| AnyHand | Both WiLoR and HaMeR fine-tuned checkpoint overrides; development+held-out |
| EgoForce | Released HALO, official camera solver, hand-only development+held-out; released forearm detector added in a development ablation; adapted hand detection |
| OmniHands | Released two-view checkpoint, predicted crops, stereo/lift variants; development+held-out |
| StableHand | Official cached-feature clip002736 demo executed. Uses GT subject shape; preprocessing TODO remains in inspected README; excluded from fair comparison |
| Dyn-HaMR | Released HMP prior and actual latent fitting executed development+held-out; full Dyn-HaMR tracker not reproduced |
| EgoHandICL | Inspected pinned source and HF dataset release. Inference references missing `handicl.egohandicl_new` and `/checkpoints/icl_arctic.pth`; usable released inference checkpoint not located. No execution claimed |
| UA-Fit / ParaFit | Released analytic solver core executed against a matched Adam objective. Full UA-Fit learned uncertainty checkpoint not located |

Exact source commits, weight hashes and HF revisions are in
`configs/models/expanded_sources.json`, `expanded_checkpoint_hashes.json`, and
`expanded_releases.json`. Existing model environments and compatibility changes
are documented in `docs/model_setup.md`.
