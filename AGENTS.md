# AGENTS.md

## Read this first
Before changing code, read these spec docs in full:
- `docs/CGC_IMPLEMENTATION_SPEC.md`
- `docs/CGC_FORMULA_CONVENTIONS.md`
- `docs/research_proposal_summary.md` (optional context only)

Then trace the current TrajFlow code path end-to-end before editing.
Do **not** modify by intuition.

## Mandatory files to inspect before editing
- `runner/train.py`
- `runner/utils/trainer.py`
- `trajflow/utils/init_objective.py`
- `trajflow/denoising/flow_matching.py`
- `trajflow/models/dmt_model.py`
- `trajflow/models/context_encoder/mtr_encoder.py`
- `trajflow/models/denoising_decoder/denoising_decoder.py`
- `trajflow/models/denoising_decoder/decoder_utils.py`
- `runner/utils/tester.py`
- `trajflow/datasets/waymo/waymo_eval.py`

## Project goal
This is **not** a generic reranking project.
This is **not** a multi-step flow-inference project.

The target is TrajFlow on WOMD with original efficient **one-step** inference preserved.
We only add a theory-aligned pipeline for:
- same-family organization monitoring
- three-way failure attribution
- offline guidance-quality labels
- lightweight late-layer correction
- candidate-wise residual reranking

The aim is to improve stability of canonical selection **without collapsing legitimate multimodality**.

## Absolute rules
1. Keep official training/inference one-step. No extra denoising loops, ODE loops, sampler loops, or repeated denoiser forwards.
2. Final output must remain **candidate-wise ranking and selection**.
3. Family is an **operational proxy**, not the final decision object.
4. Only **guidance-linked organization failure** is the direct correction target.
   - coverage failure = diagnosis only
   - ranking-only failure = diagnosis only
5. Audit mode may use more flow steps, but it must be fully isolated, optional, off by default, and excluded from main results/runtime.
6. Correction must be lightweight, local, bounded, and late-layer-first.
7. When CGC features are disabled, original repo behavior must remain unchanged.
8. Every vague concept must be operationalized by formulas, thresholds, or config fields.

## Error attribution taxonomy
For each sample, distinguish at least:
- `coverage_failure`
- `guidance_org_failure`
- `ranking_only_failure`

Only `guidance_org_failure` may produce positive correction supervision.

## Existing-code facts to confirm
Before editing, verify:
- TrajFlow is multi-modal motion prediction on WOMD.
- Config uses `NUM_QUERY=64`, `NUM_FUTURE_FRAMES=80`, `NUM_MOTION_MODES=6`.
- FlowMatcher official inference is one-step.
- DenoisingDecoder already has a decoder-layer refinement axis with `pred_scores / pred_trajs / pl_logits`.
- That decoder-layer axis is the main proxy time axis.

## Expected implementation shape
Add config-gated CGC modules under:
- `trajflow/organization/family_proxy.py`
- `trajflow/organization/proxies.py`
- `trajflow/organization/recorder.py`
- `trajflow/organization/error_attributor.py`
- `trajflow/organization/guidance_labeler.py`
- `trajflow/organization/risk_predictor.py`
- `trajflow/organization/corrector.py`
- `trajflow/organization/reranker.py`

Modify only what is needed in:
- `trajflow/models/denoising_decoder/denoising_decoder.py`
- `trajflow/models/denoising_decoder/decoder_utils.py`
- `runner/utils/trainer.py`
- `runner/utils/tester.py`
- relevant yaml config(s)

## Workflow
For any nontrivial edit:
1. Summarize the current code path.
2. List files to be touched and why.
3. Implement minimal config-gated changes.
4. Preserve both raw baseline path and org-enhanced path side-by-side.
5. Report assumptions explicitly.

## Verification / delivery
Your final output must include:
- changed file list
- concise diff summary
- how original behavior is preserved when CGC is off
- how one-step inference is preserved
- raw vs org-enhanced outputs/logs
- run instructions for AutoDL
