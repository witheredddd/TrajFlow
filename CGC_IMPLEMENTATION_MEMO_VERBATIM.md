# CGC Implementation Memo (Verbatim Markdown Copy)

> Source: exact verbatim markdown copy converted from the uploaded text memo. This file is intended to preserve the original wording and structure of the user-provided implementation instructions.

You are modifying the DSL-Lab/TrajFlow codebase.

Before changing anything, you must first read and understand these two user-provided documents in full:

1. research-proposal-Peilin_Wang.pdf
2. CGC-Flow Theory Draft.pdf

Then trace the current TrajFlow code path end-to-end before editing. Do NOT modify by intuition. You must first understand at least these files:

- runner/train.py
- runner/utils/trainer.py
- trajflow/utils/init_objective.py
- trajflow/denoising/flow_matching.py
- trajflow/models/dmt_model.py
- trajflow/models/context_encoder/mtr_encoder.py
- trajflow/models/denoising_decoder/denoising_decoder.py
- trajflow/models/denoising_decoder/decoder_utils.py
- runner/utils/tester.py
- trajflow/datasets/waymo/waymo_eval.py

The implementation must follow the proposal and theory draft faithfully. Do NOT reinterpret this as a generic reranking project, and do NOT turn it into a multi-step flow-inference method.

==================================================
0. CORE DECISION LOGIC — THIS IS THE MOST IMPORTANT
==================================================

You MUST implement error attribution, and you MUST only target the guidance-linked failure type for correction.

For every training/validation sample, do NOT treat all failures as the same. At minimum, separate them into these three categories:

1. coverage failure
   - A GT-near candidate does not appear at all, or
   - a GT-compatible coarse family does not form at all.
   - This is primarily a coverage / representation / proposal issue.
   - This is NOT the direct target of the current guidance correction.

2. guidance-linked organization failure
   - A GT-near candidate or GT-compatible family already exists,
   - but the same-family organization deteriorates along the generation process:
     pi_main drops, pi_ps rises, R_tilde rises, H_cluster rises,
     representative switching increases, or late-stage inversion appears,
   - and this ultimately leads to:
     wrong final selection, or very small final margin, or clear late-stage hesitation.
   - ONLY this failure type is the primary target of the current correction method.

3. ranking-only failure
   - Candidate coverage is already sufficient,
   - same-family organization does not clearly deteriorate,
   - but the final raw scorer / selector still picks the wrong candidate.
   - This is more of a scoring / calibration issue.
   - This is NOT the direct target of the current guidance correction.

ABSOLUTE RULE:
- The current correction module must only be trained to address type (2): guidance-linked organization failure.
- Type (1) and type (3) are for offline diagnosis, filtering, and ablation, but not the direct correction target in this stage.

==================================================
1. TRUE PROJECT GOAL
==================================================

The goal is NOT:
- to collapse multimodality into a single sharp mode,
- nor to merely improve endpoint loss.

The real goal is:

1. TrajFlow already achieves multi-modal motion prediction.
2. We now focus on:
   - coarse / family-level reachability already being present,
   - but same-family posterior organization can degrade,
   - e.g. main-peak mass gets siphoned by same-family pseudo-peak competition,
     clarity decreases,
     and the final canonical representative becomes hesitant or flips late.
3. We want to:
   - preserve legitimate multimodality,
   - preserve official single-step flow inference efficiency,
   - add a theory-aligned pipeline for:
     organization monitoring,
     error attribution,
     guidance-quality labeling,
     lightweight in-layer correction,
     and candidate-wise reranking,
   - so that selection becomes more stable while official benchmark performance is maintained or improved.

==================================================
2. HARD CONSTRAINTS
==================================================

2.1 Main platform
- The main implementation target is TrajFlow on WOMD.
- Do NOT make GoalFlow the main code target.
- GoalFlow may only be used as conceptual inspiration, not as the codebase to edit.

2.2 Official method must remain one-step flow inference
- Formal training and formal inference must NOT depend on increasing flow sampling steps.
- Do NOT turn the official method into multi-step flow inference.
- Do NOT add extra denoising / ODE integration steps.
- Do NOT add extra sampler loops.
- Do NOT repeatedly call denoiser forward to emulate process.
- Official speed, official tables, and official inference must preserve TrajFlow’s original efficient one-step flow setup.

2.3 Audit / analysis mode is allowed, but must be fully isolated
- You MAY implement a separate audit/analysis mode that temporarily uses larger flow sampling steps.
- That mode is only for:
  - trajectory evolution visualization,
  - mechanism figures,
  - appendix analysis,
  - sanity checks.
- Audit mode must NEVER be required for:
  - the main method,
  - main training,
  - main inference,
  - main tables,
  - or official runtime reporting.
- Audit mode must be disabled by default.
- Audit results must be saved separately.
- The paper-facing main method must be reproducible with audit mode fully OFF.

2.4 Family decomposition must be preserved, but family is NOT the final hard decision gate
- Family is a coarse operational partition, not the final decision object.
- Family exists for:
  a) same-family organization statistics,
  b) offline training/validation attribution and guidance labeling,
  c) local organization context for candidate-level correction/reranking.
- Final output must still be candidate-wise ranking and selection.
- Do NOT replace candidate-wise ranking with global family pooling.
- Do NOT merge all same-family candidates into one family score and use that as the final decision.
- If any design choice makes family classification “look nicer” but harms candidate-wise ranking fidelity, prioritize candidate-wise ranking fidelity.

2.5 Correction must be lightweight, local, and late-layer-first
- Correction cannot aggressively push all layers.
- It must support:
  - only_late_layers,
  - risk-threshold triggering,
  - bounded strength,
  - gate-based application.
- By default, do not strongly interfere with early layers.

2.6 Do NOT treat all bad outcomes as bad guidance
- Guidance-related labeling must be filtered through the 3-way error attribution above.
- Only guidance-linked organization failure should produce positive correction supervision.
- Coverage failures and ranking-only failures must be diagnosed and tracked, but not used as direct correction targets.

2.7 AutoDL compatibility
- Linux environment
- Keep dependencies minimal
- Reuse existing framework and configs
- Use yaml config instead of building a separate training system
- If CGC-related features are disabled, original repo behavior must remain unchanged

2.8 Every vague concept must be operationalized
Terms like:
- local
- dominant
- significant drop
- significant rise
- late-stage
- high-risk
- stable
- small margin
must always correspond to explicit formulas, thresholds, or config fields.

==================================================
3. FACTS ABOUT THE EXISTING TRAJFLOW CODE
==================================================

You must confirm these facts before editing:

1. TrajFlow is multi-modal motion prediction on WOMD.
2. Config uses:
   - NUM_QUERY = 64
   - NUM_FUTURE_FRAMES = 80
   - NUM_MOTION_MODES = 6
3. FlowMatcher official inference remains one-step.
4. DenoisingDecoder already has a multi-layer refinement axis:
   each layer has pred_scores / pred_trajs / pl_logits.
5. That decoder-layer axis is the main proxy time axis for this project.
6. Final selection must remain candidate-wise.
7. You may apply residual organization-aware reranking,
   but you may NOT replace selection with “pick a family first and only then choose inside it”.

==================================================
4. OVERALL IMPLEMENTATION ROADMAP
==================================================

The implementation must be split into two stages.

----------------------------------
Stage A: Log-only / Audit / Attribution / Label Building
----------------------------------

Goal:
- minimize invasiveness,
- record theory-aligned organization proxies over decoder layers,
- build offline error attribution,
- build offline guidance labels,
- optionally export audit-mode multi-step flow traces for analysis only.

Tasks:
A1. coarse soft family proxy
A2. theory-aligned organization proxy extraction
A3. layer-wise logging
A4. three-way error attribution
A5. guidance-quality labeling (only after attribution)
A6. optional multi-step-flow audit mode

----------------------------------
Stage B: One-step mainline correction + reranking
----------------------------------

Goal:
- keep official inference strictly one-step,
- train a lightweight online risk predictor / trigger from Stage-A labels,
- apply lightweight in-layer correction between decoder layers,
- apply candidate-wise residual reranking,
- only target guidance-linked organization failure,
- keep diversity coverage from collapsing.

==================================================
5. STAGE A — DETAILED IMPLEMENTATION
==================================================

----------------------------------
A1. Coarse soft family proxy
----------------------------------

Purpose:
Family is NOT the final scoring object.
It is a coarse operational coordinate system for:
- same-family organization statistics,
- offline guidance attribution,
- local context for candidate-wise correction/reranking.

Requirements:

1. Build a coarse family proxy for each candidate.
   It must be:
   - coarse,
   - soft,
   - computable,
   - not so fine that tiny heading differences create separate families,
   - not a hard merge of same-family candidates.

2. First version may use only trajectory geometry:
   - endpoint basin bucket
   - terminal heading bucket
   - turn-type bucket

3. Family proxy is only used to:
   - construct same-family neighborhoods
   - compute same-family organization statistics
   - provide local organization context
   - support offline attribution / labeling

Create:
trajflow/organization/family_proxy.py

Suggested interface:
build_family_proxy(pred_trajs, pred_scores, cfg)
-> family_ids, family_soft_assign, family_meta

Requirements:
- soft assignment preferred
- hard assignment only for debug/ablation
- all bucket/threshold values configurable
- family proxy across decoder layers must be stabilized
  (e.g. smoothing / prototype smoothing / anchored soft assignment)
- avoid per-layer brittle re-clustering

----------------------------------
A2. Theory-aligned organization proxy extraction
----------------------------------

This must follow the proof draft’s objects as closely as possible using operational proxies.

For each decoder layer l and each family proxy m:

1. Define three proxy regions:
   - K_main: the dominant representative’s core neighborhood
   - K_ps: other competing same-family local groups
   - R: remaining same-family region

2. Compute:
   - pi_main
   - pi_ps
   - pi_res
   - R_tilde
   - H_cluster
   - ell_leak

3. Then compute:
   - Q_main proxy
   - clarity proxy C

Create:
trajflow/organization/proxies.py

Suggested implementation:
- convert pred_scores to probabilities using softmax
- identify dominant representative inside family
- allow temporally smoothed dominant representative
- construct K_main via endpoint / trajectory proximity
- define K_ps as other local groups above min cluster mass
- define R as remainder

Suggested formulas:

pi_main = sum(prob in K_main)
pi_ps   = sum(prob in K_ps)
pi_res  = sum(prob in R)

R_tilde = weighted dispersion around family weighted center
H_cluster = entropy of local cluster masses
ell_leak = probability mass outside family tube / family neighborhood

Q_main =
    a_mass * pi_main
  - a_comp * pi_ps
  - a_var  * R_tilde
  - a_ent  * H_cluster

clarity C =
    beta_main * pi_main
  - beta_ps   * pi_ps
  - beta_res  * pi_res
  - beta_var  * R_tilde
  - beta_leak * ell_leak

Notes:
- top1-top2 margin may be exported, but it is only a secondary diagnostic.
- The main proxies must be same-family organization proxies.
- Do not try to exactly reconstruct ideal theory objects like dominant ridge or exact family tube; implement stable operational proxies.

----------------------------------
A3. Layer-wise logging
----------------------------------

During training and validation, record per decoder layer:

- family proxy
- pi_main / pi_ps / pi_res
- R_tilde / H_cluster / ell_leak
- Q_main / clarity
- representative identity
- switch count / instability summaries

Create:
trajflow/organization/recorder.py

Requirements:
- default save only summaries
- save detailed layer-wise tensors only in debug mode
- do not create heavy logging overhead

----------------------------------
A4. Three-way error attribution
----------------------------------

This is mandatory.

For each training/validation sample, do offline error attribution into:

1. coverage failure
2. guidance-linked organization failure
3. ranking-only failure

Create:
trajflow/organization/error_attributor.py

Suggested interface:
attribute_failure_type(layer_proxy_seq, final_selection_info, gt_info, cfg)
-> error_type_dict

The logic should include at least:

coverage failure:
- no GT-near candidate appears, or
- no GT-compatible coarse family forms

guidance-linked organization failure:
- GT-near candidate or GT-compatible family exists
- same-family organization deteriorates along decoder layers
- and this correlates with wrong final selection, tiny margin, or clear late-stage hesitation

ranking-only failure:
- candidate coverage is sufficient
- same-family organization is not clearly deteriorated
- but final raw ranking still picks the wrong candidate

Required outputs:
- is_coverage_failure
- is_guidance_org_failure
- is_ranking_only_failure
- gt_near_candidate_exists
- gt_family_exists
- final_margin
- instability_score

Important:
- This is strictly offline and GT-dependent.
- This is to isolate which errors guidance can actually address.
- Do NOT collapse all errors into one label.

----------------------------------
A5. Guidance-quality labeling
----------------------------------

Only after A4.

Create:
trajflow/organization/guidance_labeler.py

Suggested interface:
label_guidance_quality(error_type_dict, layer_proxy_seq, final_selection_info, gt_info, cfg)
-> guidance_label_dict

Rules:

1. For guidance-linked organization failure:
   - label as guidance-risk-positive / guidance-suspect
   - do NOT use strong causal wording like “strict bad guidance”
   - this is a proxy training label, not causal proof

2. For samples with good final result and stable organization:
   - label as guidance-stable

3. For coverage failure, ranking-only failure, or ambiguous cases:
   - label as uncertain / ignore-for-correction-learning

Outputs should include:
- is_guidance_risk_positive
- is_guidance_stable
- is_uncertain_guidance
- final_margin
- representative_switch_count
- late_stage_inversion_flag
- proxy_deterioration_score

Important:
- Stage-A labeler is strictly offline / GT-dependent
- it must NEVER be used inside the official online inference path

----------------------------------
A6. Audit mode for multi-step flow analysis
----------------------------------

This is analysis only, not the main method.

Goal:
- optionally run multi-step flow only in audit mode
- export true flow-time evolution for figures / appendix / inspection
- compare flow-time proxies against decoder-layer proxies

Audit mode requirements:
- completely optional
- disabled by default
- saved separately
- not part of main training, main inference, main tables, or runtime reports

Suggested config:
CGC.AUDIT.ENABLE = false
CGC.AUDIT.SAMPLING_STEPS = 8 or 16
CGC.AUDIT.RETURN_ALL_TIMESTEPS = true
CGC.AUDIT.RUN_ONLY_ON_SUBSET = true

Outputs:
- flow-time trajectories
- flow-time organization proxies
- comparison against decoder-layer-axis proxies

==================================================
6. STAGE B — DETAILED IMPLEMENTATION
==================================================

----------------------------------
B1. Online risk predictor / trigger
----------------------------------

Stage A is offline. Stage B must learn an online lightweight predictor.

Create:
trajflow/organization/risk_predictor.py

Suggested interface:
predict_guidance_risk(proxy_dict, local_context, layer_idx, cfg)
-> risk_score, risk_gate

Inputs:
- current layer organization proxies
- candidate/local-family context
- layer index
- optional local geometry summary

Outputs:
- risk_score
- risk_gate / trigger

Rules:
- no GT at inference
- only the online risk predictor may be used in the online path
- Stage-A offline labeler must not be called online

----------------------------------
B2. In-layer lightweight correction
----------------------------------

This is the real correction module.

Must modify:
trajflow/models/denoising_decoder/denoising_decoder.py

Goal:
- no extra flow step
- no extra denoiser forward
- correction only between existing decoder layers
- correction only on later layers and/or when risk is high

Implementation idea:

Inside apply_transformer_decoder() layer loop:

1. After each layer obtains pred_scores / pred_trajs:
   - compute family proxy
   - compute organization proxies
   - compute representative instability summaries

2. Call online risk predictor:
   risk_score, risk_gate = predictor(...)

3. Apply lightweight corrector:
   delta_q = Corrector(query_content, local_org_context, layer_idx, cfg)
   query_content = query_content + risk_gate * delta_q

Requirements:
- candidate-local / family-contextual only
- no global shrinkage of all queries
- no family merging
- only local same-family organization context may influence a candidate
- must support:
  - only_late_layers
  - risk threshold gating
  - correction strength cap
  - diversity guardrails

Create:
trajflow/organization/corrector.py

Suggested interface:
apply_org_correction(
    query_content,
    pred_scores,
    pred_trajs,
    family_proxy,
    proxy_dict,
    risk_score,
    cfg
) -> corrected_query_content, correction_aux

The corrector must remain very light:
- small MLP / linear residual + gate
- no extra attention blocks
- no extra decoder layers
- no architecture expansion that changes inference complexity meaningfully

----------------------------------
B3. Candidate-wise final reranking
----------------------------------

Final output must still be candidate-wise.

Must modify:
trajflow/models/denoising_decoder/decoder_utils.py

Goal:
- keep raw candidate scores
- add bounded organization-aware residual rerank
- do not replace candidate ranking with family ranking

Formula suggestion:
final_score_k =
    raw_score_k
  + lambda_main * local_mainness_k
  - lambda_comp * local_competition_k
  - lambda_var  * local_dispersion_k
  - lambda_ent  * local_cluster_entropy_k
  - lambda_leak * local_leak_k
  - lambda_inv  * local_instability_k

All local_* must come from the candidate’s local same-family organization context.

Create:
trajflow/organization/reranker.py

Suggested interface:
rerank_candidates_with_org_context(
    pred_scores,
    pred_trajs,
    family_proxy,
    proxy_dict,
    cfg
) -> org_scores, rerank_aux

Requirements:
- do NOT merge family scores
- keep reranking residual/bounded
- do NOT overwrite raw scores
- save both raw and org-enhanced outputs

Batch dict / outputs must preserve at least:
- pred_scores_raw
- pred_scores_org
- selected_idxs_raw
- selected_idxs_org
- family_proxy
- proxy_summary

==================================================
7. TRAINING AND EVALUATION
==================================================

----------------------------------
Training
----------------------------------

Training must support both:

- Stage A: log-only / audit / attribution / label-building
- Stage B: correction + rerank training

Requirements:
- do NOT replace original TrajFlow main loss
- optional small-weight organization auxiliary terms are allowed, but:
  - no global entropy collapse
  - no punishment of legitimate cross-family multimodality
  - only local same-family organization regularization
  - correction supervision should come mainly from guidance-linked organization failure samples

Modify:
runner/utils/trainer.py

Must log:
- coverage failure ratio
- guidance-linked organization failure ratio
- ranking-only failure ratio
- guidance-risk-positive ratio
- guidance-stable ratio
- uncertain ratio
- FCS / SFI / FLR
- RSR / SST / LIR / CP
- raw vs org rerank performance

----------------------------------
Evaluation
----------------------------------

Official main-table metrics remain the Waymo metrics:
- mAP
- minADE
- minFDE
- MissRate

Additional stability metrics:
- RSR
- SST
- LIR
- CP

Additional organization metrics:
- FCS
- SFI
- FLR

Additional diagnostics:
- top1-top2 margin
- three-way failure ratios
- guidance-risk-positive ratio
- layer-wise pi_main / pi_ps curves
- layer-wise H_cluster / R_tilde curves
- audit-mode flow-time curves

Important:
- organization and attribution metrics are theory-aligned proxy diagnostics, NOT official Waymo metrics
- do not present them as official metrics
- but they must be saved cleanly for analysis

Modify:
runner/utils/tester.py

Save:
- result_denoiser.pkl
- result_org_metrics.pkl
- result_error_attribution.pkl
- optional layerwise_summary.pkl
- optional audit_flow_traces.pkl

==================================================
8. FILES THAT MUST BE MODIFIED / ADDED
==================================================

At minimum modify:

1. runner/cfgs/waymo/trajflow+100_percent_data.yaml
   and optionally 20_percent_data.yaml

Add a CGC config block like:

CGC:
  ENABLE: true
  NO_EXTRA_FLOW_STEPS: true
  USE_LAYER_AXIS_ONLY: true

  STAGE:
    MODE: "log_only"   # or "train_correction"
    ONLY_LATE_LAYERS: true

  FAMILY_PROXY:
    USE_SOFT_ASSIGN: true
    ENDPOINT_RADIUS: ...
    HEADING_BUCKETS: ...
    TURN_THRESH: ...
    FAMILY_SMOOTHING: ...

  PROXY:
    MAIN_RADIUS: ...
    COMP_RADIUS: ...
    MIN_CLUSTER_MASS: ...
    EPS: 1e-8

  ATTRIBUTION:
    ENABLE: true
    USE_GT_FOR_OFFLINE_ONLY: true
    GT_NEAR_THRESH: ...
    ORG_FAIL_DROP_THRESH: ...
    ORG_FAIL_RISE_THRESH: ...
    SWITCH_THRESH: ...
    MARGIN_SMALL_THRESH: ...

  GUIDANCE_LABEL:
    ENABLE: true
    POSITIVE_ONLY_ON_GUIDANCE_ORG_FAIL: true
    GOOD_MARGIN_THRESH: ...
    BAD_MARGIN_THRESH: ...
    LATE_STAGE_ONLY: true

  RISK:
    ENABLE_PREDICTOR: true
    TRIGGER_THRESH: ...
    CORRECTION_CAP: ...
    ONLY_LATE_LAYERS: true

  SCORE:
    A_MASS: ...
    A_COMP: ...
    A_VAR: ...
    A_ENT: ...
    B_MAIN: ...
    B_PS: ...
    B_RES: ...
    B_VAR: ...
    B_LEAK: ...
    LAMBDA_MAIN: ...
    LAMBDA_COMP: ...
    LAMBDA_VAR: ...
    LAMBDA_ENT: ...
    LAMBDA_LEAK: ...
    LAMBDA_INV: ...

  DIVERSITY_GUARD:
    ENABLE: true
    MIN_MODE_COUNT_RATIO: ...
    MIN_SPREAD_RATIO: ...
    MAX_ORACLE_DEGRADATION: ...

  AUDIT:
    ENABLE: false
    SAMPLING_STEPS: 8
    RETURN_ALL_TIMESTEPS: true
    RUN_ONLY_ON_SUBSET: true
    SAVE_DIR: ...

2. trajflow/models/denoising_decoder/denoising_decoder.py
   - compute proxies inside layer loop
   - apply risk-triggered correction inside layer loop
   - do not alter flow structure
   - do not add decoder depth

3. trajflow/models/denoising_decoder/decoder_utils.py
   - keep raw selection path
   - add organization-aware candidate reranking path
   - do not implement hard family-first selection

4. runner/utils/trainer.py
   - Stage A / Stage B modes
   - error attribution logging
   - guidance labeling logging
   - organization metrics logging
   - optional small-weight auxiliary terms

5. runner/utils/tester.py
   - official metrics
   - organization metrics
   - attribution metrics
   - separated audit outputs

Add new directory:
trajflow/organization/
with at least:
- family_proxy.py
- proxies.py
- recorder.py
- error_attributor.py
- guidance_labeler.py
- risk_predictor.py
- corrector.py
- reranker.py

==================================================
9. EXTRA REQUIREMENTS
==================================================

1. Preserve raw baseline path and org-enhanced path side-by-side for ablation.
2. Add diversity guardrails to ensure coverage does not collapse too much.
   Save at least:
   - distinct mode count
   - average pairwise endpoint spread
   - oracle best-of-K distance
3. Do not build a separate argparse-based training system.
4. Final delivery must include:
   - real modified code
   - yaml configs
   - AutoDL run instructions
   - export instructions
   - changed file list
   - raw vs org comparison explanation

==================================================
10. FOUR PRINCIPLES THAT OVERRIDE EVERYTHING
==================================================

1. The official method must NOT depend on multi-step flow inference.
2. Final output must remain candidate-wise ranking.
3. Family decomposition is used for same-family organization statistics, offline attribution, guidance labeling, and local organization context — NOT as a direct replacement for candidate decision.
4. Only guidance-linked organization failure is the current correction target; coverage failure and ranking-only failure are only for offline diagnosis/filtering in this stage.

If any implementation detail conflicts with these four principles, preserve these four principles first.