# CGC Implementation Spec for TrajFlow (Compressed)

## 0. Purpose
Implement the project on **TrajFlow + WOMD** while preserving original efficient **one-step** flow inference.

Implementation must also read:
- `docs/CGC_FORMULA_CONVENTIONS.md`

This file is the code-level source of truth for notation, thresholds, proxy formulas, attribution rules, and reranking equations.

This project studies **posterior-organization failure under coarse guidance**:
- the correct coarse family may already be reached,
- but same-family posterior organization degrades,
- the dominant representative loses mass/clarity,
- nearby same-family competitors survive,
- and final selection becomes unstable.

The target is **clarity-preserving canonical selection**, not generic reranking and not endpoint-fit-only optimization.

---

## 1. Override precedence
When wording conflicts across source documents, obey this order:
1. **Hard implementation constraints in this spec**
2. the pasted implementation memo / decision rules
3. high-level wording from the research proposal
4. theory draft as geometric motivation / proxy guidance

In particular:
- use **family** for organization statistics, attribution, labeling, and local context,
- but keep **final online decision candidate-wise**,
- and do **not** replace candidate selection with a hard family-first decision gate.

---

## 2. Non-negotiable constraints

### 2.1 Main platform
- Main implementation target: **TrajFlow on WOMD**.
- GoalFlow is only conceptual inspiration.

### 2.2 Preserve one-step inference
Do **not**:
- increase formal flow sampling steps for the main method,
- add extra denoising / ODE integration steps,
- add sampler loops,
- repeatedly call denoiser forward to emulate process.

Official training, inference, tables, and runtime must preserve TrajFlow’s original efficient **one-step** setup.

### 2.3 Audit mode must be isolated
Optional audit mode may temporarily use larger flow steps only for:
- mechanism figures
- appendix analysis
- sanity checks
- evolution visualization

Audit mode must be:
- off by default
- saved separately
- never required for main training/inference/tables/runtime

### 2.4 Final selection must stay candidate-wise
- Family is a **coarse operational partition**, not the final decision object.
- Do **not** pool all candidates in a family into one final family score.
- Do **not** implement “pick family first, then choose inside it” as the online selection rule.
- Candidate-wise ranking fidelity has priority.

### 2.5 Correction must be light and local
Correction must be:
- late-layer-first
- risk-triggered
- bounded
- gate-based
- local to same-family organization context

Do **not** heavily interfere with early layers.

### 2.6 Only one failure type is the correction target
For each sample, separate failures into:
1. `coverage_failure`
2. `guidance_org_failure`
3. `ranking_only_failure`

Only **guidance-linked organization failure** is a positive correction target.
Coverage failure and ranking-only failure are for offline diagnosis/filtering/ablation only.

### 2.7 Compatibility
- Linux / AutoDL friendly
- minimal dependencies
- reuse current framework and yaml configs
- no separate training system
- with CGC off, repo behavior must match baseline

### 2.8 Operationalize everything
Terms like `dominant`, `late-stage`, `high-risk`, `small margin`, `significant drop/rise`, `stable`, `local` must map to explicit formulas or thresholds in config.

---

## 3. Existing code facts that must be confirmed first
Before editing, trace the code path and confirm:
- TrajFlow is multi-modal motion prediction on WOMD.
- Config uses `NUM_QUERY = 64`, `NUM_FUTURE_FRAMES = 80`, `NUM_MOTION_MODES = 6`.
- FlowMatcher official inference remains one-step.
- DenoisingDecoder already exposes a decoder-layer axis with `pred_scores`, `pred_trajs`, `pl_logits` per layer.
- That decoder-layer axis is the main proxy time axis for this project.
- Final selection remains candidate-wise.

Mandatory files to trace first:
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

---

## 4. Theory-to-implementation mapping
The proposal/theory motivate these operational objects:
- **dominant ridge** → dominant same-family representative / center neighborhood
- **same-family pseudo-peaks** → competing local same-family groups
- **width floor / residual spread** → within-family dispersion proxy
- **family peak cluster** → same-family local clusters around nearby representatives
- **clarity-preserving correction** → improve same-family organization without collapsing cross-family multimodality

Do **not** try to reconstruct exact ideal theory objects. Use stable, computable proxies.

---

## 5. Two-stage implementation roadmap

### Stage A: log-only / audit / attribution / label building
Goal:
- minimal invasiveness
- record organization proxies over decoder layers
- build offline attribution labels
- optionally export audit-mode multi-step traces

Tasks:
- A1 family proxy
- A2 organization proxies
- A3 layerwise recorder
- A4 three-way attribution
- A5 guidance labeler
- A6 isolated audit mode

### Stage B: one-step correction + rerank
Goal:
- keep mainline one-step
- learn online risk predictor from Stage-A labels
- apply lightweight in-layer correction between existing decoder layers
- add bounded candidate-wise residual reranking
- target only guidance-linked organization failure
- preserve diversity / coverage

---

## 6. Stage A details

### A1. Coarse soft family proxy
Create:
- `trajflow/organization/family_proxy.py`

Interface:
```python
build_family_proxy(pred_trajs, pred_scores, cfg)
-> family_ids, family_soft_assign, family_meta
```

Requirements:
- coarse, soft, stable, computable
- not so fine that tiny heading changes create different families
- not a hard merge of same-family candidates
- stabilize across decoder layers

First-pass geometric ingredients:
- endpoint basin bucket
- terminal heading bucket
- turn-type bucket

Use family only for:
- same-family neighborhood construction
- organization statistics
- offline attribution/labels
- local context for correction/reranking

### A2. Theory-aligned organization proxies
Create:
- `trajflow/organization/proxies.py`

For each decoder layer `l` and family proxy `m`, define proxy regions:
- `K_main`: dominant representative neighborhood
- `K_ps`: competing same-family local groups
- `R`: residual same-family region

Compute:
- `pi_main`
- `pi_ps`
- `pi_res`
- `R_tilde`
- `H_cluster`
- `ell_leak`
- `Q_main`
- `clarity`

Suggested operational formulas:
```python
prob = softmax(pred_scores)
pi_main = sum(prob in K_main)
pi_ps   = sum(prob in K_ps)
pi_res  = sum(prob in R)

R_tilde   = weighted_dispersion_around_family_center
H_cluster = entropy_of_local_cluster_masses
ell_leak  = probability_mass_outside_family_tube

Q_main = a_mass*pi_main - a_comp*pi_ps - a_var*R_tilde - a_ent*H_cluster
clarity = (
    beta_main*pi_main
    - beta_ps*pi_ps
    - beta_res*pi_res
    - beta_var*R_tilde
    - beta_leak*ell_leak
)
```

Notes:
- top1-top2 margin is only a secondary diagnostic
- main focus is same-family organization, not global mode collapse

### A3. Layerwise recorder
Create:
- `trajflow/organization/recorder.py`

Record per decoder layer:
- family proxy summary
- `pi_main / pi_ps / pi_res`
- `R_tilde / H_cluster / ell_leak`
- `Q_main / clarity`
- dominant representative identity
- switch count / instability summaries

Logging policy:
- summaries by default
- detailed tensors only in debug mode
- keep overhead low

### A4. Three-way error attribution (mandatory)
Create:
- `trajflow/organization/error_attributor.py`

Interface:
```python
attribute_failure_type(layer_proxy_seq, final_selection_info, gt_info, cfg)
-> error_type_dict
```

Required outputs:
- `is_coverage_failure`
- `is_guidance_org_failure`
- `is_ranking_only_failure`
- `gt_near_candidate_exists`
- `gt_family_exists`
- `final_margin`
- `instability_score`

Minimum logic:
- **coverage failure**: no GT-near candidate or no GT-compatible coarse family
- **guidance org failure**: GT-near candidate/family exists, same-family organization deteriorates over layers, and this aligns with wrong final selection / tiny margin / late hesitation
- **ranking-only failure**: coverage sufficient and organization not clearly deteriorated, but final raw selector still picks wrong candidate

This module is:
- offline only
- GT-dependent
- required for isolating which errors guidance can actually address

### A5. Guidance-quality labeler
Create:
- `trajflow/organization/guidance_labeler.py`

Interface:
```python
label_guidance_quality(error_type_dict, layer_proxy_seq, final_selection_info, gt_info, cfg)
-> guidance_label_dict
```

Rules:
- if `guidance_org_failure` → `guidance_risk_positive` / `guidance_suspect`
- if final result is good and organization stable → `guidance_stable`
- if coverage failure, ranking-only failure, or ambiguity → `uncertain / ignore`

Required outputs:
- `is_guidance_risk_positive`
- `is_guidance_stable`
- `is_uncertain_guidance`
- `final_margin`
- `representative_switch_count`
- `late_stage_inversion_flag`
- `proxy_deterioration_score`

Important:
- Stage-A labeler is offline and GT-dependent
- must never be called in the official online inference path

### A6. Optional audit mode
Use only for analysis.

Suggested config:
```yaml
CGC:
  AUDIT:
    ENABLE: false
    SAMPLING_STEPS: 8
    RETURN_ALL_TIMESTEPS: true
    RUN_ONLY_ON_SUBSET: true
```

Outputs:
- flow-time trajectories
- flow-time organization proxies
- comparison of flow-time vs decoder-layer proxies

Must stay fully isolated from mainline results.

---

## 7. Stage B details

### B1. Online risk predictor / trigger
Create:
- `trajflow/organization/risk_predictor.py`

Interface:
```python
predict_guidance_risk(proxy_dict, local_context, layer_idx, cfg)
-> risk_score, risk_gate
```

Inputs:
- current layer organization proxies
- local same-family context
- layer index
- optional local geometry summary

Rules:
- no GT online
- only this learned online predictor may be used online
- offline labeler must not be called online

### B2. In-layer lightweight correction
Modify:
- `trajflow/models/denoising_decoder/denoising_decoder.py`

Create:
- `trajflow/organization/corrector.py`

Interface:
```python
apply_org_correction(
    query_content,
    pred_scores,
    pred_trajs,
    family_proxy,
    proxy_dict,
    risk_score,
    cfg,
) -> corrected_query_content, correction_aux
```

Inside the existing decoder layer loop:
1. compute family proxy and organization proxies after each layer output
2. compute representative instability summaries
3. call risk predictor
4. apply a **small residual correction** to `query_content` only if gated

Constraints:
- no extra flow step
- no extra denoiser forward
- no new decoder depth
- no extra heavy attention block
- candidate-local / same-family-context only
- no global shrinkage of all queries
- no family merging

The corrector must be very light:
- small MLP or linear residual
- gated by risk and layer position
- bounded by correction cap

### B3. Candidate-wise residual reranking
Modify:
- `trajflow/models/denoising_decoder/decoder_utils.py`

Create:
- `trajflow/organization/reranker.py`

Interface:
```python
rerank_candidates_with_org_context(
    pred_scores,
    pred_trajs,
    family_proxy,
    proxy_dict,
    cfg,
) -> org_scores, rerank_aux
```

Residual rerank only:
```python
final_score_k = (
    raw_score_k
    + lambda_main * local_mainness_k
    - lambda_comp * local_competition_k
    - lambda_var  * local_dispersion_k
    - lambda_ent  * local_cluster_entropy_k
    - lambda_leak * local_leak_k
    - lambda_inv  * local_instability_k
)
```

Requirements:
- keep raw score path
- do not overwrite raw scores
- keep reranking bounded / residual
- do not merge family scores
- save both raw and org-enhanced selections

Must preserve at least:
- `pred_scores_raw`
- `pred_scores_org`
- `selected_idxs_raw`
- `selected_idxs_org`
- `family_proxy`
- `proxy_summary`

---

## 8. Training and evaluation

### Training
Modify:
- `runner/utils/trainer.py`

Support both modes:
- `log_only`
- `train_correction`

Do **not** replace TrajFlow’s original main loss.
Optional organization auxiliary terms are allowed only if they are:
- small-weight
- local to same-family organization
- not global entropy collapse
- not penalizing legitimate cross-family multimodality
- supervised mainly by guidance-linked organization failure samples

Must log:
- coverage failure ratio
- guidance-linked organization failure ratio
- ranking-only failure ratio
- guidance-risk-positive ratio
- guidance-stable ratio
- uncertain ratio
- raw vs org rerank performance
- layerwise proxy summaries
- diversity guardrail summaries

### Evaluation
Modify:
- `runner/utils/tester.py`

Official headline metrics remain Waymo metrics:
- `mAP`
- `minADE`
- `minFDE`
- `MissRate`

Additional analysis metrics may include:
- stability metrics
- organization metrics
- top1-top2 margin
- three-way failure ratios
- guidance-risk-positive ratio
- layerwise `pi_main / pi_ps`
- layerwise `H_cluster / R_tilde`
- audit-mode flow-time curves

Important:
- organization / attribution metrics are theory-aligned diagnostics
- do not present them as official Waymo metrics

Save at least:
- `result_denoiser.pkl`
- `result_org_metrics.pkl`
- `result_error_attribution.pkl`
- optional `layerwise_summary.pkl`
- optional `audit_flow_traces.pkl`

---

## 9. Config block
Modify:
- `runner/cfgs/waymo/trajflow+100_percent_data.yaml`
- optionally `trajflow+20_percent_data.yaml`

Add a config-gated block like:
```yaml
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
```

---

## 10. Files to add / modify

### Add
`trajflow/organization/`
- `family_proxy.py`
- `proxies.py`
- `recorder.py`
- `error_attributor.py`
- `guidance_labeler.py`
- `risk_predictor.py`
- `corrector.py`
- `reranker.py`

### Modify
- `trajflow/models/denoising_decoder/denoising_decoder.py`
- `trajflow/models/denoising_decoder/decoder_utils.py`
- `runner/utils/trainer.py`
- `runner/utils/tester.py`
- yaml configs

---

## 11. Diversity guardrails
Preserve multimodality and coverage.
At minimum track/save:
- distinct mode count
- average pairwise endpoint spread
- oracle best-of-K distance

Do not let organization correction collapse all candidates toward one mode.

---

## 12. Delivery checklist
Final delivery must include:
- real modified code
- yaml config(s)
- AutoDL run instructions
- export instructions
- changed file list
- concise explanation of raw vs org-enhanced comparison
- proof that one-step official path is preserved
- proof that CGC-off reproduces baseline behavior

---

## 13. Four principles that override everything
1. The official method must **not** depend on multi-step flow inference.
2. Final output must remain **candidate-wise ranking**.
3. Family decomposition is for organization statistics, offline attribution, guidance labeling, and local context — **not** as a direct replacement for candidate decision.
4. Only **guidance-linked organization failure** is the correction target in this stage.

If any implementation detail conflicts with these principles, preserve these four first.
