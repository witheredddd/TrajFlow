# CGC Formula Conventions for Codex

This file fixes the mathematical notation and operational formulas for the TrajFlow implementation.
It is **implementation-facing**, not a proof document.
When this file conflicts with informal wording elsewhere, follow this file for code-level definitions.

---

## 1. Indices, tensors, and basic notation

Let:
- `b = 1,...,B` index batch samples.
- `l = 1,...,L` index decoder layers.
- `q = 1,...,Q` index candidate queries / trajectories.
- `t = 1,...,T` index future time steps.
- `m = 1,...,M_b` index coarse family proxies for sample `b`.
- `k = 1,...,K_{b,l,m}` index same-family local clusters.

Main tensors at decoder layer `l`:
- predicted trajectory of candidate `q`:
  \[
  \hat\tau_{b,l,q} \in \mathbb{R}^{T\times 2}
  \]
- raw score / logit of candidate `q`:
  \[
  s_{b,l,q} \in \mathbb{R}
  \]
- candidate probability:
  \[
  p_{b,l,q} = \frac{\exp(s_{b,l,q})}{\sum_{q'=1}^Q \exp(s_{b,l,q'})}
  \]

For any candidate trajectory:
- endpoint:
  \[
  e_{b,l,q} = \hat\tau_{b,l,q}(T) \in \mathbb{R}^2
  \]
- terminal heading angle:
  \[
  \theta_{b,l,q} = \mathrm{atan2}(\Delta y_{T}, \Delta x_{T})
  \]
  computed from the last valid motion segment.

---

## 2. Coarse soft family proxy

For each candidate, build a coarse family proxy from three geometric attributes:
1. endpoint basin bucket,
2. terminal heading bucket,
3. turn-type bucket.

Let the soft family assignment be:
\[
A_{b,l,q,m} \in [0,1], \qquad \sum_{m=1}^{M_b} A_{b,l,q,m} = 1.
\]

Hard family id for debug / attribution only:
\[
\mathrm{fid}_{b,l,q} = \arg\max_m A_{b,l,q,m}.
\]

Family mass at layer `l`:
\[
P_{b,l,m} = \sum_{q=1}^{Q} p_{b,l,q} A_{b,l,q,m}.
\]

Implementation rule:
- Family is an **operational coordinate system**, not the final online decision object.
- Use soft assignment by default; hard assignment only for debug, attribution, and ablations.

---

## 3. Within-family normalized weights

Inside family `m`, define normalized candidate weights:
\[
\tilde p_{b,l,q\mid m} = \frac{p_{b,l,q} A_{b,l,q,m}}{P_{b,l,m}+\varepsilon}.
\]
Then:
\[
\sum_{q=1}^{Q} \tilde p_{b,l,q\mid m} = 1.
\]

All same-family organization statistics should be computed primarily from
\(\tilde p_{b,l,q\mid m}\), not from global probabilities alone.

---

## 4. Dominant representative and local groups

For each sample `b`, layer `l`, family `m`:

### 4.1 Dominant representative
Define the dominant representative index:
\[
q^*_{b,l,m} = \arg\max_q \tilde p_{b,l,q\mid m}.
\]

Optionally use temporal smoothing across layers:
- if the previous dominant representative remains within a geometry tolerance and its weight remains within a margin of the current best, keep the previous one.

### 4.2 Main neighborhood
Define the main neighborhood `K_main` around `q^*` using one of the following equivalent operational rules:

Preferred rule:
\[
K^{\text{main}}_{b,l,m} = \{q: d_{\text{traj}}(\hat\tau_{b,l,q}, \hat\tau_{b,l,q^*}) \le \delta_{\text{main}}\}.
\]

where trajectory distance can be:
\[
d_{\text{traj}}(\tau, \tau') = \frac{1}{T} \sum_{t=1}^T \|\tau(t)-\tau'(t)\|_2.
\]

Fallback cheaper rule:
\[
d_{\text{ep}}(q,q^*) = \|e_{b,l,q}-e_{b,l,q^*}\|_2,
\]
then use `d_ep <= delta_ep_main` plus optional heading consistency.

### 4.3 Same-family pseudo-peak groups
Remove `K_main`. Cluster the remaining same-family candidates by endpoint or trajectory proximity.
Any local group with normalized mass above threshold `\eta_cluster` is treated as a same-family competing group.
The union of those groups is:
\[
K^{\text{ps}}_{b,l,m}.
\]

### 4.4 Residual region
Define the remaining same-family region:
\[
R_{b,l,m} = \{1,\dots,Q\} \setminus \left(K^{\text{main}}_{b,l,m} \cup K^{\text{ps}}_{b,l,m}\right).
\]

---

## 5. Core same-family organization proxies

All quantities below are computed for each `(b,l,m)`.

### 5.1 Mass terms
\[
\pi^{\text{main}}_{b,l,m} = \sum_{q\in K^{\text{main}}_{b,l,m}} \tilde p_{b,l,q\mid m}
\]
\[
\pi^{\text{ps}}_{b,l,m} = \sum_{q\in K^{\text{ps}}_{b,l,m}} \tilde p_{b,l,q\mid m}
\]
\[
\pi^{\text{res}}_{b,l,m} = \sum_{q\in R_{b,l,m}} \tilde p_{b,l,q\mid m}
\]

These satisfy:
\[
\pi^{\text{main}} + \pi^{\text{ps}} + \pi^{\text{res}} = 1.
\]

### 5.2 Weighted family center and dispersion
Define the family-weighted center trajectory:
\[
\bar\tau_{b,l,m}(t) = \sum_{q=1}^{Q} \tilde p_{b,l,q\mid m} \, \hat\tau_{b,l,q}(t).
\]

Define within-family dispersion proxy:
\[
\widetilde R_{b,l,m} = \sum_{q=1}^{Q} \tilde p_{b,l,q\mid m} \, d_{\text{traj}}\!\left(\hat\tau_{b,l,q}, \bar\tau_{b,l,m}\right)^2.
\]

This is the implementation proxy for the theory-side width / residual spread object.

### 5.3 Cluster entropy
If the same-family candidates are partitioned into local groups `G_{b,l,m,k}` with masses
\[
\alpha_{b,l,m,k} = \sum_{q\in G_{b,l,m,k}} \tilde p_{b,l,q\mid m},
\]
define cluster entropy:
\[
H^{\text{cluster}}_{b,l,m} = - \sum_{k=1}^{K_{b,l,m}} \alpha_{b,l,m,k} \log(\alpha_{b,l,m,k}+\varepsilon).
\]

Higher entropy means stronger same-family fragmentation.

### 5.4 Family leakage
There are two valid implementations. Use one and keep it consistent.

**Default implementation (recommended):**
measure same-family leakage outside a tube around the dominant representative:
\[
\ell^{\text{leak}}_{b,l,m} = \sum_{q=1}^{Q} \tilde p_{b,l,q\mid m} \, \mathbf{1}\!
\left[d_{\text{traj}}(\hat\tau_{b,l,q}, \hat\tau_{b,l,q^*}) > \delta_{\text{tube}}\right].
\]

**Fallback implementation:**
use global out-of-family leakage:
\[
\ell^{\text{leak}}_{b,l,m} = 1 - P_{b,l,m}.
\]

If the fallback is used, name it clearly as `family_mass_leak` in code to avoid ambiguity.

---

## 6. Q_main and clarity

Define:
\[
Q^{\text{main}}_{b,l,m} =
 a_{\text{mass}}\, \pi^{\text{main}}_{b,l,m}
- a_{\text{comp}}\, \pi^{\text{ps}}_{b,l,m}
- a_{\text{var}}\, \widetilde R_{b,l,m}
- a_{\text{ent}}\, H^{\text{cluster}}_{b,l,m}.
\]

Define clarity:
\[
C_{b,l,m} =
 \beta_{\text{main}}\, \pi^{\text{main}}_{b,l,m}
- \beta_{\text{ps}}\, \pi^{\text{ps}}_{b,l,m}
- \beta_{\text{res}}\, \pi^{\text{res}}_{b,l,m}
- \beta_{\text{var}}\, \widetilde R_{b,l,m}
- \beta_{\text{leak}}\, \ell^{\text{leak}}_{b,l,m}.
\]

Sign convention:
- larger `Q_main` is better,
- larger `clarity` is better.

All coefficients must be nonnegative and live in config.

---

## 7. Dominance, margin, and inversion risk

### 7.1 Dominance
Within family `m`, define same-family top-1 and top-2 normalized weights:
\[
\tilde p^{(1)}_{b,l,m} \ge \tilde p^{(2)}_{b,l,m} \ge \cdots
\]
Then define dominance margin:
\[
D_{b,l,m} = \tilde p^{(1)}_{b,l,m} - \tilde p^{(2)}_{b,l,m}.
\]

Alternative score-space version:
\[
D^{\text{score}}_{b,l,m} = s^{(1)}_{b,l,m} - s^{(2)}_{b,l,m}.
\]
Use probability-space dominance by default.

### 7.2 Representative switch count
Let
\[
\hat q^*_{b,l,m}
\]
be the temporally smoothed dominant representative id. Then
\[
N^{\text{switch}}_{b,m} = \sum_{l=2}^{L} \mathbf{1}[\hat q^*_{b,l,m} \neq \hat q^*_{b,l-1,m}].
\]

### 7.3 Late-stage inversion risk
Define the late-stage layer set:
\[
\mathcal L_{\text{late}} = \{l: l \ge \lceil (1-\rho_{\text{late}})L \rceil\},
\]
where `rho_late \in (0,1]`, default `1/3`.

Define late-stage inversion indicator:
\[
I^{\text{late}}_{b,m} = \mathbf{1}\left[\exists l \in \mathcal L_{\text{late}} \text{ such that } \hat q^*_{b,l,m} \neq \hat q^*_{b,L,m}\right].
\]

A softer version may be the empirical late-stage switch frequency.

---

## 8. GT-near and GT-family existence

These are **offline / GT-dependent** definitions used only for attribution and label building.

Let `\tau^{gt}_b` be the ground-truth future trajectory.

Define candidate-to-GT distance with ADE/FDE:
\[
\mathrm{ADE}_{b,l,q} = \frac{1}{T} \sum_{t=1}^{T} \|\hat\tau_{b,l,q}(t) - \tau^{gt}_b(t)\|_2
\]
\[
\mathrm{FDE}_{b,l,q} = \|\hat\tau_{b,l,q}(T) - \tau^{gt}_b(T)\|_2.
\]

GT-near candidate exists if:
\[
\exists q \text{ such that } \mathrm{FDE}_{b,L,q} \le \delta_{\text{gt-fde}}
\]
or, if enabled, joint ADE/FDE thresholding.

GT-compatible family exists if at least one GT-near candidate belongs to a coarse family matching the GT family proxy:
\[
\exists q,m \text{ such that } \mathrm{FDE}_{b,L,q} \le \delta_{\text{gt-fde}},\; A_{b,L,q,m} \ge \eta_{\text{fam}},\; m = m^{gt}_b.
\]

The GT family proxy `m^{gt}_b` should be built using the same family-proxy machinery applied to the GT trajectory.

---

## 9. Deterioration score across decoder layers

For a GT-compatible family `m^{gt}_b`, define a proxy deterioration score:
\[
\Delta^{\text{org}}_b =
 w_1 \,[\pi^{\text{main}}_{b,l_0,m^{gt}} - \pi^{\text{main}}_{b,L,m^{gt}}]_+
+ w_2 \,[\pi^{\text{ps}}_{b,L,m^{gt}} - \pi^{\text{ps}}_{b,l_0,m^{gt}}]_+
+ w_3 \,[\widetilde R_{b,L,m^{gt}} - \widetilde R_{b,l_0,m^{gt}}]_+
+ w_4 \,[H^{\text{cluster}}_{b,L,m^{gt}} - H^{\text{cluster}}_{b,l_0,m^{gt}}]_+
+ w_5 \, N^{\text{switch}}_{b,m^{gt}}
+ w_6 \, I^{\text{late}}_{b,m^{gt}},
\]
where `l_0` is the first valid layer used as baseline, often `1` or an early-layer anchor.

Here `[x]_+ = \max(x,0)`.

This is the main operational summary of same-family organizational deterioration.

---

## 10. Final margin and hesitation

Let `q^(1)` and `q^(2)` be the final top-1 and top-2 candidates by raw final score.
Define final global margin:
\[
M^{\text{final}}_b = s_{b,L,q^{(1)}} - s_{b,L,q^{(2)}}.
\]

A sample is considered **small-margin** if:
\[
M^{\text{final}}_b \le \delta_{\text{margin}}.
\]

A sample is considered **late-hesitation** if either:
- `I^{late}_{b,m} = 1`, or
- the final top-ranked identity changes at least once in the late-stage layer set.

---

## 11. Three-way error attribution

These rules are offline and GT-dependent.

### 11.1 Coverage failure
Define:
\[
\mathrm{CoverageFail}_b = 1
\]
iff either:
1. no GT-near candidate exists, or
2. no GT-compatible family exists.

### 11.2 Guidance-linked organization failure
Define:
\[
\mathrm{GuidanceOrgFail}_b = 1
\]
iff all of the following hold:
1. GT-near candidate exists,
2. GT-compatible family exists,
3. deterioration score is high:
   \[
   \Delta^{\text{org}}_b \ge \delta_{\text{org-fail}},
   \]
4. and final outcome is bad or fragile:
   wrong final selection, or small final margin, or late hesitation.

### 11.3 Ranking-only failure
Define:
\[
\mathrm{RankingOnlyFail}_b = 1
\]
iff all of the following hold:
1. GT-near candidate exists,
2. GT-compatible family exists,
3. deterioration score is **not** high:
   \[
   \Delta^{\text{org}}_b < \delta_{\text{org-fail}},
   \]
4. but final selection is still wrong.

Mutual-exclusion rule:
- assign `coverage_failure` first,
- then `guidance_org_failure`,
- then `ranking_only_failure`.

---

## 12. Guidance-quality labels

These labels are used for Stage-A offline supervision only.

Define:
\[
\mathrm{GuidanceRiskPositive}_b = \mathbf{1}[\mathrm{GuidanceOrgFail}_b = 1].
\]

Define:
\[
\mathrm{GuidanceStable}_b = 1
\]
iff:
1. final selection is correct,
2. `\Delta^{\text{org}}_b < \delta_{\text{stable}}`,
3. `N^{\text{switch}}_{b,m^{gt}} \le \delta_{\text{switch-stable}}`,
4. final margin is not small.

All other ambiguous cases are:
\[
\mathrm{GuidanceUncertain}_b = 1.
\]

Do **not** use strong causal language such as “strict bad guidance.”
These are proxy labels.

---

## 13. Online risk predictor

At inference time there is **no GT**.
The online path may only use current-layer proxies and local context.

For candidate or family-local context vector `x^{org}_{b,l,m}`, predict
\[
r_{b,l,m} = \sigma(f_{\text{risk}}(x^{org}_{b,l,m})) \in [0,1].
\]

Risk gate:
\[
g_{b,l,m} = \mathbf{1}[r_{b,l,m} \ge \delta_{\text{risk}}].
\]

Use the gate only on later layers unless config explicitly says otherwise:
\[
g_{b,l,m} \leftarrow g_{b,l,m} \cdot \mathbf{1}[l \in \mathcal L_{\text{corr}}].
\]

---

## 14. In-layer correction

Let `h_{b,l,q}` denote the current query content before decoder layer update or immediately after that layer’s prediction step, depending on the exact code path.

Correction must be residual, bounded, and local:
\[
\Delta h_{b,l,q} = f_{\text{corr}}(h_{b,l,q}, \text{local-org-context}_{b,l,q})
\]
\[
\hat h_{b,l,q} = h_{b,l,q} + \lambda_{\text{corr}} \cdot g_{b,l,m(q)} \cdot \mathrm{clip}(\Delta h_{b,l,q}, -c_{\max}, c_{\max}).
\]

Rules:
- no extra denoiser forward,
- no extra decoder layer,
- no global all-candidate shrinkage,
- only same-family local context may drive correction,
- apply mostly on late layers.

---

## 15. Candidate-wise residual reranking

Final decision must remain candidate-wise.

Define a lightweight organization-aware residual rerank score:
\[
\delta s^{\text{org}}_{b,q} =
 w_C C_{b,L,m(q)}
+ w_Q Q^{\text{main}}_{b,L,m(q)}
+ w_D D_{b,L,m(q)}
- w_H H^{\text{cluster}}_{b,L,m(q)}
- w_R \widetilde R_{b,L,m(q)}
- w_I I^{\text{late}}_{b,m(q)}.
\]

Final reranked score:
\[
s'_{b,q} = s_{b,L,q} + \lambda_{\text{rr}} \, \delta s^{\text{org}}_{b,q}.
\]

Constraint:
- `\lambda_rr` must be bounded,
- reranking must not collapse to family-first selection,
- final winner is still:
\[
q^{\text{final}}_b = \arg\max_q s'_{b,q}.
\]

---

## 16. Config parameters that must exist

At minimum, expose the following in yaml:

```yaml
CGC:
  ENABLE: false
  FAMILY:
    ASSIGNMENT: soft
    ENDPOINT_RADIUS: 4.0
    HEADING_BUCKET_DEG: 20
    TURN_BUCKETS: [straight, left, right, uturn]
    STABILIZE_ALPHA: 0.8
  PROXY:
    MAIN_RADIUS: 3.0
    TUBE_RADIUS: 5.0
    MIN_CLUSTER_MASS: 0.08
    EPS: 1e-8
    A_MASS: 1.0
    A_COMP: 1.0
    A_VAR: 1.0
    A_ENT: 1.0
    BETA_MAIN: 1.0
    BETA_PS: 1.0
    BETA_RES: 0.5
    BETA_VAR: 1.0
    BETA_LEAK: 1.0
  ATTRIBUTION:
    GT_FDE_THRESH: 2.0
    GT_ADE_THRESH: 1.0
    FAMILY_MATCH_THRESH: 0.5
    ORG_FAIL_THRESH: 0.4
    STABLE_THRESH: 0.15
    SMALL_MARGIN_THRESH: 0.2
    LATE_RATIO: 0.33
    STABLE_SWITCH_MAX: 0
  CORRECTION:
    ONLY_LATE_LAYERS: true
    LATE_START_RATIO: 0.5
    RISK_THRESH: 0.6
    STRENGTH: 0.2
    DELTA_CLIP: 0.5
  RERANK:
    ENABLE: true
    LAMBDA: 0.1
  AUDIT:
    ENABLE: false
    SAMPLING_STEPS: 8
    RETURN_ALL_TIMESTEPS: true
```

All values above are defaults / placeholders and can be tuned.

---

## 17. Implementation priority

If there is tension between perfect theory-faithfulness and stable code integration, prefer:
1. explicit operational definitions,
2. stable one-step integration,
3. config-gated minimal changes,
4. preserving original baseline behavior when CGC is off.

The code should implement the formulas above faithfully enough that the proposal concepts
`dominant ridge / width floor / pseudo-peak competition / clarity / inversion risk`
are represented by reproducible proxy computations.
