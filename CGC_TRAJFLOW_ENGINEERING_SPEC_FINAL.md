# CGC_TRAJFLOW_ENGINEERING_SPEC.md

## 0. Purpose

This file is the **code-facing engineering specification** for integrating a CGC-style
(organization-aware coarse-guidance correction) module into the **TrajFlow** codebase.

This file is **not** a proof document.
It is a precise implementation spec.

It is written to satisfy the following hard requirements:

- keep the official TrajFlow mainline **one-step** flow inference,
- keep final online decision **candidate-wise**,
- use decoder layers as the main proxy time axis,
- do **three-way** failure attribution,
- train correction **only** on guidance-linked organization failure,
- do **not** replace existing TrajFlow losses,
- add only **guidance-side** losses and guidance-side modules.

The original TrajFlow code currently exposes the correct insertion points for this design:
`apply_transformer_decoder(...)` contains the per-layer decoder loop, `generate_final_prediction(...)`
contains final candidate selection, `train_model(...)` contains the main training loop,
and the default Waymo config uses `NUM_QUERY = 64`, `NUM_FUTURE_FRAMES = 80`,
`NUM_MOTION_MODES = 6`, and `DENOISING.FM.SAMPLING_STEPS = 1`.
These are the anchors for the implementation in this spec.

---

## 1. Source priority

If there is any tension between documents, use this priority:

1. **Implementation memo / prompt**: hard engineering constraints.
2. **Proposal**: problem definition, intended failure target, two-stage training write-back idea.
3. **Theory draft**: mechanism meaning, proxy direction, clarity/dominance/inversion language.

In particular:

- The proposal describes a family-level canonical-selection story.
- The implementation memo explicitly requires that the **final online output remain candidate-wise**.
- Therefore, in code, **family is only an operational structure**, not the final hard gate.


### 1.1 Approved engineering meaning of “guidance correction”

In code, “guidance correction” is operationalized as a **residual correction on guidance-conditioned decoder query updates**
inside the existing decoder-layer loop.

It is **not** implemented as:
- global overwriting of the coarse guidance tensor,
- extra flow sampling steps,
- extra denoiser forward passes,
- or a separate multi-step inference path.

This is the approved engineering realization of the proposal’s “adjust the direction and strength of guidance online”
language under the implementation memo’s one-step, local, lightweight, and candidate-wise constraints.

---

## 2. Files to modify and add

### 2.1 Existing files to modify

1. `runner/cfgs/waymo/trajflow+100_percent_data.yaml`
2. `trajflow/models/denoising_decoder/denoising_decoder.py`
3. `trajflow/models/denoising_decoder/decoder_utils.py`
4. `runner/utils/trainer.py`
5. `runner/utils/tester.py`

### 2.2 New files to add

Create a new directory:

`trajflow/organization/`

Add at least:

1. `family_proxy.py`
2. `proxies.py`
3. `recorder.py`
4. `error_attributor.py`
5. `guidance_labeler.py`
6. `experience_bank.py`
7. `risk_predictor.py`
8. `corrector.py`
9. `reranker.py`

---

## 3. Non-negotiable hard constraints

### 3.1 Keep official mainline one-step

The official TrajFlow path must remain one-step:

\[
\texttt{DENOISING.FM.SAMPLING\_STEPS} = 1
\]

Do **not** add any extra flow-sampling loop to the mainline method.

Optional multi-step analysis is allowed **only** inside a fully isolated audit mode.

### 3.2 Final online selection must remain candidate-wise

Do **not** implement:

\[
\text{family-first final decision}
\]

Do implement:

\[
\text{candidate-wise final ranking with organization-aware residual correction}
\]

Family is used only for:

- same-family organization statistics,
- offline attribution,
- offline guidance labeling,
- local correction context,
- local reranking context.

### 3.3 Only one failure type is the direct correction target

For every sample, split failures into:

1. coverage failure,
2. guidance-linked organization failure,
3. ranking-only failure.

Only type (2) is the direct correction-training target.

---

## 4. Notation

Let:

- \(b \in \{1,\dots,B\}\): batch index
- \(l \in \{1,\dots,L\}\): decoder-layer index
- \(q \in \{1,\dots,Q\}\): candidate / query index
- \(m \in \{1,\dots,M\}\): coarse family index
- \(t \in \{1,\dots,T\}\): future timestep
- \(Q = 64\), \(T = 80\) for the default Waymo setup
- \(L\): number of decoder layers in the current config
- \(\tau_{b,q}^{(l)} \in \mathbb{R}^{T \times 2}\): predicted trajectory of candidate \(q\) at layer \(l\)
- \(s_{b,q}^{(l)} \in \mathbb{R}\): raw candidate score at layer \(l\)
- \(p_{b,q}^{(l)}\): normalized candidate probability at layer \(l\)

Define:

\[
p_{b,q}^{(l)} = \frac{\exp(s_{b,q}^{(l)})}{\sum_{q'=1}^{Q} \exp(s_{b,q'}^{(l)})}
\]

All organization quantities in this document are computed from decoder-layer outputs.

---

## 5. Stage split

Training must be split into two stages.

### Stage A: `log_only`

Purpose:

- no mainline correction,
- no mainline rerank training,
- only record organization trajectories,
- perform error attribution,
- build guidance labels,
- build experience bank.

### Stage B: `train_correction`

Purpose:

- keep the same one-step mainline,
- activate online risk prediction,
- activate lightweight in-layer correction,
- activate candidate-wise residual reranking,
- optimize only added guidance-side objectives on top of original TrajFlow loss.

### Recommended stage split

Training must be controlled by config.

Let the total epoch count be:

\[
E = \texttt{OPT.NUM\_EPOCHS}
\]

Define:

\[
E_A = \lfloor \rho_A E \rfloor,\qquad
E_B = E - E_A
\]

where \(\rho_A \in (0,1)\) is a config field.

Recommended first baseline:

\[
\rho_A = 0.5
\]

For the default Waymo setup with \(E=30\), this gives:

\[
E_A = 15,\qquad E_B = 15
\]

This is a recommended baseline, not a hardwired rule.

---

## 6. Coarse family proxy

The family proxy is an operational structure.
It must be coarse, stable, and soft.

### 6.1 Family soft assignment

For each sample \(b\), layer \(l\), candidate \(q\), define a soft family assignment:

\[
A_{b,l,q,m} \in [0,1],
\qquad
\sum_{m=1}^{M} A_{b,l,q,m} = 1
\]

### 6.2 First-version family construction

Build \(A_{b,l,q,m}\) from a coarse geometry bucket composed of:

1. endpoint basin bucket,
2. terminal heading bucket,
3. turn-type bucket.

Let the endpoint of trajectory \(q\) be:

\[
e_{b,l,q} = \tau_{b,q,T}^{(l)} \in \mathbb{R}^2
\]

Let terminal heading be:

\[
\theta_{b,l,q} =
\operatorname{atan2}
\big(
\tau_{b,q,T}^{(l)}[1] - \tau_{b,q,T-1}^{(l)}[1],\,
\tau_{b,q,T}^{(l)}[0] - \tau_{b,q,T-1}^{(l)}[0]
\big)
\]

Engineering note:
if the code stores heading separately, use the stored terminal heading instead.

Turn type can be a 3-way bucket:

- left,
- straight,
- right.

For example, with heading change \(\Delta\theta_{b,l,q}\),

\[
\text{turn\_type}_{b,l,q}
=
\begin{cases}
\text{left}, & \Delta\theta_{b,l,q} > \tau_{\text{turn}} \\
\text{right}, & \Delta\theta_{b,l,q} < -\tau_{\text{turn}} \\
\text{straight}, & |\Delta\theta_{b,l,q}| \le \tau_{\text{turn}}
\end{cases}
\]

### 6.3 Family mass

\[
P_{b,l,m}
=
\sum_{q=1}^{Q} p_{b,q}^{(l)} A_{b,l,q,m}
\]

### 6.4 Within-family normalized weight

\[
\tilde p_{b,l,q \mid m}
=
\frac{p_{b,q}^{(l)} A_{b,l,q,m}}{P_{b,l,m} + \varepsilon}
\]

---

## 7. Same-family organization proxies

These must be computed for every sample \(b\), layer \(l\), family \(m\).

### 7.1 Dominant representative

\[
q^\star_{b,l,m}
=
\arg\max_{q} \tilde p_{b,l,q\mid m}
\]

### 7.2 Main neighborhood \(K_{\text{main}}\)

Let \(e^\star_{b,l,m} = e_{b,l,q^\star_{b,l,m}}\).

Define:

\[
K_{\text{main}}^{(b,l,m)}
=
\left\{
q :
\|e_{b,l,q} - e^\star_{b,l,m}\|_2 \le r_{\text{main}}
\right\}
\]

Optionally, combine endpoint distance and full-trajectory distance:

\[
d_{\text{traj}}(q,q^\star)
=
\frac{1}{T} \sum_{t=1}^{T}
\left\|
\tau_{b,q,t}^{(l)} - \tau_{b,q^\star,t}^{(l)}
\right\|_2
\]

and define:

\[
K_{\text{main}}^{(b,l,m)}
=
\left\{
q :
\|e_{b,l,q} - e^\star_{b,l,m}\|_2 \le r_{\text{main}}
\ \land\
d_{\text{traj}}(q,q^\star) \le r_{\text{traj}}
\right\}
\]

Use the second version only if it is computationally acceptable.

### 7.3 Competing same-family region \(K_{\text{ps}}\)

Within family \(m\), cluster the candidates not in \(K_{\text{main}}\) using endpoint clustering.
Let the resulting cluster index be \(c\), and let the cluster mass be:

\[
\omega_{b,l,m,c}
=
\sum_{q \in \mathcal{C}_{b,l,m,c}}
\tilde p_{b,l,q\mid m}
\]

Then define:

\[
K_{\text{ps}}^{(b,l,m)}
=
\bigcup_{c:\ \omega_{b,l,m,c} \ge \tau_{\text{cluster}}}
\mathcal{C}_{b,l,m,c}
\]

excluding the cluster that contains \(q^\star_{b,l,m}\).

### 7.4 Residual region \(R\)

\[
R^{(b,l,m)}
=
\left\{
q :
A_{b,l,q,m} > 0
\right\}
\setminus
\left(
K_{\text{main}}^{(b,l,m)} \cup K_{\text{ps}}^{(b,l,m)}
\right)
\]

### 7.5 Main / competitor / residual mass

\[
\pi_{\text{main}}^{(b,l,m)}
=
\sum_{q\in K_{\text{main}}^{(b,l,m)}}
\tilde p_{b,l,q\mid m}
\]

\[
\pi_{\text{ps}}^{(b,l,m)}
=
\sum_{q\in K_{\text{ps}}^{(b,l,m)}}
\tilde p_{b,l,q\mid m}
\]

\[
\pi_{\text{res}}^{(b,l,m)}
=
1 - \pi_{\text{main}}^{(b,l,m)} - \pi_{\text{ps}}^{(b,l,m)}
\]

### 7.6 Family center and dispersion

\[
\bar e_{b,l,m}
=
\sum_{q=1}^{Q} \tilde p_{b,l,q\mid m} e_{b,l,q}
\]

\[
R_{b,l,m}
=
\sum_{q=1}^{Q}
\tilde p_{b,l,q\mid m}
\|e_{b,l,q} - \bar e_{b,l,m}\|_2^2
\]

### 7.7 Cluster entropy

\[
H_{b,l,m}
=
-\sum_{c}
\omega_{b,l,m,c}
\log(\omega_{b,l,m,c} + \varepsilon)
\]

### 7.8 Leak

Use the engineering version:

\[
\ell_{b,l,m}
=
1 - P_{b,l,m}
\]

This is the simplest operational leak term consistent with the implementation memo.

### 7.9 Same-family dominance score

\[
Q_{\text{main}}^{(b,l,m)}
=
a_{\text{mass}}\pi_{\text{main}}^{(b,l,m)}
-
a_{\text{comp}}\pi_{\text{ps}}^{(b,l,m)}
-
a_{\text{var}}R_{b,l,m}
-
a_{\text{ent}}H_{b,l,m}
\]

with all coefficients positive.

### 7.10 Same-family clarity score

\[
C_{b,l,m}
=
\beta_{\text{main}}\pi_{\text{main}}^{(b,l,m)}
-
\beta_{\text{ps}}\pi_{\text{ps}}^{(b,l,m)}
-
\beta_{\text{res}}\pi_{\text{res}}^{(b,l,m)}
-
\beta_{\text{var}}R_{b,l,m}
-
\beta_{\text{leak}}\ell_{b,l,m}
\]

with all coefficients positive.

### 7.11 Local candidate margin

For candidate-wise diagnostics, let the within-family top-1 and top-2 candidate probabilities be
\(u_{b,l,m}^{(1)}\) and \(u_{b,l,m}^{(2)}\). Then define:

\[
\text{margin}_{b,l,m}
=
u_{b,l,m}^{(1)} - u_{b,l,m}^{(2)}
\]

### 7.12 Representative switching

Let:

\[
q^\star_{b,l,m}
\neq
q^\star_{b,l-1,m}
\]

be an indicator of dominant representative change.
Then define cumulative switch count up to layer \(l\):

\[
\text{SwitchCount}_{b,l,m}
=
\sum_{j=2}^{l}
\mathbf{1}
\left[
q^\star_{b,j,m} \neq q^\star_{b,j-1,m}
\right]
\]

### 7.13 Late-stage inversion flag

Let the final selected candidate at layer \(L\) be \(\hat q_b\), and let \(q^\star_{b,L,m}\) be the dominant
candidate of family \(m\) at the final layer.
Define a late-stage inversion flag:

\[
\text{LateInv}_{b}
=
\mathbf{1}
\left[
\hat q_b \neq q^\star_{b,L,\hat m_b}
\ \land\
\text{margin}_{b,L,\hat m_b} < \tau_{\text{margin}}
\right]
\]

where \(\hat m_b\) is the family associated with the finally selected candidate.

---

## 8. Early/late summaries

Use the decoder-layer axis as the proxy time axis.

Let the early layer set be:

\[
\mathcal{L}_{\text{early}} = \{1,\dots,\lfloor L/2 \rfloor\}
\]

Let the late layer set be:

\[
\mathcal{L}_{\text{late}} = \{\lfloor L/2 \rfloor+1,\dots,L\}
\]

For any layer quantity \(X_{b,l,m}\), define:

\[
X_{b,m}^{\text{early}}
=
\frac{1}{|\mathcal{L}_{\text{early}}|}
\sum_{l \in \mathcal{L}_{\text{early}}}
X_{b,l,m}
\]

\[
X_{b,m}^{\text{late}}
=
\frac{1}{|\mathcal{L}_{\text{late}}|}
\sum_{l \in \mathcal{L}_{\text{late}}}
X_{b,l,m}
\]

---

## 9. Three-way failure attribution

This must be done offline in Stage A and during validation.

### 9.1 GT-near candidate exists

Let \(\tau_b^{gt}\) be the ground-truth future trajectory.
Define:

\[
\text{ADE}(q,gt)
=
\frac{1}{T}
\sum_{t=1}^{T}
\left\|
\tau_{b,q,t}^{(L)} - \tau_{b,t}^{gt}
\right\|_2
\]

Then:

\[
\text{GTNearExists}_b
=
\mathbf{1}
\left[
\min_{q}
\text{ADE}(q,gt)
\le
\delta_{\text{gt}}
\right]
\]

### 9.2A GT-to-family mapping and compatibility rule

Map the GT trajectory to a coarse family \(m_b^{gt}\) using the **same** family-construction pipeline
used for predicted candidates:

1. GT endpoint basin bucket,
2. GT terminal heading bucket,
3. GT turn-type bucket.

Let the GT terminal endpoint be:

\[
e_b^{gt} = \tau_{b,T}^{gt}
\]

Let GT terminal heading be:

\[
\theta_b^{gt}
=
\operatorname{atan2}
\big(
\tau_{b,T}^{gt}[1] - \tau_{b,T-1}^{gt}[1],\,
\tau_{b,T}^{gt}[0] - \tau_{b,T-1}^{gt}[0]
\big)
\]

Define GT turn type:

\[
\text{turn\_type}_b^{gt}
=
\begin{cases}
\text{left}, & \Delta\theta_b^{gt} > \tau_{\text{turn}} \\
\text{right}, & \Delta\theta_b^{gt} < -\tau_{\text{turn}} \\
\text{straight}, & |\Delta\theta_b^{gt}| \le \tau_{\text{turn}}
\end{cases}
\]

Define family compatibility as:

\[
\text{Compat}(m,m_b^{gt})
=
\mathbf{1}
\left[
\text{endpoint\_bucket}(m)=\text{endpoint\_bucket}(m_b^{gt})
\ \land\
\text{turn\_type}(m)=\text{turn\_type}(m_b^{gt})
\ \land\
|\text{heading\_bucket}(m)-\text{heading\_bucket}(m_b^{gt})|
\le 1
\right]
\]

This explicit rule must be used both for offline attribution and for GT-family-exists checks.

### 9.2 GT-compatible family exists

Map the GT trajectory to a coarse family \(m_b^{gt}\).
Define family compatibility:

\[
\text{Compat}(m, m_b^{gt}) \in \{0,1\}
\]

Then:

\[
\text{GTFamilyExists}_b
=
\mathbf{1}
\left[
\exists m :
\text{Compat}(m, m_b^{gt}) = 1
\ \land\
P_{b,L,m} \ge \rho_{\text{fam}}
\right]
\]

### 9.3 Final margin

Let the final two highest candidate probabilities be \(p_{b,L}^{(1)}\) and \(p_{b,L}^{(2)}\).
Then:

\[
\text{margin}_{b}^{\text{final}}
=
p_{b,L}^{(1)} - p_{b,L}^{(2)}
\]

### 9.4 Wrong selection flag

Let \(q_b^{gt\star}\) be the GT-nearest candidate at the final layer.
Then define:

\[
\text{WrongSel}_b
=
\mathbf{1}
\left[
\hat q_b \neq q_b^{gt\star}
\right]
\]

where \(\hat q_b\) is the final selected candidate index.

### 9.5 Deterioration score

For the GT-compatible family \(m_b^{gt}\), define:

\[
D_b
=
\alpha_1
\left[
\pi_{\text{main},b,m_b^{gt}}^{\text{early}}
-
\pi_{\text{main},b,m_b^{gt}}^{\text{late}}
\right]_+
+
\alpha_2
\left[
\pi_{\text{ps},b,m_b^{gt}}^{\text{late}}
-
\pi_{\text{ps},b,m_b^{gt}}^{\text{early}}
\right]_+
\]

\[
\qquad
+
\alpha_3
\left[
R_{b,m_b^{gt}}^{\text{late}}
-
R_{b,m_b^{gt}}^{\text{early}}
\right]_+
+
\alpha_4
\left[
H_{b,m_b^{gt}}^{\text{late}}
-
H_{b,m_b^{gt}}^{\text{early}}
\right]_+
\]

\[
\qquad
+
\alpha_5
\cdot \text{SwitchCount}_{b,L,m_b^{gt}}
+
\alpha_6
\cdot \text{LateInv}_b
\]

where \([x]_+ = \max(x,0)\).

### 9.6 Failure type definitions

#### Coverage failure

\[
\text{CoverageFail}_b
=
\mathbf{1}
\left[
\text{GTNearExists}_b = 0
\ \lor\
\text{GTFamilyExists}_b = 0
\right]
\]

#### Guidance-linked organization failure

\[
\text{GuidanceOrgFail}_b
=
\mathbf{1}
\left[
\text{GTNearExists}_b = 1
\ \land\
\text{GTFamilyExists}_b = 1
\ \land\
D_b > \tau_{\text{org}}
\ \land\
\left(
\text{WrongSel}_b = 1
\ \lor\
\text{margin}_{b}^{\text{final}} < \tau_{\text{margin}}
\right)
\right]
\]

#### Ranking-only failure

\[
\text{RankingOnlyFail}_b
=
\mathbf{1}
\left[
\text{GTNearExists}_b = 1
\ \land\
\text{GTFamilyExists}_b = 1
\ \land\
D_b \le \tau_{\text{org}}
\ \land\
\text{WrongSel}_b = 1
\right]
\]

---

## 10. Guidance-quality labels

### 10.1 Positive risk label

\[
y_b^{\text{risk}} = \text{GuidanceOrgFail}_b
\]

### 10.2 Stable-good label

Define stable-good:

\[
y_b^{\text{pos}}
=
\mathbf{1}
\left[
\text{CoverageFail}_b = 0
\ \land\
\text{RankingOnlyFail}_b = 0
\ \land\
\text{GuidanceOrgFail}_b = 0
\ \land\
\text{margin}_{b}^{\text{final}} \ge \tau_{\text{good}}
\ \land\
D_b \le \tau_{\text{stable}}
\right]
\]

### 10.3 Ignore label

\[
y_b^{\text{ignore}}
=
1 - y_b^{\text{risk}} - y_b^{\text{pos}}
\]

Only \(y_b^{\text{risk}}\) and \(y_b^{\text{pos}}\) are used for direct guidance-side supervision.

---

## 11. Experience bank

The experience bank is a **training-side** module.
It must not become a heavy retrieval module inside official inference.

### 11.1 Stored groups

Maintain three queues:

- `bank_pos`: stable-good samples
- `bank_risk`: guidance-risk-positive samples
- `bank_ignore`: optional cache only, not used for direct supervision

### 11.2 Stored embedding

For each eligible sample \(b\), store a compact summary embedding:

\[
z_b
=
\left[
\pi_{\text{main}}^{\text{late}},
\pi_{\text{ps}}^{\text{late}},
\pi_{\text{res}}^{\text{late}},
R^{\text{late}},
H^{\text{late}},
\ell^{\text{late}},
\Delta\pi_{\text{main}},
\Delta\pi_{\text{ps}},
\Delta R,
\Delta H,
\text{SwitchCount},
\text{margin}^{\text{final}},
\text{LateInv}
\right]
\]

where:

\[
\Delta\pi_{\text{main}}
=
\pi_{\text{main}}^{\text{late}} - \pi_{\text{main}}^{\text{early}}
\]

\[
\Delta\pi_{\text{ps}}
=
\pi_{\text{ps}}^{\text{late}} - \pi_{\text{ps}}^{\text{early}}
\]

\[
\Delta R
=
R^{\text{late}} - R^{\text{early}}
\]

\[
\Delta H
=
H^{\text{late}} - H^{\text{early}}
\]

Then embed via:

\[
h_b = \operatorname{Proj}(z_b) \in \mathbb{R}^{d_h}
\]

with \(d_h = 64\) recommended for the first version.

### 11.3 Prototype means

Let:

\[
\mu_{\text{pos}}
=
\frac{1}{|bank_{\text{pos}}|}
\sum_{h \in bank_{\text{pos}}} h
\]

\[
\mu_{\text{risk}}
=
\frac{1}{|bank_{\text{risk}}|}
\sum_{h \in bank_{\text{risk}}} h
\]

Use EMA updates in code for stability.

### 11.4 Threshold rule for risk-bank admission

To avoid too few or too many risk samples, only push to `bank_risk` if:

\[
y_b^{\text{risk}} = 1
\quad\land\quad
D_b \ge \tau_{\text{bank,min}}
\quad\land\quad
D_b \le \tau_{\text{bank,max}}
\]

The thresholds must be chosen by **one and only one** of the following two modes:

**Mode A: fixed constants**
\[
\tau_{\text{bank,min}} = \tau_{\text{org}},
\qquad
\tau_{\text{bank,max}} = \tau_{\text{bank,max,const}}
\]

**Mode B: Stage-A percentile calibration**
\[
\tau_{\text{bank,min}} = \operatorname{Quantile}(D, q_{\min})
\]

\[
\tau_{\text{bank,max}} = \operatorname{Quantile}(D, q_{\max})
\]

with recommended:

\[
q_{\min}=0.60,
\qquad
q_{\max}=0.95
\]

The implementation must expose both modes in config, and use only one mode at a time.

---

## 12. Sample-matching learner and candidate-local risk predictor

Do **not** implement a heavy retrieval engine.
Use prototype-style memory matching.

### 12.1 Similarities

\[
s_b^{\text{risk-proto}}
=
\cos(h_b, \mu_{\text{risk}})
\]

\[
s_b^{\text{pos-proto}}
=
\cos(h_b, \mu_{\text{pos}})
\]

### 12.2 Candidate-local risk predictor

For each sample \(b\), current layer \(l\), and candidate \(q\), define a candidate-local risk score:

\[
r_{b,l,q}
=
\sigma
\left(
\operatorname{MLP}_{\text{risk}}
\Big(
[h_b;\ o_{b,l,q};\ s_b^{\text{risk-proto}};\ s_b^{\text{pos-proto}}]
\Big)
\right)
\]

where:

- \(h_b\) is the Stage-A / online organization summary embedding,
- \(o_{b,l,q}\) is the candidate-local organization context from Section 13.1,
- \(s_b^{\text{risk-proto}}\) and \(s_b^{\text{pos-proto}}\) are the prototype similarities.

This \(r_{b,l,q}\) is the risk score used by the correction gate.

This is the recommended first implementation.
Do **not** use kNN retrieval in the official first version.

---

## 13. In-layer correction

The correction must be inserted **inside the existing decoder-layer loop** in
`apply_transformer_decoder(...)`.

### 13.1 Candidate-local organization context

For each sample \(b\), layer \(l\), candidate \(q\), define:

\[
o_{b,l,q}
=
\left[
\pi_{\text{main}}^{(b,l,m(q))},
\pi_{\text{ps}}^{(b,l,m(q))},
\pi_{\text{res}}^{(b,l,m(q))},
R_{b,l,m(q)},
H_{b,l,m(q)},
\ell_{b,l,m(q)},
\text{margin}_{b,l,m(q)},
\text{SwitchCount}_{b,l,m(q)}
\right]
\]

where \(m(q)\) is the coarse family associated with candidate \(q\).

### 13.2 Risk gate

Correction is applied only in late layers.

Define the late-layer indicator:

\[
\mathbb{I}_{\text{late}}(l)
=
\mathbf{1}[l \ge l_0]
\]

with recommended:

\[
l_0 = \left\lfloor \frac{L}{2} \right\rfloor + 1
\]

Then define the candidate gate:

\[
g_{b,l,q}
=
\mathbb{I}_{\text{late}}(l)
\cdot
\sigma
\left(
\eta_1 r_{b,l,q}
+
\eta_2(\tau_{\text{margin}} - \text{margin}_{b,l,m(q)})
+
\eta_3 \text{SwitchCount}_{b,l,m(q)}
+
\eta_4 D_b
+
b_g
\right)
\]

### 13.3 Query residual

Let the current decoder query content be \(u_{b,l,q}\).
Define a correction residual:

\[
\Delta u_{b,l,q}
=
\gamma
\tanh
\left(
\operatorname{MLP}_{\text{corr}}
\left(
[u_{b,l,q}; o_{b,l,q}]
\right)
\right)
\]

### 13.4 Bounded update

\[
\widetilde{\Delta u}_{b,l,q}
=
\operatorname{clip}
\left(
\Delta u_{b,l,q},
-\delta_u,
\delta_u
\right)
\]

\[
u'_{b,l,q}
=
u_{b,l,q}
+
g_{b,l,q}\widetilde{\Delta u}_{b,l,q}
\]

This is the only correction update for the first implementation.

Do **not** directly overwrite the global coarse guidance tensor.
The correct operational interpretation is:
**the correction changes how the query uses guidance in late decoder layers**.

---

## 14. Candidate-wise residual reranking

This must be implemented in `decoder_utils.py` while preserving the raw path.

Let \(s_{b,q}^{\text{raw}}\) be the final raw score used by the original decoder output.

Define local rerank features for candidate \(q\):

\[
u_{b,q}^{\text{main}}
=
\pi_{\text{main}}^{(b,L,m(q))}
\]

\[
u_{b,q}^{\text{comp}}
=
\pi_{\text{ps}}^{(b,L,m(q))}
\]

\[
u_{b,q}^{\text{var}}
=
R_{b,L,m(q)}
\]

\[
u_{b,q}^{\text{ent}}
=
H_{b,L,m(q)}
\]

\[
u_{b,q}^{\text{leak}}
=
\ell_{b,L,m(q)}
\]

\[
u_{b,q}^{\text{inv}}
=
\text{SwitchCount}_{b,L,m(q)}
+
\text{LateInv}_{b}
\]

Then define the organization-enhanced score:

\[
s_{b,q}^{\text{org,pre}}
=
s_{b,q}^{\text{raw}}
+
\lambda_{\text{main}}u_{b,q}^{\text{main}}
-
\lambda_{\text{comp}}u_{b,q}^{\text{comp}}
-
\lambda_{\text{var}}u_{b,q}^{\text{var}}
-
\lambda_{\text{ent}}u_{b,q}^{\text{ent}}
-
\lambda_{\text{leak}}u_{b,q}^{\text{leak}}
-
\lambda_{\text{inv}}u_{b,q}^{\text{inv}}
\]

Apply a bounded residual clamp:

\[
\Delta s_{b,q}
=
\operatorname{clip}
\left(
s_{b,q}^{\text{org,pre}} - s_{b,q}^{\text{raw}},
-\delta_s,\,
\delta_s
\right)
\]

Final organization-aware score:

\[
s_{b,q}^{\text{org}}
=
s_{b,q}^{\text{raw}} + \Delta s_{b,q}
\]

Final online selection remains:

\[
\hat q_b
=
\arg\max_q s_{b,q}^{\text{org}}
\]

This is still **candidate-wise**, not family-first.

---

## 15. Guidance-side losses only

Do **not** change or replace the original TrajFlow loss terms.
Use the original TrajFlow loss as-is.

Let:

\[
L_{\text{base}}
=
L_{\text{TrajFlow-original}}
\]

The added losses only supervise the new guidance-side modules.

### 15.1 Mask definitions

Define the risk mask:

\[
m_b^{\text{risk}} = y_b^{\text{risk}}
\]

Define the stable mask:

\[
m_b^{\text{pos}} = y_b^{\text{pos}}
\]

Define the training mask:

\[
m_b^{\text{train}} = m_b^{\text{risk}} + m_b^{\text{pos}}
\]

Define the sample-level aggregated risk score used in the BCE loss:

\[
\bar r_b
=
\frac{1}{LQ}
\sum_{l=1}^{L}
\sum_{q=1}^{Q}
r_{b,l,q}
\]

Ignore samples where \(y_b^{\text{ignore}}=1\) for direct guidance-side losses.

### 15.2 Risk-classification loss

\[
L_{\text{risk}}
=
\frac{1}{\sum_b m_b^{\text{train}} + \varepsilon}
\sum_b
m_b^{\text{train}}
\cdot
\operatorname{BCE}(\bar r_b, y_b^{\text{risk}})
\]

### 15.3 Prototype-matching loss

\[
L_{\text{proto}}
=
\frac{1}{\sum_b m_b^{\text{train}} + \varepsilon}
\sum_b
m_b^{\text{train}}
\left[
y_b^{\text{risk}}
\|h_b - \mu_{\text{risk}}\|_2^2
+
y_b^{\text{pos}}
\|h_b - \mu_{\text{pos}}\|_2^2
\right]
\]

### 15.4A Threshold source rule for guidance-improvement targets

All thresholds appearing in \(L_{\text{guide-risk}}\) must be specified by exactly one of the following two modes:

**Mode A: fixed constants**
\[
\tau_{\pi},\ \tau_{\text{ps}},\ \tau_R,\ \tau_H,\ \tau_{\ell}
\]
are directly read from config.

**Mode B: Stage-A reference statistics**
Use Stage-A guidance-stable samples to compute reference targets:

\[
\tau_{\pi} = \operatorname{Mean}_{\text{pos}}(\pi_{\text{main}}^{\text{late}})
\]

\[
\tau_{\text{ps}} = \operatorname{Mean}_{\text{pos}}(\pi_{\text{ps}}^{\text{late}})
\]

\[
\tau_R = \operatorname{Mean}_{\text{pos}}(R^{\text{late}})
\]

\[
\tau_H = \operatorname{Mean}_{\text{pos}}(H^{\text{late}})
\]

\[
\tau_{\ell} = \operatorname{Mean}_{\text{pos}}(\ell^{\text{late}})
\]

The implementation must expose a config switch choosing between fixed-threshold mode and Stage-A-statistics mode.

### 15.4 Guidance-improvement loss on risk samples

This loss is used only on risk-positive samples.

\[
L_{\text{guide-risk}}
=
\frac{1}{\sum_b m_b^{\text{risk}} + \varepsilon}
\sum_b
m_b^{\text{risk}}
\cdot
\Big(
\omega_1[\tau_{\pi} - \pi_{\text{main},b,m_b^{gt}}^{\text{late}}]_+
+
\omega_2[\pi_{\text{ps},b,m_b^{gt}}^{\text{late}} - \tau_{\text{ps}}]_+
\]

\[
\qquad\qquad
+
\omega_3[R_{b,m_b^{gt}}^{\text{late}} - \tau_R]_+
+
\omega_4[H_{b,m_b^{gt}}^{\text{late}} - \tau_H]_+
+
\omega_5[\ell_{b,m_b^{gt}}^{\text{late}} - \tau_{\ell}]_+
+
\omega_6[\tau_{\text{margin}} - \text{margin}_{b}^{\text{final}}]_+
+
\omega_7 \text{SwitchCount}_{b,L,m_b^{gt}}
+
\omega_8 \text{LateInv}_b
\Big)
\]

Interpretation:
if the correction does not improve the target guidance-related supervision quantities,
this loss stays large.

### 15.5 Stable-preservation loss on good samples

On stable-good samples, the corrector should not overreact.
Therefore:

\[
L_{\text{guide-pos}}
=
\frac{1}{\sum_b m_b^{\text{pos}} + \varepsilon}
\sum_b
m_b^{\text{pos}}
\left(
\nu_1 \cdot \overline{g}_b^2
+
\nu_2 \cdot \frac{1}{LQ} \sum_{l,q} \|\widetilde{\Delta u}_{b,l,q}\|_2^2
\right)
\]

where:

\[
\overline{g}_b
=
\frac{1}{LQ}
\sum_{l=1}^{L}
\sum_{q=1}^{Q}
g_{b,l,q}
\]

This keeps the corrector near-off on already good guidance.

### 15.6 Rerank pairwise loss

Let \(q_b^{+}\) be the GT-near candidate and \(q_b^{-}\) the hardest wrongly favored candidate.
Then:

\[
L_{\text{rerank}}
=
\frac{1}{\sum_b m_b^{\text{risk}} + \varepsilon}
\sum_b
m_b^{\text{risk}}
\cdot
\left[
\kappa
-
s_{b,q_b^{+}}^{\text{org}}
+
s_{b,q_b^{-}}^{\text{org}}
\right]_+
\]

### 15.7 Total added guidance loss

\[
L_{\text{guidance}}
=
\lambda_{\text{risk}} L_{\text{risk}}
+
\lambda_{\text{proto}} L_{\text{proto}}
+
\lambda_{\text{grisk}} L_{\text{guide-risk}}
+
\lambda_{\text{gpos}} L_{\text{guide-pos}}
+
\lambda_{\text{rerank}} L_{\text{rerank}}
\]

### 15.8 Final total loss

\[
L_{\text{total}}
=
L_{\text{base}}
+
L_{\text{guidance}}
\]

This satisfies the requirement:
**do not modify other losses; only add guidance-side losses.**

---

## 16. Decoder integration points

This is the exact logic that must be inserted into
`trajflow/models/denoising_decoder/denoising_decoder.py`.

### 16.1 Stage A behavior inside decoder loop

For each layer \(l\):

1. obtain `pred_scores`, `pred_trajs`,
2. build family proxy,
3. compute organization proxies,
4. record layer summary.

No correction is applied.

### 16.2 Stage B behavior inside decoder loop

For each layer \(l\):

1. obtain `pred_scores`, `pred_trajs`,
2. build family proxy,
3. compute organization proxies,
4. update layer summary,
5. compute sample embedding \(h_b\),
6. compute risk score \(r_b\),
7. compute candidate gates \(g_{b,l,q}\),
8. if \(l \ge l_0\), apply bounded residual correction \(u'_{b,l,q}\).

### 16.3 Pseudocode

```python
for layer_idx in range(self.num_decoder_layers):
    pred_scores, pred_trajs, ... = current_layer_outputs

    family_proxy = build_family_proxy(pred_trajs, pred_scores, cfg)
    proxy_dict = compute_org_proxies(pred_trajs, pred_scores, family_proxy, cfg)
    recorder.update(layer_idx, family_proxy, proxy_dict)

    if stage_mode == "train_correction":
        h = summarize_proxy_sequence_current(proxy_dict, layer_idx, cfg)
        risk_score = risk_predictor(h, proxy_dict, exp_bank)
        gate = build_risk_gate(risk_score, proxy_dict, layer_idx, cfg)
        query_content = apply_org_correction(query_content, gate, proxy_dict, cfg)
```

---

## 17. Final-selection integration point

This is the exact logic that must be inserted into
`trajflow/models/denoising_decoder/decoder_utils.py`.

Keep the raw path.

### 17.1 Raw path

\[
s^{\text{raw}} = \sigma(\texttt{pred\_scores})
\]

### 17.2 Organization-enhanced path

Use \(s^{\text{org}}\) from Section 14.

### 17.3 Save both outputs

The final batch / return dict must keep at least:

- `pred_scores_raw`
- `pred_scores_org`
- `selected_idxs_raw`
- `selected_idxs_org`
- `proxy_summary`
- `family_proxy`

### 17.4 Pseudocode

```python
pred_scores_raw = torch.sigmoid(pred_scores)

pred_scores_org, rerank_aux = rerank_candidates_with_org_context(
    pred_scores_raw, pred_trajs, family_proxy, proxy_summary, cfg
)

selected_idxs_raw = topk_or_nms(pred_scores_raw, pred_trajs, num_motion_modes)
selected_idxs_org = topk_or_nms(pred_scores_org, pred_trajs, num_motion_modes)
```

---

## 18. Training loop behavior

Modify `runner/utils/trainer.py`.

### 18.1 Stage A

- run original TrajFlow forward,
- log proxies,
- attribute failure type,
- assign guidance label,
- push samples to experience bank,
- do **not** activate correction loss,
- do **not** activate rerank loss.

### 18.2 Stage B

- run corrected forward,
- compute original TrajFlow loss,
- compute added guidance losses,
- total loss is \(L_{\text{base}} + L_{\text{guidance}}\),
- continue updating experience bank online or freeze it after warm start.

### 18.3 Recommended Stage-B bank policy

Use:

- Stage A: build bank
- Stage B first half: bank is frozen
- Stage B second half: optionally allow EMA prototype updates only

This keeps learning stable.

---

## 19. Required logging

At minimum, print and save:

### Attribution statistics

- coverage failure ratio
- guidance-linked organization failure ratio
- ranking-only failure ratio

### Label statistics

- guidance-stable ratio
- guidance-risk-positive ratio
- ignore ratio

### Proxy statistics

- early / late \(\pi_{\text{main}}\)
- early / late \(\pi_{\text{ps}}\)
- early / late \(R\)
- early / late \(H\)
- early / late \(\ell\)
- switch count mean
- late inversion ratio
- final margin mean

### Risk / correction statistics

- mean \(r_b\)
- mean \(\overline g_b\)
- mean correction norm
- prototype similarities

### Performance statistics

- raw vs org minADE
- raw vs org minFDE
- raw vs org miss rate
- raw vs org top-1 margin

### Diversity guardrails

- distinct mode count
- pairwise endpoint spread
- oracle best-of-K distance

---

## 20. YAML config block

Add a `CGC` block to the Waymo YAML.

```yaml
CGC:
  ENABLE: true
  STAGE:
    MODE: "log_only"           # "log_only" or "train_correction"
    USE_RATIO_SPLIT: true
    STAGE_SPLIT_RATIO: 0.5
    ONLY_LATE_LAYERS: true
    LATE_START_RATIO: 0.5

  FAMILY_PROXY:
    USE_SOFT_ASSIGN: true
    ENDPOINT_RADIUS: 4.0
    TRAJ_RADIUS: 6.0
    TURN_THRESH: 0.25
    HEADING_BUCKETS: 8
    MIN_FAMILY_MASS: 0.05

  PROXY:
    MAIN_RADIUS: 4.0
    TRAJ_RADIUS: 6.0
    MIN_CLUSTER_MASS: 0.08
    EPS: 1e-8
    A_MASS: 1.0
    A_COMP: 1.0
    A_VAR: 0.2
    A_ENT: 0.2
    B_MAIN: 1.0
    B_PS: 1.0
    B_RES: 0.5
    B_VAR: 0.2
    B_LEAK: 0.2

  ATTRIBUTION:
    GT_NEAR_THRESH: 2.0
    GT_FAMILY_MIN_MASS: 0.10
    ORG_FAIL_THRESH: 0.60
    FINAL_MARGIN_THRESH: 0.05
    GOOD_MARGIN_THRESH: 0.10
    STABLE_THRESH: 0.20

  EXP_BANK:
    DIM: 64
    MAX_POS: 4096
    MAX_RISK: 4096
    MAX_IGNORE: 4096
    THRESH_MODE: "percentile"   # "fixed" or "percentile"
    BANK_MIN_D: 0.60
    BANK_MAX_D: 1.20
    BANK_Q_MIN: 0.60
    BANK_Q_MAX: 0.95
    EMA_MOMENTUM: 0.99
    FREEZE_AFTER_STAGE_A: true

  RISK:
    BETA_RISK_PROTO: 1.0
    BETA_POS_PROTO: 1.0
    ONLY_LATE_LAYERS: true
    GATE_BIAS: -1.0

  CORRECTOR:
    HIDDEN_DIM: 128
    GAMMA: 0.10
    DELTA_U_CLAMP: 0.10

  RERANK:
    LAMBDA_MAIN: 0.30
    LAMBDA_COMP: 0.30
    LAMBDA_VAR: 0.05
    LAMBDA_ENT: 0.05
    LAMBDA_LEAK: 0.05
    LAMBDA_INV: 0.05
    DELTA_S_CLAMP: 0.10
    PAIR_MARGIN: 0.05

  LOSS:
    LAMBDA_RISK: 0.20
    LAMBDA_PROTO: 0.05
    LAMBDA_GUIDE_RISK: 0.20
    LAMBDA_GUIDE_POS: 0.05
    LAMBDA_RERANK: 0.10
    TARGET_MODE: "stageA_stats"   # "fixed" or "stageA_stats"
    TAU_PI: 0.55
    TAU_PS: 0.20
    TAU_R: 10.0
    TAU_H: 1.0
    TAU_LEAK: 0.50

  DIVERSITY_GUARD:
    ENABLE: true
    MIN_MODE_COUNT_RATIO: 0.70
    MIN_SPREAD_RATIO: 0.80
    MAX_ORACLE_DEGRADATION: 0.05

  AUDIT:
    ENABLE: false
    SAMPLING_STEPS: 8
    RETURN_ALL_TIMESTEPS: true
```

All numeric values above are first-version engineering defaults and may be tuned.

---

## 21. What must not be changed

1. Do not replace original TrajFlow objective.
2. Do not add mainline multi-step flow inference.
3. Do not convert final online selection into family-first hard gating.
4. Do not make the experience bank a heavy inference-time retrieval system.
5. Do not treat all bad samples as bad guidance.
6. Do not train correction directly on coverage failure or ranking-only failure.

---

## 22. Final implementation summary

The final implementation must satisfy all of the following:

- Stage A records same-family posterior-organization trajectories.
- Stage A performs three-way failure attribution.
- Stage A builds an experience bank containing stable-good and guidance-risk-positive samples.
- Stage B uses prototype-style sample matching to predict guidance risk.
- Stage B adds a bounded, late-layer, candidate-local correction inside the existing decoder loop.
- Stage B adds a bounded candidate-wise reranking path.
- The original TrajFlow loss is preserved.
- Only guidance-side losses are added.
- The official mainline remains one-step.
- Final online selection remains candidate-wise.

That is the complete engineering spec.
