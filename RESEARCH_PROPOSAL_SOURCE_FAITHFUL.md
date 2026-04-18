# research-proposal-Peilin_Wang.pdf
> Source-faithful markdown conversion from the PDF text layer. Page order is preserved. Only very light cleanup was applied to remove obvious PDF line-break artifacts and trailing page-number-only lines. No conceptual paraphrasing was introduced.


<!-- Page 1 -->

Research Project Description

Beyond Family-Level Reachability
Posterior Clarity for Canonical Selection in
Coarse-Guided Conditional Flow Generation

Abstract
In conditional flow generation under weak or coarse guidance, models can often reach the
correct coarse solution family, yet still fail to maintain a dominant canonical representative
within that family. This failure can make the final selection unstable. This project
studies this underexplored problem, which we call posterior-organization failure. Existing
conditional flow methods are often effective at family-level formation. However, their
training objectives mainly ensure that generation enters a plausible solution family. They
do not guarantee that the posterior within that family will further contract into a clear
and stable canonical representative. In other words, reaching the correct family does
not mean that the model has achieved effective within-family organization. Even when
the coarse semantic structure is already correct, probability mass may still spread across
several nearby candidates. As a result, the dominant representative becomes less salient,
the ranking margin keeps shrinking, and the final selection becomes unstable.
Motivated by this observation, this project does not follow the usual path of treating
the problem as terminal mismatch or as a post-generation reranking issue. Instead, it
shifts the object of study forward to generation-time posterior organization. The project
proposes an organization-aware coarse-guidance correction framework. During generation,
the framework continuously monitors three signals: dominance, clarity, and inversion risk.
These signals describe whether the dominant representative remains concentrated, clear,
and stably selectable. The framework then uses them to adjust the direction and strength
of guidance online.
At the methodological level, this project further aims to write the correction mechanism
back into training. The model should learn not only from terminal outcomes, but also from
intermediate organizational patterns that lead either to stable selection or to selection
inversion. This design creates a consistent pipeline from process diagnosis, to guidance
correction, to canonical selection. As a result, the final decision will no longer rely only
on terminal fit. It will also consider whether the coarse family is reasonable and whether
that family contains a sufficiently clear and dominant canonical representative.
For evaluation, this project will study the framework from three perspectives: mechanism
visibility, effectiveness on representative tasks, and cross-domain transferability. The goal


<!-- Page 2 -->

is to test whether the framework can consistently reduce within-family probability mass
splitting, improve the identifiability of the dominant representative, and lower the risk of
selection inversion under weak guidance.
Overall, this project aims to address the black-box nature and mismatch issues introduced
by flow-based generative models and human-provided priors from a statistical and dynamical perspective. It treats the generation process itself as an object that can be modeled,
interpreted, and diagnosed, rather than as a hidden path that only leads to an output.
It also introduces stronger process-level criteria and decision principles for evaluating
generation quality. More broadly, the project seeks to reveal the incomplete-matching
mechanisms that commonly arise in generative AI during denoising and transport. In this
way, it aims to push generation beyond coarse feasibility toward finer, more stable, and
more trustworthy inference and decision-making.
Keywords: Conditional Flow Matching; Posterior Organization; Canonical Selection;
Coarse Guidance; Generative AI.

1. Background and Research Problem
This project studies a conditional generative modeling problem under weak or coarse
guidance. The goal is not to predict one deterministic output, but to learn a conditional
distribution of structured outputs under partial information. In many modern AI tasks, the
available condition is already strong enough to identify a correct broad family of admissible
solutions, yet it remains too weak to determine one unique fine-grained realization. Here,
a family means a high-level class of valid outputs that is globally compatible with the
condition, such as a goal region, a route type, or a semantic mode. Multimodality
is therefore not a nuisance. It is a structural feature of conditional prediction under
incomplete information. Flow-based generative methods are especially attractive in this
setting because they provide a clean and scalable way to transport a simple reference
distribution toward a complex conditional target, while also making the evolution of
probability mass explicit (Lipman et al., 2023, 2024; Liu et al., 2022; Albergo et al., 2023).
The central difficulty, however, is not only whether a model can reach a plausible coarse
family, but whether it can organize probability mass well enough within that family for
reliable final selection. In many tasks, coarse guidance already pushes generation into
the correct broad family, yet it does not preserve one dominant canonical representative
inside that family. The main peak may remain broad, nearby side peaks may survive, and
the score margin of the correct representative may gradually shrink during generation.
Once this happens, final ranking can be inverted. The model may then output either an
incorrect family or a non-canonical member of the correct family.
Recent trajectory-flow studies make this failure mode concrete. GoalFlow shows that
multimodal trajectory generation already faces trajectory divergence, guidance inconsistency, and candidate selection complexity (Xing et al., 2025). TrajFlow further shows
that generating multiple plausible futures is not enough by itself, and therefore introduces
an additional ranking loss to improve uncertainty estimation and downstream selection


<!-- Page 3 -->

Figure 1: Example of family-level formation without stable within-family concentration.
Although the model has already reached a plausible coarse maneuver family, probability mass
remains distributed across several nearby trajectories within that family instead of collapsing
onto one dominant canonical representative. This intra-family competition is precisely the type
of posterior-organization failure studied in this project, because it weakens the main trajectory,
splits score mass, and increases the risk of unstable final selection (Yan et al., 2025).

(Yan et al., 2025). These studies are important for the present project because they
suggest that practical failure often appears after the correct broad family has already
been reached. What remains unstable is the internal organization of the posterior and the
final choice made from it.
This project addresses that gap by shifting correction from terminal mismatch to generationtime organization. The key question is not simply how to rerank outputs after generation,
but whether a failure that cannot be repaired reliably from the final match can instead be
diagnosed and corrected from the evolving process itself. This direction is consistent with a
broader trend in generative modeling: recent work increasingly studies guidance correction,
training–inference consistency, and path-level reliability rather than only endpoint quality
(Liu et al., 2024; Xia et al., 2025; Saini et al., 2025; Wang et al., 2026; Harris, 2026). In
particular, Planner Aware Path Learning (PAPL) shows that once planning changes the
reverse denoising path at inference time, part of that path-level logic should be written
back into training (Peng et al., 2025). Building on this trend, the present project develops
an organization-aware coarse-guidance correction framework. The core modeling storyline,
supported by the accompanying proof draft, is that standard conditional flow mainly
learns family-level averages; unresolved refined subfamily structure leaves a non-vanishing
width floor around the dominant ridge; local nonlinear effects can then turn a broad
peak into a family peak cluster of nearby competitors; and the right objective therefore
becomes clarity-preserving canonical selection rather than endpoint fit alone. In practice,


<!-- Page 4 -->

the method monitors three coupled quantities during generation—dominance, clarity, and
inversion risk —and uses them to repair guidance before the final decision fails.

Figure 2: Synthetic visualization of the core mechanism studied in this project. All three rows
are transported toward the same coarse terminal family, but they exhibit markedly different
middle-stage organization. The top row shows canonical transport, where probability mass
remains well organized and the family representative stays stable. The middle row introduces a
same-family perturbation through a local biased fold, which creates a nearby side mode and
fragments posterior mass without changing the coarse terminal family. The bottom row applies
progressive correction, which suppresses the spurious same-family competitor and restores a
cleaner dominant representative.

2. Review of the Literature
2.1 Advanced applications have already exposed the practical difficulty
Recent application studies already show that the practical bottleneck is no longer generation alone, but generation plus final selection. In trajectory prediction, motion planning,
and goal-conditioned generation, modern flow-based methods can already produce several
plausible futures efficiently. However, once multiple plausible candidates are available,
a second problem immediately arises: which candidate should be trusted and finally
selected? This issue becomes especially visible when guidance is partial, scene information
is imperfect, or the downstream selection rule is weak.
Recent trajectory-flow work makes this point concrete. GoalFlow explicitly motivates
multimodal trajectory generation by noting that autonomous driving rarely admits
a single suitable future, while also highlighting trajectory selection complexity, high
trajectory divergence, and inconsistency between guidance and scene information as
practical obstacles (Xing et al., 2025). TrajFlow likewise shows that generating multiple
plausible futures is not enough by itself, and therefore introduces an additional ranking
loss to improve uncertainty estimation and downstream candidate ordering (Yan et al.,
2025). Taken together, these studies suggest that the empirical difficulty is not simply
whether the model can reach a plausible coarse family, but whether it can preserve one
sufficiently dominant representative after that family has already formed.


<!-- Page 5 -->

This distinction matters because it changes the object of study. If the main issue were
only family misspecification, then stronger guidance or a more expressive generator might
already suffice. By contrast, once the model has entered the correct family, the bottleneck
shifts to the posterior organization within that family: whether the dominant mode
remains sharp, whether nearby competitors absorb its mass, and whether the final ranking
still respects the correct representative. Existing application papers clearly reveal that
this problem is real, but they do not yet isolate it as an independent theoretical and
methodological target.
2.2 A broader trend is moving from pure generation toward guidance-aware
alignment
A broader trend in diffusion- and flow-based research is that guidance is increasingly
treated as something that must be aligned, corrected, and written back into training,
rather than as a fixed external signal. Recent work on diffusion alignment emphasizes
that modern generative systems are no longer judged only by raw sample quality, but
also by whether their outputs match intended properties, constraints, and downstream
preferences (Liu et al., 2024). In parallel, recent flow work shows that the field is moving
beyond pure generation toward steering, safety, path geometry, and interpretable transport,
as illustrated by rectified flow, stochastic interpolants, FlowChef, Safe Flow Matching,
stream-level flow matching, and Koopman-based flow models (Liu et al., 2022; Albergo
et al., 2023; Albergo and Vanden-Eijnden, 2022; Patel et al., 2025; Dai et al., 2025; Wei and
Ma, 2025; Turan et al., 2025). Guidance-specific corrections then appear in methods such
as rectified diffusion guidance, Rectified-CFG++, and CFG-Ctrl, which treat inference
steering itself as an object of modeling rather than a purely external control signal (Xia
et al., 2025; Saini et al., 2025; Wang et al., 2026).
This shift is especially relevant for the present project because it changes where correction
should happen. If planning, guidance, or control modifies the actual generation path,
then endpoint-level evaluation alone may no longer be sufficient. Instead, one must ask
whether the evolving transport process itself is being organized in the right way. In this
sense, guidance is no longer just an input; it becomes a dynamical object whose quality
must be assessed through the trajectory of generation.
Planner Aware Path Learning (PAPL) makes this logic especially explicit. It argues that
once planning changes the reverse denoising path at inference time, the original training
objective no longer matches the actual inference dynamics, and part of that path-level
logic should therefore be written back into training (Peng et al., 2025). This idea is
highly relevant here. It suggests a more general principle: when final mismatch cannot be
repaired reliably from endpoint fit alone, one should ask whether the failure can instead
be diagnosed and corrected from the evolving process itself. The present project follows
exactly this direction, but targets a different object, namely posterior-organization failure
under coarse guidance inside a fixed family.


<!-- Page 6 -->

2.3 The remaining gap: posterior-organization failure under coarse guidance
What remains insufficiently addressed in the current literature is not multimodality alone,
nor ranking alone, nor training–inference mismatch alone, but the problem of guidance
sufficiency within a fixed family. Even when coarse guidance is already adequate at the
family level, it may still remain insufficient at the representative level. In that case, the
dominant peak may fail to monopolize probability mass, nearby sibling peaks may survive,
and final selection may become unstable even though the correct coarse family has already
been reached.
This is the specific gap that motivates the present project. The literature already contains
neighboring concerns in the form of posterior collapse, latent non-identifiability, steering
mismatch, and weak ranking supervision (Lucas et al., 2019; Wang et al., 2021; Yan et al.,
2025; Peng et al., 2025). However, what is still missing is a guidance-aware correction
mechanism that directly targets the within-family degradation of posterior organization.
In other words, existing work typically improves generation quality, sampling efficiency,
or downstream ordering, but it does not yet directly model how a coarse family that
has already formed can still lose its canonical representative through local broadening,
competition, and fragmentation.
The present project therefore introduces a conceptually different target: clarity-preserving
correction. Instead of only optimizing endpoint fit or global trajectory plausibility, it
studies how to maintain a clear dominant ridge, suppress family-internal peak competition,
and prevent selection inversion. This shift is important because it connects three layers
that are often studied separately: distributional geometry, generation dynamics, and
downstream decision-making.

3. Theoretical Orientation
3.1 Why family-level reachability is not enough
Standard conditional flow objectives resolve family-level conditional means, but they do
not by themselves identify fixed-family residual structure. Let Ut denote the fine-grained
target field and let C denote coarse conditioning. Under the standard regression-style
training objective used in conditional flow matching, the optimal learnable field under
squared loss is the conditional mean
uC
t (Xt ) = E[Ut | Xt , C],
while the irreducible error is the corresponding conditional variance term (Lipman et al.,
2023, 2024; Albergo and Vanden-Eijnden, 2022; Albergo et al., 2023; Liu et al., 2022).
Thus, conditioning strictly improves the regressibility of the target field, but it does not
automatically determine the full residual covariance structure inside a fixed coarse family.
As a consequence, standard conditional flow training can be very effective at forming
high-level families, yet it may still leave substantial uncertainty unresolved within those
families.
This limitation becomes sharper once a coarse family label M is introduced. The corre6


<!-- Page 7 -->

sponding residual uncertainty decomposes naturally into a family-level component and
a fixed-family component. The first explains why conditional generation can quickly
form broad modes such as left turn versus right turn, or one goal basin versus another.
The second explains why, even after the correct family has formed, one may still observe
unresolved variation within that family, such as early turn versus late turn, different
entry angles, or different interaction rhythms. The key implication is therefore precise:
family-level formation is not the same as fixed-family concentration.
A further consequence is that standard conditional flow objectives are not guaranteed to
distinguish different fixed-family residual organizations that share the same family-level
conditional mean. Two systems may induce the same optimal regression target while still
differing in their fixed-family residual covariance, and hence in whether the corresponding
family is internally sharp or already contains nearby competing substructure. For the
present project, this is the critical starting point. The problem is not whether coarse
guidance can form the correct family at all, but whether the formed family is internally
organized well enough to support stable final selection.
3.2 Dominant ridge, width floor, and family peak cluster
The natural object of study after family formation is the local geometry of the fixed(m)
family conditional density itself. Fix a coarse family M = m and let ρt (x) denote the
corresponding family-conditional density. Rather than treating the final output as a point
(m)
estimate detached from geometry, the framework studies the local peak structure of ρt .
In particular, a dominant ridge is the local center line along which the family-conditional
density attains a stable peak in the transverse direction. Around that ridge one can
define a transverse fiber law and a transverse width, and this width measures whether the
dominant representative is sharply concentrated or still broad and unstable.
The accompanying proof draft shows that this width does not vanish automatically once
the correct family has been reached. If refined subfamily structure remains unresolved
inside the fixed family, then even local contraction can only shrink the transverse spread
down to a strictly positive width floor rather than collapse it into an ideal sharp spike. In
the proof draft, this floor is tied to unresolved subfamily separation through a fixed-family
Wasserstein-radius decomposition, which separates within-subfamily concentration from
between-subfamily separation and naturally places the argument in a Wasserstein transport
geometry (Villani, 2009; Santambrogio, 2015). The resulting conclusion is structural:
whenever finer subfamily organization is still present but not resolved by coarse guidance,
the dominant peak retains a positive geometric width floor.
This width floor matters because a broad dominant peak is not a harmless approximation
error. Once local nonlinear effects act on such a broad peak, the peak may bifurcate
into nearby side wells, shoulder structures, or redundant candidates. These are not new
legal coarse families. They are competitors that arise within the already-correct family.
Collectively, they form what the proof draft calls a family peak cluster. At a broader
mathematical level, the separation between contraction-driving and rearrangement-like
effects is also compatible with classical Helmholtz-type decompositions of vector fields
into gradient-like and divergence-free components (Gl”otzl and Richters, 2020). The


<!-- Page 8 -->

theoretical question is therefore not merely whether an extra side peak can appear, but
how the emergence of a family peak cluster changes the quality of the representative that
should ultimately be selected.
3.3 From peak existence to dominant-quality degradation
The central theoretical object in this project is not the mere existence of spurious peaks,
but the degradation of dominant-peak quality once family-internal competitors appear.
A statement of the form “a side peak may exist” is too weak for the present purpose,
because the practical decision problem is not to count peaks, but to explain why the
correct representative loses its advantage even after the correct family has already formed.
Once a family peak cluster emerges, the dominant representative is weakened through
three coupled channels. First, nearby competitors divert probability mass away from the
main peak. Second, the transverse width around the dominant ridge increases, which
makes the dominant representative less sharp and less stable. Third, the peak distribution
inside the family becomes more fragmented, which raises cluster-level entropy and makes
the final choice more sensitive to ranking noise or local perturbation. These three effects
jointly reduce the dominance margin of the canonical representative.
This reformulation changes what the theory is supposed to explain. The goal is not to
prove that coarse guidance sometimes produces local irregularities in a generic sense. The
goal is to explain why a family that is already correct at the coarse level can still fail at
the representative level. In the language of the proof draft, the key transition is from
family-level formation to fixed-family concentration, and then from broad dominant ridge
to family peak cluster. Once that transition happens, the practical failure is no longer one
of family misspecification, but one of dominant-quality degradation and eventual selection
inversion.
As illustrated in Figure 3, the key distinction is not whether the transport eventually
reaches a legal coarse family, but whether the posterior remains well organized within
that family during the middle stage of transport. The canonical row preserves a single
dominant representative throughout the evolution, whereas the perturbed row develops a
same-family side mode that fragments posterior mass and weakens the main peak. The
corrected row shows the intended effect of the proposed method: it does not redefine
the terminal family, but progressively suppresses the same-family competitor before final
selection becomes unstable.


<!-- Page 9 -->

Figure 3: Synthetic visualization of the core mechanism studied in this project. All three rows
are transported toward the same coarse terminal family, but they exhibit markedly different
middle-stage organization. The top row shows canonical transport, where probability mass
remains well organized and the family representative stays stable. The middle row introduces a
same-family perturbation through a local biased fold, which creates a nearby side mode and
fragments posterior mass without changing the coarse terminal family. The bottom row applies
progressive correction, which suppresses the spurious same-family competitor and restores a
cleaner dominant representative. The figure therefore illustrates the central distinction of this
project: the key failure is not necessarily missing the correct family, but losing the correct
within-family organization during transport.

3.4 Main theoretical consequences and methodological implication
The preceding analysis yields three main theoretical consequences and one methodological
implication.
C1. Standard conditional flow objectives can achieve family-level formation without
resolving fixed-family residual organization. In particular, they identify conditional mean
structure at the family level, but they do not automatically recover the full residual
covariance structure within a fixed family.
C2. If refined subfamily structure remains unresolved, the dominant ridge retains a
strictly positive width floor rather than collapsing into a unique sharp representative.
Consequently, correct family formation alone does not imply internal concentration onto


<!-- Page 10 -->

a canonical representative.
C3. Once local nonlinear geometry acts on such a broad peak, a family peak cluster
may emerge, and the dominant representative is then weakened by mass splitting, peak
competition, and increased fragmentation. The theoretical problem is therefore not merely
whether nearby peaks can appear, but how these same-family competitors jointly degrade
dominant-peak quality.
I1. The natural methodological response is not to rely on endpoint fit alone, but to
monitor quantities that are directly tied to representative stability during generation.
This is why the present project moves toward dominance, clarity, and inversion risk
as organization-aware signals for correction: if the failure mechanism is process-level
posterior-organization degradation within a fixed family, then the correction mechanism
should also operate at the level of evolving posterior organization rather than only at the
terminal output.
Standard conditional flow
forms coarse family

Fixed-family residual
structure remains

Width floor around
dominant ridge
under local
nonlinear interaction

Selection inversion
risk rises

Dominant quality
degrades

Family peak cluster
emerges

Figure 4: Theoretical storyline of the proposal. Standard conditional flow can form the correct
coarse family without resolving fixed-family residual organization. If unresolved refined subfamily
structure leaves a positive width floor around the dominant ridge, local nonlinear effects can
generate a family peak cluster. The resulting competition degrades dominant-peak quality and
eventually raises selection inversion risk.

4. Methodology
4.1 Methodological principle
The methodology of this project is to correct coarse guidance before posterior-organization
failure fully propagates to the final decision. The key idea is not to wait until the end
of generation to rerank candidates, but to monitor how guidance reshapes posterior
organization during transport itself. Once the correct coarse family has already been
reached, the practical question is no longer whether generation remains broadly plausible,
but whether one dominant representative can stay sufficiently sharp and stable inside that
family.
This principle follows directly from the theory in Section 3. If coarse guidance mainly
secures family-level reachability while leaving fixed-family residual structure unresolved,
then the correction mechanism should target the internal organization of that family
rather than only terminal fit. In other words, the project treats guidance not as a fixed
external signal, but as a revisable object whose quality should be judged by the posterior
organization it induces.


<!-- Page 11 -->

4.2 Organization-aware signals and guidance correction
The proposed framework uses three compact organization-aware signals: dominance,
clarity, and inversion risk. Dominance measures whether the main representative still
stands above nearby competitors, for example through the score or probability margin
between the top-ranked and second-ranked candidates within the same family. Clarity
measures whether probability mass remains concentrated around the dominant ridge
rather than dispersing across family-internal side peaks or residual regions; in practice, it
can be quantified through family-level posterior entropy or the local mass concentration
around the dominant candidate. Inversion risk measures whether the current deterioration
is already severe enough to threaten the final decision; operationally, it can be tracked
through the frequency or probability of top-ranked identity switches during generation.
Together, these quantities translate the geometric picture from Section 3 into operational
signals for guidance correction.
On this basis, the project introduces an organization-aware coarse-guidance correction
step. Rather than using raw coarse guidance directly, the method refines it according
to the organization it currently induces. If the present guidance weakens the dominant
representative, enlarges nearby competitors, reduces clarity, or raises inversion risk, the
correction step adjusts guidance toward a more favorable direction. The purpose is not
to replace the original guidance wholesale, but to repair the part of it that generates
within-family degradation.
4.3 Training write-back and canonical selection
The correction mechanism is then written back into training through a late-stage consistency design. Training proceeds in two linked stages. In the first stage, the model
runs under raw coarse guidance and records how posterior organization evolves during
generation, especially whether dominance and clarity remain high or begin to collapse. In
the second stage, these trajectories are used to learn the correction mechanism: trajectories with stable organization and correct final selection act as positive targets, whereas
trajectories that develop fragmentation and lead to unstable ranking act as warning
cases. In this way, the method learns not only from terminal outcomes, but from the
organizational patterns that precede them.
At inference time, the final decision follows a two-stage canonical selection rule. First,
among goal-feasible or terminal-feasible families, the model selects the family whose overall
organization is most favorable. Second, within that family, it selects the candidate that
remains closest to the dominant ridge and exhibits the most stable local organization. This
rule is not an extra heuristic added after the fact. It is the decision-level counterpart of
the theory: if final risk decomposes into family-level risk and within-family representative
risk, then the final output should be chosen by first selecting the family and then selecting
its canonical representative.


<!-- Page 12 -->

5. Data and Experimental Setting
The experimental program is designed to test the proposal at three progressively stronger
levels: synthetic mechanism validation, benchmark-level application validation, and a
finance-related transfer setting. This structure is deliberate. The first level makes the
theoretical mechanism directly visible, the second checks whether it remains useful in
representative flow-based applications, and the third tests whether the same correction
logic still matters in a low-signal, high-noise environment.
5.1 Synthetic validation
The first level uses low-dimensional synthetic tasks. The purpose of this setting is not to
maximize task performance, but to make the proposed mechanism directly observable. In
particular, the synthetic experiments are designed to exhibit the following sequence clearly:
the correct coarse family is already reachable, the dominant representative remains broad
because refined subfamily structure is unresolved, nearby same-family competitors emerge
under local nonlinear interaction, and final selection becomes unstable unless guidance is
corrected.
This setting serves two roles. First, it provides visual and quantitative validation of the
theoretical storyline. Second, it provides a controlled environment for testing whether
the proposed organization-aware signals are stable enough to be used in practice. The
main outputs at this stage are interpretable mechanism figures, organization metrics, and
initial validation of the correction logic.
5.2 Representative flow-based benchmarks
The second level is the main application validation. It will focus on representative flowbased motion-generation benchmarks associated with GoalFlow and TrajFlow, namely
NAVSIM and the Waymo Open Motion Dataset (WOMD).1 2 These benchmarks are
especially suitable because they combine multimodal candidate generation with the need
for final selection, which is exactly the setting in which posterior-organization failure
becomes operationally important.
Evaluation will be conducted at two levels. At the task level, standard motion-prediction
metrics such as ADE, FDE, minADE, and minFDE will be used to verify that the
correction mechanism does not sacrifice basic performance. At the organization level, the
experiments will examine whether correction improves representative stability, reduces
family-internal competition, and lowers selection inversion risk. To make the comparison
concrete, the main baselines will include GoalFlow-style coarse goal-guided flow generation,
TrajFlow-style multimodal flow generation with ranking-enhanced selection, a raw coarseguided version of our own pipeline without organization-aware correction, and selection- or
ranking-enhanced variants that improve downstream ordering without explicit posteriororganization repair. The central comparison will therefore ask not only whether the
proposed method preserves standard task performance, but also whether it yields a
1

Waymo Open Motion Dataset official page: https://waymo.com/open/data/motion/
NAVSIM official benchmark page: https://github.com/autonomousvision/navsim

2


<!-- Page 13 -->

larger Top-1 versus Top-2 score margin, lower family-level entropy, stronger local mass
concentration, and fewer ranking inversions throughout generation.
5.3 Fintech transfer setting
The third level is a finance-related transfer setting based on CSMAR and Tonghuashun
iFinD.3 4 This part is not included as a superficial application add-on. Its purpose is to test
whether the proposed correction mechanism remains useful when conditional generation
is embedded in a noisier and lower-signal environment.
Finance provides a natural low-SNR and weak-guidance testbed because macro signals,
relational priors, and market regimes often identify only a broad plausible family, while
the fine-grained dynamics of price movement, order flow, or risk propagation remain highly
noisy and easily admit unstable within-family competition. Conceptually, this extension
asks whether the distinction between reaching a plausible family and preserving a dominant
representative becomes even more important when posterior peaks are easily buried under
low signal-to-noise ratios. If the answer is positive, then the project contributes not only to
motion-generation settings, but also to a broader class of structured conditional inference
problems in which reliable selection under weak guidance is central.

6. Expected Contribution
This project aims to contribute at three levels.
First, it makes a theoretical contribution by identifying posterior-organization failure as
an independent object of study in conditional flow generation. Rather than reducing
failure to path error or endpoint miss, it explains how family-level reachability can coexist
with fixed-family degradation, and why dominant ridges, width floors, and family peak
clusters become the relevant geometric objects.
Second, it makes a methodological contribution by proposing an organization-aware
correction framework for coarse guidance. The method does not rely only on endpoint
reranking. Instead, it monitors dominance, clarity, and inversion risk during generation
and uses these signals to repair guidance before deterioration fully propagates to the final
output.
Third, it makes a decision-level contribution by introducing a canonical selection principle
for flow-based conditional generation. This principle matters because it links posterior
geometry to the practical act of selecting one output, and therefore aligns distributional
structure with downstream decision reliability.
More broadly, the project aims to show that conditional generation should not be judged
only by whether it can produce plausible samples, but also by whether it can preserve a
stable dominant representative under weak or coarse guidance.
3
4

Tonghuashun iFinD official platform: https://ft.10jqka.com.cn/
CSMAR official database portal: https://data.csmar.com/


<!-- Page 14 -->

7. Research Program Timelines and Milestones
To keep the project feasible as a research assistantship agenda, the work plan is organized
into four linked stages.
7.1 Stage 1: theory consolidation
The first stage focuses on consolidating the theoretical narrative. The main tasks are
to align the proposal with the supplementary proof draft, sharpen the definitions of
dominance, clarity, and inversion risk, and tighten the connection between family-level
formation and within-family degradation. The expected output is a clean and coherent
theory section.
7.2 Stage 2: synthetic mechanism verification
The second stage builds the synthetic validation environment. The goal is to make the
full mechanism directly visible, including broad peaks, same-family side-mode emergence,
mass splitting, and clarity-driven repair. The expected output is a stable synthetic
implementation together with an initial figure set and interpretable mechanism diagnostics.
7.3 Stage 3: benchmark evaluation
The third stage integrates the correction framework into representative flow-based benchmarks. This stage includes the main benchmark runs, the core comparisons against
baseline systems, and a first round of diagnostic analyses on selection stability and organization quality. The expected output is an initial benchmark result set together with a
clearer picture of which gains come from correction rather than ranking alone.
7.4 Stage 4: transfer test and manuscript integration
The final stage extends the framework to the finance-related transfer setting and integrates
the results into a unified write-up. The expected output is a polished proposal-to-paper
transition: a refined theory section, a benchmark-oriented empirical section, and an
extension section testing low-SNR transferability.

References
Michael S. Albergo and Eric Vanden-Eijnden. Building normalizing flows with stochastic
interpolants. arXiv preprint arXiv:2209.15571, 2022.
Michael S. Albergo, Nicholas M. Boffi, and Eric Vanden-Eijnden. Stochastic interpolants:
A unifying framework for flows and diffusions. arXiv preprint arXiv:2303.08797, 2023.
Xinyu Dai, Donghao Yu, Shilei Zhang, and Zhibin Yang. Safe flow matching: Robot
motion planning with control barrier functions. arXiv preprint arXiv:2504.08661, 2025.
Erhard Gl”otzl and Oliver Richters. Helmholtz decomposition and rotation potentials in
n-dimensional cartesian coordinates. arXiv preprint arXiv:2012.13157, 2020.


<!-- Page 15 -->

Trevor Harris.
Flow-based conformal predictive distributions.
arXiv:2602.07633, 2026.

arXiv preprint

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le.
Flow matching for generative modeling. In International Conference on Learning
Representations, 2023.
Yaron Lipman, Marton Havasi, Peter Holderrieth, Neta Shaul, Matt Le, Brian Karrer,
Ricky T. Q. Chen, David Lopez-Paz, Heli Ben-Hamu, and Itai Gat. Flow matching
guide and code. arXiv preprint arXiv:2412.06264, 2024.
B. Liu, Y. Xie, et al. Alignment of diffusion models: Fundamentals, challenges, and future.
arXiv preprint arXiv:2409.07253, 2024.
Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to
generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003, 2022.
James Lucas, George Tucker, Roger Grosse, and Mohammad Norouzi. Understanding posterior collapse in generative latent variable models. In Deep Generative Models for Highly
Structured Data Workshop, International Conference on Learning Representations, 2019.
Manan Patel, Shaoxuan Wen, Dimitris N. Metaxas, and Yezhou Yang. Flowchef: Steering
of rectified flow models for controlled generations. In Proceedings of the IEEE/CVF
International Conference on Computer Vision, 2025.
Fred Zhangzhi Peng, Zachary Bezemek, Jarrid Rector-Brooks, Shuibai Zhang, Anru R.
Zhang, Michael Bronstein, Avishek Joey Bose, and Alexander Tong. Planner aware
path learning in diffusion language models training. arXiv preprint arXiv:2509.23405,
2025.
S. Saini, S. Gupta, and A. C. Bovik. Rectified-cfg++ for flow based models. In Advances
in Neural Information Processing Systems, 2025.
Filippo Santambrogio. Optimal Transport for Applied Mathematicians: Calculus of
Variations, PDEs, and Modeling. Birkhäuser, 2015.
Emre Turan, Alexandros Siozopoulos, Laura Martinez, Jean Gaubil, Emma Pierson,
and Maks Ovsjanikov. Unfolding generative flows with koopman operators: Fast and
interpretable sampling. arXiv preprint arXiv:2506.22304, 2025.
Cédric Villani. Optimal Transport: Old and New. Springer, 2009.
H. Wang, Y. Liu, J. Chi, F. Liu, R. Xue, and Y. Duan. Cfg-ctrl: Control-based classifierfree diffusion guidance. arXiv preprint arXiv:2603.03281, 2026.
Yixin Wang, David M. Blei, and John P. Cunningham. Posterior collapse and latent
variable non-identifiability. In Advances in Neural Information Processing Systems,
2021.


<!-- Page 16 -->

G. Wei and L. Ma. Stream-level flow matching with gaussian processes. In Proceedings of
the 42nd International Conference on Machine Learning, 2025.
Menghan Xia, Nan Xue, Yujun Shen, Ran Yi, Tingting Gong, and Yu-Jie Liu. Rectified diffusion guidance for conditional generation. In Proceedings of the IEEE/CVF
Conference on Computer Vision and Pattern Recognition, 2025.
Zebin Xing, Xingyu Zhang, Yang Hu, Bo Jiang, Tong He, Qian Zhang, Xiaoxiao Long, and
Wei Yin. Goalflow: Goal-driven flow matching for multimodal trajectories generation
in end-to-end autonomous driving. arXiv preprint arXiv:2503.05689, 2025.
Qi Yan, Brian Zhang, Yutong Zhang, Daniel Yang, Joshua White, Di Chen, Jiachao Liu,
Langechuan Liu, Binnan Zhuang, Shaoshuai Shi, and Renjie Liao. Trajflow: Multi-modal
motion prediction via flow matching. arXiv preprint arXiv:2506.08541, 2025.
