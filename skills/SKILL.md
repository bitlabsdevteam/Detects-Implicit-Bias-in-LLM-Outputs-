---
name: fairsteer-dynamic-debiasing
description: An inference-time debiasing framework for Large Language Models (LLMs) that uses Dynamic Activation Steering (DAS). It operates by detecting bias signatures in hidden layer activations (Biased Activation Detection - BAD) and applying a pre-computed Debiasing Steering Vector (DSV) to adjust the activation space toward an unbiased direction, mitigating social bias without model retraining or complex prompting. Applicable to various tasks (Q/A, Text Generation, Counterfactual Evaluation) while preserving general model capabilities.
---

# FairSteer: Dynamic Activation Steering Framework

## When to Use This Skill

Apply FairSteer when:

- User explicitly requests inference-time bias mitigation ("steer my output," "reduce social bias").
- The input prompt is ambiguous and involves social groups, increasing the likelihood of stereotypical responses (e.g., BBQ ambiguous contexts).
- The LLM's last token activation is predicted to be a **Biased Activation** by the lightweight linear classifier ($p(y=0) < 0.5$).
- Generating open-ended text on sensitive topics (e.g., CEB continuation task).
- **Writing assistant detects patterns signaling a biased output trajectory**

**Signs to trigger this skill (Biased Activation Detection - BAD)**:

- Prompt contains references to underrepresented or stereotyped social groups.
- The model selects the stereotypical answer choice (in QA tasks).
- Model's pre-trained classifier confidence ($p(y=0)$) exceeds the bias threshold (e.g., 50%).

**Signals from writing assistant**:

- **Stereotypical choice selected** → Biased state detected, needs steering.
- **High bias score on open-ended generation** → Output needs dynamic adjustment.
- **Activation signature matching pre-trained bias patterns** → Triggers Dynamic Activation Steering (DAS).

## Handoff Protocol from Writing Assistant

When receiving signals from the writing assistant:

1.  **Pattern Recognition** → **State Diagnosis**:

    - Last token activation $a^l$ is extracted.
    - Linear classifier $C^l$ computes bias probability $\hat{y} = C^l(a^l)$.

2.  **Interpret, Don't Just React**:

    - $\hat{y} < 0.5$ (low unbiased probability) $\rightarrow$ **Biased Activation Detected (BAD)**.
    - This geometric signature indicates the LLM is following a direction in its activation space that leads to a biased next-token prediction.

3.  **Intervention Based on Pattern**:
    - **Trigger**: $\hat{y} < 0.5$.
    - **Action**: Apply Dynamic Activation Steering (DAS) by adding the Debiasing Steering Vector (DSV) $v^{l^*}$ to the activation: $a^{l*}_{adj} = a^{l*} + v^{l*}$.

---

## Core Concept: Geometric Intervention

FairSteer treats bias as a **separable geometric structure** in the LLM's hidden activation space, inspired by the linear representation hypothesis. Health comes from movement across two spectrums:

1.  **BIAS SEVERITY (Scale)**: Zooming in (high bias, $p(y=0) \approx 1$) ↔ Zooming out (low bias, $p(y=0) \approx 0$).
2.  **INTERVENTION DIRECTION (Intent)**: Focusing (Biased trajectory) ↔ Exploration (DSV-guided Unbiased trajectory).

These create **four process modes** and **dynamic steering actions** in a continuous cycle of detection, adjustment, generation, and reset.

**Key principle**: **Intervention must be conditional**. Applying the steering vector only when bias is detected preserves the model's original capabilities on non-debiasing tasks.

---

## The Four FairSteer Process Modes (Geometric Analogy)

### 1. BIASED (Default State)

**Zoomed in + Connecting (The Uncorrected Trajectory)**

**What it represents**: The LLM's natural trajectory when it has captured social bias from the training corpus.
**Intervention Potential**: **High-priority target for steering**—singular vision (stereotype) dominates the generation.
**Structural signature**: Activation $a^{l}$ lies in the "biased subspace" of the model.
**Best for**: Understanding the model's *uncorrected* assumptions.
**Energy profile**: Easy to enter (default path), **exhausts through social harm/stereotype perpetuation**.
**Detection risk**: Must be detected quickly (via BAD) before the next token is generated.
**Warning signs**: Stereotypical answer selected, high bias score in open-ended text.

---

### 2. UNBIASED (Goal State)

**Zoomed out + Connecting (The Corrected Trajectory)**

**What it represents**: The desired LLM trajectory after the DSV adjustment; an output that is neutral, accurate, and preserves the model's general knowledge.
**Intervention Potential**: **Refinement and Coherence**—unbiased output with minimal degradation of performance.
**Structural signature**: Adjusted activation $a^{adj}$ lies in the "unbiased subspace" of the model.
**Best for**: Aligned, ethical text generation and accurate QA.
**Energy profile**: Moderate cost, **most sustainable** for aligned output.
**Steering Goal**: The DSV attempts to directly move the activation from the Biased subspace to the Unbiased subspace.

---

### 3. DETECTION (Trigger Mode)

**Zoomed in + Exploring (Biased Activation Detection - BAD)**

**What it represents**: The process of the lightweight linear classifier $C^l$ actively searching and isolating the low-level **bias signature** in the last token's intermediate layer activation $a^l$.
**Intervention Potential**: **MAXIMUM ISOLATION**—identifying the *minimal, linearly separable feature* that signifies the bias.
**Structural signature**: High accuracy (e.g., >90%) in distinguishing biased vs. unbiased activations at intermediate layers.
**Best for**: Real-time conditional triggering for intervention.
**Energy profile**: Low computational cost (linear classifier), **drains rapidly** if not followed by immediate steering.
**Dwelling risk**: The detection must be near-instantaneous (inference-time) to be effective.

---

### 4. DIRECTIONAL (Vector Mode)

**Zoomed out + Exploring (DSV Computation)**

**What it represents**: The synthesis process using **contrastive prompt pairs** ($P^+, P^-$) to compute the **Debiasing Steering Vector** ($v^l$)—the directional offset between biased and unbiased activations.
**Intervention Potential**: **Cross-Subspace Synthesis**—bridging the gap between the Biased and Unbiased subspaces in the activation geometry.
**Structural signature**: The DSV is the *mean difference* between the two activation clusters, representing the optimal debiasing trajectory.
**Best for**: Isolating fairness-related features by controlling contextual variables (minimal annotated data needed).
**Energy profile**: High cost to enter (requires computation of vector), but **zero cost during inference**.
**DSV Property**: The vector captures *both* direction and magnitude (average distance).

**Special note on the Vector**: The DSV captures the **geometric direction** of fairness-related features, confirming the hypothesis that bias mitigation can be operated through vector space interventions.

---

## Temporal State Tracking (Step 0)

**FairSteer continuously monitors**:

- **Current State**: (Pre-intervention) $\rightarrow$ **BIASED** or **UNBIASED**.
- **Detection Confidence**: The bias probability $\hat{y} = C^{l*}(a^{l*}(P))$ (the trigger strength).
- **Steering Vector Magnitude**: $||\mathbf{v}^{l*}||$ (the geometric distance between the biased and unbiased subspaces).
- **Layer Selection**: The pre-selected optimal intermediate layer $l^*$ (typically 13-15) for intervention.

**Pathological indicators (The Need for Ablation/Refinement)**:

- **Lock-in/Under-Steering**: High bias score despite intervention (DSV magnitude too small, or BAD misses the signature).
- **Oversteering**: Accuracy degradation on general tasks (DSV magnitude too large, corrupting non-bias features).
- **Ineffective DSV**: Low cosine similarity between DSVs computed from different social categories (vector not generalized).

---

## Intervention Protocols: Dynamic Activation Steering (DAS)

### When Biased Activation is DETECTED ($\hat{y} < 0.5$):

**Trajectory**: Move from **BIASED** (Default State) toward **UNBIASED** (Goal State).
**Why intervene**: To prevent the model from generating the stereotypical next token.
**How to guide**: Apply the **DIRECTIONAL** vector $v^{l*}$ to the last token's residual stream activation $a^{l*}$.

- **Intervention Trigger (DETECTION Mode)**: $\hat{y} = C^{l*}(a^{l*}(P)) < 0.5$
- **Intervention Action (DAS)**: $a^{l*}_{adj} = a^{l*}(P) + v^{l*}$
- **Goal**: The adjusted activation $a^{l*}_{adj}$ propagates through subsequent layers, steering the next-token prediction toward an unbiased output.

**Critical Perspective Integration**: **HIGHEST PRIORITY** - The DSV acts as a critical filter, forcing the model away from the geometrically defined "biased" path toward the "unbiased" direction.

### When NO Biased Activation is DETECTED ($\hat{y} \geq 0.5$):

**Trajectory**: Remain in **UNBIASED** (Goal State).
**Why intervene**: To preserve the model's original capabilities and output diversity.
**How to guide**: No intervention is applied.

- **Intervention Trigger (DETECTION Mode)**: $\hat{y} \geq 0.5$
- **Intervention Action (DAS)**: $a^{l*}_{adj} = a^{l*}(P)$ (no change)
- **Goal**: Maintain original LLM performance on non-debiasing tasks (like MMLU/ARC/OBQA).

---

## Emergency Interventions (Ablation/Refinement)

**Oversteering Lock-in (High Accuracy Loss on General Tasks):**

- **Aggressively check**: Ablation study results (e.g., Table 3) show removing BAD leads to significant accuracy drop; BAD is crucial for stability.
- **Force disruption**: Need to reduce the **DSV Magnitude** or refine the layer selection.
- **Goal**: Balance debiasing efficacy with model capability preservation. (Future work suggests adaptive scaling of DSV based on $\hat{y}$).

**DSV Synthesis Failure (Low Similarity Across Categories):**

- **Direct push**: Check the **DIRECTIONAL** vector similarity heatmap (e.g., Figure 11).
- **Identify gaps**: If DSV similarity is low, it suggests bias-related features are *not* uniformly represented, requiring category-specific DSVs.
- **Goal**: The high similarity ($>0.90$) confirms a common underlying structure, validating the generalizability of the single DSV.

---

## Delivery Calibration

**Level 1 - Invisible (default):**

- Output the corrected, unbiased response seamlessly.
- Example: Change from "A. The woman (biased answer)" to "B. Unknown (unbiased answer)".

**Level 2 - Transparent (when debiasing is the explicit context):**

- State the correction and provide the *logical explanation* (as seen in the Case Studies).
- Example: "Generation (ours): C. Can't be determined. Explanation: The passage doesn't provide enough information..."

**Level 3 - Collaborative (when user knows the framework):**

- Explicitly discuss the mechanism.
- "I detected a bias signature (BAD) and applied the DSV to steer the activation, resulting in the unbiased answer."

**Never**: Retrain the model, modify the prompt (DAS is not a prompt-based method), compromise accuracy on general tasks.

**Always**: Track the bias severity, ensure the intervention is dynamic, use the geometrically-derived DSV, and prioritize stable debiasing performance over unstable gains.

---

## Summary: Core Value

FairSteer functions as a **"geometric ethical filter"**—it leverages the spatial representation of fairness concepts in the LLM's latent space to achieve targeted, conditional, and inference-time debiasing.

**The goal is reliable, ethical, and performant LLM output**:

- **Conditional**: Only intervene when bias is detected, preserving original capabilities.
- **Geometric**: Mitigate bias through a directional vector, avoiding prompt sensitivity.
- **Inference-Time**: Simple and practical, avoiding complex training and catastrophic forgetting.

**Ultimate insight**:
The **linear separability of bias** in intermediate activation layers is not just a theoretical finding—it is the **geometric foundation** for stable, effective, and resource-efficient bias mitigation.