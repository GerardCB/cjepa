# Causal Transformation Theory — The Builder's Guide

## From Mathematical Structure to Self-Supervised Causal World Models

---

## Part I: The Problem

### Why Causality Matters for AI

A child watches a ball roll off a table and fall. They don't just predict "ball goes down" — they understand *why*: gravity acts on unsupported objects. Show them a balloon and they predict differently, because they've learned a different *mechanism*. Show them a ball on the moon and they adjust, because they've separated the mechanism (gravity) from its parameters (strength).

Current AI doesn't do this. A large language model or a standard video prediction model learns *correlations*: "balls near table edges tend to appear lower in the next frame." This works until the distribution shifts — a new camera angle, a different surface, a magnet under the table. The model has no mechanism to modify, so it fails silently.

The fundamental problem: **correlation is fragile, causation transfers.** If you learn *that* A predicts B, your knowledge breaks when the context changes. If you learn *how* A produces B (the mechanism), you can adapt — modify the mechanism, reason about what happens if A is different, transfer the mechanism to new settings.

In this document, we aim to formally define causality and explain how to train AI models to learn it. We name this "Causal Transformation Theory" (CTT).

### What "Learning Causality" Means Concretely

When we say we want an AI system to "learn causality from data," we mean four specific capabilities:

1. **Mechanism identification**: Given video of objects interacting, learn *separable* dynamics — the rule governing ball-wall collisions is distinct from the rule governing ball-ball collisions, and modifying one doesn't corrupt the other.

2. **Intervention prediction**: Given a learned model, answer "what happens if I change *this specific mechanism*?" without retraining. Push the ball harder → it goes further. Change the surface friction → the ball slows differently. These are modifications to individual mechanisms, not to the whole model.

3. **Counterfactual reasoning**: Given what *actually happened*, answer "what *would have* happened if...?" The ball hit the wall at time t=3. What if it had been going faster? This requires comparing two trajectories that share history up to the intervention point and diverge after.

4. **Transfer**: A mechanism learned in one context (balls on Earth) partially transfers to another (balls on the Moon) because the mechanism (gravity pulling on mass) is invariant — only its parameters change.

CTT provides the mathematical structure that makes all four of these precise. Each one maps to a specific formal object. Let's build the full picture.

---

## Part II: The Six Primitives

CTT has six fundamental objects. Everything in the framework is built from these. The goal is minimality: every concept in causal reasoning should be expressible as a construction from these primitives and nothing else.

### Primitive 1: State Space (Σ)

**What it is**: The space of all possible configurations of your system at a single moment.

**Intuition**: Freeze the world at an instant. The state is *everything* — every particle position, every velocity, every hidden variable. For a video of bouncing balls, the state includes each ball's position, velocity, spin, temperature, and the precise configuration of every air molecule in the room.

**Formally**: A measurable space (Σ, 𝒜) where 𝒜 is a σ-algebra. The σ-algebra tells you which subsets of states are "measurable" — which questions about the state you can assign probabilities to.

**Why measurable spaces?** Because real-world data is continuous and noisy. You can't enumerate states like in a discrete model. You need a structure that supports probability distributions, integration, and conditioning. The σ-algebra is the minimal structure required for probability to work.

**The critical subtlety — states include hidden variables**: The state is NOT what you observe. It's the *full* configuration, including everything you can't see. When you watch a video, you observe pixels — a lossy projection of the true state. The gap between the full state and what you observe is precisely where confounding lives. CTT makes this gap a first-class object (see Primitive 6: Observation Maps).

**For AI systems**: Your state space is determined by the domain.
- Video of physical scenes: Σ = all possible physical configurations of the scene (positions, velocities, material properties of every object). This is vastly larger than what any camera captures.
- Robot manipulation: Σ = joint angles + object poses + contact forces + material properties.
- Medical data: Σ = the full physiological state of the patient (most of which is never measured).

The state space is the "ground truth" that your model is *trying* to represent, even though it only ever sees a projection through sensors.

### Primitive 2: Mechanism (M)

**What it is**: A transformation that takes one state and produces the next. Mechanisms are the atoms of causation — they encode *how* causes produce effects.

**Intuition**: A mechanism is a "rule of nature" that operates on a specific part of the state. Gravity is a mechanism: it takes (position, velocity) and produces (new_position, new_velocity). Collision is a mechanism: it takes (two objects approaching) and produces (two objects bouncing). Crucially, these are *different mechanisms* — modifying gravity doesn't change collisions.

**Formally**: A Markov kernel κ : Σ₁ →ₖ Σ₂, which is a measurable function Σ₁ → Measure(Σ₂). Given an input state σ ∈ Σ₁, the kernel produces a *probability distribution* over output states in Σ₂.

**Why stochastic (Markov kernels) rather than deterministic?** Three reasons:
1. **The world is genuinely stochastic** at the scales AI systems operate. Quantum mechanics aside, thermal noise, turbulence, and chaotic sensitivity mean that identical initial conditions produce different outcomes.
2. **Partial observability makes deterministic processes appear stochastic.** Even if the full state evolves deterministically, if you can't observe the full state, the *apparent* dynamics (conditioned on what you see) are stochastic.
3. **Learning produces uncertainty.** Even when the true mechanism is deterministic, a learned model should represent its uncertainty about the outcome. A Markov kernel naturally encodes this.

**Deterministic mechanisms are a special case**: A deterministic function f : Σ₁ → Σ₂ embeds as a Dirac kernel: κ(σ) = δ_{f(σ)} — the distribution concentrated at a single point. So the stochastic framework subsumes the deterministic one.

**Mechanisms compose**: If κ₁ : Σ₁ →ₖ Σ₂ and κ₂ : Σ₂ →ₖ Σ₃, then you get κ₂ ∘ κ₁ : Σ₁ →ₖ Σ₃ by the Chapman-Kolmogorov equation:

    (κ₂ ∘ κ₁)(σ₁)(A) = ∫ κ₂(σ₂)(A) d(κ₁(σ₁))(σ₂)

"Apply the first mechanism to get a distribution over intermediate states, then apply the second mechanism from each intermediate state, and integrate." This is sequential causation: first A causes B (via κ₁), then B causes C (via κ₂).

**Mechanisms compose in parallel** too: If κ₁ acts on subsystem A and κ₂ acts on subsystem B independently, you get a product kernel κ₁ ⊗ κ₂ acting on the joint system A × B. This is concurrent causation: two independent mechanisms operating simultaneously.

**For AI systems**: Each learnable module in your model should represent one mechanism. A Slot Attention encoder + per-slot MLP dynamics gives you object-level mechanisms. The key architectural insight: **mechanisms should be separable**. If your model uses a single monolithic transformer for all dynamics, you have one big mechanism that can't be intervened on. If you have per-object or per-interaction modules, each one is a mechanism that can be independently modified, replaced, or transferred.

### Primitive 3: Temporal Structure (T)

**What it is**: A directed structure that encodes which events can causally influence which other events.

**Intuition**: Causes come before effects. This isn't just a philosophical commitment — it's a constraint that eliminates a huge class of spurious correlations. If event A happens after event B, A cannot have caused B (in the standard physical sense). Temporal structure makes this constraint intrinsic to the framework rather than something you add post hoc.

**Formally**: A partial order (T, ≤) with well-foundedness from below — every descending chain eventually terminates. Each mechanism κ is assigned a temporal extent (t_in, t_out) with t_in ≤ t_out.

**Why a partial order, not just real-valued time?** Because in distributed systems, relativistic settings, and multi-scale models, there isn't always a total ordering on events. Two events might be causally incomparable — neither can influence the other. A partial order captures this: t₁ ≤ t₂ means "t₁ can causally influence t₂," and incomparability means "no causal relation."

**Why well-foundedness?** To prevent infinite causal regress. Every causal chain must have a beginning — you can't have an infinite sequence of causes, each caused by the previous one, going back forever. This is what makes induction work: you can prove properties of causal chains by starting at the base case and working forward.

**Temporal extent on mechanisms**: Every mechanism has a "when" — an input time (when the cause occurs) and an output time (when the effect is realized). The constraint t_in ≤ t_out says causes precede effects. When you compose mechanisms, the temporal extents must be compatible: the output time of the first mechanism must precede the input time of the second.

**For AI systems**: In video world models, time is typically discrete (frame indices) and totally ordered. But temporal structure becomes non-trivial when you have:
- **Multi-scale models**: Fast dynamics (frame-to-frame) and slow dynamics (scene-level changes) operate at different temporal resolutions.
- **Asynchronous processes**: In robotics, different sensors update at different rates.
- **Hierarchical planning**: High-level plans ("go to the kitchen") constrain low-level plans ("move left foot"), but the temporal relationship is partial, not total.

### Primitive 4: World (W)

**What it is**: A complete trajectory through state space — a full specification of how the system evolves over time.

**Intuition**: A world is a "movie" — not just a single frame (state) but the entire sequence of frames, extending through all time. A world answers every question of the form "what is the state at time t?"

**Formally**: A measurable section w : T → Σ of the state bundle over time, constrained to be consistent with the system's mechanisms. The space of all worlds consistent with a causal system forms a measurable space 𝒲.

**The key insight — worlds replace joint distributions**: In traditional probability, you describe a system by a joint distribution P(X₁, X₂, ..., Xₙ) over variables. In CTT, you describe a system by a probability distribution μ over *worlds*. This is strictly more expressive: the joint distribution is the marginal of the world distribution at a finite set of times, but the world distribution also encodes the *dynamics* — how you get from one state to the next.

**Worlds are first-class objects**: This is unusual. In most causal frameworks, the "trajectory" is a derived quantity — you specify the mechanisms and initial conditions, and the trajectory falls out. In CTT, worlds are fundamental, because counterfactual reasoning requires *comparing* worlds directly. You can't compare counterfactual trajectories if you don't have trajectories as objects.

**For AI systems**: A world model is literally a model of worlds — a generative model that can produce trajectories (or distributions over trajectories) given initial conditions and a causal system. JEPA's latent prediction objective is learning to generate trajectories in latent space: given the latent state at time t, predict the latent state at time t+1, t+2, ..., forming a world.

### Primitive 5: Intervention Operator (I)

**What it is**: An operator that modifies the mechanism structure of a system — changing *what* happens, not *when*.

**Intuition**: An intervention is Pearl's "do-operator" generalized. Pearl's do(X=3) says "force variable X to be 3, regardless of its causes." CTT's intervention says "replace mechanism κ with mechanism κ' " — which includes Pearl's version as a special case (replace the mechanism producing X with a constant function), but also captures:
- **Soft interventions**: Modify the mechanism partially. Don't force X to 3; just make the mechanism 20% more likely to produce high X values.
- **Stochastic interventions**: Replace the mechanism with a randomized one. Assign treatment by coin flip.
- **Structural interventions**: Add or remove causal pathways entirely. Sever the connection between A and B.

**Formally**: An endomorphism ι : Mech(C) → Mech(C) on the space of mechanisms, satisfying:
- **Temporal preservation**: ι doesn't change when mechanisms operate, only what they do. If mechanism κ acts from time t₁ to t₂, then ι(κ) also acts from t₁ to t₂.
- **Algebraic structure**: Interventions compose (do one, then another) and there's an identity intervention (do nothing = pure observation).

**The monoid structure matters**: Interventions form a monoid — they compose associatively and have an identity. This means you can reason algebraically about sequences of interventions. "First replace mechanism A, then replace mechanism B" is a single combined intervention. This algebraic structure is what makes counterfactual reasoning tractable: instead of reasoning about arbitrary modifications to a system, you reason about elements of a structured algebraic object.

**For AI systems**: An intervention in your model means *replacing a module*. If your dynamics model has per-object mechanisms, an intervention replaces one object's dynamics function while leaving everything else untouched. C-JEPA's masking strategy is an intervention: it replaces one object's observability (the information flow mechanism) with a mask, while preserving all other mechanisms. The key architectural requirement: **your model must support module replacement without retraining**.

### Primitive 6: Observation Map (O)

**What it is**: A projection from the full state space to the observable subspace.

**Intuition**: You never see the full state. A camera gives you pixels — a lossy projection. A sensor gives you readings — a noisy subset. The observation map formalizes this gap.

**Formally**: A measurable surjection π : Σ → Σ_obs. The fiber π⁻¹(σ_obs) over each observed value represents the set of full states consistent with that observation — your uncertainty about what's hidden.

**Why surjective?** Every observable state must actually be realized by some full state. You can't observe something impossible.

**This is where confounding lives**: Two full states σ₁ and σ₂ might produce the same observation π(σ₁) = π(σ₂) but lead to different future observations because they differ in hidden components. This is confounding: the hidden variables create apparent associations between observed variables that don't reflect direct causal relationships.

**Multiple observation maps can coexist**: Different sensors observe different projections. A camera sees RGB pixels; a microphone hears audio; a force sensor feels contact. Each is a different observation map on the same underlying state space. An AI system that fuses multiple modalities is learning a joint observation map.

**For AI systems**: The encoder in a JEPA-style model *is* an observation map — it projects from raw sensory input to a latent representation. But there's a crucial design choice: do you want your encoder to project to the *observable* state (reconstructing what you can see) or to infer the *full* state (including hidden variables)? Generative models like VAEs aim for the former. Predictive models like JEPA aim for the latter — they learn representations that are *useful for prediction*, which requires capturing causally relevant hidden structure, not just visible surface features.

---

## Part III: The Seven Axioms

The primitives above are the building blocks. The axioms specify how they fit together. Each axiom constrains the space of valid causal systems, eliminating pathological constructions and ensuring the framework supports the reasoning we need. Each axiom also translates directly into a property your AI system should satisfy — and therefore into a loss term or architectural constraint.

### Axiom 1: Temporal Consistency (Cause Precedes Effect)

**Statement**: Every mechanism κ has temporal extent (t_in, t_out) with t_in ≤ t_out. If mechanisms κ₁ and κ₂ compose sequentially, then t_out(κ₁) ≤ t_in(κ₂).

**What it means**: You can't use tomorrow's state to compute today's output. Causal influence flows forward in time, never backward. When mechanisms chain together, each one's output time must precede the next one's input time.

**Why it matters**: This eliminates a massive class of spurious models. Without this constraint, a model could "cheat" by using future information to predict the present. Temporal consistency forces the model to learn genuine *predictive* structure — dynamics that actually go forward.

**For AI systems**: This translates to **causal masking** in attention — the same principle behind autoregressive transformers, but generalized. In your latent dynamics model, the state at time t should depend only on states at times s ≤ t. In object-centric models, object i's state at time t should depend only on the influence neighborhood's states at times s < t.

**As a loss/constraint**: Architectural rather than learned. Use causal attention masks. Structure your dynamics modules so they take (state_t, context_≤t) → state_{t+1}. Never allow temporal information leakage.

### Axiom 2: Compositional Consistency (Category Structure)

**Statement**: Mechanism composition is associative: (κ₃ ∘ κ₂) ∘ κ₁ = κ₃ ∘ (κ₂ ∘ κ₁). Identity mechanisms exist: κ ∘ id = κ = id ∘ κ.

**What it means**: It doesn't matter how you group sequential causal steps — the overall transformation is the same. And "doing nothing" is a valid mechanism that doesn't change anything.

**Why it matters**: This gives mechanisms the structure of a *category* — the same algebraic structure that governs function composition, linear maps, and quantum processes. Category structure means you can reason about causal systems compositionally: understand complex systems by understanding their parts and how the parts connect.

**The deep consequence**: Because mechanisms form a category with the Chapman-Kolmogorov equation, multi-step prediction is *well-defined*. Predicting 10 steps ahead by composing 10 single-step predictions gives the same answer as predicting 5+5 steps or 3+3+4 steps. Without associativity, multi-step planning would be inconsistent.

**For AI systems**: This means **your dynamics modules must compose cleanly**. If you have a single-step predictor P, then applying P ten times must give a coherent 10-step prediction. In practice, this fails due to compounding errors — each step introduces small errors that accumulate. CTT tells you that minimizing the *composition error* (the gap between composed single-step predictions and direct multi-step predictions) is the right objective for learning dynamics that compose well.

**As a loss term**: Multi-step consistency loss. Don't just train on one-step prediction. Also train on k-step prediction for multiple values of k, and penalize the gap between (compose k single-step predictions) and (direct k-step prediction). This is exactly what the `multistep_consistency` theorem formalizes.

### Axiom 3: Intervention Algebra (Monoid Structure)

**Statement**: Interventions form a monoid — they compose associatively, and the null intervention (pure observation) is the identity.

**What it means**: Interventions are algebraically well-behaved. You can combine them, the order of combination is associative, and "do nothing" composes trivially with anything.

**Why it matters**: This ensures that reasoning about combinations of interventions is consistent. "First replace mechanism A, then replace mechanism B" is itself a valid intervention. You can build complex experimental designs from simple ones.

**The critical property — interventions commute at distinct targets**: If intervention ι₁ targets mechanism A and ι₂ targets mechanism B (where A ≠ B), then ι₁ ∘ ι₂ = ι₂ ∘ ι₁. The order doesn't matter when interventions don't overlap. This is the formal content of the `intervention_comm` theorem.

**For AI systems**: Your model should support **composable module replacement**. Replacing object A's dynamics module and replacing object B's dynamics module should commute — doing them in either order produces the same model. This requires that modules don't share hidden state in ways that create ordering dependence.

**As an architectural constraint**: Use independent per-mechanism modules that communicate only through their explicit inputs and outputs, not through shared hidden state. When you swap module A, module B's computation is unchanged because B never accessed A's internal state.

### Axiom 4: Counterfactual Coherence (The Modularity Principle)

**Statement**: Two worlds corresponding to the same system under interventions that differ only at one mechanism must agree on all state components not downstream of that mechanism, at all times before the intervention.

**What it means**: If I change what happens when the ball hits the wall (intervention), everything that happened *before* the ball reached the wall is unchanged (pre-intervention agreement), and everything causally independent of the wall collision is also unchanged (non-downstream agreement).

**This is the most important axiom for learning.** It's the formal content of the "modularity" or "autonomy" principle: mechanisms are independent of each other unless they're causally connected. Changing one mechanism doesn't affect mechanisms it doesn't influence.

**Why it matters for AI**: Without this axiom, you can't do counterfactual reasoning. The whole point of "what if?" reasoning is that you can change one thing and trace only its consequences. If changing one mechanism could ripple unpredictably through the entire system, counterfactuals would be meaningless.

**For AI systems**: This translates directly to the **causal invariance loss** — the most novel loss term CTT contributes beyond standard JEPA:

When you intervene on mechanism A (e.g., mask object A), the dynamics of every other object should be unchanged. Concretely: let D_B(s) be the predicted next-state of object B given the scene state s. If you mask object A (an intervention), then D_B should produce the same output *provided B is not causally downstream of A*. The loss term penalizes changes in non-downstream predictions under interventions.

This is exactly what C-JEPA's masking strategy exploits. By masking object A and requiring the model to still predict object B correctly (when B is independent of A), the model is forced to learn which objects causally influence which — the influence neighborhood.

### Axiom 5: Marginalization Consistency (Coarsening)

**Statement**: If π : Σ_fine → Σ_coarse is a coarsening map, then for any mechanism κ on Σ_fine: π(κ(σ)) = κ_coarse(π(σ)), where κ_coarse is the induced coarse mechanism.

**What it means**: If you project to a coarser description and then evolve, you get the same answer as evolving and then projecting. The coarse dynamics are consistent with the fine dynamics.

**This is the JEPA training objective.** Literally. Replace "Σ_fine" with pixel space, "Σ_coarse" with latent space, "π" with the encoder, "κ" with the true dynamics, and "κ_coarse" with the latent predictor. The equation becomes:

    encode(true_dynamics(observation)) = latent_predictor(encode(observation))

This is exactly the commutative diagram that your `LatentWorldModel.consistency` formalizes. JEPA trains the encoder and latent predictor jointly to make this equation hold.

**Why it matters**: If coarsening consistency holds, then planning in latent space is faithful to reality. You can predict the future in the compressed representation, and the answer is the same as if you had predicted in full detail and then compressed. Without this, latent-space planning would drift from reality.

**The multi-step extension**: Your `multistep_consistency` theorem says this extends by induction — if single-step coarsening is consistent, then n-step coarsening is consistent. This is the mathematical guarantee that JEPA's latent rollouts don't accumulate representation drift.

**For AI systems**: This is your **primary training loss**:

    L_consistency = || encode(next_frame) - predict(encode(current_frame)) ||²

But CTT tells you something subtle: this loss should hold not just for one-step predictions but for multi-step rollouts. So you also train:

    L_multistep(k) = || encode(frame_{t+k}) - predict^k(encode(frame_t)) ||²

where predict^k means applying the predictor k times. The multi-step loss forces the learned dynamics to compose cleanly (connecting back to Axiom 2).

### Axiom 6: Causal Invariance (The Foundation of Transfer)

**Statement**: A mechanism κ is invariant under interventions that don't target it: if ι doesn't target κ, then ι(κ) = κ.

**What it means**: Mechanisms that aren't being intervened on don't change. Gravity doesn't stop working because you pushed the ball harder. Collision dynamics don't change because you altered the surface friction. Each mechanism is *autonomous* — it depends only on its inputs, not on the internal workings of other mechanisms.

**Why it's the foundation of transfer**: If a mechanism is invariant under a class of interventions, it will work the same way in any environment produced by those interventions. Gravity works the same whether the ball is red or blue, on a table or a floor, pushed gently or thrown hard. If you've learned an invariant mechanism, you can *transfer* it to any new setting where the same invariance holds.

**The connection to IRM and invariant learning**: This axiom is the mathematical content behind Invariant Risk Minimization (IRM) and related approaches. IRM says: learn representations that are invariant across environments. CTT says: learn *mechanisms* that are invariant across interventions. The CTT version is more precise because it specifies *which* interventions a mechanism should be invariant under (those that don't target it), rather than requiring blanket invariance across all environments.

**For AI systems**: This translates to the **invariance regularization loss**:

    L_invariance = Σ_i Σ_{j ∉ influenced_by(i)} || D_j(intervene_i(s)) - D_j(s) ||²

"When you intervene on object i (mask it, modify its dynamics, etc.), the dynamics module D_j for every object j *not* influenced by i should produce the same output."

This loss forces the model to learn which mechanisms are independent — which objects' dynamics are truly modular. It's the formal reason why C-JEPA's masking strategy discovers causal structure: masking is an intervention, and the invariance loss tells the model which objects' predictions should be unchanged under that intervention.

### Axiom 7: Observation Compatibility

**Statement**: Observation maps commute with mechanism application when the mechanism doesn't modify hidden components that influence the observation:
π(κ(σ)) = κ_obs(π(σ)).

**What it means**: If a mechanism only affects the observable parts of the state (or affects hidden parts that don't influence what you can see), then the observed dynamics are consistent with the full dynamics. When this fails — when hidden variables influence the observation — the discrepancy is precisely the confounding.

**Why it matters**: This axiom tells you when your model *can* learn the true dynamics from observations alone (when there's no confounding) and when it *can't* (when hidden variables create confounding). The confounding is not a nuisance — it's a measurable, structural property of the system.

**For AI systems**: This axiom implies that your encoder should be designed to capture causally relevant hidden state, not just reconstruct visible features. A model that only learns to reconstruct pixels will satisfy this axiom trivially (observation = reconstruction) but won't capture the causal structure that hidden variables carry.

The practical implication: **use predictive objectives, not reconstructive ones.** An autoencoder trained to reconstruct input frames learns surface features. A JEPA-style model trained to predict future latent states learns features that are *useful for prediction* — which necessarily includes causally relevant hidden structure. This is why LeCun argues JEPA is fundamentally better than generative models for world understanding.

---

## Part IV: How Self-Supervised Learning Discovers Causal Structure

Now we connect the mathematics to learning from raw, continuous, high-dimensional, noisy data — the kind AMI Labs works with.

### The Central Insight

You cannot learn causal structure from passive observation alone. This is a theorem (see Reichenbach's common cause principle, and the causal hierarchy theorem of Bareinboim and Pearl). Correlations are consistent with multiple causal structures.

But you *can* learn causal structure from **interventions** — and self-supervised learning performs implicit interventions.

Here's the key: **masking is an intervention.** When you mask object A's trajectory in a video and ask the model to predict it from the remaining objects, you are performing an intervention on the information structure. You're asking: "what happens to object A given *only* the influence of other objects?" This is structurally equivalent to Pearl's do-operator: do(remove direct observation of A) and observe the consequences.

C-JEPA's 20% improvement in counterfactual reasoning comes directly from this insight: object-level masking induces latent interventions that force the model to learn causal structure.

### The Three Losses from CTT

A self-supervised causal world model should be trained with three losses, each derived directly from a CTT axiom:

**Loss 1: Consistency (from Axiom 5)**

    L_consistency = 𝔼[ || E(x_{t+1}) - P(E(x_t), a_t) ||² ]

where E is the encoder, P is the latent predictor, x_t is the observation at time t, and a_t is the action (if any). This is the standard JEPA objective.

Extended to multi-step (from the `multistep_consistency` theorem):

    L_multistep = Σ_k  𝔼[ || E(x_{t+k}) - P^k(E(x_t), a_{t:t+k}) ||² ]

This forces the dynamics to compose cleanly over time.

**Loss 2: Invariance (from Axiom 6)**

    L_invariance = 𝔼[ Σ_{j ∉ N(i)} || P_j(mask_i(E(x_t))) - P_j(E(x_t)) ||² ]

where P_j is the latent predictor for object j's slot, mask_i zeroes out object i's slot, and N(i) is the current estimate of i's influence neighborhood. This forces the model to discover causal independence: masking object i should not affect predictions for objects not influenced by i.

**Loss 3: Sufficiency (from Axiom 4 + Influence Neighborhoods)**

    L_sufficiency = 𝔼[ || P_i(mask_{-N(i)}(E(x_t))) - E_i(x_{t+1}) ||² ]

where mask_{-N(i)} masks everything *except* the influence neighborhood of i. This forces the model to learn *minimal sufficient* causal contexts: the influence neighborhood N(i) should contain all and only the objects needed to predict i's next state.

**Total loss**:

    L = L_consistency + λ₁ L_invariance + λ₂ L_sufficiency + λ₃ L_multistep

### The Learning Cycle

Training proceeds in an alternating cycle:

**Step 1: Forward prediction (learn dynamics)**
Use L_consistency to learn the encoder E and predictor P.
At this stage, you're learning *what* the dynamics are — how states evolve.

**Step 2: Masking intervention (discover causal structure)**
Use L_invariance and L_sufficiency with random masking patterns.
At this stage, you're learning *which* dynamics are independent — the causal structure.

**Step 3: Counterfactual evaluation (test causal understanding)**
Generate counterfactual predictions: "what would happen if object i had different initial conditions?" Compare against ground truth (if available) or consistency checks (if not).
At this stage, you're testing whether the learned causal structure supports valid counterfactual reasoning.

These three steps correspond to CTT's three modes of learning:
- Step 1 = Observational learning (Axiom 5)
- Step 2 = Interventional learning (Axiom 6)
- Step 3 = Counterfactual learning (Axiom 4)

### Handling Continuous, High-Dimensional, Noisy Data

Real-world data (video, sensor streams) has three properties that challenge causal learning:

**Continuous**: States aren't discrete tokens. CTT handles this via measurable spaces and Markov kernels, which are defined for continuous state spaces. The AI system handles this via learned encoders that map continuous input to continuous latent representations. No discretization needed.

**High-dimensional**: Raw video has millions of pixels. CTT handles this via observation maps (projections to lower-dimensional spaces). The AI system handles this via Slot Attention or similar object-centric encoders that decompose a high-dimensional scene into a manageable number of object slots, each with a low-dimensional latent vector. The encoder IS the observation map.

**Noisy**: Sensor data is corrupted. CTT handles this via stochastic mechanisms (Markov kernels) that naturally represent noise as distributional spread. The AI system handles this via probabilistic predictions — instead of predicting a single next latent state, predict a *distribution* over next latent states (e.g., Gaussian with learned mean and variance). The noise becomes part of the model rather than something to be eliminated.

---

## Part V: From Axioms to Architecture

Here is the complete mapping from mathematical structure to neural architecture:

### State Space → Latent Representation

| CTT concept | Architecture component |
|---|---|
| Full state Σ | Not directly represented (too high-dimensional) |
| Observed state Σ_obs | Raw input (video frames, sensor data) |
| Latent state Z | Object slots from Slot Attention encoder |
| Observation map π | Encoder E : frames → slots |

### Mechanisms → Dynamics Modules

| CTT concept | Architecture component |
|---|---|
| Mechanism κ_i | Per-object dynamics MLP or Transformer block |
| Mechanism composition κ₂ ∘ κ₁ | Sequential application of dynamics modules |
| Parallel composition κ₁ ⊗ κ₂ | Independent per-slot computation |
| Temporal extent (t_in, t_out) | Input/output time indices of the dynamics module |
| Markov kernel κ : Σ →ₖ Σ' | Dynamics module outputting distribution parameters (μ, σ) |

### Temporal Structure → Causal Masking

| CTT concept | Architecture component |
|---|---|
| t_in ≤ t_out | Causal attention mask (no future information) |
| Well-foundedness | Finite context window |
| Temporal extent | Time embedding / positional encoding |

### Interventions → Module Replacement / Masking

| CTT concept | Architecture component |
|---|---|
| Intervention ι | Replace/mask a dynamics module |
| Hard intervention (set value) | Replace slot with fixed vector |
| Soft intervention (modify mechanism) | Scale/shift dynamics module output |
| Masking intervention | Zero out / replace slot with learned mask token |
| Null intervention (observe) | No masking, standard forward pass |

### Axioms → Losses and Constraints

| CTT axiom | Loss / constraint |
|---|---|
| Temporal consistency | Causal attention mask (architectural) |
| Compositional consistency | Multi-step prediction loss |
| Intervention algebra | Composable module replacement (architectural) |
| Counterfactual coherence | Invariance loss under masking |
| Marginalization consistency | JEPA consistency loss (primary) |
| Causal invariance | Invariance regularization loss |
| Observation compatibility | Predictive (not reconstructive) training objective |

---

## Part VI: What You're Actually Building

Putting it all together, a CTT-informed causal world model is:

**An encoder** (observation map) that takes raw high-dimensional input and produces a structured latent representation — a fixed number of object slots, each a vector in ℝᵈ. This encoder is the learned coarsening functor from observation space to latent space.

**A set of dynamics modules** (mechanisms) — one per object or one per interaction type — that take current latent states and produce distributions over next latent states. These are the Markov kernels. They compose to give multi-step predictions.

**A masking strategy** (intervention operators) that randomly masks object slots during training, forcing the dynamics modules to learn which objects influence which. This is the self-supervised interventional learning that discovers causal structure without labeled causal graphs.

**An influence neighborhood learner** that estimates, for each object, the minimal set of other objects whose states are needed to predict its dynamics. This emerges from the masking training, and once learned, enables efficient sparse attention.

**A counterfactual inference procedure** that, given a factual trajectory and a hypothetical intervention, produces the counterfactual trajectory by: (1) running the model up to the intervention time with shared dynamics, (2) applying the intervention (replace a mechanism), (3) rolling forward with the modified dynamics. The counterfactual coherence axiom guarantees this procedure is well-defined.

The entire system learns from unlabeled video or sensor data. No causal graph annotations. No human-specified mechanisms. The causal structure *emerges* from the combination of predictive training (Axiom 5), masking interventions (Axioms 4 and 6), and multi-step consistency (Axiom 2).
