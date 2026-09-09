# Experimental Protocol — Goal-Information Decay and Repair Routing for LLM Agents

**Audience:** the implementing engineer (human or Claude Code).
**Status:** execution plan. Read Section 0 before writing any code.

---

## 0. What this project is, and the one thing that can kill it

### 0.1 The claim

Agent context degrades during execution for **two distinguishable reasons**, and the distinction determines which repair works:

- **Goal loss** — the objective is no longer strongly recoverable from the agent's state.
- **Clutter accumulation** — task-irrelevant material has grown to compete with the objective.

We measure both at every step, and use the pair to *select* a repair action rather than applying one fixed strategy.

### 0.2 What is NOT novel (do not claim these)

Online failure detection, activation probing of agents, failure taxonomies, runtime repair, and state-preserving context editing are all published. The contribution is **diagnosis-conditioned repair selection**, plus the two-channel decomposition that makes selection possible. Every experiment below exists to support that claim and nothing wider.

### 0.3 The kill gates

This project has three points where it should be abandoned. They are placed early and deliberately. **Do not skip forward past a failed gate.**

| Gate | Phase | Condition to abandon |
|---|---|---|
| **K1** | 1 | A trivial embedding-similarity baseline matches the probe |
| **K2** | 2 | Goal-recoverability adds nothing over trajectory length |
| **K3** | 3 | G and C are not separable (|ρ| > 0.8) |

Each gate is cheap relative to what follows it. Running them in order is the single most important process decision in this document.

---

## 1. Environment and infrastructure

### 1.1 Model selection

**Primary: `Qwen3.8-27B`** (Apache 2.0, dense, ~262K context).

Rationale, in the order a reviewer will ask:

1. **Dense, not MoE.** Activation probing on a mixture-of-experts model is confounded by routing — the residual stream at layer *L* reflects different expert subsets across tokens. Every probing paper worth citing uses dense models. This is not negotiable.
2. **Open weights.** The method reads residual-stream activations. Closed models are impossible: Anthropic's API does not expose token logprobs either, so there is no black-box fallback for Claude, GPT, or Gemini.
3. **Hostable.** 27B in bf16 ≈ 54 GB → fits one 80GB H100 or two 40GB A100s. Frontier open models like Kimi K3 (2.8T params) are rack-scale and out of reach.
4. **Permissive license.** Apache 2.0 avoids a licensing footnote in the paper.

**Secondary (generalization only, Phase 5): one model from a different family** — a `gpt-oss` checkpoint (Apache 2.0) or a Gemma-family dense model. Purpose is solely to show the effect is not Qwen-specific. Do not attempt this before Phase 5.

> **Verify before ordering compute.** The open-weight landscape moves fast and these specific checkpoints may have been superseded. Confirm current availability, exact context window, and license on the model card. The requirements that matter are: dense, open weights, ≥128K context, permissive license, hostable on your GPUs.

### 1.2 Serving

Use **vLLM** for trajectory generation (throughput) and **raw HuggingFace Transformers with hooks** for activation extraction (vLLM does not expose intermediate activations cleanly).

This means a **two-pass architecture**:
- **Pass A** — generate trajectories with vLLM, log everything to disk.
- **Pass B** — replay each logged step through HF Transformers with forward hooks, cache activations.

Pass B is teacher-forced replay of text the model itself generated, so activations are on-policy. State this explicitly in the paper; a reviewer will ask.

### 1.3 Activation caching — plan storage before you start

Naive caching will fill a disk in hours. Do the arithmetic first:

```
trajectories × steps/traj × layers × d_model × 2 bytes
2000 × 40 × 64 × 5120 × 2 ≈ 52 TB   ← unacceptable
```

**Required reductions:**
- **Layers:** cache 4 only — depth fractions 0.25, 0.5, 0.75, 1.0. Middle layers carry the most linearly-decodable semantic content; the sweep over which is best is a small ablation, not the main result.
- **Positions:** cache the **last token of each step** plus a mean-pool over that step's generated tokens. Not every token.
- **Precision:** float16.

```
2000 × 40 × 4 × 5120 × 2 × 2 (last + mean) ≈ 6.5 GB   ← fine
```

Store as sharded `.npy` or `safetensors` keyed by `(task_id, seed, step_idx)`. Never pickle.

### 1.4 Repository structure

```
goal-decay/
  configs/          # YAML, one per experiment; no hardcoded hyperparameters
  src/
    rollout/        # trajectory generation (vLLM)
    extract/        # activation caching (HF + hooks)
    probes/         # G and C estimators
    repair/         # repair primitives + router
    ledger/         # execution ledger
    eval/           # metrics, statistics
  data/
    raw/            # generated trajectories (immutable once written)
    activations/
    splits/         # frozen split files, committed to git
  results/          # one subdir per run, config hash in name
  paper/
```

**Determinism requirements:** fixed seeds everywhere, `temperature` logged per run, all splits written to disk and committed **before** any model touches them. Log the git commit hash into every results directory.

---

## 2. Phase 0 — Corpus construction (Week 1–2)

### 2.1 Primary benchmark: τ²-bench

**Why this one, defensibly:**

- **Free trajectory-level labels.** τ²-bench computes a programmatic reward — 1 if the final environment state matches the ground-truth task specification, 0 otherwise. No LLM judge, no human annotation, no judge-bias critique.
- **Stateful with real side effects.** The agent mutates a database. This is what makes the Effect Preservation Rate metric meaningful and the "restart is not free" argument concrete.
- **Directly comparable.** AgentTether, our main repair baseline, reports on τ-bench.
- **Public.** Domain data and official evaluation code are available.

Use all three domains: **retail, airline, telecom.** Telecom is dual-control (user and agent both call tools), which produces longer, noisier trajectories — useful for the long-context arm.

### 2.2 Secondary benchmark: GAIA (Phase 5 only)

Longer horizons, web tools, more accumulated noise. Used for cross-domain generalization only.

### 2.3 Do NOT use pre-annotated trajectory datasets as your primary corpus

You will be tempted by Who&When (184 human-annotated traces, MIT) and AgentProcessBench (1,000 trajectories, 89.1% inter-annotator agreement). **They were generated by other models.** You cannot extract *your* model's activations from *their* trajectories without replaying them off-policy, which breaks the on-policy claim underpinning RQ1.

Use them for exactly one thing: a **secondary validation** in Phase 5, replaying them through your model to check that goal-recoverability correlates with human-annotated decisive error steps. Label it clearly as off-policy replay.

### 2.4 Generation protocol

```
for domain in [retail, airline, telecom]:
  for task in domain.tasks:
    for seed in [0,1,2,3,4]:
      run agent (Qwen3.8-27B, ReAct scaffold, temp=0.7)
      log: full message list, tool calls, tool returns,
           per-step token counts, final programmatic reward
```

**Target: ≥2,000 trajectories with ≥25% failures.** If the failure rate is under 15%, the signal will be too sparse — increase task difficulty or raise temperature. If it is over 60%, the scaffold is broken; fix it before proceeding.

**Record per step:** step index, cumulative tokens, tokens added this step, tool name, whether the tool call errored, and message role.

### 2.5 Frozen splits — do this now, not later

Split **by task**, never by step.

> Steps within one trajectory are highly correlated. A random step-level split puts steps 4 and 5 of the same trajectory in train and test, and your probe will report near-perfect accuracy that is pure leakage. This is the single most common fatal error in probing papers, and a reviewer will look for it.

Produce three split files, committed to git:

- `split_random.json` — grouped by `task_id`, 60/20/20.
- `split_domain.json` — train on retail+airline, test on telecom. Tests cross-domain generalization.
- `split_length.json` — length-stratified, for the confound control in §4.3.

---

## 3. Phase 1 — The kill test (Week 3) 🚩 GATE K1

**Run this before building anything else.**

### 3.1 The baseline that could end the project

Compute, with no probes and no model internals:

```
cos_sim(embed(goal_text), embed(context_at_step_t))
```

using any off-the-shelf sentence embedder. Evaluate its AUROC for predicting trajectory failure at 25%, 50%, 75%, and 100% of trajectory progress.

### 3.2 Decision rule (fix these numbers now, before seeing results)

- Cosine baseline AUROC@50% **≥ 0.65** → it already works. Your probe must beat it by ≥0.05 AUROC with non-overlapping bootstrap CIs, or the contribution collapses to "we reproduced cosine similarity with extra steps." **Stop and reconsider the project.**
- Cosine baseline AUROC@50% **< 0.60** → proceed. You now also have your headline baseline for the paper.
- In between → proceed, but the bar for the probe is now this number plus 0.05.

Write the result to `results/gate_k1/` regardless of outcome. If the project dies here, it dies having cost three weeks, and the negative result is itself worth a workshop note.

---

## 4. Phase 2 — The G probe and RQ1 (Weeks 4–7)

### 4.1 Defining the target — the design decision reviewers will scrutinize

`I(S_t; G)` as written in the original idea is **undefined**: mutual information needs a distribution over goals, and within one episode the goal is a single fixed string. Do not write it that way.

Use **V-usable information** (Xu et al., ICLR 2020; Ethayarajh et al., 2022): information extractable by a computationally bounded predictor family V.

```
I_V(S_t → Y) = H(Y) − H_V(Y | S_t)
```

where `H_V(Y|S_t)` is the minimum cross-entropy achievable by any probe in V. Report in **bits**.

**What is Y?** Use structured goal-slot decoding. τ²-bench tasks have machine-readable specifications, so extract slots per domain:

- retail: `{action_type, order_id, item_id, constraint}`
- airline: `{action_type, flight_id, passenger_constraint, date}`
- telecom: `{issue_type, device_id, resolution_target}`

`Y` = the tuple of slot values. `G(t) = I_V(S_t → Y)` is then literally "how many bits of the task specification are still recoverable from the agent's state at step t."

This is defensible in a way a single scalar similarity is not: it is grounded in the task's own ground-truth spec, not in a similarity heuristic.

**Sanity check (run alongside):** contrastive goal identification — given `S_t`, pick the true task from 32 candidates. Should track `G(t)` closely. If it doesn't, the slot decoder is broken.

### 4.2 Probe family and required controls

- **V = linear probes** (one linear layer, no hidden nonlinearity). Simple family = cleaner V-information semantics and a stronger claim: if a *linear* probe finds it, the information is genuinely there.
- **Selectivity control (Hewitt & Liang 2019).** Train an identical probe on **shuffled labels** and report the gap. Without this, a reviewer will say your probe memorized rather than decoded. Non-negotiable.
- **Probe train/test hygiene.** Probe training tasks must be disjoint from evaluation tasks. Use the frozen splits.
- Train separately per layer; report the layer sweep as a small ablation.

### 4.3 The confound that will sink you if unaddressed 🚩 GATE K2

**Failing trajectories are longer.** More steps, more retries, more tool output. Any signal that decays with length will correlate with failure for trivial reasons. Published step-level analysis confirms interaction length correlates strongly with task difficulty and outcome, and that unsuccessful trajectories involve more steps on average.

**Three mandatory controls:**

1. **Length-matched strata.** Bin trajectories by total steps; report AUROC within each bin. If the signal only works across bins and not within them, you have measured length.
2. **Length as an explicit baseline.** Include `step_index` and `cumulative_tokens` as standalone predictors in every comparison table. Your probe must beat them.
3. **Dual evaluation points.** Report at fixed *fraction* of progress (25/50/75%) **and** at fixed *absolute* step (10, 20, 30). These dissociate the two explanations.

**Gate K2:** if `G(t)` does not beat `cumulative_tokens` alone by ≥0.05 AUROC at 50% progress with non-overlapping CIs, **stop**. You are measuring trajectory length with an expensive instrument.

### 4.4 The bar for RQ1

The target is set by published work: across ten uncertainty metrics including semantic and predictive entropy, no metric exceeded mean AUROC 0.60 at 50% trajectory progress, and all stayed below 0.70 even at 90%.

**Success = AUROC@50% > 0.60 with a bootstrap CI excluding 0.60**, at negligible per-step cost compared with an LLM auditor.

**Baselines table (all required):**

| Baseline | Why it must be there |
|---|---|
| Cosine similarity (§3.1) | The trivial control |
| `cumulative_tokens` | The length confound |
| Semantic entropy | The published uncertainty family |
| Verbal confidence | Strongest reported uncertainty baseline |
| Linear success/failure probe | The obvious simpler alternative — you must show G is not just this |
| AgentForesight-style LLM auditor | The cost comparison |

The success/failure probe row is the one reviewers will fixate on. If a generic probe matches your goal-specific probe, N1 is dead and only the routing claim survives. Know this number early.

### 4.5 Statistics

- **Cluster bootstrap over tasks** (not steps), 10,000 resamples. Steps are not independent.
- **5 seeds minimum**, report mean ± CI across seeds.
- **Effect sizes with CIs.** No bare p-values.
- Pre-register thresholds in `configs/` before running. Commit them.

---

## 5. Phase 3 — The C channel and causal validation, RQ2 (Weeks 8–11)

### 5.1 Defining C

Keep it simple and auditable. Partition context tokens into three classes using the execution ledger (§6.1):

- **goal-relevant** — the objective statement and re-anchors
- **effect-relevant** — committed tool calls and verified results referenced in the ledger
- **residual** — everything else

```
C(t) = residual_token_mass(t), weighted by attention mass received
```

Report both the raw token-share version and the attention-weighted version. The raw version needs no model internals and makes the metric partially reproducible on closed models — worth a sentence in the paper.

### 5.2 Separability 🚩 GATE K3

Compute Spearman ρ between `G(t)` and `C(t)` across all steps.

- **|ρ| > 0.8** → the plane collapses to one dimension. The routing claim (N3) is dead. **Stop, and write up the single-channel finding as a shorter paper.**
- **|ρ| < 0.6** → proceed with confidence.
- In between → proceed but report prominently; a reviewer will compute this themselves.

### 5.3 Causal intervention — the experiment that separates this from correlational work

Every competing paper is correlational. This is your differentiator. Use a **paired within-task design**: same task, same seed, same prefix, one manipulated condition.

| Manipulation | Predicted ΔG | Predicted ΔC | Predicted Δfailure |
|---|---|---|---|
| Inject irrelevant tool output (500 tok) | ↓ mild | ↑ strong | ↑ |
| Move objective to low-salience position | ↓ strong | — | ↑ |
| Re-anchor objective at step *k* | ↑ | — | ↓ |
| Prune residual spans at step *k* | ↑ mild | ↓ strong | ↓ |
| **Placebo: inject *goal-relevant* text of equal length** | — | — | — |

**The placebo row is essential.** Without it, "we added 500 tokens and it got worse" is explained by length alone. Equal-length relevant injection must NOT produce the same degradation. If it does, you have rediscovered context length.

**Analysis:** paired tests over matched task pairs, effect sizes with CIs, ≥200 pairs per manipulation.

---

## 6. Phase 4 — Repair routing, RQ3 (Weeks 12–16)

This phase carries the paper. Everything before it is instrumentation.

### 6.1 The Execution Ledger

Append-only record of committed effects, structurally separate from the reasoning context:

```json
{
  "step": 12,
  "type": "tool_call",
  "tool": "update_order",
  "args": {...},
  "result_ref": "blob://a3f2",
  "verified": true,
  "reversible": false
}
```

**Invariant: repair rewrites the context (a view over the ledger); repair never rewrites the ledger.** This is what makes mid-flight repair safe and is the mechanism behind the progress-preservation claim. Enforce it with an assertion in code, not a convention.

Compaction folds a verbose tool return into a reference into the ledger, so the payload stays addressable rather than lost — the recoverable-sidecar pattern Self-GC demonstrated.

### 6.2 Repair primitives

Implement three, each independently switchable:

1. **`reanchor()`** — re-render the objective at a high-salience position (context start; head and tail positions are attended more strongly than the middle).
2. **`compact()`** — fold residual spans into ledger references, preserving anchors.
3. **`delegate()`** — spawn a sub-agent with a clean context and a scoped subtask; ledger persists.

Plus **`escalate()`** — halt and surface to a human.

### 6.3 The router

```
if G_low and C_low:    reanchor()
if G_low and C_high:   compact()
if G_high and C_high:  delegate()
if G_high and C_low:   no action
```

**Escalation is not a quadrant.** It fires on `dG/dt` below threshold **after** a routed repair has already failed to restore G. Implement it as a second-level rule, not a fifth cell. Thresholds for "low"/"high" are fit on the *training* split only and frozen before test evaluation.

### 6.4 Arms (all required)

| Arm | Purpose |
|---|---|
| No repair | Floor |
| Full restart | The current default practice |
| Always `reanchor()` | Isolates routing from the primitive |
| Always `compact()` | Isolates routing from the primitive |
| Always `delegate()` | Isolates routing from the primitive |
| **Random routing** | **Isolates routing from *repairing at the right time*** |
| **Routed (ours)** | The claim |
| AgentTether-style guidance injection | Published baseline |

> The **random-routing** arm is the one that makes RQ3 credible. Without it, any gain could come from intervening at the right moment rather than choosing the right action. Random routing fires the same repairs at the same trigger points with the action chosen at random. If routed does not beat random, the diagnosis is doing no work and only the timing matters — which would be a real finding, but a different and much smaller paper.

### 6.5 Metrics

- **Task success rate** (τ²-bench programmatic reward)
- **Total tokens to completion** — repair must not cost more than it saves
- **Effect Preservation Rate** — fraction of validated committed effects retained across repair, versus restart. This metric is ours; define it precisely and justify it.
- **Repair trigger precision/recall** — how often repair fired on trajectories that would have succeeded anyway
- **Wall-clock overhead**

### 6.6 Success criterion

Routed beats the **best** single-strategy arm **and** random routing on success rate, with non-overlapping cluster-bootstrap CIs, without increasing total tokens.

---

## 7. Phase 5 — Generalization and the window-size question (Weeks 17–19)

### 7.1 Context window scaling

Anticipates the strongest framing objection: *"won't bigger context windows make this obsolete?"*

Hold the task fixed; vary the available context budget (32K / 128K / 262K) on the same model. Ask whether `G` decay tracks **absolute tokens consumed** or **fraction of window used**.

Degradation is known to begin well before any window limit is reached, because attention is a normalized aggregation in which all tokens compete regardless of remaining headroom. If decay tracks absolute tokens, larger windows do not rescue the agent and the contribution survives the next model generation. Half a page, high value.

### 7.2 Cross-model and cross-domain

- Second model family → the effect is not Qwen-specific.
- GAIA → not τ²-specific.
- Off-policy replay of Who&When → `G` should dip near human-annotated decisive error steps. Label the off-policy caveat explicitly.

---

## 8. Threats to validity — write this section early

Draft it in Week 4, not Week 18. It shapes the experiments.

1. **Length confound** → §4.3 controls.
2. **Probe capacity** → selectivity control, §4.2.
3. **Single model family** → §7.2.
4. **Single benchmark** → GAIA arm.
5. **Off-policy replay** in the Who&When validation → labeled.
6. **Threshold overfitting** → thresholds fit on train split only, frozen before test.
7. **White-box requirement** → limits deployment to open models; state plainly. Prior work on early failure prediction has the same restriction.
8. **Simulated users** in τ²-bench → not real user behavior.

---

## 9. Order of operations — the short version

1. **Weeks 1–2** — infra, generate ≥2,000 τ²-bench trajectories, freeze splits.
2. **Week 3** — 🚩 **K1**: cosine baseline. Possible stop.
3. **Weeks 4–7** — G probe, selectivity control, length controls. 🚩 **K2**. Possible stop.
4. **Weeks 8–11** — C channel, separability 🚩 **K3**, causal interventions with placebo.
5. **Weeks 12–16** — ledger, three primitives, router, all eight arms including random routing.
6. **Weeks 17–19** — window scaling, second model, GAIA, Who&When replay.
7. **Weeks 20–22** — writing.

---

## 10. Instructions for Claude Code

- **Never** split probe train/test by step. Always group by `task_id`. Assert this in the data loader.
- **Never** report a metric without a cluster bootstrap CI over tasks.
- **Never** silently drop trajectories. Log every exclusion with a reason; report counts in the paper.
- **Always** write config + git hash into each results directory.
- **Always** run the shuffled-label control alongside every probe.
- Treat `data/raw/` as immutable after generation.
- Ledger writes are append-only. Add an assertion that fails loudly on mutation.
- When a gate fails, **stop and report**. Do not proceed to the next phase and do not tune until the gate passes — that converts a kill gate into a fishing expedition, and it is the failure mode this document exists to prevent.
