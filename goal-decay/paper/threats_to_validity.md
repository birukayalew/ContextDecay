# Threats to validity — draft (start Week 4 per protocol §8)

1. **Length confound** → §4.3 controls (length-matched strata, `cumulative_tokens` baseline, dual fraction/absolute evaluation points).
2. **Probe capacity** → selectivity control on shuffled labels, §4.2.
3. **Single model family** → §7.2, second-family generalization arm.
4. **Single benchmark** → GAIA arm, §7.2.
5. **Off-policy replay** in the Who&When validation → labeled explicitly as off-policy.
6. **Threshold overfitting** → router thresholds fit on train split only, frozen before test.
7. **White-box requirement** → limits deployment to open models; state plainly.
8. **Simulated users** in τ²-bench → not real user behavior.

## 9. Hybrid-attention architecture (new — Qwen3.8-27B specific)

Qwen3.8-27B is not uniform softmax attention across all 64 layers: 48 are
Gated DeltaNet (linear-attention) layers and only 16 are full Gated
Attention layers, interleaved. The attention-dilution mechanism motivating
this whole project (§7.1 — "attention is a normalized aggregation in which
all tokens compete regardless of remaining headroom") is a **softmax
attention** story. Linear-attention layers (Gated DeltaNet) do not perform
the same normalized all-to-all competition; they maintain a recurrent
state that is updated multiplicatively/additively rather than computing a
softmax over the full context.

**Consequence for methodology:** the four cached layers (protocol §1.3)
must not be chosen by depth fraction alone. `src/extract/layer_select.py`
enforces that at least 2 of the 4 cached layers are full-attention layers,
determined by reading `config.json`'s layer-type field on the actual
checkpoint, not assumed from position. The chosen indices and their
attention type are logged to `layer_selection.json` alongside every
extraction run.

**Consequence for the paper — this may be a finding, not just a caveat:**
whether `G(t)` decays similarly when probed from linear-attention layers
vs. full-attention layers is an open empirical question worth reporting
directly. Candidate framings once Phase 2 data exists:

- If G decays comparably at both layer types → suggests the effect is not
  purely a softmax-competition artifact, which *strengthens* the general
  claim (goal information degrades for reasons beyond attention dilution
  specifically).
- If G decays only/mostly at full-attention layers → directly confirms
  the attention-dilution mechanism and is worth a dedicated subsection
  with the layer-type contrast as the key plot.
- If linear-attention layers show *no* decay signal at all → worth
  flagging as a distinct limitation of probing hybrid architectures, and
  motivates being explicit in the intro that the mechanism claim is
  scoped to full-attention layers even though the deployed repair system
  operates on the whole model.

Do not pre-commit to one of these framings before the data exists. Log
both layer-type-stratified AUROC curves in Phase 2 regardless of which
story they support.
