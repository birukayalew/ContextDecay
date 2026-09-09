# Remote (GPU server) handoff

Everything below runs on the GPU server, not the Windows dev machine.
Copy the repo over first (`git clone`/`git pull` this repo), then run
these in order. **Stop after the dry run and report back before running
the full generation** — do not proceed to the full run on your own
judgment call.

## 0. Environment

```bash
# Model: Qwen/Qwen3.8-27B (HF), 27B dense hybrid-attention (48 linear + 16
# full Gated Attention layers), Apache 2.0, 262144 context, ~55.6GB bf16.
# Only the Qwen org HF repo is genuine -- verify the org before downloading.
huggingface-cli download Qwen/Qwen3.8-27B --local-dir ./models/Qwen3.8-27B

# tau2-bench
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
git rev-parse HEAD   # <-- record this, it must go into configs/rollout_dryrun.yaml's pinned_commit
uv sync              # NOT pip install -e . -- needs Python >=3.12,<3.14
cd ..

# vLLM + serving stack (versions TBD -- check current vLLM compatibility
# with Qwen3.8-27B's hybrid attention architecture; Gated DeltaNet layers
# may need a vLLM version recent enough to support them -- confirm before
# assuming vanilla vLLM handles this checkpoint out of the box)
pip install vllm transformers accelerate safetensors
```

**Before going further:** confirm vLLM actually loads and serves
Qwen3.8-27B correctly given the hybrid linear/full attention layers. If
vLLM doesn't yet support Gated DeltaNet layers, generation may need to
fall back to raw HF Transformers (slower) for Pass A too, not just Pass B.
Report this back before writing throughput estimates into the paper.

## 1. Pin the tau2-bench commit

Edit `configs/rollout_dryrun.yaml`:
```yaml
benchmark:
  pinned_commit: "<hash from git rev-parse HEAD above>"
```

## 2. Inspect one real tau2-bench trial log BEFORE running our wrapper

```bash
cd tau2-bench
tau2 run --help   # confirm actual flag names (--seed may not exist -- check)
tau2 run --domain retail --agent-llm Qwen/Qwen3.8-27B --user-llm Qwen/Qwen3.8-27B --num-trials 1 --num-tasks 1
```

Find wherever this writes its output and paste (or send as a file) the
full JSON/log structure of **one trial**. `src/rollout/generate.py`'s
`parse_tau2_trial()` is currently a stub that raises `NotImplementedError`
on purpose — it needs the real schema filled in, not a guess, because a
wrong field mapping would silently corrupt every trajectory in the corpus
the paper depends on. Send that one trial log back first.

## 3. Dry run (after parse_tau2_trial is filled in)

```bash
python -m src.rollout.generate \
  --config configs/rollout_dryrun.yaml \
  --tau2-repo-path ./tau2-bench
```

This runs retail only, 5 tasks, 1 trial (per instructions). It will:
- refuse to run if the tau2-bench checkout isn't at the pinned commit
- refuse to overwrite `data/raw/trajectories_dryrun.jsonl` if it already exists
- log every excluded/unparseable trial to `results/rollout_dryrun/exclusions.jsonl` with a reason (never silently dropped)
- print a summary and explicitly say DRY RUN COMPLETE — do not proceed past this

## 4. What to send back after the dry run

- `data/raw/trajectories_dryrun.jsonl` (or just its first record + one full step, if the whole file is unwieldy)
- `results/rollout_dryrun/exclusions.jsonl` if non-empty
- Confirmation of which required fields (§2.4: step_idx, cumulative_tokens, tokens_added_this_step, tool_name, tool_errored, message_role, programmatic_reward) were actually populated vs. missing/null
- The tau2-bench commit hash actually used
- Wall-clock time for the 5-task dry run, so we can estimate the full 2,000-trajectory run before committing to it

Once that's reviewed, we'll fill in `configs/rollout_full.yaml` and green-light the full generation.

## Layer selection for activation caching (Phase 2, later — not needed for the dry run)

Before Pass B activation extraction starts, run:

```bash
python -c "
import json, sys
sys.path.insert(0, '.')
from src.extract.layer_select import classify_layers, select_cache_layers, write_layer_selection
with open('models/Qwen3.8-27B/config.json') as f:
    hf_config = json.load(f)
layer_types = classify_layers(hf_config)
choices = select_cache_layers(layer_types, min_full_attention=2)
write_layer_selection(choices, 'results/layer_selection.json')
print(choices)
"
```

If this raises `KeyError` about a missing `layer_types` field, open
`config.json` by hand, find whatever field actually distinguishes Gated
DeltaNet (linear) from Gated Attention (full) layers, and report the
field name and its values back — don't guess at the split from layer
position, since the 16 full-attention layers are interleaved.
