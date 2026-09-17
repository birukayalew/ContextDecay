# Remote (GPU server) handoff

Everything below runs on NC State's Hazel HPC cluster (Slurm), not the
Windows dev machine. This doc reflects what has actually been verified
working as of 2026-09-17, after a full session of debugging the real
setup end to end. If you're restarting from a dead session, jump to
**"Restarting after a disconnect"** below — the one-time setup
(model download, conda env, tau2-bench install) does not need to be
redone, only the runtime steps (GPU allocation, vLLM server, env vars).

## One-time setup (already done, documented for reference / a fresh machine)

### Login node vs. compute node

**Compute nodes have no internet access.** Downloads (model weights, pip
packages, git clones) must happen on a login node. Only `srun`/`sbatch`
jobs get GPU access, and those run on compute nodes.

### Hazel-specific Slurm quirks

- `si --gpus --qos gpu` and `si --gpus --qos short_gpu` show live GPU
  availability (not `sinfo`).
- **`--qos gpu` is batch-only** (for `sbatch`). Interactive sessions
  (`srun --pty bash`) must use `--qos short` (CPU) or `--qos short_gpu` (GPU).
  Using `--qos gpu` with `srun` queues forever without erroring clearly.
- H200s (48GB... actually 143GB/~140GiB usable) live under
  `--partition=gpu_partners --qos=short_gpu --gres=gpu:h200:1`.
- Session example that worked:
  ```bash
  srun --partition=gpu_partners --qos=short_gpu --gres=gpu:h200:1 \
    --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
  ```

### Cache/quota traps (both hit during setup — avoid repeating)

- `/home` has a **15GB space quota AND a 10,000-file count quota**. The
  file-count quota is the one that actually bites: pip/uv/conda/vllm/
  flashinfer/torch-inductor caches all default to subdirectories of
  `~/.cache` and can silently fail mid-job with `OSError: [Errno 122]
  Disk quota exceeded` even when space usage looks fine.
- **Fix applied:** moved `~/.cache` contents to scratch and symlinked:
  ```bash
  mkdir -p /share/agenticsystems/$USER/project/home_cache
  mv ~/.cache/* /share/agenticsystems/$USER/project/home_cache/
  rmdir ~/.cache
  ln -s /share/agenticsystems/$USER/project/home_cache ~/.cache
  ```
  Do this once per account; it's the more robust fix than exporting
  individual tools' cache-dir env vars (several tools, including
  `flashinfer`, don't fully respect `XDG_CACHE_HOME`).
- `/share/agenticsystems` (group quota, shared across the whole lab) is
  also a **file-count** quota (1,000,000 files) and was observed at
  ~95% during setup. This is not just our usage — flag to PI/labmates if
  it becomes a blocker; a `uv`/`pip` cache alone can add tens of
  thousands of files.

### Model and benchmark

```bash
# Model: Qwen/Qwen3.8-27B (HF), confirmed working. 27B dense, HYBRID
# attention (48 Gated DeltaNet linear-attention layers + 16 full Gated
# Attention layers -- resolved by vLLM as Qwen3_5ForConditionalGeneration).
# Apache 2.0, 262144 native context, ~55.6GB bf16 across 18 shards.
hf download Qwen/Qwen3.8-27B --local-dir /share/agenticsystems/$USER/project/models/Qwen3.8-27B
# (huggingface-cli is deprecated in favor of `hf`; same tool)

# tau2-bench, pinned commit (confirmed working, all three domains present):
git clone https://github.com/sierra-research/tau2-bench.git
cd tau2-bench
git checkout 2174a603f6d014ef94473ffa95957f6ce27100db  # or verify HEAD is already this
```

**Apply the required patch before installing** (see
`configs/tau2_bench.patch` in this repo):

```bash
git apply /path/to/goal-decay/configs/tau2_bench.patch
```

This patch does two things, both required:
1. Adds `websockets` to `pyproject.toml` — without it, `import tau2` itself
   crashes (`tau2/__init__.py` unconditionally imports a voice/audio-native
   module that needs `websockets`, even though we never use `--audio-native`).
2. Changes `DEFAULT_LLM_NL_ASSERTIONS` in `src/tau2/config.py` from
   `"gpt-4.1-2025-04-14"` to `"hosted_vllm/Qwen/Qwen3.8-27B"`. tau2-bench's
   NL-assertions evaluator (marked "experimental/WIP" in its own
   `AGENTS.md`) grades some tasks' natural-language criteria via a
   hardcoded LLM call with no CLI/env override — left at the default it
   requires a real `OPENAI_API_KEY` and silently fails 3/5 dry-run tasks
   with `infrastructure_error`. Patched to reuse our own hosted model
   instead of requiring a paid external API key.

Then install:
```bash
pip install uv
export UV_CACHE_DIR=/share/agenticsystems/$USER/project/.uv_cache  # belt-and-suspenders even with the ~/.cache symlink fix
mkdir -p $UV_CACHE_DIR
uv sync   # NOT pip install -e . -- needs Python >=3.12,<3.14
```

### Conda environment

No shared env exists at `/usr/local/usrapps/agenticsystems` on this
cluster (checked, not present). Built our own via Miniforge in scratch
space (downloaded on the **login node**, since compute nodes have no
internet):

```bash
cd /share/agenticsystems/$USER/project
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O miniforge.sh
bash miniforge.sh -b -p /share/agenticsystems/$USER/project/miniforge3
source /share/agenticsystems/$USER/project/miniforge3/bin/activate
conda create -y -n goaldecay python=3.12   # base conda's own python was 3.14, too new
conda activate goaldecay
```

Then (also on the login node, for internet access):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install vllm transformers accelerate safetensors "huggingface_hub[cli]"
```
Note: installing `vllm` force-upgrades torch to its own preferred version
(observed: torch 2.13.0+cu130) — this is expected and fine, don't fight it.

**libstdc++ fix required for vLLM to even import:**
```bash
conda install -y -n goaldecay -c conda-forge libstdcxx-ng
```
Without this, `vllm serve` crashes on `import sqlite3` with
`ImportError: /lib64/libstdc++.so.6: version 'CXXABI_1.3.15' not found`
— the compute node's system libstdc++ is older than what conda's bundled
`libicui18n` needs. The conda-forge `libstdcxx-ng` package provides a
newer one; it must then be found *first* on the linker path (see
`LD_LIBRARY_PATH` in the restart steps below).

## Restarting after a disconnect (the common case)

GPU allocations expire (2hr in our sessions) and SSH sessions drop. When
that happens, nothing needs to be re-downloaded or re-installed — just
restore the runtime environment.

### 1. Get a GPU allocation

```bash
ssh <netid>@login.hpc.ncsu.edu
srun --partition=gpu_partners --qos=short_gpu --gres=gpu:h200:1 \
  --cpus-per-task=8 --mem=64G --time=02:00:00 --pty bash
```

### 2. Restore the environment (on the compute node)

```bash
source /share/agenticsystems/$USER/project/miniforge3/bin/activate goaldecay
module load cuda/13.2
export CUDA_HOME=/usr/local/apps/cuda/cuda-13.2
export LD_LIBRARY_PATH=/share/agenticsystems/$USER/project/miniforge3/envs/goaldecay/lib:$LD_LIBRARY_PATH
cd /share/agenticsystems/$USER/project
```

### 3. Start the vLLM server

```bash
vllm serve models/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
  --host 0.0.0.0 --port 8000 --dtype bfloat16 \
  --enable-auto-tool-choice --tool-call-parser qwen3_xml
```

**Both flags are required** — without `--enable-auto-tool-choice
--tool-call-parser qwen3_xml`, tau2-bench's agent tool calls fail with
`"auto" tool choice requires --enable-auto-tool-choice and
--tool-call-parser to be set`. The parser MUST be `qwen3_xml`, not the
more commonly-recommended `hermes` — confirmed by reading
`vllm/parser/qwen3.py`'s own docstring: Qwen3's native tool-call format
is `<tool_call><function=NAME><parameter=KEY>VALUE</parameter></function></tool_call>`,
an XML format, not Hermes's JSON format. (`qwen3_coder` is an alias for
the same parser class; `qwen3_xml` is the clearer name to use.)

Leave this running in its own terminal/session. Note the compute node's
hostname (`hostname` command) — you'll need it in the next step.
Startup takes ~2-3 minutes (JIT kernel warmup, torch.compile, CUDA graph
capture). Wait for `Application startup complete.`

### 4. Point tau2-bench at it (from a login-node session)

```bash
ssh <netid>@login.hpc.ncsu.edu   # separate tab/session
source /share/agenticsystems/$USER/project/miniforge3/bin/activate goaldecay
export HOSTED_VLLM_API_BASE=http://<gpu-node-hostname>:8000/v1
cd /share/agenticsystems/$USER/project/ContextDecay/goal-decay
```

Sanity check the server is reachable:
```bash
curl -s http://<gpu-node-hostname>:8000/v1/models
```

### 5. Run the dry run (or full run)

```bash
python -m src.rollout.generate \
  --config configs/rollout_dryrun.yaml \
  --tau2-repo-path /share/agenticsystems/$USER/project/tau2-bench
```

Model strings passed to tau2-bench must use the `hosted_vllm/` prefix
(LiteLLM's convention for a self-hosted OpenAI-compatible endpoint) --
`generate.py` adds this automatically from `model.name` in the config.

## What "dry run complete" looks like

A working dry run (5 retail tasks, 1 trial) produces a mix of outcomes —
not all tasks are expected to succeed (that's real task difficulty, not
a bug):
- `user_stop` / `agent_stop` termination — normal completion, reward
  computed either way (0.0 is a legitimate "task failed" result, not an error)
- `infrastructure_error` — NOT normal. If you see this, something in the
  pipeline (vLLM connectivity, tau2-bench config, missing patch) is
  broken. Check `sim["info"]["error"]` in `results.json` for the traceback.

## Known non-blocking noise

- LiteLLM logs `This model isn't mapped yet. model=Qwen/Qwen3.8-27B,
  custom_llm_provider=hosted_vllm` repeatedly — this is only a cost-
  tracking warning (LiteLLM has no pricing data for a self-hosted model),
  not a failure. Noisy across 2,000 trajectories but harmless.

## Layer selection for activation caching (Phase 2, later — not needed yet)

```bash
python -c "
import json, sys
sys.path.insert(0, '.')
from src.extract.layer_select import classify_layers, select_cache_layers, write_layer_selection
with open('/share/agenticsystems/$USER/project/models/Qwen3.8-27B/config.json') as f:
    hf_config = json.load(f)
layer_types = classify_layers(hf_config)
choices = select_cache_layers(layer_types, min_full_attention=2)
write_layer_selection(choices, 'results/layer_selection.json')
print(choices)
"
```

If this raises `KeyError` about a missing `layer_types` field, open
`config.json` by hand, find whatever field actually distinguishes Gated
DeltaNet (linear) from Gated Attention (full) layers, and report back —
don't guess at the split from layer position, since the 16 full-attention
layers are interleaved, not contiguous.
