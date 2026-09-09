# goal-decay

Implementation for "Goal-Information Decay and Repair Routing for LLM Agents".
Full experimental design, kill gates, and phase-by-phase plan: see
[`../EXPERIMENT_PROTOCOL.md`](../EXPERIMENT_PROTOCOL.md). Read that before
touching this code — it defines the invariants this code enforces.

## Split: local vs. GPU server

This repo is developed on a Windows machine with no local GPU. Two classes
of work:

- **Local (this machine, CPU-only):** repo scaffolding, config/ledger/split
  logic and their tests, Gate K1 (cosine-similarity baseline), stats/eval
  code, the router logic, paper writing.
- **Remote (GPU server, run by a collaborator):** trajectory generation
  (vLLM), activation extraction (HF Transformers + hooks), anything that
  loads Qwen3.8-27B or another 27B-class checkpoint.

Remote-bound work is specified as an exact command block in
`README_REMOTE.md` (or in the PR/message accompanying it) — copy that
verbatim to whoever is running the GPU box rather than improvising flags.

## Layout

See protocol §1.4. `configs/` holds one YAML per experiment (no hardcoded
hyperparameters in code). `data/raw/` is immutable once written. `results/`
gets one subdir per run, each stamped with `config.yaml` + `provenance.json`
(git commit hash, dirty-tree flag, timestamp) via `src/util.stamp_results_dir`.

## Invariants enforced in code, not just convention

- `src/ledger/ledger.py` — append-only; reopening an existing ledger file
  for writing, or appending an out-of-order step, raises.
- `src/probes/splits.py` — frozen splits are grouped by `task_id`; any
  attempt to evaluate a probe on a split that leaks a task across
  train/val/test raises `SplitLeakageError`.
- `src/eval/bootstrap.py` — cluster bootstrap resamples whole tasks, never
  individual steps.

## Running tests

```
pip install -r requirements.txt
pytest tests/ -v
```

## Gate K1 (first thing to run against real data)

```
python -m src.eval.gate_k1 --config configs/gate_k1.yaml
```

CPU-only, no GPU server needed. Requires `data/raw/trajectories.jsonl` and
`data/splits/split_random.json` to exist first (Phase 0).
