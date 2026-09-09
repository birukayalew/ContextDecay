"""Phase 0 trajectory generation (protocol §2.4). GPU-server script --
requires vllm, tau2-bench (uv sync'd, pinned commit), and CUDA. Does NOT
run on the Windows dev machine.

Wraps `tau2 run` per domain/seed, parses tau2-bench's own trial logs, and
re-emits one JSONL record per trajectory in the schema the rest of this
repo expects (Gate K1, probes, ledger construction all read this format).

Usage (dry run):
    python -m src.rollout.generate --config configs/rollout_dryrun.yaml

Usage (full run, after dry run is verified):
    python -m src.rollout.generate --config configs/rollout_full.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.util import git_commit_hash, load_config, log_exclusion, stamp_results_dir


def get_tau2_commit(repo_path: str) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def run_tau2(domain: str, agent_llm: str, user_llm: str, num_trials: int,
             num_tasks: int | None, tau2_repo_path: str, seed: int) -> str:
    """Invoke `tau2 run` for one domain, return path to its raw output dir.

    tau2-bench writes its own results; this function only shells out and
    locates the output. Adjust `--seed` flag name if tau2-bench's actual
    CLI differs -- verify with `tau2 run --help` on the server first.
    """
    cmd = [
        "tau2", "run",
        "--domain", domain,
        "--agent-llm", agent_llm,
        "--user-llm", user_llm,
        "--num-trials", str(num_trials),
        "--seed", str(seed),
    ]
    if num_tasks is not None:
        cmd += ["--num-tasks", str(num_tasks)]

    print(f"[rollout] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=tau2_repo_path, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"tau2 run failed for domain={domain} seed={seed}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print(result.stdout, file=sys.stderr)
    # tau2-bench's output location must be confirmed on the server --
    # placeholder path, update after first dry-run invocation.
    return os.path.join(tau2_repo_path, "results", domain)


def parse_tau2_trial(trial_path: str, domain: str, seed: int) -> dict | None:
    """Parse one tau2-bench trial log into our trajectory schema.

    This is a stub -- the exact structure of tau2-bench's trial JSON must
    be inspected on the server (it will be present after the first `tau2
    run` invocation) and this function filled in to match. Field mapping
    required, per protocol §2.4:

        step_idx              <- index into the message/tool-call sequence
        cumulative_tokens     <- running total, computed here if tau2
                                  doesn't report it directly
        tokens_added_this_step<- per-step delta
        tool_name              <- from the tool-call message
        tool_errored           <- bool, from the tool return status
        message_role           <- user / agent / tool

        task_id, domain, seed, goal_text, full_message_list,
        programmatic_reward   <- tau2's own pass/fail grading (1/0)

    Raise, do not silently return a malformed record -- the caller logs
    exclusions explicitly (protocol §10: never silently drop trajectories).
    """
    raise NotImplementedError(
        "Fill in after inspecting one real tau2-bench trial log on the "
        "server. Do not guess the schema -- this is exactly the kind of "
        "thing the dry run at --num-tasks 5 --num-trials 1 exists to "
        "surface. Print the raw trial JSON structure first."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--tau2-repo-path", required=True, help="Local path to the sierra-research/tau2-bench checkout on this server")
    args = ap.parse_args()

    cfg = load_config(args.config)
    stamp_results_dir(cfg["output_dir"], cfg)

    tau2_commit = get_tau2_commit(args.tau2_repo_path)
    pinned = cfg["benchmark"].get("pinned_commit")
    if pinned and pinned != tau2_commit:
        raise RuntimeError(
            f"Config pins tau2-bench commit {pinned!r} but the checkout "
            f"at {args.tau2_repo_path} is at {tau2_commit!r}. Checkout "
            "the pinned commit before generating trajectories -- protocol "
            "requires all trajectories come from one named benchmark version."
        )
    if not pinned:
        print(
            f"WARNING: config has no pinned_commit set. Running against "
            f"{tau2_commit}. Set benchmark.pinned_commit to this value "
            "in the config before the full run.",
            file=sys.stderr,
        )

    is_dry_run = "dry_run" in cfg
    run_cfg = cfg["dry_run"] if is_dry_run else cfg["full_run"]
    num_trials = run_cfg.get("num_trials", 1)
    num_tasks = run_cfg.get("num_tasks")

    out_path = cfg["raw_trajectories_path"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        raise FileExistsError(
            f"{out_path} already exists. data/raw/ is immutable once "
            "written (protocol §10) -- move or rename the existing file "
            "if you intend to regenerate."
        )

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for domain in cfg["domains"]:
            for seed in range(num_trials):
                trial_dir = run_tau2(
                    domain=domain,
                    agent_llm=cfg["model"]["name"],
                    user_llm=cfg["model"]["name"],
                    num_trials=1,  # one seed at a time for clean per-seed dirs
                    num_tasks=num_tasks,
                    tau2_repo_path=args.tau2_repo_path,
                    seed=seed,
                )
                for fname in sorted(os.listdir(trial_dir)):
                    trial_path = os.path.join(trial_dir, fname)
                    try:
                        record = parse_tau2_trial(trial_path, domain, seed)
                    except NotImplementedError:
                        raise
                    except Exception as e:
                        log_exclusion(cfg["output_dir"], f"{domain}/{fname}", f"parse error: {e}")
                        continue
                    if record is None:
                        log_exclusion(cfg["output_dir"], f"{domain}/{fname}", "parser returned None")
                        continue

                    missing = [
                        f for f in cfg["logging"]["required_trajectory_fields"]
                        if f not in record
                    ]
                    if missing:
                        log_exclusion(cfg["output_dir"], f"{domain}/{fname}", f"missing fields: {missing}")
                        continue

                    out_f.write(json.dumps(record) + "\n")
                    n_written += 1

    print(f"[rollout] wrote {n_written} trajectories to {out_path}")
    print(f"[rollout] tau2-bench commit: {tau2_commit}")
    print(f"[rollout] goal-decay commit: {git_commit_hash('.')}")

    if is_dry_run:
        print(
            "\nDRY RUN COMPLETE. Do not proceed to the full run yet.\n"
            f"Inspect {out_path} by hand -- print the first record's keys "
            "and one full step, confirm every required_step_fields entry "
            "is populated (not just present), and confirm "
            "programmatic_reward is 0/1 and matches tau2-bench's own "
            "grading output for the same trials before scaling up."
        )


if __name__ == "__main__":
    main()
