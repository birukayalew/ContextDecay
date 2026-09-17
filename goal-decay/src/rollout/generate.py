"""Phase 0 trajectory generation (protocol §2.4). GPU-server script --
requires vllm, tau2-bench (uv sync'd, pinned commit), and CUDA. Does NOT
run on the Windows dev machine.

Wraps `tau2 run` per domain/seed, parses tau2-bench's `results.json`
output, and re-emits one JSONL record per trajectory in the schema the
rest of this repo expects (Gate K1, probes, ledger construction all read
this format).

Confirmed tau2-bench output shape (verified 2026-09-17 against a real
--save-to run on the GPU server, commit 2174a603f6d014ef94473ffa95957f6ce27100db):

    data/simulations/<save_to>/results.json  (one file per `tau2 run` call)
        {
          "timestamp": ..., "info": {...}, "simulation_index": {...},
          "tasks": [ {id, description, user_scenario, ticket,
                      initial_state, evaluation_criteria, issues,
                      required_documents, user_tools}, ... ],
          "simulations": [
            {
              "id", "task_id", "trial", "seed", "duration",
              "termination_reason", "agent_cost", "user_cost",
              "reward_info": {
                  "reward": 0.0 or 1.0,
                  "db_check": {"db_match": bool, "db_reward": float},
                  "action_checks": [ {"action": {...}, "action_match": bool,
                                       "action_reward": float,
                                       "tool_type": "read"|"write"}, ... ]
              },
              "messages": [
                {
                  "role": "assistant"|"user"|"tool",
                  "content": str,  # NOTE: for Qwen3.8-27B this includes the
                                   # model's raw reasoning trace inline, not
                                   # just the final utterance -- see note below
                  "tool_calls": [ {"id", "name", "arguments", "requestor"} ] or null,
                  "turn_idx": int,
                  "timestamp": str,
                  "cost": float,
                  "usage": {"completion_tokens": int, "prompt_tokens": int} or null,
                  "error": bool,  # only present on role="tool" messages
                }, ...
              ],
              ...
            }, ...
          ]
        }

Notes worth keeping in mind downstream (paper §8 threats-to-validity
candidates):
  - `usage` is null on non-assistant turns (user-simulator messages don't
    report token counts in this tau2-bench version) -- cumulative_tokens
    is therefore computed only from assistant-turn usage, which
    undercounts user-simulator tokens actually in context. Note this as
    an approximation, not a silent gap.
  - `content` on assistant turns is NOT pure "what was said" -- Qwen3.8-27B
    emits visible chain-of-thought reasoning inline before/around the
    actual reply and before tool calls. This is arguably part of "agent
    state" for the G-probe (protocol §4), but anyone building a goal-slot
    decoder off `content` alone needs to know reasoning text is mixed in.

Usage (dry run):
    python -m src.rollout.generate --config configs/rollout_dryrun.yaml --tau2-repo-path /path/to/tau2-bench

Usage (full run, after dry run is verified):
    python -m src.rollout.generate --config configs/rollout_full.yaml --tau2-repo-path /path/to/tau2-bench
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from typing import Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.util import git_commit_hash, load_config, log_exclusion, stamp_results_dir


def get_tau2_commit(repo_path: str) -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo_path, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def run_tau2(domain: str, agent_llm: str, user_llm: str,
             tau2_repo_path: str, seed: int, save_to: str,
             num_tasks: int | None = None, task_ids: list[str] | None = None,
             concurrency: int | None = None) -> str:
    """Invoke `tau2 run` for one domain/seed. Returns the path to the
    results.json it writes (data/simulations/<save_to>/results.json,
    resolved relative to tau2_repo_path since tau2 writes there).

    tau2-bench calls out to the LLM via LiteLLM's hosted_vllm provider;
    the caller must have HOSTED_VLLM_API_BASE set in the environment
    pointing at a running `vllm serve` instance with
    --enable-auto-tool-choice --tool-call-parser qwen3_xml (confirmed
    correct parser for Qwen3.8-27B / Qwen3_5ForConditionalGeneration --
    NOT hermes, which is for a different tool-call format).

    Pass exactly one of num_tasks or task_ids. task_ids is preferred for
    any domain with more tasks than we intend to sample: tau2-bench's
    --num-tasks does tasks[:num_tasks], a plain deterministic slice in
    file order (verified against src/tau2/runner/helpers.py) -- NOT a
    random or representative sample, and NOT seed-dependent. For a
    domain like telecom (2,285 tasks), "first N in file order" risks
    systematic bias; task_ids lets us pass our own pre-sampled,
    documented, git-committed set (see sample_tasks.py).
    """
    if num_tasks is not None and task_ids is not None:
        raise ValueError("pass at most one of num_tasks, task_ids")

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", domain,
        "--agent-llm", agent_llm,
        "--user-llm", user_llm,
        "--num-trials", "1",
        "--seed", str(seed),
        "--save-to", save_to,
    ]
    if num_tasks is not None:
        cmd += ["--num-tasks", str(num_tasks)]
    if task_ids is not None:
        cmd += ["--task-ids"] + list(task_ids)
    if concurrency is not None:
        cmd += ["--max-concurrency", str(concurrency)]

    print(f"[rollout] running: {' '.join(cmd)}", file=sys.stderr)
    result = subprocess.run(cmd, cwd=tau2_repo_path, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"tau2 run failed for domain={domain} seed={seed}:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    print(result.stdout[-4000:], file=sys.stderr)  # tail only, this is verbose

    results_path = os.path.join(tau2_repo_path, "data", "simulations", save_to, "results.json")
    if not os.path.exists(results_path):
        raise RuntimeError(f"Expected tau2 output at {results_path}, not found.")
    return results_path


def _message_step_fields(msg: dict) -> dict:
    """Extract the protocol §2.4 required per-step fields from one
    tau2-bench message dict.

    Confirmed against real trial data: `role: "tool"` messages carry an
    explicit `error: bool` field (verified both error=false cases; a
    true case has not yet been observed in a sample trial, but the field
    is unambiguously present and typed, so no guessing is involved).
    Non-tool messages (assistant/user) have no `error` field and are not
    tool errors by construction -- tool_errored is None for those rows,
    not False, so "not a tool call" stays distinguishable from "tool
    call that succeeded."
    """
    usage = msg.get("usage") or {}
    tool_calls = msg.get("tool_calls") or []
    role = msg["role"]
    return {
        "step_idx": msg["turn_idx"],
        "message_role": role,
        "tool_name": tool_calls[0]["name"] if tool_calls else None,
        "tool_errored": msg.get("error") if role == "tool" else None,
        "tokens_added_this_step": usage.get("completion_tokens", 0),
        # prompt_tokens on an assistant turn already reflects cumulative
        # context at that point; fall back to running total for turns
        # with no usage reported (user/tool turns in this tau2-bench version).
        "_prompt_tokens": usage.get("prompt_tokens"),
        "content": msg.get("content"),
        "tool_calls_raw": tool_calls,
    }


def parse_tau2_trial(sim: dict, task: dict, domain: str, seed: int, tau2_commit: str) -> dict:
    """Convert one tau2-bench `simulations[i]` entry (+ its matching task
    spec) into our trajectory schema. Raises on any structural surprise
    rather than guessing -- protocol §10: never silently drop/corrupt.
    """
    messages = sim["messages"]
    if not messages:
        raise ValueError(f"simulation {sim.get('id')} has no messages")

    steps = []
    running_tokens = 0
    for msg in messages:
        f = _message_step_fields(msg)
        if f["_prompt_tokens"] is not None:
            # cumulative context size as reported at this assistant turn
            running_tokens = f["_prompt_tokens"] + f["tokens_added_this_step"]
        else:
            running_tokens += f["tokens_added_this_step"]

        steps.append({
            "step_idx": f["step_idx"],
            "cumulative_tokens": running_tokens,
            "tokens_added_this_step": f["tokens_added_this_step"],
            "tool_name": f["tool_name"],
            "tool_errored": f["tool_errored"],
            "message_role": f["message_role"],
            "context_text": f["content"] or "",
        })

    reward_info = sim["reward_info"]
    return {
        "task_id": sim["task_id"],
        "domain": domain,
        "seed": seed,
        "trial": sim.get("trial"),
        "goal_text": task.get("description", ""),
        "full_message_list": messages,
        "programmatic_reward": reward_info["reward"],
        "reward_info": reward_info,
        "termination_reason": sim.get("termination_reason"),
        "steps": steps,
        "tau2_commit": tau2_commit,
    }


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

    if "HOSTED_VLLM_API_BASE" not in os.environ:
        raise RuntimeError(
            "HOSTED_VLLM_API_BASE is not set. Start `vllm serve` on the "
            "GPU node first (--enable-auto-tool-choice --tool-call-parser "
            "qwen3_xml) and export HOSTED_VLLM_API_BASE=http://<node>:8000/v1 "
            "before running this script."
        )

    is_dry_run = "dry_run" in cfg
    run_cfg = cfg["dry_run"] if is_dry_run else cfg["full_run"]
    num_trials = run_cfg.get("num_trials", 1)
    num_tasks = run_cfg.get("num_tasks")
    concurrency = run_cfg.get("concurrency")

    sampled_task_ids: dict[str, list[str]] = {}
    sample_file = run_cfg.get("sampled_task_ids_file")
    if sample_file is not None:
        if num_tasks is not None:
            raise ValueError(
                "config sets both num_tasks and sampled_task_ids_file -- "
                "pass at most one. sampled_task_ids_file takes precedence "
                "for reproducible, non-file-order-biased sampling."
            )
        with open(sample_file, "r", encoding="utf-8") as f:
            sample_data = json.load(f)
        sampled_task_ids = sample_data["sampled_task_ids"]

    out_path = cfg["raw_trajectories_path"]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    if os.path.exists(out_path):
        raise FileExistsError(
            f"{out_path} already exists. data/raw/ is immutable once "
            "written (protocol §10) -- move or rename the existing file "
            "if you intend to regenerate."
        )

    agent_llm = f"hosted_vllm/{cfg['model']['name']}"
    user_llm = f"hosted_vllm/{cfg['model']['name']}"
    run_tag = cfg["experiment"]

    n_written = 0
    with open(out_path, "w", encoding="utf-8") as out_f:
        for domain in cfg["domains"]:
            domain_task_ids = sampled_task_ids.get(domain) if sampled_task_ids else None
            for seed in range(num_trials):
                save_to = f"{run_tag}_{domain}_seed{seed}_{uuid.uuid4().hex[:8]}"
                results_path = run_tau2(
                    domain=domain,
                    agent_llm=agent_llm,
                    user_llm=user_llm,
                    num_tasks=num_tasks if domain_task_ids is None else None,
                    task_ids=domain_task_ids,
                    concurrency=concurrency,
                    tau2_repo_path=args.tau2_repo_path,
                    seed=seed,
                    save_to=save_to,
                )

                with open(results_path, "r", encoding="utf-8") as f:
                    results = json.load(f)

                tasks_by_id = {t["id"]: t for t in results["tasks"]}

                for sim in results["simulations"]:
                    trial_key = f"{domain}/{save_to}/{sim.get('id')}"
                    try:
                        task = tasks_by_id[sim["task_id"]]
                        record = parse_tau2_trial(sim, task, domain, seed, tau2_commit)
                    except Exception as e:
                        log_exclusion(cfg["output_dir"], trial_key, f"parse error: {e}")
                        continue

                    missing = [
                        f for f in cfg["logging"]["required_trajectory_fields"]
                        if f not in record
                    ]
                    if missing:
                        log_exclusion(cfg["output_dir"], trial_key, f"missing fields: {missing}")
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
