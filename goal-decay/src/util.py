"""Shared utilities: config loading and results-directory provenance stamping.

Protocol §1.4 / §10: every results directory must carry the config that
produced it and the git commit hash, and determinism requires seeds to be
logged per run.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Config file {path} is empty")
    return cfg


def git_commit_hash(repo_root: str | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except Exception:
        return "UNKNOWN_NO_GIT"


def git_is_dirty(repo_root: str | None = None) -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        return bool(out.stdout.strip())
    except Exception:
        return True


def stamp_results_dir(results_dir: str, config: dict, repo_root: str | None = None) -> None:
    """Write config.yaml and provenance.json into results_dir. Call this
    at the start of every experiment script, before any results are written.
    """
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    provenance = {
        "git_commit": git_commit_hash(repo_root),
        "git_dirty": git_is_dirty(repo_root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "argv": sys.argv,
    }
    with open(os.path.join(results_dir, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump(provenance, f, indent=2)

    if provenance["git_dirty"]:
        print(
            f"WARNING: git working tree is dirty. Results in {results_dir} "
            "were produced with uncommitted changes; commit hash in "
            "provenance.json will not fully reproduce this run.",
            file=sys.stderr,
        )


def log_exclusion(results_dir: str, trajectory_id: str, reason: str) -> None:
    """Append a (trajectory_id, reason) row to exclusions.jsonl.

    Protocol §10: never silently drop trajectories. Every filtering step
    (failed parse, missing reward, corrupt activation cache, etc.) must
    call this instead of just `continue`-ing past the record.
    """
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, "exclusions.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"trajectory_id": trajectory_id, "reason": reason}) + "\n")
