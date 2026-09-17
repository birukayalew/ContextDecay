"""Sample a fixed, balanced set of task IDs per domain for Phase 0
corpus generation (protocol §2.1, §2.4).

Real task counts per domain in tau2-bench (pinned commit
2174a603f6d014ef94473ffa95957f6ce27100db), discovered 2026-09-17:
retail=114, airline=50, telecom=2285. Airline's 50 sets the ceiling for
balanced per-domain sampling (protocol requires no single domain
dominate the corpus, and cross-domain generalization tests in §2.5's
split_domain.json depend on comparable per-domain sample sizes).

tau2-bench's own --num-tasks flag takes tasks[:num_tasks] -- a plain
deterministic slice in file order, NOT a random sample and NOT
seed-dependent (verified by reading src/tau2/runner/helpers.py). For a
domain with thousands of tasks (telecom), "first N in file order" risks
an unrepresentative, systematically biased slice. This script samples
task IDs ourselves with a fixed, logged seed, so the same 50 IDs per
domain are used consistently across all trial seeds, and the exact
sample is committed to git and auditable -- matching the protocol's
"frozen splits" philosophy (§2.5).

Run this ONCE, commit the output, and never resample -- resampling
after seeing results is exactly the kind of thing that turns a
pre-registered corpus into a fishing expedition.

Usage (on the GPU server, needs the tau2-bench checkout, no GPU needed):
    python -m src.rollout.sample_tasks \
        --tau2-repo-path /path/to/tau2-bench \
        --n-per-domain 50 \
        --seed 42 \
        --out configs/sampled_task_ids.json
"""
from __future__ import annotations

import argparse
import json
import os
import random


DOMAIN_TASK_FILES = {
    "retail": "data/tau2/domains/retail/tasks.json",
    "airline": "data/tau2/domains/airline/tasks.json",
    "telecom": "data/tau2/domains/telecom/tasks.json",
}


def load_task_ids(tau2_repo_path: str, domain: str) -> list[str]:
    path = os.path.join(tau2_repo_path, DOMAIN_TASK_FILES[domain])
    with open(path, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    ids = [t["id"] for t in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate task ids found in {path}")
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau2-repo-path", required=True)
    ap.add_argument("--n-per-domain", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise FileExistsError(
            f"{args.out} already exists. This sample is frozen once "
            "created (protocol §2.5 philosophy) -- delete it explicitly "
            "if you really intend to resample, and note why in the "
            "commit message."
        )

    rng = random.Random(args.seed)
    result = {
        "seed": args.seed,
        "n_per_domain": args.n_per_domain,
        "tau2_task_file_counts": {},
        "sampled_task_ids": {},
    }

    for domain in DOMAIN_TASK_FILES:
        all_ids = load_task_ids(args.tau2_repo_path, domain)
        result["tau2_task_file_counts"][domain] = len(all_ids)
        n = min(args.n_per_domain, len(all_ids))
        if n < args.n_per_domain:
            print(
                f"WARNING: domain {domain} has only {len(all_ids)} tasks, "
                f"requested {args.n_per_domain}. Using all {n}."
            )
        sampled = sorted(rng.sample(all_ids, n))
        result["sampled_task_ids"][domain] = sampled
        print(f"{domain}: sampled {len(sampled)} of {len(all_ids)} tasks")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}. Commit this file -- it is the frozen task sample.")


if __name__ == "__main__":
    main()
