"""Gate K1 -- the kill test (protocol §3).

Runs BEFORE anything else in the project. No probes, no model internals:
cos_sim(embed(goal_text), embed(context_at_step_t)), evaluated for AUROC
against trajectory failure at 25/50/75/100% of progress.

CPU-only. Runnable locally -- does not need the GPU server.

Usage:
    python -m src.eval.gate_k1 --config configs/gate_k1.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.eval.bootstrap import cluster_bootstrap_ci
from src.probes.splits import load_split
from src.util import load_config, log_exclusion, stamp_results_dir


def load_trajectories(path: str) -> list[dict]:
    """Each line: {task_id, seed, goal_text, steps: [{step_idx, context_text}], success}"""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def context_at_fraction(steps: list[dict], fraction: float) -> str | None:
    if not steps:
        return None
    idx = min(len(steps) - 1, int(round(fraction * (len(steps) - 1))))
    return steps[idx]["context_text"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    stamp_results_dir(cfg["output_dir"], cfg)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg["embedder"]["name"])

    trajectories = load_trajectories(cfg["data"]["trajectories_path"])
    split = load_split(cfg["data"]["split_path"])

    # Gate K1 is evaluated on the frozen test split only -- it is a
    # baseline check, not something we get to tune, so there is nothing
    # to fit on train.
    kept = []
    for r in trajectories:
        try:
            if split.assign(r["task_id"]) != "test":
                continue
        except KeyError:
            log_exclusion(cfg["output_dir"], r.get("task_id", "UNKNOWN"), "task_id not in split file")
            continue
        if "steps" not in r or not r["steps"]:
            log_exclusion(cfg["output_dir"], r["task_id"], "no steps recorded")
            continue
        if "success" not in r:
            log_exclusion(cfg["output_dir"], r["task_id"], "missing success label")
            continue
        kept.append(r)

    results = {}
    for frac in cfg["progress_fractions"]:
        task_ids, y_true, contexts, goals = [], [], [], []
        for r in kept:
            ctx = context_at_fraction(r["steps"], frac)
            if ctx is None:
                continue
            task_ids.append(r["task_id"])
            y_true.append(0 if r["success"] else 1)  # 1 = failure, matches "predicting failure"
            contexts.append(ctx)
            goals.append(r["goal_text"])

        if len(set(y_true)) < 2:
            results[f"frac_{frac}"] = {"error": "degenerate: only one class present"}
            continue

        goal_emb = model.encode(goals, normalize_embeddings=True, show_progress_bar=False)
        ctx_emb = model.encode(contexts, normalize_embeddings=True, show_progress_bar=False)
        cos = np.sum(goal_emb * ctx_emb, axis=1)
        score = -cos  # lower goal-context similarity -> predicts higher failure prob

        task_ids = np.array(task_ids)
        y_true = np.array(y_true)

        ci = cluster_bootstrap_ci(
            task_ids, y_true, score, roc_auc_score,
            n_resamples=cfg["bootstrap"]["n_resamples"], seed=cfg["seed"],
        )
        results[f"frac_{frac}"] = ci
        print(f"progress={frac:.2f}  AUROC={ci['point']:.3f}  CI=[{ci['ci_lo']:.3f}, {ci['ci_hi']:.3f}]  n_tasks={ci['n_tasks']}")

    auroc_50 = results.get("frac_0.5", {}).get("point")
    decision = None
    if auroc_50 is not None:
        th = cfg["decision_thresholds"]
        if auroc_50 >= th["reconsider_at_or_above"]:
            decision = "RECONSIDER: cosine baseline already works (>=0.65). Probe must beat it by >=0.05 AUROC, non-overlapping CIs."
        elif auroc_50 < th["proceed_below"]:
            decision = "PROCEED: cosine baseline is weak (<0.60). This becomes the headline baseline."
        else:
            decision = f"PROCEED WITH CAUTION: in between. Probe bar is now {auroc_50 + 0.05:.3f}."
        print(f"\nDECISION: {decision}")

    out = {"per_fraction": results, "auroc_at_50pct": auroc_50, "decision": decision}
    with open(os.path.join(cfg["output_dir"], "gate_k1_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\nWrote results to {cfg['output_dir']}/gate_k1_results.json regardless of outcome.")


if __name__ == "__main__":
    main()
