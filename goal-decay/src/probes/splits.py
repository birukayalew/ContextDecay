"""Frozen split loading with a hard task-level grouping assertion.

Protocol §2.5 / §10: splits must be grouped by task_id, never by step, and
never by trajectory if a task has multiple seeds -- otherwise steps (or
seeds) of the same task leak across train/test and probe AUROC becomes
pure leakage. This module is the single choke point every probe trainer
must go through; it refuses to hand back a split that violates the
constraint.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


class SplitLeakageError(RuntimeError):
    """Raised when a split file or a proposed (train, test) pairing shares task_ids."""


@dataclass(frozen=True)
class Split:
    name: str
    train_task_ids: frozenset
    val_task_ids: frozenset
    test_task_ids: frozenset

    def __post_init__(self):
        pairs = [
            ("train", self.train_task_ids),
            ("val", self.val_task_ids),
            ("test", self.test_task_ids),
        ]
        for (name_a, a), (name_b, b) in [
            (pairs[0], pairs[1]),
            (pairs[0], pairs[2]),
            (pairs[1], pairs[2]),
        ]:
            overlap = a & b
            if overlap:
                raise SplitLeakageError(
                    f"Split '{self.name}': {name_a} and {name_b} share "
                    f"{len(overlap)} task_id(s): {sorted(overlap)[:5]}..."
                )

    def assign(self, task_id: str) -> str:
        if task_id in self.train_task_ids:
            return "train"
        if task_id in self.val_task_ids:
            return "val"
        if task_id in self.test_task_ids:
            return "test"
        raise KeyError(f"task_id {task_id!r} not present in split '{self.name}'")


def load_split(path: str) -> Split:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    required = {"name", "train_task_ids", "val_task_ids", "test_task_ids"}
    missing = required - d.keys()
    if missing:
        raise ValueError(f"Split file {path} missing keys: {missing}")
    return Split(
        name=d["name"],
        train_task_ids=frozenset(d["train_task_ids"]),
        val_task_ids=frozenset(d["val_task_ids"]),
        test_task_ids=frozenset(d["test_task_ids"]),
    )


def assert_no_step_level_leakage(records: list[dict], split: Split) -> None:
    """Given flat per-step records with 'task_id' and 'split_assignment'
    fields, verify every record's assignment matches the frozen split and
    that no task_id appears under more than one assignment.
    """
    seen: dict[str, str] = {}
    for r in records:
        tid = r["task_id"]
        expected = split.assign(tid)
        actual = r.get("split_assignment")
        if actual is not None and actual != expected:
            raise SplitLeakageError(
                f"Record for task_id={tid} carries split_assignment="
                f"{actual!r} but the frozen split says {expected!r}."
            )
        if tid in seen and seen[tid] != expected:
            raise SplitLeakageError(
                f"task_id={tid} assigned to both {seen[tid]!r} and "
                f"{expected!r} -- this should be impossible and indicates "
                "a corrupted split file."
            )
        seen[tid] = expected
