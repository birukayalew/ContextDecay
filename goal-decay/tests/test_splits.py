import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.probes.splits import Split, SplitLeakageError, assert_no_step_level_leakage, load_split


def test_valid_split_ok():
    s = Split("test", frozenset(["t1", "t2"]), frozenset(["t3"]), frozenset(["t4"]))
    assert s.assign("t1") == "train"
    assert s.assign("t4") == "test"


def test_overlapping_split_rejected():
    with pytest.raises(SplitLeakageError):
        Split("bad", frozenset(["t1", "t2"]), frozenset(["t2"]), frozenset(["t4"]))


def test_load_split_roundtrip(tmp_path):
    path = str(tmp_path / "split.json")
    with open(path, "w") as f:
        json.dump({
            "name": "split_random",
            "train_task_ids": ["a", "b"],
            "val_task_ids": ["c"],
            "test_task_ids": ["d"],
        }, f)
    s = load_split(path)
    assert s.assign("a") == "train"
    assert s.assign("d") == "test"


def test_load_split_missing_keys_rejected(tmp_path):
    path = str(tmp_path / "bad.json")
    with open(path, "w") as f:
        json.dump({"name": "x", "train_task_ids": []}, f)
    with pytest.raises(ValueError):
        load_split(path)


def test_step_level_leakage_detected():
    s = Split("t", frozenset(["taskA"]), frozenset(), frozenset(["taskB"]))
    # A record claims taskA is in the test split, contradicting the frozen split.
    records = [{"task_id": "taskA", "split_assignment": "test"}]
    with pytest.raises(SplitLeakageError):
        assert_no_step_level_leakage(records, s)


def test_consistent_records_pass():
    s = Split("t", frozenset(["taskA"]), frozenset(), frozenset(["taskB"]))
    records = [
        {"task_id": "taskA", "split_assignment": "train"},
        {"task_id": "taskA", "split_assignment": "train"},  # same task, multiple steps -- fine
        {"task_id": "taskB", "split_assignment": "test"},
    ]
    assert_no_step_level_leakage(records, s)  # should not raise


def test_unknown_task_id_raises():
    s = Split("t", frozenset(["taskA"]), frozenset(), frozenset())
    with pytest.raises(KeyError):
        s.assign("taskZ")
