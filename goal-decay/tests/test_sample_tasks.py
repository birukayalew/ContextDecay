import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.rollout.sample_tasks import load_task_ids


def make_tasks_file(tmp_path, domain, n):
    d = tmp_path / f"data/tau2/domains/{domain}"
    d.mkdir(parents=True)
    tasks = [{"id": f"{domain}_{i}"} for i in range(n)]
    (d / "tasks.json").write_text(json.dumps(tasks))
    return tmp_path


def test_load_task_ids_reads_all_ids(tmp_path):
    root = make_tasks_file(tmp_path, "retail", 5)
    ids = load_task_ids(str(root), "retail")
    assert ids == [f"retail_{i}" for i in range(5)]


def test_load_task_ids_rejects_duplicates(tmp_path):
    d = tmp_path / "data/tau2/domains/retail"
    d.mkdir(parents=True)
    (d / "tasks.json").write_text(json.dumps([{"id": "a"}, {"id": "a"}]))
    with pytest.raises(ValueError):
        load_task_ids(str(tmp_path), "retail")


def test_sampling_is_deterministic_given_seed(tmp_path):
    import random
    root = make_tasks_file(tmp_path, "telecom", 100)
    ids = load_task_ids(str(root), "telecom")

    sample_a = sorted(random.Random(42).sample(ids, 50))
    sample_b = sorted(random.Random(42).sample(ids, 50))
    assert sample_a == sample_b


def test_sampling_respects_cap_when_fewer_tasks_than_requested(tmp_path):
    root = make_tasks_file(tmp_path, "airline", 30)
    ids = load_task_ids(str(root), "airline")
    n = min(50, len(ids))
    assert n == 30
