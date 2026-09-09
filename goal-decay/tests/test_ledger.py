import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.ledger.ledger import ExecutionLedger, LedgerEntry, LedgerMutationError


def make_entry(step, verified=True):
    return LedgerEntry(
        step=step, type="tool_call", tool="update_order",
        args={"id": step}, result_ref=f"blob://{step}",
        verified=verified, reversible=False,
    )


def test_append_and_read_back(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ledger = ExecutionLedger(path=path)
    ledger.append(make_entry(1))
    ledger.append(make_entry(2))
    ledger.close()

    reloaded = ExecutionLedger.load(path)
    assert len(reloaded) == 2
    assert [e.step for e in reloaded] == [1, 2]


def test_cannot_reopen_existing_file_for_writing(tmp_path):
    path = str(tmp_path / "ledger.jsonl")
    ExecutionLedger(path=path).close()
    with pytest.raises(LedgerMutationError):
        ExecutionLedger(path=path)


def test_out_of_order_step_rejected(tmp_path):
    ledger = ExecutionLedger(path=str(tmp_path / "ledger.jsonl"))
    ledger.append(make_entry(5))
    with pytest.raises(LedgerMutationError):
        ledger.append(make_entry(3))


def test_entries_are_frozen_dataclasses():
    entry = make_entry(1)
    with pytest.raises(Exception):
        entry.step = 99  # frozen=True must forbid this


def test_as_of_does_not_mutate(tmp_path):
    ledger = ExecutionLedger(path=str(tmp_path / "ledger.jsonl"))
    ledger.append(make_entry(1))
    ledger.append(make_entry(2))
    snapshot = ledger.as_of(1)
    assert len(snapshot) == 1
    assert len(ledger) == 2  # original untouched


def test_verified_effects_filters(tmp_path):
    ledger = ExecutionLedger(path=str(tmp_path / "ledger.jsonl"))
    ledger.append(make_entry(1, verified=True))
    ledger.append(make_entry(2, verified=False))
    assert len(ledger.verified_effects()) == 1
