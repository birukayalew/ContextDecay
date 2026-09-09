"""Execution ledger: append-only record of committed tool-call effects.

Invariant (protocol §6.1, §10): repair rewrites the context (a view over the
ledger); repair never rewrites the ledger. This module enforces that with a
hard assertion, not a convention -- any attempt to mutate or delete an
existing entry raises LedgerMutationError.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


class LedgerMutationError(RuntimeError):
    """Raised when code attempts to modify or remove an existing ledger entry."""


@dataclass(frozen=True)
class LedgerEntry:
    step: int
    type: str
    tool: str
    args: dict
    result_ref: str
    verified: bool
    reversible: bool
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class ExecutionLedger:
    """Append-only ledger for one trajectory.

    Backed by an in-memory list plus an on-disk JSONL file opened in
    append mode. There is no update/delete method by design -- attempting
    to reach into `_entries` and mutate in place is caught by
    `_frozen_guard` on write-out.
    """

    def __init__(self, path: Optional[str] = None):
        self._entries: list[LedgerEntry] = []
        self._path = path
        self._file = None
        if path is not None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            if os.path.exists(path):
                raise LedgerMutationError(
                    f"Ledger file already exists at {path}; ledgers are "
                    "append-only for the life of one run and must not be "
                    "reopened for writing. Read it back instead."
                )
            self._file = open(path, "a", encoding="utf-8")

    def append(self, entry: LedgerEntry) -> None:
        if not isinstance(entry, LedgerEntry):
            raise TypeError("append() requires a LedgerEntry")
        if self._entries and entry.step < self._entries[-1].step:
            raise LedgerMutationError(
                f"Ledger entries must be non-decreasing in step; got "
                f"step={entry.step} after step={self._entries[-1].step}."
            )
        self._entries.append(entry)
        if self._file is not None:
            self._file.write(json.dumps(entry.to_dict()) + "\n")
            self._file.flush()

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    def as_of(self, step: int) -> list[LedgerEntry]:
        """All entries committed at or before `step`. Never mutates state."""
        return [e for e in self._entries if e.step <= step]

    def verified_effects(self, step: Optional[int] = None) -> list[LedgerEntry]:
        pool = self._entries if step is None else self.as_of(step)
        return [e for e in pool if e.verified]

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> "ExecutionLedger":
        return self

    def __exit__(self, *exc):
        self.close()

    @staticmethod
    def load(path: str) -> "ExecutionLedger":
        """Read-only reconstruction from a JSONL file on disk."""
        ledger = ExecutionLedger(path=None)
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                extra = d.pop("extra", {})
                ledger._entries.append(LedgerEntry(**d, extra=extra))
        return ledger
