"""Shared synchronous execution receipts, independent of scientific adjudication.

Backends retain their own sandbox and result validation. Every call is a new
attempt, even for an identical specification. A started attempt without a
terminal receipt is UNKNOWN after interruption, never implicitly retried.
This is a local service, not a remote scheduler or an exactly-once guarantee.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar, Generic

from .workspace import exclusive_workspace

T = TypeVar("T")


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_digest(path: Path) -> str:
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def inspect_attempt(records: Path, attempt_id: str) -> dict[str, Any]:
    """Read a receipt without executing anything; reject damage, expose unknowns.

    Digests detect inconsistency, not a malicious writer who rewrites all files.
    'interrupted' reports interruption of the caller, not proof all descendants
    stopped. No status here authorizes an automatic retry.
    """
    if len(attempt_id) != 32 or any(c not in "0123456789abcdef" for c in attempt_id):
        raise ValueError("invalid_experiment_attempt_id")
    folder = Path(records) / attempt_id
    started = json.loads((folder / "started.json").read_text())
    if (not isinstance(started, dict) or started.get("attempt_id") != attempt_id
            or started.get("version") != "experiment-receipt-v1" or started.get("status") != "started"
            or started.get("spec_digest") != fingerprint(started.get("spec"))):
        raise ValueError("experiment_started_receipt_corrupt")
    terminal_path = folder / "terminal.json"
    if not terminal_path.exists():
        return dict(started, status="unknown", execution_may_have_occurred=True)
    terminal = json.loads(terminal_path.read_text())
    if (not isinstance(terminal, dict)
            or any(terminal.get(key) != value for key, value in started.items() if key != "status")
            or terminal.get("status") not in {"completed", "failed", "interrupted"}
            or ("result" in terminal and terminal.get("result_digest") != fingerprint(terminal["result"]))
            or (terminal.get("status") == "completed" and "result" not in terminal)
            or ("result" not in terminal and not terminal.get("error_type"))):
        raise ValueError("experiment_terminal_receipt_corrupt")
    return terminal


def _write_new(path: Path, value: Any) -> None:
    # Exclusive creation: never overwrite the record of an earlier execution.
    with path.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


@dataclass(frozen=True)
class ExecutionResult(Generic[T]):
    value: T
    receipt: dict[str, Any]


def execute(spec: Mapping[str, Any], *, records: Path, backend: Callable[[], T],
            summarize: Callable[[T], Mapping[str, Any]],
            succeeded: Callable[[T], bool]) -> ExecutionResult[T]:
    """Freeze intent before calling a backend; preserve its failure semantics.

    ``records`` belongs to one backend work directory: concurrent calls for it
    are refused by the shared advisory lease, while separate jobs can run in
    parallel. The receipt binds supplied identities, not unlisted dependencies. Backends
    must separately attest their data and isolation. A receipt is not evidence
    of independence, scientific validity, or a supported hypothesis.
    """
    with exclusive_workspace(records):
        return _execute(spec, records=records, backend=backend, summarize=summarize, succeeded=succeeded)


def _execute(spec: Mapping[str, Any], *, records: Path, backend: Callable[[], T],
             summarize: Callable[[T], Mapping[str, Any]],
             succeeded: Callable[[T], bool]) -> ExecutionResult[T]:
    frozen = json.loads(canonical(dict(spec)))
    if not frozen.get("backend") or not frozen.get("backend_digest"):
        raise ValueError("experiment_requires_backend_identity")
    attempt_id = uuid.uuid4().hex
    folder = Path(records) / attempt_id
    folder.mkdir(parents=True, exist_ok=False)
    identity = {"version": "experiment-receipt-v1", "attempt_id": attempt_id,
                "spec_digest": fingerprint(frozen), "spec": frozen,
                "runtime": {"python": platform.python_version(), "platform": platform.system()},
                "scientific_verdict": "not_assessed",
                "independence_verified": False}
    _write_new(folder / "started.json", dict(identity, status="started"))
    started = time.monotonic()
    try:
        value = backend()
        summary = dict(summarize(value))
        # Validate serializability before acknowledging completion.
        summary = json.loads(canonical(summary))
        status = "completed" if succeeded(value) else "failed"
    except BaseException as exc:
        _write_new(folder / "terminal.json", dict(identity, status="interrupted" if
            isinstance(exc, (KeyboardInterrupt, SystemExit)) else "failed",
            error_type=type(exc).__name__, elapsed_seconds=time.monotonic() - started))
        raise
    receipt = dict(identity, status=status, result=summary,
                   result_digest=fingerprint(summary), elapsed_seconds=time.monotonic() - started)
    _write_new(folder / "terminal.json", receipt)
    return ExecutionResult(value, receipt)
