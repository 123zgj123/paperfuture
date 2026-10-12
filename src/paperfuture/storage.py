"""Atomic, content-addressed artifacts and process-safe run ownership."""
from __future__ import annotations

import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator


class IntegrityError(ValueError):
    pass


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def timestamp(value: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError("timestamps must be ISO 8601 strings with a timezone")
    result = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError(f"timestamp needs an explicit timezone: {value}")
    return result.astimezone(dt.timezone.utc)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".writing-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def seal(path: Path, payload: Any) -> Any:
    envelope = {"sha256": digest(payload), "payload": payload}
    if path.exists():
        if unseal(path) != payload:
            raise IntegrityError(f"immutable artifact already exists with different content: {path}")
    else:
        atomic_json(path, envelope)
    return payload


def unseal(path: Path) -> Any:
    envelope = read_json(path)
    if digest(envelope["payload"]) != envelope["sha256"]:
        raise IntegrityError(f"artifact hash mismatch: {path}")
    return envelope["payload"]


@contextlib.contextmanager
def exclusive(path: Path, *, blocking=False) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"another process owns this workspace/run: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
