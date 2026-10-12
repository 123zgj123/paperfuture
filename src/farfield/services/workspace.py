"""One active public research controller per workspace, across processes.

The advisory lease coordinates FarField entry points. It does not replace
experiment isolation or prevent unrelated programs from editing files.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
from functools import wraps
from inspect import signature
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable, Iterator


class WorkspaceBusy(RuntimeError):
    """Another controller still owns this workspace; do not replay or overwrite it."""


@contextmanager
def exclusive_workspace(workspace: Path) -> Iterator[Path]:
    root = Path(workspace).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Keep this inode after release. Unlinking it would let a waiting process
    # lock the old inode while a new process locked a newly created one.
    fd = os.open(root / '.research.lock', os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'r+', encoding='utf-8') as lease:
        if not stat.S_ISREG(os.fstat(lease.fileno()).st_mode):
            raise ValueError('research workspace lease must be a regular file')
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise WorkspaceBusy(f'research workspace is already running: {root}') from None
        try:
            lease.seek(0)
            lease.truncate()
            json.dump({'pid': os.getpid(), 'scope': 'public_research_controller'}, lease)
            lease.flush()
            yield root
        finally:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)


def workspace_lease(function: Callable[..., Iterator[Any]]) -> Callable[..., Iterator[Any]]:
    """Hold the lease for a streamed run, releasing on exhaustion, error or close."""
    parameters = signature(function)
    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Iterator[Any]:
        arguments = parameters.bind(*args, **kwargs)
        with exclusive_workspace(arguments.arguments['workspace']):
            yield from function(*args, **kwargs)
    return guarded
