"""Typed scientific events. Executors propose; the kernel admits; the reducer applies."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping


EVENT_LOG_FILE = "SCIENTIFIC_EVENTS.json"


class EventLogUnreadable(ValueError):
    """The log exists and cannot be read. It is the authority, so nothing proceeds."""


def write_file_atomically(path: Path, text: str) -> None:
    """Replace `path` with `text` all at once, or leave it exactly as it was.

    Written beside the target, flushed to disk, then renamed over it, so a crash
    or a full disk leaves the old file or the new one — never half of either.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # One scratch file per write, not per process: two threads saving the same
    # workspace shared a pid-named file, and one rename found it already gone.
    handle_fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.",
                                       suffix=".partial")
    scratch = Path(name)
    try:
        with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(scratch, path)
    finally:
        if scratch.exists():
            scratch.unlink()

EVENT_TYPES = (
    "QuestionCreated",
    "QuestionResolved",
    "QuestionPartiallyResolved",
    "QuestionBlocked",
    "TheoryCreated",
    "TheoryWeakened",
    "TheorySupported",
    "TheoryContradicted",
    "EvidenceAdded",
    "EvidenceRejected",
    "WorldCreated",
    "WorldAttested",
    "WorldDeprecated",
    "CapabilityProposed",
    "CapabilityValidated",
    "CapabilityTrusted",
    "CapabilityRevoked",
    "CapabilityDeprecated",
    "HarnessPatchProposed",
    "HarnessAdmitted",
    "ActionFailed",
    "ActionBlocked",
    "ContradictionDetected",
    "LineageForked",
    "LineageMerged",
    "FrontierChanged",
    "DebtUpdated",
    "TickAdvanced",
    "ArtifactAdded",
    "GoalSet",
    # Project-level control. These change what the project *is*, not only what
    # it currently knows.
    "ContractSet",
    "ContractAmended",
    "ReviewRecorded",
    "ReviewResolved",
    "ReviewRepairRefused",
    "DesignRevised",
    "LineageRetired",
    "ProjectPivoted",
    "DecisionRecorded",
)

MEANINGFUL_EVENTS = frozenset(
    {
        "QuestionResolved",
        "QuestionPartiallyResolved",
        "TheoryCreated",
        "TheoryWeakened",
        "TheorySupported",
        "TheoryContradicted",
        "EvidenceAdded",
        "WorldCreated",
        "WorldAttested",
        "CapabilityValidated",
        "CapabilityTrusted",
        "HarnessAdmitted",
        "ContradictionDetected",
        "FrontierChanged",
        # Recognising that a design is wrong, or that a direction is not worth
        # continuing, is research progress. Repeating a broken design is not.
        "ReviewRecorded",
        "ReviewResolved",
        "DesignRevised",
        "ProjectPivoted",
    }
)


@dataclass(frozen=True)
class ScientificEvent:
    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    tick: int = 0
    action_type: str = ""
    frontier_target_id: str = ""
    event_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "payload": dict(self.payload),
            "seq": self.seq,
            "tick": self.tick,
            "action_type": self.action_type,
            "frontier_target_id": self.frontier_target_id,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "ScientificEvent":
        row = dict(payload or {})
        body = row.get("payload")
        return cls(
            event_type=str(row.get("event_type") or ""),
            payload=dict(body) if isinstance(body, Mapping) else {},
            seq=int(row.get("seq") or 0),
            tick=int(row.get("tick") or 0),
            action_type=str(row.get("action_type") or ""),
            frontier_target_id=str(row.get("frontier_target_id") or ""),
            event_id=str(row.get("event_id") or ""),
        )

    def stamped(self, *, seq: int, tick: int = 0) -> "ScientificEvent":
        eid = self.event_id or f"e{seq:06d}-{self.event_type}"
        return replace(self, seq=seq, tick=tick or self.tick, event_id=eid)


def make_event(
    event_type: str,
    payload: Mapping[str, Any] | None = None,
    *,
    action_type: str = "",
    frontier_target_id: str = "",
    tick: int = 0,
) -> ScientificEvent:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown scientific event {event_type!r}")
    return ScientificEvent(
        event_type=event_type,
        payload=dict(payload or {}),
        action_type=action_type,
        frontier_target_id=frontier_target_id,
        tick=tick,
    )


@dataclass
class EventLog:
    """Append-only log plus the state snapshot events were applied against."""

    baseline: dict[str, Any] = field(default_factory=dict)
    events: list[ScientificEvent] = field(default_factory=list)

    def next_seq(self) -> int:
        return len(self.events) + 1

    def append(self, event: ScientificEvent, *, tick: int = 0) -> ScientificEvent:
        stamped = event.stamped(seq=self.next_seq(), tick=tick)
        self.events.append(stamped)
        return stamped

    def extend(self, events: Iterable[ScientificEvent], *, tick: int = 0) -> list[ScientificEvent]:
        return [self.append(item, tick=tick) for item in events]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": dict(self.baseline),
            "events": [item.to_dict() for item in self.events],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | None) -> "EventLog":
        row = dict(payload or {})
        rows = list(row.get("events") or [])
        # A row that is not an event is damage, not noise. Skipping it replayed
        # a history with a hole in it as if it were whole.
        for index, item in enumerate(rows):
            if not isinstance(item, Mapping):
                raise EventLogUnreadable(f"event {index + 1} of the log is not an event")
        events = [ScientificEvent.from_dict(item) for item in rows]
        baseline = dict(row.get("baseline") or {}) if isinstance(row.get("baseline"), Mapping) else {}
        return cls(baseline=baseline, events=events)


def load_event_log(workspace: Path | None) -> EventLog:
    """The log, or an empty one for a workspace that has never written one.

    A log that exists and cannot be read is an error. Reading it as empty made
    a torn write look like a fresh project: the state still loaded, its own
    snapshot became the new baseline, and the next save replaced the history.
    """
    if workspace is None:
        return EventLog()
    path = Path(workspace) / EVENT_LOG_FILE
    if not path.is_file():
        return EventLog()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EventLogUnreadable(f"{path} cannot be read: {exc}") from exc
    if not isinstance(payload, dict):
        raise EventLogUnreadable(f"{path} is not an event log")
    return EventLog.from_dict(payload)


def save_event_log(workspace: Path | None, log: EventLog) -> Path | None:
    if workspace is None:
        return None
    path = Path(workspace) / EVENT_LOG_FILE
    write_file_atomically(path, json.dumps(log.to_dict(), ensure_ascii=False, indent=2) + "\n")
    return path
