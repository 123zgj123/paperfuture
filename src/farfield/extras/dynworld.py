"""A frozen world, awakened: runtime-owned dynamics over attested bytes.

The catalog freeze made worlds *real* — retrieved, hashed, bound before
the claim. It did not make them *analyzable*: a fixture was a byte blob
a model-written script read however it liked, so a token stream from a
novel could dress up as the world of any sequence-shaped claim. Schema
match is not scientific match, and nothing downstream could tell.

This module wraps each schema family with a dynamics bridge, in the
spirit of EnvHarness's "wrapping, not authoring": the frozen bytes and
their digest are untouched; what is added is a deterministic interface
the *trusted runtime* owns —

- **levers**: named interventions with a bounded intensity (edge
  dropout, token dropout, base mutation, row noise, transition
  dropout, ...). A lever is the only way a mechanism can act on this
  world; a claim whose mechanism corresponds to no lever has no handle
  here and must say so.
- **observables**: named measurements the runtime computes from the
  world state. An observable is a verifier the model did not write.
- **forward simulation**: execute the registered lever on the *evolving*
  state until a runtime stop (absorbing configuration, observable
  plateau, or horizon). The trajectory is the idea analysis on this
  object — not a dose ramp reapplied to frozen bytes, and not a model
  narrative. Deterministic given (world digest, lever, seed).

The model never writes a transition rule. A dynamics authored after
seeing the claim would just be a larger invented world — the same hole
the network ban closed for data. Bridges live here, versioned with the
runtime, written before any particular hypothesis exists.

Epistemic status: the trajectory may veto a probe (no handle, no room
to move) and must appear in the research plan. It may never climb the
ladder — `supports` still comes only from the registered two-arm probe
on the same attested bytes.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .validity import ratio
from .world import WorldFixture, digest_files, schema_family

# Loading caps keep a scout pass cheap on the biggest frozen slices.
_MAX_EDGES = 20_000
_MAX_TOKENS = 20_000
_MAX_BASES = 40_000
_MAX_ROWS = 4_000
_MAX_TRANSITIONS = 10_000
_MAX_TRACES = 250
_MAX_TRACE_STEPS = 64

# Projected trace payloads are heavy (the Live-SWE freeze is tens of MB).
# Scout reloads the same digest once per lever; cache the projection.
_TRACE_STATE_CACHE: dict[str, dict[str, Any]] = {}

# Step 0 is the untouched world. Later steps apply the lever to the
# *previous* state at this working intensity until a stop condition.
DEFAULT_INTENSITY = 0.6
DEFAULT_HORIZON = 8
PLATEAU_STEPS = 2

# Below this relative change over the full ramp a lever/observable pair
# is reported inert — pulling it would probably yield an uninformative
# probe.
RESPONSE_FLOOR = 0.03

LEVER_NONE = "none"

# Literature phrases a verified paper may name to rehearse a *declared*
# lever. This is not a second compile menu: generation still welds onto
# the lever name (`truncate`, `mask_tools`, …). Rehearsal retrieval
# looks these phrases up; it may not invent `code_rollout` as a handle.
LEVER_LITERATURE: dict[str, tuple[str, ...]] = {
    "truncate": (
        "truncate",
        "process reward",
        "full-trace",
        "next-event",
        "prefix-to-next",
        "outcome reward",
    ),
    "mask_failures": (
        "mask_failures",
        "prefix monitor",
        "early failure",
        "fail-fast",
        "mid-trajectory",
        "observable prefix",
    ),
    "mask_tools": (
        "mask_tools",
        "adaptive sampling",
        "query budget",
        "bandit",
        "acquisition",
    ),
    "dropout": ("dropout",),
    "hub_removal": ("hub_removal", "hub removal"),
    "rewire": ("rewire",),
    "window_shuffle": ("window_shuffle", "window shuffle"),
    "point_mutation": ("point_mutation", "point mutation"),
    "noise": ("noise",),
    "walk": ("walk", "protocol walk"),
    "drop_majority": ("drop_majority", "majority class"),
}

# Object-family literature for the freeze schema, still not a lever.
# A labeled_traces idea may rehearse code world models; a graph idea may not.
SCHEMA_LITERATURE: dict[str, tuple[str, ...]] = {
    "labeled_traces": (
        "code world model",
        "world model",
        "observation-action",
        "forward simulation",
    ),
    "program_state": (
        "self-improvement",
        "self-modifying code",
        "code world model",
        "acceptance test",
        "program state",
    ),
}


def literature_phrases(
    *,
    levers: tuple[str, ...] | list[str] = (),
    schema: str = "",
    claim: str = "",
    mechanism: str = "",
) -> tuple[str, ...]:
    """Phrases rehearsal may search. Empty when nothing on the menu matches."""
    phrases: list[str] = []
    named = [
        str(item).strip()
        for item in levers
        if str(item).strip() and str(item).strip() != LEVER_NONE
    ]
    family = schema_family(schema) if schema else ""
    if named or family or schema:
        for lever in named:
            phrases.extend(LEVER_LITERATURE.get(lever, (lever,)))
        key = family or schema
        if key:
            phrases.extend(SCHEMA_LITERATURE.get(key, ()))
    else:
        blob = f"{claim} {mechanism}".lower()
        for group in (*LEVER_LITERATURE.values(), *SCHEMA_LITERATURE.values()):
            if any(phrase.lower() in blob for phrase in group):
                phrases.extend(group)
    seen: set[str] = set()
    ordered: list[str] = []
    for phrase in phrases:
        label = phrase.lower()
        if label and label not in seen:
            seen.add(label)
            ordered.append(phrase)
    return tuple(ordered)


class DynamicsUnavailable(Exception):
    """No runtime bridge exists for this fixture (schema or payload)."""


# ---------------------------------------------------------------------------
# payload loading (frozen bytes -> in-memory world state)


def graph_node_id(item: Any) -> str:
    """Coerce a GENERATED or freeze node to a hashable id.

    LLM-written graph.json often emits `{"id": "n0", "label": "..."}`
    objects. Putting those in a set is `unhashable type: 'dict'`.
    """
    if item is None or isinstance(item, bool):
        return ""
    if isinstance(item, dict):
        for key in ("id", "name", "label", "node"):
            value = item.get(key)
            if value is not None and not isinstance(value, (dict, list, tuple)):
                text = str(value).strip()
                if text:
                    return text
        return ""
    if isinstance(item, (list, tuple)):
        return graph_node_id(item[0]) if item else ""
    return str(item).strip()


def graph_edge_pair(item: Any) -> tuple[str, str] | None:
    """Coerce an edge to a pair of string ids, or None if unusable."""
    if isinstance(item, dict):
        src = (
            item.get("src")
            or item.get("source")
            or item.get("u")
            or item.get("from")
            or item.get("a")
        )
        dst = (
            item.get("dst")
            or item.get("target")
            or item.get("v")
            or item.get("to")
            or item.get("b")
        )
        left, right = graph_node_id(src), graph_node_id(dst)
        if left and right:
            return left, right
        return None
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        left, right = graph_node_id(item[0]), graph_node_id(item[1])
        if left and right:
            return left, right
    return None


def graph_nodes_edges(
    payload: dict[str, Any] | None, *, max_edges: int = _MAX_EDGES
) -> tuple[list[str], list[tuple[str, str]]]:
    """String node ids and undirected pairs. Dict-shaped JSON cannot enter a set."""
    data = payload if isinstance(payload, dict) else {}
    seen: dict[str, None] = {}
    edges: list[tuple[str, str]] = []

    def _add(nid: str) -> None:
        if nid and nid not in seen:
            seen[nid] = None

    for item in data.get("nodes") or []:
        _add(graph_node_id(item))
    for item in data.get("edges") or []:
        pair = graph_edge_pair(item)
        if pair is None:
            continue
        _add(pair[0])
        _add(pair[1])
        edges.append(pair)
        if len(edges) >= max_edges:
            break
    return list(seen), edges


def _string_set(items: Any) -> set[str]:
    """Feature / tool vocabularies must be strings, never JSON objects."""
    out: set[str] = set()
    if isinstance(items, dict):
        items = items.keys()
    for item in items or ():
        if isinstance(item, dict):
            label = graph_node_id(item)
            if label:
                out.add(label)
            continue
        text = str(item).strip()
        if text:
            out.add(text)
    return out


def _load_graph(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "graph.json").read_text(encoding="utf-8"))
    nodes, edges = graph_nodes_edges(
        payload if isinstance(payload, dict) else {}, max_edges=_MAX_EDGES
    )
    return {"nodes": sorted(nodes), "edges": edges}


def _load_stream(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "stream.json").read_text(encoding="utf-8"))
    return {"tokens": [str(t) for t in payload["tokens"][:_MAX_TOKENS]]}


def _load_fasta(root: Path) -> dict[str, Any]:
    name = next(
        (n for n in ("sequence.fasta", "genome.fasta") if (root / n).is_file()),
        None,
    )
    if name is None:
        candidates = sorted(root.glob("*.fasta"))
        if not candidates:
            raise DynamicsUnavailable("no fasta file in fixture")
        name = candidates[0].name
    seq = "".join(
        line.strip()
        for line in (root / name).read_text(encoding="utf-8").splitlines()
        if line and not line.startswith(">")
    )
    return {"sequence": seq[:_MAX_BASES]}


def _table_payload_name(root: Path) -> str | None:
    """The fixture file that holds the table, chosen by what its bytes hold.

    `table.json` by convention, else the one JSON payload whose body carries a
    `rows` list. This used to name one dataset's file outright, which made a
    general loader quietly special to that dataset.
    """
    if (root / "table.json").is_file():
        return "table.json"
    skip = {"manifest.json", "origin.json", "seed.json", "world.json"}
    for path in sorted(root.glob("*.json")):
        if path.name in skip:
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
            return path.name
    return None


def _load_table(root: Path) -> dict[str, Any]:
    name = _table_payload_name(root)
    if name is None:
        raise DynamicsUnavailable("no table payload in fixture")
    payload = json.loads((root / name).read_text(encoding="utf-8"))
    declared = [str(col) for col in (payload.get("columns") or [])]
    rows = _table_row_dicts(payload.get("rows") or [], declared)[:_MAX_ROWS]
    columns = [
        str(col)
        for col in declared
        if rows and isinstance(rows[0].get(str(col)), (int, float))
        and not isinstance(rows[0].get(str(col)), bool)
    ]
    if not columns and rows:
        columns = [
            key
            for key, value in rows[0].items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
    vectors = [
        tuple(float(row.get(col) or 0.0) for col in columns) for row in rows
    ]
    return {"columns": columns, "vectors": vectors}


def _table_row_dicts(
    raw_rows: list[Any], declared: list[str]
) -> list[dict[str, Any]]:
    """Accept freeze dict-rows or GENERATED list-rows without crashing scout."""
    rows: list[dict[str, Any]] = []
    for item in raw_rows:
        if isinstance(item, dict):
            rows.append(item)
            continue
        if not isinstance(item, (list, tuple)):
            continue
        if declared:
            width = min(len(declared), len(item))
            rows.append({declared[i]: item[i] for i in range(width)})
        else:
            rows.append({f"c{i}": value for i, value in enumerate(item)})
    return rows


def _load_trace(root: Path) -> dict[str, Any]:
    payload = json.loads((root / "world.json").read_text(encoding="utf-8"))
    states = [
        str(s.get("id") if isinstance(s, dict) else s)
        for s in payload["states"]
    ]
    transitions = [
        (str(t["src"]), str(t["dst"]))
        for t in payload["transitions"][:_MAX_TRANSITIONS]
    ]
    initial = str(payload.get("initial") or (states[0] if states else ""))
    return {
        "states": states,
        "transitions": transitions,
        "initial": initial,
        "current": initial,
        "history": [initial] if initial else [],
    }


def _trace_rows(payload: dict[str, Any]) -> list[Any]:
    rows = payload.get("traces")
    return list(rows) if isinstance(rows, list) else []


def _parse_step_signals(step: dict[str, Any]) -> tuple[int | None, float | None, int]:
    """Choice / payoff / action length from an attested step, if present."""
    chars = int(step.get("action_chars") or 0)
    action = step.get("action")
    if not chars and isinstance(action, str):
        chars = len(action)
    choice: int | None = None
    payoff: float | None = None
    for blob in (action, step.get("output_excerpt")):
        if not isinstance(blob, str) or not blob.startswith("{"):
            continue
        try:
            payload = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        raw_choice = payload.get("choice", payload.get("own_choice"))
        if raw_choice is not None and choice is None:
            try:
                choice = int(raw_choice)
            except (TypeError, ValueError):
                choice = None
        raw_payoff = payload.get("own_payoff")
        if raw_payoff is not None and payoff is None:
            try:
                payoff = float(raw_payoff)
            except (TypeError, ValueError):
                payoff = None
    return choice, payoff, chars


def _load_labeled_traces(root: Path) -> dict[str, Any]:
    """Project attested traces into scout state.

    Labels, failures, created_tools, and per-step action_chars stay in.
    The runtime does not replay bash. Official order prefix — same
    honesty as other freeze slices — keeps the scout cheap on the
    1500-trace parent.
    """
    cache_key = f"{root.resolve()}::v2"
    cached = _TRACE_STATE_CACHE.get(cache_key)
    if cached is not None:
        return {
            "traces": [dict(row) for row in cached["traces"]],
            "features": set(cached["features"]),
        }
    payload: dict[str, Any] = {}
    for name in ("world.json", "traces.json"):
        path = root / name
        if not path.is_file():
            continue
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict) and _trace_rows(raw):
            payload = raw
            break
    traces: list[dict[str, Any]] = []
    features: set[str] = set()
    labels: set[str] = set()
    for raw in _trace_rows(payload)[:_MAX_TRACES]:
        if not isinstance(raw, dict):
            continue
        steps = raw.get("steps") if isinstance(raw.get("steps"), list) else []
        fails: list[int] = []
        action_chars: list[int] = []
        choices: list[int] = []
        payoffs: list[float] = []
        for step in steps[:_MAX_TRACE_STEPS]:
            if not isinstance(step, dict):
                fails.append(0)
                action_chars.append(0)
                continue
            code = step.get("returncode")
            failed = int(code not in (None, 0))
            fails.append(failed)
            if failed:
                features.add("failures")
            choice, payoff, chars = _parse_step_signals(step)
            action_chars.append(chars)
            if chars:
                features.add("actions")
            if choice is not None:
                choices.append(choice)
                features.add("choices")
            if payoff is not None:
                payoffs.append(payoff)
                features.add("payoffs")
        tools_raw = raw.get("created_tools")
        tools = (
            [str(item) for item in tools_raw[:32] if str(item).strip()]
            if isinstance(tools_raw, list)
            else []
        )
        n_tools = len(tools) if tools else int(
            (raw.get("derived") or {}).get("created_tool_count") or 0
        )
        if n_tools:
            features.add("tools")
        label = str(raw.get("label") or "")
        if label:
            labels.add(label)
        traces.append(
            {
                "id": str(raw.get("id") or f"t{len(traces)}"),
                "label": label,
                "n_steps": len(steps) or len(fails),
                "n_fail": sum(fails),
                "n_tools": n_tools,
                "fails": fails,
                "tools": tools,
                "action_chars": action_chars,
                "n_cooperate": sum(1 for item in choices if item == 0),
                "n_choice": len(choices),
                "payoff_sum": sum(payoffs),
                "n_payoff": len(payoffs),
            }
        )
    if len(labels) >= 2:
        features.add("labels")
    if any(row["n_steps"] > 0 for row in traces):
        features.add("steps")
    _TRACE_STATE_CACHE[cache_key] = {
        "traces": [dict(row) for row in traces],
        "features": set(features),
    }
    return {
        "traces": [dict(row) for row in traces],
        "features": set(features),
    }


def _read_world_payload(root: Path) -> dict[str, Any]:
    """Prefer `data/world.json` when `world.json` is only the fixture card.

    bind_world writes the manifest at the root and the scientific bytes
    under `data/`. Reading the card as the payload made a constructed
    program_state world look leverless (`dropout` only, no cycles).
    """
    candidates = (root / "data" / "world.json", root / "world.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if (
            payload.get("updates")
            or payload.get("cells")
            or payload.get("traces")
            or payload.get("states")
        ):
            return payload
    return {}


def _load_program(root: Path) -> dict[str, Any]:
    """Project a program_state world into scout state.

    Cells, validators, and the ordered update history stay in. The
    runtime does not re-execute the program; `divergent` is instance
    ground truth recorded at construction, the same way returncodes are
    on a trace freeze.
    """
    payload = _read_world_payload(root)
    cells = [str(item) for item in (payload.get("cells") or [])]
    validators: list[dict[str, Any]] = []
    for raw in payload.get("validators") or []:
        if not isinstance(raw, dict):
            continue
        validators.append(
            {
                "id": str(raw.get("id") or f"v{len(validators)}"),
                "reads": [str(cell) for cell in (raw.get("reads") or [])],
            }
        )
    reads_of = {row["id"]: set(row["reads"]) for row in validators}
    updates: list[dict[str, Any]] = []
    features: set[str] = set()
    for raw in payload.get("updates") or []:
        if not isinstance(raw, dict):
            continue
        writes = {str(cell) for cell in (raw.get("writes") or [])}
        validator_writes = [str(vid) for vid in (raw.get("validator_writes") or [])]
        on_cycle = any(
            writes & reads_of.get(vid, set()) for vid in validator_writes
        )
        row = {
            "id": str(raw.get("id") or f"u{len(updates)}"),
            "epoch": int(raw.get("epoch") or len(updates)),
            "episode": str(raw.get("episode") or ""),
            "writes": sorted(writes),
            "reads": [str(cell) for cell in (raw.get("reads") or [])],
            "validator_writes": validator_writes,
            "accepted": bool(raw.get("accepted")),
            "divergent": bool(raw.get("divergent")),
            "task_return": float(raw.get("task_return") or 0.0),
            "on_cycle": on_cycle,
        }
        updates.append(row)
        if on_cycle:
            features.add("cycles")
        if validator_writes:
            features.add("validator_writes")
        if row["divergent"]:
            features.add("divergent")
    if updates:
        features.add("updates")
    # Whether an outcome was recorded at all, which is a different question
    # from whether that outcome is independent of the gate (`oracle`).
    if any("task_return" in row for row in updates):
        features.add("task_return")
    oracle = payload.get("oracle") if isinstance(payload.get("oracle"), dict) else {}
    # An oracle the gate never saw (hidden tests) makes false acceptance a
    # real quantity; an evaluator's own score does not.
    oracle_independent = bool(oracle.get("independent_of_gate", False)) and len(
        {row["task_return"] for row in updates}
    ) >= 2
    if oracle_independent:
        features.add("oracle")
    episodes = Counter(row["episode"] for row in updates if row["episode"])
    # How many self-modifications each update's episode made, as a property of
    # the update: a stratum defined on it survives subsampling unchanged.
    for row in updates:
        row["episode_updates"] = episodes.get(row["episode"], 0)
    if any(count >= HEAVY_EPISODE_UPDATES for count in episodes.values()) and any(
        count < HEAVY_EPISODE_UPDATES for count in episodes.values()
    ):
        features.add("episode_burden")
    return {
        "cells": cells,
        "validators": validators,
        "updates": updates,
        "features": features,
        "oracle_independent": oracle_independent,
    }


def _payload_kind(root: Path) -> str:
    """What the bytes actually are. Schema name is not enough.

    A GENERATED traces.json and an attested Live-SWE world.json both
    host labeled-trace dynamics. A novel's stream.json must not.
    """
    payload = _read_world_payload(root)
    if payload:
        if _trace_rows(payload):
            return "labeled_traces"
        if isinstance(payload.get("updates"), list) and payload.get("updates"):
            return "program_state"
        if payload.get("states") and payload.get("transitions"):
            return "symbolic_trace"
        if payload.get("acquisition") == "incomplete":
            return ""
    if (root / "traces.json").is_file():
        return "labeled_traces"
    if (root / "graph.json").is_file():
        return "undirected_graph"
    if (root / "stream.json").is_file():
        return "text_stream"
    if any(root.glob("*.fasta")):
        return "fasta"
    if _table_payload_name(root):
        return "numeric_table"
    return ""


# ---------------------------------------------------------------------------
# observables (runtime-owned verifiers)


def _observe_graph(state: dict[str, Any]) -> dict[str, float]:
    nodes = state["nodes"]
    edges = state["edges"]
    adjacency: dict[Any, list[Any]] = {}
    for a, b in edges:
        adjacency.setdefault(a, []).append(b)
        adjacency.setdefault(b, []).append(a)
    seen: set[Any] = set()
    largest = 0
    for start in nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for peer in adjacency.get(node, ()):
                if peer not in seen:
                    seen.add(peer)
                    stack.append(peer)
        largest = max(largest, size)
    total = max(1, len(nodes))
    isolated = sum(1 for node in nodes if not adjacency.get(node))
    return {
        "largest_component_fraction": largest / total,
        "mean_degree": 2.0 * len(edges) / total,
        "isolated_fraction": isolated / total,
    }


def _observe_stream(state: dict[str, Any]) -> dict[str, float]:
    tokens = state["tokens"]
    total = max(1, len(tokens))
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    counts = Counter(tokens)
    top = counts.most_common(1)[0][1] if counts else 0
    return {
        "distinct_fraction": len(counts) / total,
        "adjacent_repeat_rate": repeats / max(1, total - 1),
        "top_token_share": top / total,
    }


def _observe_fasta(state: dict[str, Any]) -> dict[str, float]:
    seq = state["sequence"]
    total = max(1, len(seq))
    k = 8
    kmers = {seq[i : i + k] for i in range(0, max(0, len(seq) - k + 1))}
    gc = sum(1 for base in seq if base in "GCgc")
    return {
        "kmer8_diversity": len(kmers) / max(1, total - k + 1),
        "gc_fraction": gc / total,
    }


def _observe_table(state: dict[str, Any]) -> dict[str, float]:
    vectors = state["vectors"]
    if not vectors:
        return {"mean_pairwise_distance": 0.0, "mean_column_spread": 0.0}
    dims = len(vectors[0])
    # Deterministic pairwise sample: consecutive pairs wrap around.
    dist = 0.0
    count = 0
    for i in range(len(vectors)):
        a = vectors[i]
        b = vectors[(i + 1) % len(vectors)]
        dist += sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
        count += 1
    spreads = []
    for d in range(dims):
        column = [vec[d] for vec in vectors]
        spreads.append(max(column) - min(column))
    return {
        "mean_pairwise_distance": dist / max(1, count),
        "mean_column_spread": sum(spreads) / max(1, dims),
    }


def _toolset_jaccard(left: list[str], right: list[str]) -> float:
    a, b = _string_set(left), _string_set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _observe_traces(state: dict[str, Any]) -> dict[str, float]:
    traces = state.get("traces") or []
    features = _string_set(state.get("features") or ())
    n = max(1, len(traces))
    labels = Counter(str(row.get("label") or "") for row in traces)
    labels.pop("", None)
    majority = max(labels.values()) if labels else 0
    steps = sum(int(row.get("n_steps") or 0) for row in traces)
    fails = sum(int(row.get("n_fail") or 0) for row in traces)
    tooled = sum(1 for row in traces if int(row.get("n_tools") or 0) > 0)
    totals = [
        sum(int(item) for item in (row.get("action_chars") or ()))
        for row in traces
    ]
    mean_chars = sum(totals) / n
    if len(traces) >= 2:
        pairs = [
            _toolset_jaccard(row.get("tools") or [], traces[(i + 1) % len(traces)].get("tools") or [])
            for i, row in enumerate(traces)
        ]
        jaccard = sum(pairs) / max(1, len(pairs))
    else:
        jaccard = 1.0
    observed = {
        "n_traces": float(len(traces)),
        "majority_label_share": majority / n,
        "mean_step_count": steps / n,
        "failed_step_fraction": fails / max(1, steps),
        "tooled_fraction": tooled / n,
        "mean_toolset_jaccard": jaccard,
        "mean_action_chars": mean_chars,
    }
    if "choices" in features or any(int(row.get("n_choice") or 0) for row in traces):
        chosen = sum(int(row.get("n_choice") or 0) for row in traces)
        observed["cooperate_share"] = (
            sum(int(row.get("n_cooperate") or 0) for row in traces) / max(1, chosen)
        )
    if "payoffs" in features or any(int(row.get("n_payoff") or 0) for row in traces):
        paid = sum(int(row.get("n_payoff") or 0) for row in traces)
        observed["mean_own_payoff"] = (
            sum(float(row.get("payoff_sum") or 0.0) for row in traces) / max(1, paid)
        )
    return observed


def _observe_program(state: dict[str, Any]) -> dict[str, float]:
    updates = state.get("updates") or []
    n = max(1, len(updates))
    accepted = [row for row in updates if row.get("accepted")]
    # False acceptance is gate-accepted and oracle-rejected when the world
    # carries an oracle the gate never saw; only a world without one falls
    # back to the replay-divergence proxy. The proxy was the measure that
    # sat blind to `task_return` for three WORLD probes.
    if state.get("oracle_independent"):
        false_accepts = sum(
            1 for row in accepted if float(row.get("task_return") or 0.0) == 0.0
        )
    else:
        false_accepts = sum(1 for row in accepted if row.get("divergent"))
    cycles = sum(1 for row in updates if row.get("on_cycle"))
    returns = [float(row.get("task_return") or 0.0) for row in accepted]
    return {
        "n_updates": float(len(updates)),
        "accepted_fraction": len(accepted) / n,
        "false_accept_fraction": false_accepts / max(1, len(accepted)),
        "cycle_fraction": cycles / n,
        "mean_accepted_return": sum(returns) / max(1, len(returns)),
    }


def _observe_trace(state: dict[str, Any]) -> dict[str, float]:
    states = state["states"]
    transitions = state["transitions"]
    adjacency: dict[str, list[str]] = {}
    for src, dst in transitions:
        adjacency.setdefault(src, []).append(dst)
    seen = {state["initial"]} if state["initial"] else set()
    stack = list(seen)
    while stack:
        node = stack.pop()
        for peer in adjacency.get(node, ()):
            if peer not in seen:
                seen.add(peer)
                stack.append(peer)
    total = max(1, len(states))
    dead = sum(1 for s in states if not adjacency.get(s))
    current = str(state.get("current") or state.get("initial") or "")
    absorbing = 1.0 if current and not adjacency.get(current) else 0.0
    hops = max(0, len(state.get("history") or []) - 1)
    return {
        "reachable_fraction": len(seen) / total,
        "dead_end_fraction": dead / total,
        "mean_out_degree": len(transitions) / total,
        "at_absorbing": absorbing,
        "walk_hops": float(hops),
    }


# ---------------------------------------------------------------------------
# levers (the only ways a mechanism can act on the world)


def _keep(items: list, fraction_removed: float, rng: random.Random) -> list:
    if fraction_removed <= 0:
        return list(items)
    kept = [item for item in items if rng.random() >= fraction_removed]
    return kept if kept else list(items[:1])


def _lever_graph(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    nodes, edges = state["nodes"], list(state["edges"])
    if lever == "dropout":
        return {"nodes": nodes, "edges": _keep(edges, intensity, rng)}
    if lever == "hub_removal":
        degree: Counter = Counter()
        for a, b in edges:
            degree[a] += 1
            degree[b] += 1
        cut = max(0, int(len(nodes) * intensity * 0.2))
        hubs = {node for node, _ in degree.most_common(cut)}
        return {
            "nodes": nodes,
            "edges": [e for e in edges if e[0] not in hubs and e[1] not in hubs],
        }
    if lever == "rewire":
        out = []
        # sorted, not list(set): the generator is seeded from the world digest
        # but `choice` indexes into this pool, and a set of string node ids
        # iterates in an order that changes between processes. Two runs of the
        # same simulation returned different graphs and different
        # `report_digest`s — an experiment whose result depends on which
        # process ran it cannot support a claim, and the whole design rests on
        # a replay reproducing what happened.
        pool = sorted({n for e in edges for n in e})
        for a, b in edges:
            if rng.random() < intensity and pool:
                out.append((a, rng.choice(pool)))
            else:
                out.append((a, b))
        return {"nodes": nodes, "edges": out}
    raise DynamicsUnavailable(f"unknown graph lever {lever!r}")


def _lever_stream(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    tokens = list(state["tokens"])
    if lever == "dropout":
        return {"tokens": _keep(tokens, intensity, rng)}
    if lever == "window_shuffle":
        window = max(2, int(4 + intensity * 60))
        out = []
        for i in range(0, len(tokens), window):
            chunk = tokens[i : i + window]
            if rng.random() < intensity:
                rng.shuffle(chunk)
            out.extend(chunk)
        return {"tokens": out}
    raise DynamicsUnavailable(f"unknown stream lever {lever!r}")


def _lever_fasta(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    seq = state["sequence"]
    if lever == "dropout":
        return {
            "sequence": "".join(
                base for base in seq if rng.random() >= intensity
            )
            or seq[:1]
        }
    if lever == "point_mutation":
        alphabet = "ACGT"
        out = [
            rng.choice(alphabet) if rng.random() < intensity else base
            for base in seq
        ]
        return {"sequence": "".join(out)}
    raise DynamicsUnavailable(f"unknown fasta lever {lever!r}")


def _lever_table(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    vectors = list(state["vectors"])
    if lever == "dropout":
        return {"columns": state["columns"], "vectors": _keep(vectors, intensity, rng)}
    if lever == "noise":
        if not vectors:
            return dict(state)
        dims = len(vectors[0])
        spans = []
        for d in range(dims):
            column = [vec[d] for vec in vectors]
            spans.append((max(column) - min(column)) or 1.0)
        out = [
            tuple(
                value + rng.uniform(-1, 1) * intensity * spans[d]
                for d, value in enumerate(vec)
            )
            for vec in vectors
        ]
        return {"columns": state["columns"], "vectors": out}
    raise DynamicsUnavailable(f"unknown table lever {lever!r}")


def _lever_trace(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    transitions = list(state["transitions"])
    if lever == "walk":
        current = str(state.get("current") or state.get("initial") or "")
        outgoing = [dst for src, dst in transitions if src == current]
        if not outgoing:
            return dict(state)
        nxt = rng.choice(outgoing)
        history = list(state.get("history") or [])
        history.append(nxt)
        return {**state, "current": nxt, "history": history}
    if lever == "dropout":
        return {**state, "transitions": _keep(transitions, intensity, rng)}
    if lever == "rewire":
        out = []
        for src, dst in transitions:
            if rng.random() < intensity and state["states"]:
                out.append((src, rng.choice(state["states"])))
            else:
                out.append((src, dst))
        return {**state, "transitions": out}
    raise DynamicsUnavailable(f"unknown trace lever {lever!r}")


def _levers_traces(state: dict[str, Any]) -> tuple[str, ...]:
    """Only advertise handles the payload can actually feel.

    A stub without returncodes must not offer mask_failures — that would
    be a menu the diagnosis cannot satisfy, the same contradiction the
    object lock used to have on pairings.
    """
    features = _string_set(state.get("features") or ())
    levers = ["dropout"]
    if "failures" in features:
        levers.append("mask_failures")
    if "tools" in features:
        levers.append("mask_tools")
    if "labels" in features:
        levers.append("drop_majority")
    if "steps" in features:
        levers.append("truncate")
    return tuple(levers)


def _lever_traces(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    traces = [dict(row) for row in (state.get("traces") or [])]
    if lever == "dropout":
        traces = _keep(traces, intensity, rng)
    elif lever == "mask_failures":
        for row in traces:
            if rng.random() < intensity:
                row["n_fail"] = 0
                row["fails"] = [0] * len(row.get("fails") or [])
    elif lever == "mask_tools":
        for row in traces:
            if rng.random() < intensity:
                row["n_tools"] = 0
                row["tools"] = []
    elif lever == "drop_majority":
        counts = Counter(str(row.get("label") or "") for row in traces)
        counts.pop("", None)
        if counts:
            majority = max(counts, key=lambda name: (counts[name], name))
            traces = [
                row
                for row in traces
                if str(row.get("label") or "") != majority or rng.random() >= intensity
            ]
            traces = traces or list(state["traces"][:1])
    elif lever == "truncate":
        keep = max(1, int(round((1.0 - min(0.95, intensity)) * 64)))
        for row in traces:
            fails = list(row.get("fails") or [])[:keep]
            chars = list(row.get("action_chars") or [])[:keep]
            row["fails"] = fails
            row["action_chars"] = chars
            row["n_steps"] = min(int(row.get("n_steps") or 0), keep)
            row["n_fail"] = min(int(row.get("n_fail") or 0), sum(fails))
    else:
        raise DynamicsUnavailable(f"unknown traces lever {lever!r}")
    return {"traces": traces, "features": set(state.get("features") or ())}


# Observational handles on a replayed history. `contrast:<stratum>` splits
# the records the world already recorded: treatment = records in the
# stratum, control = the rest, matched on the world's natural strata. A
# replayed history cannot feel an intervention (its acceptances are facts),
# so these are the handles that ask about *its* dynamics; interventional
# levers only subsample it.
@dataclass(frozen=True)
class WorldCapability:
    """One thing a schema can do, with the payload it needs, in one object.

    Every defect this registry replaces came from the same shape: a name
    written in one place, the payload it needs written in another, and the code
    that performs it in a third, with nothing tying the three together.

    A world advertised `cycle_fraction` on a freeze carrying no `on_cycle`
    field, so the measure read zero in every stratum and a live hypothesis was
    spent predicting that it would move. A world advertised `freeze_validators`
    as its one manipulable handle when the field its effect is read through was
    absent, so the lever applied cleanly and changed nothing at any intensity.
    Both were fixed by adding a fourth table — which made the surface larger
    rather than smaller.

    Here the name cannot be written without a body, the body cannot be
    registered without saying what it needs, and what is advertised is derived
    by asking. "Advertised but inert" stops being a bug to detect and becomes a
    state with nowhere to live.

    ``requires`` are features the payload must all carry; ``requires_any`` is
    satisfied by any one of them, for a measure with two possible sources.
    """

    name: str
    requires: tuple[str, ...] = ()
    requires_any: tuple[str, ...] = ()
    meaning: str = ""

    def available(self, features: Any) -> bool:
        have = set(features or ())
        if not set(self.requires).issubset(have):
            return False
        return not self.requires_any or bool(have & set(self.requires_any))


@dataclass(frozen=True)
class Measure(WorldCapability):
    """A number a schema reports, and the payload that lets it be computed.

    ``compute`` returns None where the number is undefined on this payload — a
    fraction of accepted updates when none were accepted — and an undefined
    measure is left out of the observation, never read as 0. ``describes_size``
    marks a count of what was observed rather than an outcome of it: two
    strata of different size differ in it by construction, so it is not an
    effect.
    """

    compute: Callable[[dict[str, Any]], float | None] | None = None
    describes_size: bool = False


@dataclass(frozen=True)
class Lever(WorldCapability):
    """A handle on a payload. ``requires`` is what its effect is *read
    through*, not what it writes — gating `freeze_validators` on the field it
    writes is exactly how an inert intervention came to be advertised."""

    apply: Callable[[list[dict[str, Any]], float, random.Random,
                     dict[str, Any]], list[dict[str, Any]]] | None = None


# Observational handles on a replayed history. `contrast:<stratum>` splits
# the records the world already recorded: treatment = records in the
# stratum, control = the rest, matched on the world's natural strata. A
# replayed history cannot feel an intervention (its acceptances are facts),
# so these are the handles that ask about *its* dynamics; interventional
# levers only subsample it.
CONTRAST_PREFIX = "contrast:"

# The recorded properties of an update that a contrast is matched on. A
# difference between a stratum and the rest could be the stratum, or the kind
# of episode its updates came from, so the comparison is made within cells of
# these. Each reads only fields the update carries. An update with no recorded
# episode has neither — grouping every such update into one pseudo-episode gave
# them a template and a burden they do not have — so its strata are undefined
# and it cannot be matched.
PROGRAM_STRATA: dict[str, Callable[[dict[str, Any]], str | None]] = {
    "episode_template": lambda row: (_episode_repo(str(row["episode"]))
                                     if row.get("episode") else None),
    "updates_per_episode": lambda row: (_burden_bucket(int(row.get("episode_updates") or 0))
                                        if row.get("episode") else None),
}

# An episode that made this many self-modifications or more is a heavy one.
HEAVY_EPISODE_UPDATES = 4


@dataclass(frozen=True)
class Contrast(WorldCapability):
    """An observational handle: the updates in a stratum against the rest.

    ``member`` decides from fields the update itself carries, so membership
    does not change when the history is subsampled. ``defined_by`` names the
    matching strata the contrast *is*: a contrast cannot be matched on the
    variable that defines it — no cell could hold both arms — so it is matched
    on the others and reports the one it could not be.
    """

    member: Callable[[dict[str, Any]], bool] | None = None
    defined_by: tuple[str, ...] = ()

    def matched_on(self) -> tuple[str, ...]:
        return tuple(name for name in PROGRAM_STRATA if name not in self.defined_by)


PROGRAM_CONTRASTS: tuple[Contrast, ...] = (
    Contrast("contrast:validator_write", requires=("validator_writes",),
             member=lambda row: bool(row.get("validator_writes")),
             meaning="updates that rewrote one of the program's own validators vs "
                     "updates that did not"),
    Contrast("contrast:divergent", requires=("divergent",),
             member=lambda row: bool(row.get("divergent")),
             meaning="updates whose cell content diverged from its previous version vs "
                     "updates that did not"),
    # Membership is a fact about the update's episode in the recorded history.
    # It used to test `bool(burden) >= 4`, which no boolean satisfies, or else a
    # count nothing ever recorded — so the stratum was always empty, and a lever
    # that deleted the whole history was reported as the world's strongest.
    Contrast("contrast:heavy_episode", requires=("episode_burden",),
             member=lambda row: int(row.get("episode_updates") or 0) >= HEAVY_EPISODE_UPDATES,
             defined_by=("updates_per_episode",),
             meaning="updates from episodes with four or more self-modifications vs "
                     "episodes with three or fewer"),
)
_CONTRASTS = {row.name: row for row in PROGRAM_CONTRASTS}


def is_contrast(lever: str) -> bool:
    return str(lever or "").startswith(CONTRAST_PREFIX)


def contrast_of(lever: str) -> Contrast | None:
    return _CONTRASTS.get(str(lever or ""))


def contrast_meaning(lever: str) -> str:
    row = contrast_of(lever)
    return row.meaning if row else ""


def matching_strata(lever: str) -> tuple[str, ...]:
    """The strata a contrast on this history is compared within."""
    row = contrast_of(lever)
    return row.matched_on() if row else tuple(PROGRAM_STRATA)


def _accepted(state: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in (state.get("updates") or []) if row.get("accepted")]


def _false_accepts(state: dict[str, Any]) -> int:
    accepted = _accepted(state)
    if state.get("oracle_independent"):
        return sum(1 for row in accepted if float(row.get("task_return") or 0.0) == 0.0)
    return sum(1 for row in accepted if row.get("divergent"))


PROGRAM_MEASURES: tuple[Measure, ...] = (
    Measure("n_updates", describes_size=True,
            compute=lambda s: float(len(s.get("updates") or ()))),
    Measure("accepted_fraction", requires=("updates",),
            compute=lambda s: ratio(len(_accepted(s)), len(s.get("updates") or ()))),
    Measure("cycle_fraction", requires=("cycles",),
            compute=lambda s: ratio(sum(1 for r in (s.get("updates") or ()) if r.get("on_cycle")),
                                    len(s.get("updates") or ()))),
    # `oracle` means the outcome is independent of the gate, which is a
    # different question from whether an outcome was recorded at all. Requiring
    # it here dropped this measure from a world that records returns under its
    # own gate, where the mean is still a real number about that world.
    Measure("mean_accepted_return", requires=("task_return",),
            compute=lambda s: ratio(sum(float(r.get("task_return") or 0.0) for r in _accepted(s)),
                                    len(_accepted(s)))),
    # False acceptance is gate-accepted and oracle-rejected when the world
    # carries an oracle the gate never saw; a world without one falls back to
    # the replay-divergence proxy. Either source will do; neither, and the
    # measure has nothing to read.
    Measure("false_accept_fraction", requires_any=("oracle", "divergent"),
            compute=lambda s: ratio(_false_accepts(s), len(_accepted(s)))),
)


def _apply_dropout(updates, intensity, rng, state):
    return _keep(updates, intensity, rng)


def _apply_replay_gate(updates, intensity, rng, state):
    # Intercept the write–read dependency cycle at runtime: an update that
    # mutates its own validator is replayed against the frozen pre-update state
    # before acceptance. Replay catches divergence; non-divergent self-edits
    # keep their acceptance.
    for row in updates:
        if not (row.get("on_cycle") and row.get("accepted")):
            continue
        if row.get("divergent") and rng.random() < intensity:
            row["accepted"] = False
    return updates


def _apply_freeze_validators(updates, intensity, rng, state):
    # Validators become immutable: the update's own edit to its checker never
    # lands, so a divergent self-approval loses its green light.
    for row in updates:
        if not row.get("validator_writes"):
            continue
        if rng.random() < intensity:
            row["validator_writes"] = []
            if row.get("on_cycle"):
                row["on_cycle"] = False
                if row.get("divergent"):
                    row["accepted"] = False
    return updates


def _apply_stratum(contrast: Contrast):
    def run(updates, intensity, rng, state):
        return [row for row in updates if contrast.member(row)]
    return run


PROGRAM_LEVERS: tuple[Lever, ...] = (
    Lever("dropout", apply=_apply_dropout,
          meaning="subsample the recorded history"),
    Lever("replay_gate", requires=("cycles",), apply=_apply_replay_gate,
          meaning="replay a self-mutating update against its pre-update state"),
    # Freezing validators lands on `validator_writes` and `on_cycle`; only
    # `on_cycle` is read by any measure, so a payload without cycles cannot
    # feel this lever however cleanly it applies.
    Lever("freeze_validators", requires=("validator_writes", "cycles"),
          apply=_apply_freeze_validators,
          meaning="the update's edit to its own checker never lands"),
) + tuple(
    Lever(row.name, requires=row.requires, apply=_apply_stratum(row), meaning=row.meaning)
    for row in PROGRAM_CONTRASTS
)


def _observe_program(state: dict[str, Any]) -> dict[str, float]:
    """Only the measures this payload can actually produce, where they are defined."""
    features = _string_set(state.get("features") or ())
    observed: dict[str, float] = {}
    for row in PROGRAM_MEASURES:
        if not row.available(features):
            continue
        value = row.compute(state)
        if value is not None:
            observed[row.name] = float(value)
    return observed


def _levers_program(state: dict[str, Any]) -> tuple[str, ...]:
    """Only the handles this payload can actually feel."""
    features = _string_set(state.get("features") or ())
    return tuple(row.name for row in PROGRAM_LEVERS if row.available(features))


def _lever_program(
    state: dict[str, Any], lever: str, intensity: float, rng: random.Random
) -> dict[str, Any]:
    """Apply the registered handle. There is nowhere here to name a lever the
    registry does not define, which is what the old if/elif chain allowed."""
    known = {row.name: row for row in PROGRAM_LEVERS}
    entry = known.get(str(lever or ""))
    if entry is None or entry.apply is None:
        raise DynamicsUnavailable(f"unknown program lever {lever!r}")
    updates = entry.apply([dict(row) for row in (state.get("updates") or [])],
                          intensity, rng, state)
    return {
        "cells": list(state.get("cells") or []),
        "validators": [dict(row) for row in (state.get("validators") or [])],
        "updates": updates,
        "features": set(state.get("features") or ()),
        "oracle_independent": bool(state.get("oracle_independent")),
    }


# ---------------------------------------------------------------------------
# the bridge registry

_BRIDGES: dict[str, dict[str, Any]] = {
    "undirected_graph": {
        "load": _load_graph,
        "observe": _observe_graph,
        "apply": _lever_graph,
        "levers": ("dropout", "hub_removal", "rewire"),
    },
    "text_stream": {
        "load": _load_stream,
        "observe": _observe_stream,
        "apply": _lever_stream,
        "levers": ("dropout", "window_shuffle"),
    },
    "fasta": {
        "load": _load_fasta,
        "observe": _observe_fasta,
        "apply": _lever_fasta,
        "levers": ("dropout", "point_mutation"),
    },
    "numeric_table": {
        "load": _load_table,
        "observe": _observe_table,
        "apply": _lever_table,
        "levers": ("dropout", "noise"),
    },
    "symbolic_trace": {
        "load": _load_trace,
        "observe": _observe_trace,
        "apply": _lever_trace,
        "levers": ("walk", "dropout", "rewire"),
    },
    "labeled_traces": {
        "load": _load_labeled_traces,
        "observe": _observe_traces,
        "apply": _lever_traces,
        "levers": ("dropout",),
        "levers_for": _levers_traces,
    },
    "program_state": {
        "load": _load_program,
        "observe": _observe_program,
        "apply": _lever_program,
        "levers": ("dropout",),
        "levers_for": _levers_program,
    },
}
_BRIDGES["undirected_named_graph"] = _BRIDGES["undirected_graph"]


def _episode_repo(episode: str) -> str:
    """`astropy__astropy-12907` -> `astropy__astropy`. The template, not the task."""
    text = str(episode or "")
    return text.rsplit("-", 1)[0] if "-" in text else text


def _burden_bucket(count: int) -> str:
    """Updates per episode, in the bands the contrast table already uses."""
    if count <= 1:
        return "1"
    if count <= 3:
        return "2-3"
    if count <= 7:
        return "4-7"
    return "8+"


def matched_contrast(
    fixture: WorldFixture, lever: str, observable: str
) -> dict[str, Any]:
    """A stratum against its complement *within* matched cells.

    The world names episode template and updates-per-episode as the strata a
    contrast on this history has to match on. Comparing the stratum against
    the whole history left a difference ambiguous between the stratum and
    which episodes it came from — and on the Live-SWE-agent freeze updates
    from heavy episodes shift false acceptance *more* than validator-writing
    updates do, which is exactly that confound.

    Each cell is the tuple of the strata the contrast is not defined by. A
    contrast defined by a stratum is not matched on it — no cell could hold
    both arms — and ``not_matched_on`` says so. Cells with both arms populated
    contribute; cells with one arm, or where the observable is undefined in an
    arm, are reported as dropped rather than averaged in. The control side is
    standardised to the treatment's cell distribution, so the arms answer the
    same question about the same kinds of episode. A count of the records
    compared is refused as an outcome: two strata differ in size by
    construction.

    Deterministic, and no model call. It is still observational: matching on
    recorded strata does not make a replayed history an experiment.
    """
    bridge = _bridge_of(fixture)
    contrast = contrast_of(lever)
    if not bridge or contrast is None:
        return {"status": "not_a_contrast", "lever": lever}
    outcome = next((row for row in PROGRAM_MEASURES if row.name == observable), None)
    if outcome is not None and outcome.describes_size:
        return {"status": "not_an_outcome", "lever": lever, "observable": observable,
                "reason": f"{observable} counts the records compared; two strata "
                          "differ in it by construction"}
    state = bridge["load"](fixture.root)
    updates = [dict(row) for row in (state.get("updates") or [])]
    if not updates:
        return {"status": "no_updates", "lever": lever}
    strata = contrast.matched_on()

    def cell_of(row: dict[str, Any]) -> tuple[str, ...] | None:
        key = tuple(PROGRAM_STRATA[name](row) for name in strata)
        return None if any(part is None for part in key) else key

    def measure(rows: list[dict[str, Any]]) -> float | None:
        if not rows:
            return None
        value = bridge["observe"]({**state, "updates": rows}).get(observable)
        return None if value is None else float(value)

    cells: dict[tuple[str, ...], dict[str, list[dict[str, Any]]]] = {}
    unmatchable = {"treatment": 0, "control": 0}
    for row in updates:
        arm = "treatment" if contrast.member(row) else "control"
        key = cell_of(row)
        if key is None:
            unmatchable[arm] += 1
            continue
        cells.setdefault(key, {"treatment": [], "control": []})[arm].append(row)

    used, dropped = [], []
    treated_total = 0
    for cell, arms in sorted(cells.items()):
        if not arms["treatment"] or not arms["control"]:
            dropped.append({"cell": list(cell),
                            "n_treatment": len(arms["treatment"]),
                            "n_control": len(arms["control"])})
            continue
        left, right = measure(arms["treatment"]), measure(arms["control"])
        if left is None or right is None:
            dropped.append({"cell": list(cell), "reason": "observable_undefined"})
            continue
        treated_total += len(arms["treatment"])
        used.append({"cell": list(cell), "weight": len(arms["treatment"]),
                     "treatment": left, "control": right,
                     "n_treatment": len(arms["treatment"]),
                     "n_control": len(arms["control"])})
    if not used or not treated_total:
        return {"status": "no_matched_cell", "lever": lever,
                "observable": observable, "dropped_cells": len(dropped)}

    treatment = sum(row["treatment"] * row["weight"] for row in used) / treated_total
    control = sum(row["control"] * row["weight"] for row in used) / treated_total
    return {
        "status": "matched",
        "rule": "matched-contrast-v1",
        "lever": lever,
        "meaning": contrast_meaning(lever),
        "observable": observable,
        "matched_on": list(strata),
        "not_matched_on": list(contrast.defined_by),
        "unmatchable": ({**unmatchable, "reason": "no recorded episode, so the strata it "
                                                  "would be matched on are undefined"}
                        if any(unmatchable.values()) else {}),
        "treatment": round(treatment, 6),
        "control": round(control, 6),
        "difference": round(treatment - control, 6),
        "n_treatment": sum(row["n_treatment"] for row in used),
        "n_control": sum(row["n_control"] for row in used),
        "cells_used": len(used),
        "cells_dropped": len(dropped),
        "dropped_examples": dropped[:5],
        "coverage": round(treated_total / sum(1 for row in updates if contrast.member(row)), 6),
        "causal_scope": "none",
        "interpretation": "matched observational contrast on a replayed history; "
                          "matching two recorded strata does not make it an experiment",
    }


def _bridge_of(fixture: WorldFixture) -> dict[str, Any] | None:
    """Prefer the payload's shape over the manifest schema name.

    A freeze or GENERATED construction of traces must not inherit the
    novel stream bridge just because someone said 'token'. An incomplete
    acquisition has no handle.
    """
    kind = _payload_kind(fixture.root)
    if kind:
        return _BRIDGES.get(kind)
    return _BRIDGES.get(schema_family(fixture.schema)) or _BRIDGES.get(fixture.schema)


def has_dynamics(fixture: WorldFixture | None) -> bool:
    return fixture is not None and _bridge_of(fixture) is not None


def levers_of(fixture: WorldFixture) -> tuple[str, ...]:
    bridge = _bridge_of(fixture)
    if not bridge:
        return ()
    advertised = bridge.get("levers_for")
    if callable(advertised):
        return tuple(advertised(bridge["load"](fixture.root)))
    return tuple(bridge["levers"])


def compose_simulation(fixture: WorldFixture | None) -> dict[str, Any]:
    """What forward simulation may honestly run on this binding.

    Origin is `bound` (attested freeze) or `constructed` (GENERATED of
    the claim's family). It is never a borrowed novel of another field.
    Unavailable means skip — do not pick Pride as a playground.
    """
    if fixture is None:
        return {"available": False, "reason": "no_object", "origin": ""}
    if str(getattr(fixture, "role", "") or "") == "test":
        return {
            "available": False,
            "reason": "test_fixture",
            "origin": "",
            "world_id": fixture.id,
        }
    if not has_dynamics(fixture):
        return {
            "available": False,
            "reason": "no_handle",
            "origin": "",
            "world_id": fixture.id,
            "schema": fixture.schema,
        }
    origin = (
        "constructed"
        if str(fixture.role or "") in {"generated", "placebo"}
        else "bound"
    )
    try:
        levers = list(levers_of(fixture))
        observables = list(observables_of(fixture))
    except Exception:
        return {
            "available": False,
            "reason": "payload_unreadable",
            "origin": origin,
            "world_id": fixture.id,
            "schema": fixture.schema,
        }
    return {
        "available": True,
        "reason": "",
        "origin": origin,
        "world_id": fixture.id,
        "schema": fixture.schema,
        "levers": levers,
        "observables": observables,
    }


def observables_of(fixture: WorldFixture) -> tuple[str, ...]:
    """Measures this world can produce.

    The observer returns only what the payload supports, so this is its keys —
    there is no separate gate to fall out of step with it.
    """
    bridge = _bridge_of(fixture)
    if not bridge:
        return ()
    state = bridge["load"](fixture.root)
    return tuple(sorted(bridge["observe"](state)))


def _rng(fixture: WorldFixture, lever: str, seed: int, step: int) -> random.Random:
    key = f"{fixture.digest}:{lever}:{seed}:{step}"
    return random.Random(int(hashlib.sha256(key.encode()).hexdigest()[:16], 16))


def _relative_delta(base: float, end: float) -> float:
    if abs(base) > 1e-12:
        return (end - base) / abs(base)
    return end - base


def _observables_plateau(
    previous: dict[str, float], current: dict[str, float]
) -> bool:
    # An observable that became undefined, or became defined, is a change.
    if not previous or set(previous) != set(current):
        return False
    for key, value in current.items():
        delta = abs(_relative_delta(float(previous.get(key, value)), float(value)))
        if delta >= RESPONSE_FLOOR:
            return False
    return True


def _analysis_of(trajectory: dict[str, Any]) -> dict[str, Any]:
    """Compile the idea-analysis product from a finished trajectory.

    Runtime text, not a model narrative. Goes into the packet. Cannot climb.
    """
    steps = list(trajectory.get("steps") or [])
    response = dict(trajectory.get("response") or {})
    room = any(delta is not None and abs(float(delta)) >= RESPONSE_FLOOR
               for delta in response.values())
    stop = str(trajectory.get("stop_reason") or "horizon")
    lever = str(trajectory.get("lever") or "")
    world_id = str(trajectory.get("world_id") or "")
    last_t = steps[-1]["t"] if steps else 0
    if stop == "no_handle":
        summary = (
            f"On {world_id}, the registered prediction has no runtime handle; "
            "the idea cannot be executed on this object."
        )
    elif stop == "no_object":
        summary = (
            "No attested freeze matches this claim; inventing data is not "
            "verification and is not an idea analysis."
        )
    elif not room:
        summary = (
            f"On {world_id} under {lever}, observables did not move by step "
            f"{last_t} ({stop}); the prediction has no trajectory here."
        )
    else:
        moved = ", ".join(
            f"{name} {delta:+.1%}"
            for name, delta in sorted(response.items())
            if delta is not None and abs(float(delta)) >= RESPONSE_FLOOR
        )
        summary = (
            f"On {world_id} under {lever}, execution stopped at t={last_t} "
            f"({stop}). Moved: {moved}."
        )
    return {
        "stop_reason": stop,
        "room_to_move": room,
        "n_steps": len(steps),
        "summary": summary,
    }


def forward_simulate(
    fixture: WorldFixture,
    lever: str,
    *,
    intensity: float = DEFAULT_INTENSITY,
    seed: int = 0,
    horizon: int = DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Execute `lever` on the evolving world until a runtime stop.

    Step 0 is the untouched world. Each later step applies the lever to
    the *previous* state at `intensity` — a trajectory of the prediction
    on this object, not a dose ramp from frozen bytes. Stops on an
    absorbing configuration, an observable plateau, or the horizon.
    Deterministic given (world digest, lever, seed, horizon).
    """
    bridge = _bridge_of(fixture)
    if bridge is None:
        raise DynamicsUnavailable(f"no bridge for schema {fixture.schema!r}")
    state = bridge["load"](fixture.root)
    allowed = (
        tuple(bridge["levers_for"](state))
        if callable(bridge.get("levers_for"))
        else tuple(bridge["levers"])
    )
    if lever not in allowed:
        raise DynamicsUnavailable(f"unknown lever {lever!r} for {fixture.schema}")
    steps: list[dict[str, Any]] = []
    stop_reason = "horizon"
    plateau_streak = 0
    for t in range(horizon + 1):
        if t == 0:
            current = state
            level = 0.0
        else:
            level = intensity
            current = bridge["apply"](
                state, lever, level, _rng(fixture, lever, seed, t)
            )
        observables = {
            key: round(value, 6)
            for key, value in bridge["observe"](current).items()
        }
        row: dict[str, Any] = {
            "t": t,
            "intensity": round(level, 6),
            "observables": observables,
        }
        if current.get("current"):
            row["current"] = current["current"]
        steps.append(row)
        if t > 0:
            if current == state:
                stop_reason = "absorbing"
                break
            if _observables_plateau(steps[t - 1]["observables"], observables):
                plateau_streak += 1
                if plateau_streak >= PLATEAU_STEPS:
                    stop_reason = "plateau"
                    break
            else:
                plateau_streak = 0
        state = current
    first, last = steps[0]["observables"], steps[-1]["observables"]
    # An observable the lever left undefined has no response to report. Reading
    # its absence as the starting value called it unmoved.
    response: dict[str, float | None] = {}
    for key, base in first.items():
        response[key] = (round(_relative_delta(base, last[key]), 6)
                         if key in last else None)
    trajectory = {
        "world_id": fixture.id,
        "world_digest": fixture.digest,
        "schema": fixture.schema,
        "lever": lever,
        "intensity": intensity,
        "seed": seed,
        "horizon": horizon,
        "stop_reason": stop_reason,
        "steps": steps,
        "response": response,
    }
    trajectory["analysis"] = _analysis_of(trajectory)
    trajectory["report_digest"] = hashlib.sha256(
        json.dumps(
            {key: trajectory[key] for key in trajectory if key != "report_digest"},
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return trajectory


def response_card(
    fixture: WorldFixture, *, seed: int = 0, horizon: int = 3
) -> dict[str, Any]:
    """One forward execution per lever: the world's measured dynamics.

    The card is what a diagnosis gets to read — which levers exist,
    which observables they move, and by how much. `inert` collects the
    lever/observable pairs whose response stayed under the floor; an
    experiment built on an inert pair is pre-announced uninformative.
    """
    if not has_dynamics(fixture):
        raise DynamicsUnavailable(f"no bridge for schema {fixture.schema!r}")
    levers: dict[str, dict[str, float]] = {}
    for lever in levers_of(fixture):
        trajectory = forward_simulate(
            fixture, lever, seed=seed, horizon=horizon
        )
        levers[lever] = trajectory["response"]
    observables = sorted(next(iter(levers.values()), {}))
    card = {
        "world_id": fixture.id,
        "world_digest": fixture.digest,
        "schema": fixture.schema,
        "seed": seed,
        "horizon": horizon,
        "levers": levers,
        "observables": observables,
        "inert": sorted(
            f"{lever}/{obs}"
            for lever, resp in levers.items()
            for obs, delta in resp.items()
            if delta is not None and abs(delta) < RESPONSE_FLOOR
        ),
        "undefined": sorted(
            f"{lever}/{obs}"
            for lever, resp in levers.items()
            for obs, delta in resp.items()
            if delta is None
        ),
    }
    card["report_digest"] = hashlib.sha256(
        json.dumps(card, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return card


def card_lines(card: dict[str, Any]) -> str:
    """Render the response card for a prompt: measured, not asserted."""
    lines = []
    for lever, response in sorted(card.get("levers", {}).items()):
        moved = ", ".join(
            f"{obs} {delta:+.1%}"
            for obs, delta in sorted(response.items())
            if delta is not None and abs(delta) >= RESPONSE_FLOOR
        )
        undefined = [obs for obs, delta in sorted(response.items()) if delta is None]
        meaning = f" [{contrast_meaning(lever)}]" if is_contrast(lever) else ""
        reading = moved or ("inert (no observable moved)" if not undefined else "")
        if undefined:
            reading = "; ".join(filter(None, [reading, f"undefined after it: {', '.join(undefined)}"]))
        lines.append(f"- {lever}{meaning}: {reading}")
    return "\n".join(lines)


def scout_summary(card: dict[str, Any], lever: str = "") -> str:
    """A compact world-response quote for a redesign prompt.

    Quoting this is safe where treatment/control numbers are not: these
    are properties of the frozen world under runtime levers, computed
    with no reference to any claim's verdict.
    """
    parts = [f"forward simulation of world {card.get('world_id')}:"]
    parts.append(card_lines(card))
    chosen = card.get("levers", {}).get(lever)
    if lever and chosen is not None:
        strongest = max(((obs, delta) for obs, delta in chosen.items() if delta is not None),
                        key=lambda kv: abs(kv[1]), default=None)
        if strongest and abs(strongest[1]) < RESPONSE_FLOOR:
            parts.append(
                f"the lever you registered ({lever}) is measured inert in this"
                " world — choose a lever the simulation shows responsive"
            )
    return "\n".join(parts)


def materialize_placebo(
    fixture: WorldFixture, dest: Path, *, seed: int = 0
) -> WorldFixture:
    """Write a structure-destroying copy of the freeze. Same filenames.

    Schema-preserving: a graph stays a graph, a stream stays tokens.
    The scientific object is ruined (rewired, shuffled, column-permuted)
    so a script that only uses schema shape produces the same numbers.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(seed))
    family = schema_family(fixture.schema)
    for name in fixture.files:
        src = fixture.root / name
        out = dest / name
        if not src.is_file():
            continue
        if family == "undirected_graph" and name.endswith(".json"):
            payload = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and "edges" in payload:
                nodes = list(payload.get("nodes") or [])
                edges = list(payload.get("edges") or [])
                if nodes and edges:
                    payload["edges"] = [
                        [rng.choice(nodes), rng.choice(nodes)] for _ in edges
                    ]
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "text_stream" and name.endswith(".json"):
            payload = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("tokens"), list):
                tokens = list(payload["tokens"])
                rng.shuffle(tokens)
                payload["tokens"] = tokens
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "text_stream" and name.endswith(".txt"):
            words = src.read_text(encoding="utf-8").split()
            rng.shuffle(words)
            out.write_text(" ".join(words), encoding="utf-8")
            continue
        if family == "fasta" and name.endswith(".fasta"):
            lines = src.read_text(encoding="utf-8").splitlines()
            header = next((line for line in lines if line.startswith(">")), ">placebo")
            seq = list(
                "".join(line.strip() for line in lines if line and not line.startswith(">"))
            )
            rng.shuffle(seq)
            body = "".join(seq)
            wrapped = "\n".join(body[i : i + 70] for i in range(0, len(body), 70))
            out.write_text(header + "\n" + wrapped + "\n", encoding="utf-8")
            continue
        if family == "numeric_table" and name.endswith(".json"):
            payload = json.loads(src.read_text(encoding="utf-8"))
            rows = list(payload.get("rows") or []) if isinstance(payload, dict) else []
            if rows and isinstance(rows[0], dict):
                keys = [
                    key
                    for key, value in rows[0].items()
                    if isinstance(value, (int, float)) and not isinstance(value, bool)
                ]
                for key in keys:
                    column = [row.get(key) for row in rows]
                    rng.shuffle(column)
                    for row, value in zip(rows, column):
                        row[key] = value
                payload["rows"] = rows
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "labeled_traces" and name in {"world.json", "traces.json"}:
            payload = json.loads(src.read_text(encoding="utf-8"))
            rows = list(payload.get("traces") or []) if isinstance(payload, dict) else []
            labels = [str(row.get("label") or "") for row in rows if isinstance(row, dict)]
            rng.shuffle(labels)
            for row, label in zip(rows, labels):
                if isinstance(row, dict):
                    row["label"] = label
                    steps = row.get("steps")
                    if isinstance(steps, list):
                        rng.shuffle(steps)
                        row["steps"] = steps
            if isinstance(payload, dict):
                payload["traces"] = rows
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "program_state" and name == "world.json":
            # Destroy the structure, keep every marginal: each structural
            # field (which cells were written / read, whether a validator
            # was rewritten, whether the cell diverged, whether the gate
            # accepted, which episode) is permuted independently across
            # updates, so a stratum no longer lines up with its outcomes
            # and no dependency survives. A measure that reads the world
            # must move; one that only reads its own arm flag must not.
            # Without this branch the placebo was a byte-identical copy and
            # every program_state `supports` died as "did not consume".
            payload = json.loads(src.read_text(encoding="utf-8"))
            rows = list(payload.get("updates") or []) if isinstance(payload, dict) else []
            dict_rows = [row for row in rows if isinstance(row, dict)]
            for field in ("writes", "reads", "validator_writes", "divergent", "accepted", "episode", "sha256"):
                if not any(field in row for row in dict_rows):
                    continue
                values = [row.get(field) for row in dict_rows]
                rng.shuffle(values)
                for row, value in zip(dict_rows, values):
                    row[field] = value
            if isinstance(payload, dict):
                payload["updates"] = rows
                out.write_text(
                    json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
                )
                continue
        if family == "symbolic_trace" and name == "world.json":
            payload = json.loads(src.read_text(encoding="utf-8"))
            states = [
                str(item.get("id") if isinstance(item, dict) else item)
                for item in (payload.get("states") or [])
            ]
            for edge in payload.get("transitions") or []:
                if isinstance(edge, dict) and states:
                    edge["dst"] = rng.choice(states)
            out.write_text(
                json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            continue
        shutil.copy2(src, out)
    files = list(fixture.files)
    digest = digest_files(dest, files)
    return WorldFixture(
        id=f"{fixture.id}-placebo",
        title=f"placebo of {fixture.id}",
        source="placebo",
        retrieved_at=fixture.retrieved_at,
        digest=digest,
        files=tuple(files),
        root=dest,
        domains=fixture.domains,
        role="placebo",
        schema=fixture.schema,
        load_hint=fixture.load_hint,
    )
