"""A dedicated frontier-model role for provisional technical understanding."""
from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from .inquiry import strings, text


TECHNICAL_MAP_SYSTEM = (
    "You reconstruct technical development to guide research reading, not to judge an idea. "
    "First sketch competing accounts of how this field developed, its current bottlenecks, "
    "failed routes and disputes. Then, when sources are supplied, revise that sketch against "
    "their exact passages. Separate changes in mechanism from changes in data, compute or "
    "evaluation. A citation or later date does not prove causal progress. Missing dates must "
    "remain missing. Preserve disagreements, negative results and unknowns. "
    "Return JSON {bottlenecks:[], disputes:[], failed_routes:[], open_questions:[], "
    "search_questions:[], revisions:[], transitions:[{earlier_source_id, later_source_id, "
    "mechanism_delta, preserved, assumptions:[], confounders:[], failure_conditions:[], "
    "passages:[{source_id, quote}]}]}. All plural fields other than transitions/passages are "
    "lists of strings. Before retrieval, use empty source IDs and passages and mark the "
    "transitions as hypotheses in mechanism_delta. After retrieval, cite only supplied IDs "
    "and exact verbatim quotes. Propose at most three focused search questions; they must "
    "clarify the user's problem, a candidate source mechanism, or a disputed assumption. "
    "You may conclude that no historical transition is established by the available sources."
)


def technical_map(client: Any, question: str, *, sources: Sequence[Mapping[str, Any]] = (),
                  previous: Mapping[str, Any] | None = None) -> dict[str, Any]:
    from ..llm import complete_call
    from .ideation import parse_json_object

    supplied = [{"source_id": str(s.get("source_id") or s.get("cite_id") or ""),
                 "title": text(s.get("title")), "text": str(s.get("text") or s.get("abstract") or "")[:12000],
                 "published": text(s.get("published"))} for s in sources[:8]]
    supplied = [s for s in supplied if s["source_id"] and s["text"]]
    lookup = {s["source_id"]: s for s in supplied}
    phase = "revise" if supplied else "sketch"
    reply = complete_call(client, json.dumps({"research_question": question, "sources": supplied,
                          "previous_map": dict(previous or {}), "phase": phase}, ensure_ascii=False),
                          purpose=f"research_technical_map_{phase}", system=TECHNICAL_MAP_SYSTEM)
    raw = parse_json_object(reply.text)
    transitions, gaps = [], []
    items = raw.get("transitions") or []
    for item in items[:8] if isinstance(items, list) else []:
        if not isinstance(item, Mapping):
            continue
        ids = [text(item.get(k)) for k in ("earlier_source_id", "later_source_id")]
        passages = []
        raw_passages = item.get("passages") or []
        for passage in raw_passages[:6] if isinstance(raw_passages, list) else []:
            if not isinstance(passage, Mapping):
                continue
            sid, quote = text(passage.get("source_id")), text(passage.get("quote"), 2400)
            if sid in lookup and quote and quote in " ".join(lookup[sid]["text"].split()):
                passages.append({"source_id": sid, "quote": quote})
            else:
                gaps.append({"source_id": sid, "reason": "quote_not_in_supplied_source"})
        referenced = len(set(ids)) == 2 and all(i in lookup for i in ids)
        dated = bool(referenced and all(lookup[i]["published"] for i in ids)
                     and lookup[ids[0]]["published"] < lookup[ids[1]]["published"])
        covered = set(ids) <= {p["source_id"] for p in passages}
        transitions.append({"earlier_source_id": ids[0] if ids[0] in lookup else "",
                            "later_source_id": ids[1] if ids[1] in lookup else "",
                            "unresolved_source_ids": [i for i in ids if i and i not in lookup],
                            "mechanism_delta": text(item.get("mechanism_delta")),
                            "preserved": text(item.get("preserved")),
                            **{k: strings(item.get(k)) for k in ("assumptions", "confounders", "failure_conditions")},
                            "passages": passages,
                            "status": "source_linked_interpretation" if dated and covered else "unverified_hypothesis",
                            "causal_attribution": "unverified", "epistemic": "GENERATED"})
    return {"version": "technical-map-v1", "phase": phase,
            **{k: strings(raw.get(k), 3 if k == "search_questions" else 12)
               for k in ("bottlenecks", "disputes", "failed_routes", "open_questions", "search_questions", "revisions")},
            "transitions": transitions, "source_ids": list(lookup), "reference_gaps": gaps,
            "model": str(getattr(reply, "model", "")),
            "completion_digest": str(getattr(reply, "digest", "")),
            "epistemic": "GENERATED", "scientific_validation": False}


def can_spend(client: Any, calls: int) -> bool:
    ledger = getattr(client, "ledger", None)
    remaining = ledger.remaining().get("api_calls") if callable(getattr(ledger, "remaining", None)) else 0
    return isinstance(remaining, (int, float)) and not isinstance(remaining, bool) and remaining >= calls
