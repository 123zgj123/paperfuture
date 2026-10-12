"""Evidence-grounded technical stages and ordinal directed transitions."""
from __future__ import annotations

from graphlib import CycleError, TopologicalSorter

from .contracts import ContractError, GRAPH, require_refs, validate
from .storage import digest, timestamp


def transition_weight(delta: dict) -> int | None:
    if any(x is None for x in delta.values()):
        return None
    if max(delta["mechanism"], delta["assumption"]) == 2:
        return 4 if delta["capability"] == 2 else 3
    return 2 if max(delta.values()) == 2 or sum(delta.values()) >= 3 else 1


def validate_graph(graph: dict, snapshot, selected: dict) -> None:
    validate(graph, GRAPH)
    stages = {n["id"]: n for n in graph["stages"]}
    if len(stages) != len(graph["stages"]):
        raise ContractError("duplicate stage IDs")
    keys = set(selected["selected_keys"])
    used = set()
    for node in stages.values():
        require_refs(node["paper_keys"], keys, "stage paper keys", nonempty=True)
        require_refs(node["evidence_ids"], snapshot.evidence, "stage evidence", nonempty=True)
        if any(snapshot.evidence[e]["paper_key"] not in node["paper_keys"] for e in node["evidence_ids"]):
            raise ContractError("stage evidence must come from its member papers")
        if set(node["paper_keys"]) - {snapshot.evidence[e]["paper_key"] for e in node["evidence_ids"]}:
            raise ContractError("each stage member paper must contribute at least one evidence span")
        used.update(node["paper_keys"])
    if used != keys:
        raise ContractError("every selected paper must be represented; explicitly report limitations without silently dropping papers")
    dependencies = {key: set() for key in stages}
    edge_ids, pairs = set(), set()
    for edge in graph["edges"]:
        if edge["id"] in edge_ids:
            raise ContractError("duplicate edge IDs")
        edge_ids.add(edge["id"])
        source, target = edge["source"], edge["target"]
        if source not in stages or target not in stages or source == target:
            raise ContractError("edge endpoints must be distinct known stages")
        if (source, target) in pairs:
            raise ContractError("duplicate directed stage pair; describe multiple changes on one edge")
        pairs.add((source, target))
        require_refs(edge["evidence_ids"], snapshot.evidence, "edge evidence", nonempty=True)
        a, b = edge["source_witness"], edge["target_witness"]
        if a not in stages[source]["evidence_ids"] or b not in stages[target]["evidence_ids"]:
            raise ContractError("edge witnesses must be evidence belonging to the corresponding stages")
        if not {a, b} <= set(edge["evidence_ids"]):
            raise ContractError("edge evidence must include both temporal witnesses")
        if any(snapshot.evidence[e]["paper_key"] not in keys for e in edge["evidence_ids"]):
            raise ContractError("edge evidence is outside the selected corpus")
        old = snapshot.get(snapshot.evidence[a]["paper_key"])
        new = snapshot.get(snapshot.evidence[b]["paper_key"])
        if timestamp(old["version_public_at"]) >= timestamp(new["version_public_at"]):
            raise ContractError("edge requires a strictly older source contribution witness")
        dependencies[target].add(source)
    try:
        tuple(TopologicalSorter(dependencies).static_order())
    except CycleError as exc:
        raise ContractError("technical development graph contains a cycle") from exc


def build_graph(snapshot, selected, config, gateway) -> dict:
    keys = selected["selected_keys"]
    evidence = {e: span for key in keys for e, span in snapshot.spans(key).items()
                if e in selected["extractions"][key]["evidence_ids"]}
    raw = gateway.structured("graph", {"question": config["question"], "cutoff": config["cutoff"],
        "papers": [{"paper_key": k, "published": snapshot.get(k)["version_public_at"],
                    "references_inside_snapshot": snapshot.references(k), "extraction": selected["extractions"][k]} for k in keys],
        "evidence": evidence, "selection_coverage_gaps": selected["coverage_gaps"]},
        "Build a directed ACYCLIC graph of technical stages, not years or a forced linear chain. Every selected paper must belong to at least one stage with its evidence. A stage describes a problem, mechanism, assumptions, demonstrated capabilities and bottlenecks. Branches and merges are allowed. Add an edge only when the target evidence supports a technical development relationship to the source; similarity alone is insufficient and citations cannot be invented. Source witness must be strictly older than target witness. Include both witnesses in edge evidence_ids. Do not manufacture edges to connect disconnected evidence. Describe exact inherited and changed elements. delta has five coordinates: 0=no change, 1=local change, 2=major change, null=not established. capability=2 requires demonstrated capability-boundary change, not speculation. Confidence is an ordinal evidence judgment. Abstract-only uncertainty must appear in limitations/coverage_gaps.",
        identity=f"graph:{snapshot.id}:{digest(selected)}", check=lambda value: validate_graph(value, snapshot, selected))
    weighted = {**raw, "edges": [{**e, "weight": transition_weight(e["delta"])} for e in raw["edges"]]}
    weighted.update(snapshot_id=snapshot.id, selected_hash=digest(selected),
                    weight_semantics="ordinal_technical_change_not_distance_or_success_probability",
                    evidence_status="LLM_extracted_and_structurally_validated_not_human_verified")
    weighted["graph_id"] = digest(weighted)
    return weighted


def graph_context(graph: dict, snapshot, selected: dict) -> dict:
    evidence_ids = {e for stage in graph["stages"] for e in stage["evidence_ids"]}
    evidence_ids.update(e for edge in graph["edges"] for e in edge["evidence_ids"])
    return {"snapshot_id": snapshot.id, "graph_id": graph["graph_id"], "cutoff": snapshot.payload["cutoff"],
            "complete_stage_graph": graph, "evidence": {e: snapshot.evidence[e] for e in sorted(evidence_ids)},
            "papers": [{"key": k, "title": snapshot.get(k)["title"], "date": snapshot.get(k)["version_public_at"],
                        "source_url": snapshot.get(k)["source_url"], "content_level": snapshot.get(k)["content_level"]}
                       for k in selected["selected_keys"]],
            "limitations": selected["limitations"] + graph["coverage_gaps"]}
