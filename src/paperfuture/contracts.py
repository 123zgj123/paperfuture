"""JSON contracts shared by prompts, runtime validation and exported schemas.

These checks establish structural/provenance validity, not scientific truth.
"""
from __future__ import annotations

import math
from typing import Any


class ContractError(ValueError):
    pass


def string():
    return {"type": "string", "minLength": 1}


def enum(*values):
    return {"enum": list(values)}


def number(low, high, *, integer=False, nullable=False):
    result = {"type": "integer" if integer else "number", "minimum": low, "maximum": high}
    return {"anyOf": [result, {"type": "null"}]} if nullable else result


def array(item=None, minimum=0, maximum=10000):
    return {"type": "array", "items": item or string(), "minItems": minimum, "maxItems": maximum}


def obj(**fields):
    return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}


def validate(value: Any, schema: dict, path="$") -> None:
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate(value, option, path)
                return
            except ContractError:
                pass
        raise ContractError(f"{path}: no allowed type matched")
    if "enum" in schema:
        if value not in schema["enum"]:
            raise ContractError(f"{path}: expected one of {schema['enum']}")
        return
    kind = schema["type"]
    matches = {"object": isinstance(value, dict), "array": isinstance(value, list),
               "string": isinstance(value, str), "integer": type(value) is int,
               "number": type(value) in (float, int), "boolean": type(value) is bool,
               "null": value is None}
    if not matches[kind]:
        raise ContractError(f"{path}: expected {kind}")
    if kind == "object":
        fields = schema["properties"]
        if set(value) != set(fields):
            raise ContractError(f"{path}: missing={sorted(set(fields)-set(value))}, unexpected={sorted(set(value)-set(fields))}")
        for key, item in value.items():
            validate(item, fields[key], f"{path}.{key}")
    elif kind == "array":
        if not schema["minItems"] <= len(value) <= schema["maxItems"]:
            raise ContractError(f"{path}: invalid array length")
        for i, item in enumerate(value):
            validate(item, schema["items"], f"{path}[{i}]")
    elif kind == "string" and not value.strip():
        raise ContractError(f"{path}: empty string")
    elif kind in ("integer", "number"):
        if not math.isfinite(value) or not schema["minimum"] <= value <= schema["maximum"]:
            raise ContractError(f"{path}: outside [{schema['minimum']}, {schema['maximum']}]")


DELTA = obj(**{k: number(0, 2, integer=True, nullable=True) for k in
               ("problem", "assumption", "mechanism", "capability", "resource")})
EVIDENCE = array(string(), 1)
EXTRACTION = obj(relevance=number(0, 4, integer=True), quality=number(0, 4, integer=True),
    selection_reason=string(), problem=string(), mechanism=string(), assumptions=array(),
    capabilities=array(), limitations=array(), bottlenecks=array(), subtopics=array(string(), 1),
    evidence_ids=array(string(), 1, 8))
STAGE = obj(id=string(), label=string(), problem=string(), mechanism=string(), assumptions=array(),
    capabilities=array(), limitations=array(), bottlenecks=array(), paper_keys=array(string(), 1),
    evidence_ids=EVIDENCE)
EDGE = obj(id=string(), source=string(), target=string(),
    relation=enum("refine", "replace_mechanism", "relax_assumption", "change_objective", "transfer_setting", "combine", "limit_or_refute"),
    inherited=string(), changed=string(), bottleneck=string(), delta=DELTA,
    rationale=string(), confidence=enum("high", "medium", "low", "unknown"),
    source_witness=string(), target_witness=string(), evidence_ids=EVIDENCE)
GRAPH = obj(stages=array(STAGE, 2), edges=array(EDGE), coverage_gaps=array())
ASSUMPTION = obj(id=string(), statement=string(), origin=enum("literature", "hypothesis"), evidence_ids=array())
IDEA = obj(title=string(), problem=string(), mechanism=string(), difference=string(), minimal_test=string())
PROPOSAL = obj(source_stages=array(string(), 1), source_edges=array(), bottleneck=string(),
    historical_pattern=string(), operation=string(), target_delta=DELTA,
    target_weight=number(1, 4, integer=True), magnitude_reason=string(), evidence_ids=EVIDENCE,
    assumptions=array(ASSUMPTION), idea=IDEA)
SCENARIO = obj(kind=enum("success", "partial", "failure"), conditions=array(string(), 1),
    mechanism_chain=array(string(), 1), observable_predictions=array(string(), 1),
    competing_explanation=string(), next_bottleneck=string(), evidence_ids=array(), assumption_ids=array())
SIMULATION = obj(epistemic_status=enum("conditional_simulation_not_evidence"), scenarios=array(SCENARIO, 3, 3))
OBJECTION = obj(id=string(), claim=string(), kind=enum("evidenced_conflict", "unknown", "taste"),
    severity=number(0, 1), condition=string(), evidence_ids=array(), assumption_ids=array(), repair=string())
CRITIQUE = obj(preserved_novelty=string(), objections=array(OBJECTION), missed_opportunities=array())
VALUE_SCORE = obj(score=number(0, 4, integer=True, nullable=True), reason=string(), evidence_ids=array())
RESOLUTION = obj(objection_id=string(), disposition=enum("resolved", "unresolved", "not_applicable"),
    conditions_match={"type": "boolean"}, reason=string(), evidence_ids=array())
ASSESSMENT = obj(novelty=VALUE_SCORE, importance=VALUE_SCORE, potential_gain=VALUE_SCORE,
    feasibility=enum("testable", "missing_conditions", "substantive_obstacle"),
    feasibility_reason=string(), mechanism_clear={"type": "boolean"}, validation_path_clear={"type": "boolean"},
    resolutions=array(RESOLUTION), remaining_unknowns=array(), next_question=string(),
    suggested_action=enum("revise", "continue_leap", "switch"), suggested_change=string(),
    parent_round=number(1, 4, integer=True), preserve_assumption_ids=array(),
    mechanism_signature=string(), assumption_signature=string())
RERANK = obj(results=array(obj(paper_key=string(), relevance=number(0, 4, integer=True), reason=string()), 1))
SCHEMAS = {"extract": EXTRACTION, "rerank": RERANK, "graph": GRAPH, "propose": PROPOSAL,
           "simulate": SIMULATION, "critique": CRITIQUE, "assess": ASSESSMENT}


def require_refs(ids: list, available: dict | set, context: str, nonempty=False) -> None:
    if nonempty and not ids:
        raise ContractError(f"{context}: evidence is required")
    if len(ids) != len(set(ids)):
        raise ContractError(f"{context}: duplicate identifiers")
    missing = set(ids) - set(available)
    if missing:
        raise ContractError(f"{context}: unknown references {sorted(missing)}")
